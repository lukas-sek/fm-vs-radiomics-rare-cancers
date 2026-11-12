"""Capped gradual unfreeze + LLRD + EMA/raw early stopping for Primus/ViT (40 blocks)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


_HEAD_SUBSTRINGS = ("cls_head", "classifier", "head")
_NUM_BLOCKS = 40
_LAST_BLOCK_IDX = _NUM_BLOCKS - 1  # 39

def stage_for_epoch(epoch: int) -> int:
    # Epochs 0–9: head only.
    # Epochs 10–19: unfreeze last block (39).
    # Epochs 20+: also unfreeze second-last block (38) at lower LLRD LR.
    if epoch < 10:
        return 0
    if epoch < 20:
        return 1
    return 2


def base_lr_for_stage(stage: int) -> float:
    return 1e-4 if stage == 0 else 5e-5


def blocks_for_stage(stage: int) -> list[int]:
    if stage <= 0:
        return []
    if stage == 1:
        return [_LAST_BLOCK_IDX]
    return [_LAST_BLOCK_IDX - 1, _LAST_BLOCK_IDX]


def _resolve_backbone(model: nn.Module) -> nn.Module:
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "model"):
        return encoder.model
    backbone = getattr(model, "model", None)
    if backbone is not None:
        return backbone
    return model


def get_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    backbone = _resolve_backbone(model)
    eva = getattr(backbone, "eva", None)
    blocks = None
    if eva is not None:
        blocks = getattr(eva, "blocks", None) or getattr(eva, "layers", None)
    if blocks is None:
        blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError(
            f"Could not find transformer blocks on backbone type {type(backbone).__name__}."
        )
    if len(blocks) != _NUM_BLOCKS:
        raise ValueError(
            f"Expected {_NUM_BLOCKS} transformer blocks, found {len(blocks)}."
        )
    return blocks


def _is_head_param(name: str) -> bool:
    return any(sub in name for sub in _HEAD_SUBSTRINGS)


class SmoothedEarlyStopping:
    """EMA + raw validation AUROC early stopping with best-weight caching (higher is better).

    Stops when either EMA AUROC plateaus for ``patience`` epochs or raw val AUROC
    plateaus for ``raw_patience`` epochs. Best weights are still those with best EMA.
    """

    def __init__(self, patience: int = 7, alpha: float = 0.7, raw_patience: int = 7):
        self.patience = int(patience)
        self.raw_patience = int(raw_patience)
        self.alpha = float(alpha)
        self.ema_val: float | None = None
        self.best_ema: float | None = None
        self.bad_epochs = 0
        self.best_raw: float | None = None
        self.raw_bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_epoch: int | None = None
        self.stop_reason: str | None = None

    def step(self, model: nn.Module, val_auroc: float, epoch: int) -> bool:
        """Update EMA and raw AUROC trackers. Returns True if training should stop."""
        raw = float(val_auroc)
        if self.ema_val is None:
            self.ema_val = raw
        else:
            self.ema_val = self.alpha * self.ema_val + (1.0 - self.alpha) * raw

        improved = self.best_ema is None or self.ema_val > self.best_ema
        if improved:
            self.best_ema = self.ema_val
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1

        raw_improved = self.best_raw is None or raw > self.best_raw
        if raw_improved:
            self.best_raw = raw
            self.raw_bad_epochs = 0
        else:
            self.raw_bad_epochs += 1

        ema_stop = self.bad_epochs >= self.patience
        raw_stop = self.raw_bad_epochs >= self.raw_patience
        if ema_stop and raw_stop:
            self.stop_reason = "ema+raw"
        elif ema_stop:
            self.stop_reason = "ema"
        elif raw_stop:
            self.stop_reason = "raw"
        else:
            self.stop_reason = None
        return ema_stop or raw_stop

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is None:
            return
        model.load_state_dict(self.best_state)


def build_unfreeze_optimizer(
    model: nn.Module,
    stage: int,
    base_lr: float | None = None,
    decay_factor: float = 0.75,
    weight_decay: float = 0.01,
) -> AdamW:
    """Freeze/unfreeze by stage and return a fresh AdamW with LLRD param groups."""
    stage = int(stage)
    if base_lr is None:
        base_lr = base_lr_for_stage(stage)
    active_blocks = blocks_for_stage(stage)
    blocks = get_transformer_blocks(model)

    for p in model.parameters():
        p.requires_grad = False

    head_module = getattr(model, "cls_head", None) or getattr(model, "head", None)
    head_params: list[nn.Parameter] = []
    if head_module is not None:
        for param in head_module.parameters():
            param.requires_grad = True
            head_params.append(param)
    else:
        for name, param in model.named_parameters():
            if _is_head_param(name):
                param.requires_grad = True
                head_params.append(param)

    block_param_lists: dict[int, list[nn.Parameter]] = {i: [] for i in active_blocks}
    for idx in active_blocks:
        for param in blocks[idx].parameters():
            param.requires_grad = True
            block_param_lists[idx].append(param)

    groups: list[dict[str, Any]] = []
    if head_params:
        groups.append(
            {"params": head_params, "lr": float(base_lr), "weight_decay": weight_decay, "name": "head"}
        )

    for idx in sorted(active_blocks):
        params = block_param_lists[idx]
        if not params:
            continue
        # LR(layer_l) = base_lr * (ξ ** (39 - l + 1))
        lr = float(base_lr) * (float(decay_factor) ** (_LAST_BLOCK_IDX - idx + 1))
        groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": f"block_{idx}",
            }
        )

    if not groups:
        raise RuntimeError("No trainable parameter groups; check head naming on the model.")

    return AdamW(groups, lr=float(base_lr), weight_decay=weight_decay)


def _amp_dtype(precision: str) -> torch.dtype | None:
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return None


@torch.no_grad()
def _eval_val(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Return (mean val loss, AUROC). AUROC uses P(class=1) for binary, OVR macro otherwise."""
    model.eval()
    total = 0.0
    n = 0
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    use_amp = amp_dtype is not None and device.type == "cuda"
    for batch in val_loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Expected (x, y) batches from val_loader.")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long().view(-1)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(x, target=y)
            if logits.ndim == 1 or (logits.ndim > 1 and logits.shape[-1] == 1):
                raise ValueError(
                    "Expected multiclass logits [B, C] for CrossEntropyLoss; "
                    f"got shape {tuple(logits.shape)}."
                )
            loss = criterion(logits, y)
        total += float(loss.detach()) * max(1, int(y.shape[0]))
        n += max(1, int(y.shape[0]))
        probs = F.softmax(logits.float(), dim=-1)
        all_labels.append(y.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    val_loss = total / max(1, n)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    try:
        if probs.shape[1] == 2:
            auroc = float(roc_auc_score(labels, probs[:, 1]))
        else:
            auroc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")
    return val_loss, auroc


def _trainable_param_stats(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


def _save_weights(
    path: Path,
    model: nn.Module,
    *,
    epoch: int,
    stage: int,
    ema_val_auroc: float | None,
    val_auroc: float | None,
    val_loss: float | None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = _trainable_param_stats(model)
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "ema_val_auroc": ema_val_auroc,
            "val_auroc": val_auroc,
            "val_loss": val_loss,
            "model": model.state_dict(),
            **stats,
        },
        path,
    )
    if path.name == "best.pt":
        sidecar = {
            **stats,
            "epoch": int(epoch),
            "stage": int(stage),
            "ema_val_auroc": ema_val_auroc,
            "val_auroc": val_auroc,
            "val_loss": val_loss,
        }
        (path.parent / "best_trainable_params.json").write_text(
            json.dumps(sidecar, indent=2) + "\n"
        )


def train_pipeline(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int = 100,
    device: str | torch.device = "cuda",
    *,
    ckpt_dir: Path | str | None = None,
    decay_factor: float = 0.75,
    ema_alpha: float = 0.7,
    patience: int = 7,
    raw_patience: int = 7,
    max_norm: float = 1.0,
    label_smoothing: float = 0.1,
    precision: str = "fp16",
    weight_decay: float = 0.05,
) -> nn.Module:
    """Run capped unfreeze + LLRD + EMA/raw early stopping. Restores best EMA weights."""
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    stopper = SmoothedEarlyStopping(
        patience=patience, alpha=ema_alpha, raw_patience=raw_patience
    )
    ckpt_dir_path = Path(ckpt_dir) if ckpt_dir is not None else None
    # Honor training.precision (same role as Accelerate mixed_precision in Trainer).
    amp_dtype = _amp_dtype(precision) if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    print(
        f"[capped_unfreeze] device={device} precision={precision} amp={use_amp} "
        f"weight_decay={weight_decay:g}"
    )

    current_stage = -1
    optimizer: AdamW | None = None

    for epoch in range(max_epochs):
        stage = stage_for_epoch(epoch)
        if stage != current_stage:
            current_stage = stage
            base_lr = base_lr_for_stage(stage)
            optimizer = build_unfreeze_optimizer(
                model,
                stage,
                base_lr=base_lr,
                decay_factor=decay_factor,
                weight_decay=weight_decay,
            )
            active = blocks_for_stage(stage)
            group_lrs = {
                g.get("name", i): g["lr"] for i, g in enumerate(optimizer.param_groups)
            }
            print(
                f"[stage {stage}] epoch={epoch} base_lr={base_lr:g} "
                f"blocks={active} lrs={group_lrs}"
            )

        assert optimizer is not None
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("Expected (x, y) batches from train_loader.")
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long().view(-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x, target=y)
                if logits.ndim == 1 or (logits.ndim > 1 and logits.shape[-1] == 1):
                    raise ValueError(
                        "Expected multiclass logits [B, C] for CrossEntropyLoss; "
                        f"got shape {tuple(logits.shape)}."
                    )
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=max_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.detach()) * max(1, int(y.shape[0]))
            train_n += max(1, int(y.shape[0]))

        val_loss, val_auroc = _eval_val(
            model, val_loader, criterion, device, amp_dtype=amp_dtype
        )
        if val_auroc != val_auroc:  # NaN (e.g. single-class val batch epoch)
            print(
                f"epoch {epoch:03d}/{max_epochs - 1} stage={current_stage} "
                f"val_AUROC=nan — skipping EMA update"
            )
            should_stop = False
        else:
            should_stop = stopper.step(model, val_auroc, epoch)
        train_loss = train_loss_sum / max(1, train_n)

        ema_s = f"{stopper.ema_val:.4f}" if stopper.ema_val is not None else "nan"
        best_s = f"{stopper.best_ema:.4f}" if stopper.best_ema is not None else "nan"
        auroc_s = f"{val_auroc:.4f}" if val_auroc == val_auroc else "nan"
        best_raw_s = f"{stopper.best_raw:.4f}" if stopper.best_raw is not None else "nan"
        print(
            f"epoch {epoch:03d}/{max_epochs - 1} stage={current_stage} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_AUROC={auroc_s} best_val_AUROC={best_raw_s} "
            f"ema_AUROC={ema_s} best_ema_AUROC={best_s} "
            f"bad_ema={stopper.bad_epochs}/{patience} "
            f"bad_raw={stopper.raw_bad_epochs}/{raw_patience}"
        )

        if ckpt_dir_path is not None:
            _save_weights(
                ckpt_dir_path / "last.pt",
                model,
                epoch=epoch,
                stage=current_stage,
                ema_val_auroc=stopper.ema_val,
                val_auroc=val_auroc,
                val_loss=val_loss,
            )
            if stopper.bad_epochs == 0 and stopper.best_state is not None:
                _save_weights(
                    ckpt_dir_path / "best.pt",
                    model,
                    epoch=epoch,
                    stage=current_stage,
                    ema_val_auroc=stopper.best_ema,
                    val_auroc=val_auroc,
                    val_loss=val_loss,
                )

        if should_stop:
            reason = stopper.stop_reason or "unknown"
            print(
                f"Early stop at epoch {epoch} ({reason}): "
                f"ema_bad={stopper.bad_epochs}/{patience}, "
                f"raw_bad={stopper.raw_bad_epochs}/{raw_patience} "
                f"(best_epoch={stopper.best_epoch}, best_ema_AUROC={stopper.best_ema:.4f}, "
                f"best_val_AUROC={stopper.best_raw:.4f})."
            )
            break

    stopper.restore_best(model)
    if ckpt_dir_path is not None and stopper.best_state is not None:
        _save_weights(
            ckpt_dir_path / "best.pt",
            model,
            epoch=stopper.best_epoch if stopper.best_epoch is not None else -1,
            stage=stage_for_epoch(stopper.best_epoch or 0),
            ema_val_auroc=stopper.best_ema,
            val_auroc=stopper.best_ema,
            val_loss=None,
        )
    return model
