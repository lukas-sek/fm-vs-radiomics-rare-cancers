"""LoRA adapters for Primus/EVA transformer blocks (pure PyTorch).

Used as ``LoRA.__init__(model, ...)`` from the PEFT registry (``self`` is the
full classification model). Adapters are injected only into the selected
block indices; the classification head stays trainable.

Does **not** require the HuggingFace ``peft`` package (avoids fragile peft /
transformers version coupling on aimlab).
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HEAD_SUBSTRINGS = ("cls_head", "classifier", "head")
_DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "proj",
    "fc1_g",
    "fc1_x",
    "fc2",
)


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
    return blocks


def _is_head_param(name: str) -> bool:
    return any(sub in name for sub in _HEAD_SUBSTRINGS)


def _leaf_target_names(target_modules: Sequence[str]) -> list[str]:
    """Normalize 'attn.q_proj' → 'q_proj' for leaf matching."""
    leaves: list[str] = []
    for name in target_modules:
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaves:
            leaves.append(leaf)
    return leaves


def _trainable_param_stats(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update (α/r)·B A x."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base).__name__}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        # A: (r, in), B: (out, r) — B starts at 0 so forward is unchanged at init.
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        dx = self.lora_dropout(x)
        # x @ A^T @ B^T  ==  (x @ A.T) @ B.T
        result = result + (F.linear(F.linear(dx, self.lora_A), self.lora_B) * self.scaling)
        return result


def _replace_linears_with_lora(
    module: nn.Module,
    target_leaves: Sequence[str],
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Replace matching child Linears in-place. Returns number replaced."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name in target_leaves:
            setattr(
                module,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced += 1
        else:
            replaced += _replace_linears_with_lora(
                child,
                target_leaves,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
    return replaced


class LoRA:
    """Inject pure-PyTorch LoRA into selected Primus blocks; train head + adapters."""

    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.01,
        lora_target_modules: Iterable[str] | None = None,
        lora_layers: Sequence[int] | None = None,
        use_dora: bool = False,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
        lora_bias: str = "none",
    ) -> None:
        # Registry calls LoRA.__init__(model, ...), so ``self`` is the model.
        model: nn.Module = self  # type: ignore[assignment]

        if use_dora:
            raise ValueError("use_dora=True is not supported by the pure-PyTorch LoRA path.")
        if use_rslora:
            raise ValueError("use_rslora=True is not supported by the pure-PyTorch LoRA path.")
        if lora_bias not in ("none", None):
            raise ValueError(
                f"lora_bias={lora_bias!r} is not supported; use lora_bias='none'."
            )
        del init_lora_weights  # always zero-init B as above

        target_modules = list(lora_target_modules or _DEFAULT_TARGET_MODULES)
        target_leaves = _leaf_target_names(target_modules)
        layers = list(lora_layers) if lora_layers is not None else [39]

        blocks = get_transformer_blocks(model)
        n_blocks = len(blocks)
        for idx in layers:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(
                    f"lora_layers index {idx} out of range for {n_blocks} transformer blocks."
                )

        for param in model.parameters():
            param.requires_grad = False

        matched_total = 0
        for idx in layers:
            n_match = _replace_linears_with_lora(
                blocks[idx],
                target_leaves,
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
            )
            if n_match == 0:
                raise RuntimeError(
                    f"No Linear modules matching {target_leaves} in transformer "
                    f"block {idx}. Inspect block named_modules() and adjust "
                    f"--peft.lora_target_modules."
                )
            matched_total += n_match

        for name, param in model.named_parameters():
            if _is_head_param(name) or "lora_A" in name or "lora_B" in name:
                param.requires_grad = True

        trainable, total = _trainable_param_stats(model)
        pct = 100.0 * trainable / total if total else 0.0
        msg = (
            f"[lora] layers={layers} rank={lora_rank} alpha={lora_alpha} "
            f"dropout={lora_dropout} matched_linears={matched_total} "
            f"trainable={trainable:,}/{total:,} ({pct:.3f}%)"
        )
        logger.info(msg)
        print(msg)
