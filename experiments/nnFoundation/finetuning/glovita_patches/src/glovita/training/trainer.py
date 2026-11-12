"""Main training loop built on HuggingFace Accelerate.

Design principles:
  - Explicit over magic: every step of the training loop is visible here.
  - All distributed / mixed-precision / gradient-accumulation concerns are
    handled by Accelerate; the rest of the code is plain PyTorch.
  - Experiment logging goes through a small backend-neutral abstraction.
  - NaN detection replaces the old NaNLossCallback.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from itertools import islice
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from torchmetrics import MetricCollection

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:
    Progress = None

from glovita.augmentation.mixup import mixup_criterion, mixup_data
from glovita.metrics.balanced_accuracy import BalancedAccuracy
from glovita.metrics.conf_mat import ConfusionMatrix
from glovita.models.peft.gps import maybe_apply_gps
from glovita.models.peft.gradual_unfreeze import get_freeze_state, update_for_epoch
from glovita.configs.data import DataConfig
from glovita.configs.peft import GradualUnfreezeConfig, PeftConfig
from glovita.configs.task import TaskConfig
from glovita.configs.training import TrainingConfig
from glovita.logging.base import ExperimentLogger


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

def build_metrics(task_config: TaskConfig, data_config: DataConfig, prefix: str) -> MetricCollection:
    """Build a ``MetricCollection`` from the task + data configuration."""
    from torchmetrics import (
        AUROC, Accuracy, AveragePrecision, F1Score,
        MeanAbsoluteError, MeanSquaredError, Precision, Recall,
    )

    num_classes = data_config.num_classes
    task = data_config.task
    subtask = data_config.subtask
    metric_task = subtask  # "multiclass" | "multilabel"
    metrics_dict = {}
    # Inputs are already gathered via `gather_for_metrics`; torchmetrics must not sync them again.
    sync_kwargs = {"sync_on_compute": False}

    if task == "Classification":
        if "acc" in task_config.metrics:
            metrics_dict["Accuracy"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes, **sync_kwargs
            )
        if "balanced_acc" in task_config.metrics:
            if metric_task == "multilabel":
                metrics_dict["Balanced_Accuracy"] = BalancedAccuracy(
                    num_classes=num_classes, task="multilabel", **sync_kwargs,
                )
            else:
                metrics_dict["Balanced_Accuracy"] = Accuracy(
                    task=metric_task, num_classes=num_classes, num_labels=num_classes,
                    average="macro", **sync_kwargs,
                )
        if "f1" in task_config.metrics:
            metrics_dict["F1"] = F1Score(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro", **sync_kwargs,
            )
        if "f1_per_class" in task_config.metrics:
            metrics_dict["F1_per_class"] = F1Score(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average=None, **sync_kwargs,
            )
        if "pr" in task_config.metrics:
            metrics_dict["Precision"] = Precision(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro", **sync_kwargs,
            )
            metrics_dict["Recall"] = Recall(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro", **sync_kwargs,
            )
        if "top5acc" in task_config.metrics:
            metrics_dict["Accuracy_top5"] = Accuracy(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                top_k=5, **sync_kwargs,
            )
        if "auroc" in task_config.metrics:
            metrics_dict["AUROC"] = AUROC(
                task=metric_task, num_classes=num_classes, num_labels=num_classes,
                average="macro", **sync_kwargs,
            )
        if "ap" in task_config.metrics:
            metrics_dict["AP"] = AveragePrecision(
                task=metric_task, num_classes=num_classes, num_labels=num_classes, **sync_kwargs,
            )

    elif task == "Regression":
        if "mse" in task_config.metrics:
            metrics_dict["MSE"] = MeanSquaredError(**sync_kwargs)
        if "mae" in task_config.metrics:
            metrics_dict["MAE"] = MeanAbsoluteError(**sync_kwargs)

    return MetricCollection(metrics_dict, prefix=prefix)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the full training lifecycle for a single fold.

    Usage::

        trainer = Trainer(training_cfg, task_cfg, data_cfg, optimizer_cfg)
        model = trainer.fit(model, optimizer, scheduler, train_loader, val_loader)
    """

    NAN_PATIENCE = 3  # consecutive NaN/Inf batches before aborting

    def __init__(
        self,
        training_config: TrainingConfig,
        task_config: TaskConfig,
        data_config: DataConfig,
        log_dir: Path,
        logger: ExperimentLogger,
        peft_config: PeftConfig | None = None,
    ):
        self.cfg = training_config
        self.task = task_config
        self.data = data_config
        self.log_dir = log_dir
        self.logger = logger
        self.peft_config = peft_config
        self._gradual_unfreeze = isinstance(peft_config, GradualUnfreezeConfig)
        self.metrics_path = log_dir / "metrics.json"
        self._metrics_history: dict[str, list[dict[str, float | int]]] = {
            "train": [],
            "val": [],
        }

        kwargs_handlers = []
        # Gradual unfreeze leaves some params without grads until unlocked.
        find_unused = training_config.ddp_find_unused_parameters or self._gradual_unfreeze
        if find_unused:
            kwargs_handlers.append(DistributedDataParallelKwargs(find_unused_parameters=True))
        self.accelerator = Accelerator(
            mixed_precision=training_config.precision if training_config.precision != "no" else "no",
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            kwargs_handlers=kwargs_handlers,
        )

        # Criterion
        subtask = data_config.subtask
        task = data_config.task
        if task == "Classification":
            if subtask == "multiclass":
                self.criterion = nn.CrossEntropyLoss(
                    label_smoothing=task_config.label_smoothing
                )
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        # Metrics (moved to device later in fit())
        self.train_metrics = build_metrics(task_config, data_config, prefix="train_")
        self.val_metrics = build_metrics(task_config, data_config, prefix="val_")

        # Optional confusion matrix / scatter tracking
        self.val_conf_mat: Optional[ConfusionMatrix] = None
        self.train_conf_mat: Optional[ConfusionMatrix] = None
        if task == "Classification":
            if task_config.result_plot in ("val", "all"):
                self.val_conf_mat = ConfusionMatrix(num_classes=data_config.num_classes)
            if task_config.result_plot == "all":
                self.train_conf_mat = ConfusionMatrix(num_classes=data_config.num_classes)

        # Regression scatter state
        self.val_preds: list = []
        self.val_labels: list = []
        self.train_preds: list = []
        self.train_labels: list = []

        # Best-checkpoint tracking: choose a logged validation metric key
        # explicitly when provided, otherwise fall back to the first logged one.
        self._best_metric_key = training_config.best_checkpoint_metric
        self._best_val_metric: float | None = None
        self._epochs_without_improvement = 0
        # Keep the top-K validation checkpoints (plus last.pt), not every epoch.
        self._top_k_checkpoints = 3
        self._top_checkpoints: list[dict] = []  # {epoch, metric, path}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        val_loader,
    ) -> nn.Module:
        """Train *model* and return the trained model."""

        # Compile before prepare so Accelerate wraps the compiled model
        if self.cfg.compile:
            model = torch.compile(model, mode="default")

        # Accelerate takes ownership of distribution / device placement.
        # The scheduler is not prepared: a prepared scheduler steps num_processes times per call.
        model, optimizer, train_loader, val_loader = (
            self.accelerator.prepare(model, optimizer, train_loader, val_loader)
        )

        maybe_apply_gps(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            loss_fn=self._loss_for_calibration,
            accelerator=self.accelerator,
        )

        # Move metrics to the right device
        device = self.accelerator.device
        self.train_metrics = self.train_metrics.to(device)
        self.val_metrics = self.val_metrics.to(device)
        if self.val_conf_mat is not None:
            self.val_conf_mat = self.val_conf_mat.to(device)
        if self.train_conf_mat is not None:
            self.train_conf_mat = self.train_conf_mat.to(device)

        # Optional sanity val
        if self.cfg.num_sanity_val_steps > 0:
            self._val_epoch(
                model,
                optimizer,
                val_loader,
                epoch=-1,
                sanity_steps=self.cfg.num_sanity_val_steps,
            )

        for epoch in range(self.cfg.epochs):
            self._maybe_update_gradual_unfreeze(model, epoch)
            self._train_epoch(model, optimizer, train_loader, epoch)
            improved = self._val_epoch(model, optimizer, val_loader, epoch)
            if scheduler is not None:
                scheduler.step()
            if self.cfg.enable_checkpointing and self.accelerator.is_main_process:
                self._save_checkpoint(model, optimizer, epoch, is_best=False)

            if self.cfg.early_stopping_patience is not None:
                min_epochs = int(getattr(self.cfg, "early_stopping_min_epochs", 0) or 0)
                # Warm up: keep updating best.pt, but do not start the patience
                # counter until min_epochs have completed.
                if epoch + 1 < min_epochs:
                    self._epochs_without_improvement = 0
                elif improved:
                    self._epochs_without_improvement = 0
                elif improved is False:
                    self._epochs_without_improvement += 1
                if (
                    epoch + 1 >= min_epochs
                    and self._epochs_without_improvement
                    >= self.cfg.early_stopping_patience
                ):
                    self.accelerator.print(
                        f"Early stopping at epoch {epoch + 1}: no improvement for "
                        f"{self.cfg.early_stopping_patience} epochs after "
                        f"min_epochs={min_epochs} "
                        f"(metric={self._best_metric_key!r})."
                    )
                    break

        if self.cfg.val_steps_per_epoch is not None:
            self._final_val(model, optimizer, val_loader)

        return self.accelerator.unwrap_model(model)

    # ------------------------------------------------------------------
    # Internal loop methods
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        epoch: int,
    ) -> None:
        model.train()
        total_loss = 0.0
        nan_count = 0
        num_batches = 0
        mode = self.task.metric_computation_mode

        max_batches = None
        if self.cfg.train_steps_per_epoch is not None:
            max_batches = self.cfg.train_steps_per_epoch * self.cfg.gradient_accumulation_steps
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)

        pbar = self._make_progress(
            loader,
            f"Epoch {epoch + 1}/{self.cfg.epochs} [train]",
            rich_style="cyan",
            split="train",
            max_steps=max_batches,
        )

        for x, y in pbar:
            with self.accelerator.accumulate(model):
                if self.task.mixup:
                    if not torch.is_tensor(x):
                        raise ValueError("Mixup currently only supports tensor inputs, not MIL bag dictionaries.")
                    x_mix, targets_a, targets_b, lam = mixup_data(x, y, alpha=self.task.mixup_alpha)
                    y_hat = model(x_mix)
                    loss = mixup_criterion(self.criterion, y_hat, targets_a, targets_b, lam)
                else:
                    y_hat = model(x, target=y)
                    if self.data.num_classes == 1:
                        y_hat = y_hat.view(-1)
                    target = y.float() if self.data.subtask == "multilabel" else y
                    loss = self.criterion(y_hat, target)
                    aux_loss = self._get_model_aux_loss(model)
                    if aux_loss is not None:
                        loss = loss + aux_loss

                # NaN / Inf detection (replaces NaNLossCallback)
                loss_scalar = float(loss.detach())
                if math.isnan(loss_scalar) or math.isinf(loss_scalar):
                    nan_count += 1
                    if nan_count >= self.NAN_PATIENCE:
                        raise RuntimeError(
                            f"Loss has been NaN/Inf for {self.NAN_PATIENCE} consecutive "
                            "batches — aborting training."
                        )
                else:
                    nan_count = 0

                self.accelerator.backward(loss)
                if self.cfg.gradient_clip_val is not None:
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), self.cfg.gradient_clip_val
                    )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.detach().float()

            gathered_y_hat, gathered_y = self._gather_for_metrics(y_hat, y)
            if mode == "stepwise":
                self._update_metrics(
                    self.train_metrics,
                    gathered_y_hat,
                    gathered_y,
                    self.train_conf_mat,
                )
                if self.task.result_plot == "all" and self.data.task == "Regression":
                    self.train_preds.extend(gathered_y_hat.detach().cpu().tolist())
                    self.train_labels.extend(gathered_y.detach().cpu().tolist())
            else:
                self.train_metrics.update(gathered_y_hat, gathered_y)

            pbar.set_postfix(**self._progress_postfix(loss, self.train_metrics, split="train"))

            num_batches += 1

        avg_loss = (total_loss / num_batches).item()

        if self.accelerator.is_main_process:
            log_dict = {"train_loss": avg_loss, "epoch": epoch}
            if mode == "epochwise":
                log_dict.update(self._compute_and_reset(self.train_metrics))
                if self.train_conf_mat is not None:
                    self.train_conf_mat.log_to_logger("train", self.logger, step=epoch)
                    self.train_conf_mat.reset()
                if self.data.task == "Regression" and self.train_preds:
                    self._log_scatter("Train", self.train_labels, self.train_preds, step=epoch)
                    self.train_preds, self.train_labels = [], []
            else:
                # stepwise: metrics already computed per-step; log epoch summary
                log_dict.update(self._compute_and_reset(self.train_metrics))
                if self.train_conf_mat is not None:
                    self.train_conf_mat.log_to_logger("train", self.logger, step=epoch)
                    self.train_conf_mat.reset()
            self.logger.log_metrics(log_dict, step=epoch)
            self._append_local_metrics("train", log_dict)
        else:
            self.train_metrics.reset()

    def _val_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader,
        epoch: int,
        sanity_steps: int = 0,
        final: bool = False,
    ) -> bool | None:
        """Run validation. Returns True/False if a best-metric comparison was made, else None."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        mode = self.task.metric_computation_mode
        max_batches = None if final else self.cfg.val_steps_per_epoch
        improved: bool | None = None

        with torch.no_grad():
            pbar = self._make_progress(
                loader,
                "Final [val]" if final else f"Epoch {epoch + 1}/{self.cfg.epochs} [val]",
                rich_style="magenta",
                split="val",
                max_steps=max_batches,
            )

            for step, (x, y) in enumerate(pbar):
                if sanity_steps > 0 and step >= sanity_steps:
                    break

                y_hat = model(x, target=y)
                if self.data.num_classes == 1:
                    y_hat = y_hat.view(-1)
                target = y.float() if self.data.subtask == "multilabel" else y
                loss = self.criterion(y_hat, target)
                aux_loss = self._get_model_aux_loss(model)
                if aux_loss is not None:
                    loss = loss + aux_loss
                total_loss += loss.detach().float()

                gathered_y_hat, gathered_y = self._gather_for_metrics(y_hat, y)
                if mode == "stepwise":
                    self._update_metrics(
                        self.val_metrics,
                        gathered_y_hat,
                        gathered_y,
                        self.val_conf_mat,
                    )
                    if self.task.result_plot in ("val", "all") and self.data.task == "Regression":
                        self.val_preds.extend(gathered_y_hat.detach().cpu().tolist())
                        self.val_labels.extend(gathered_y.detach().cpu().tolist())
                else:
                    self.val_metrics.update(gathered_y_hat, gathered_y)
                    if self.val_conf_mat is not None:
                        self.val_conf_mat.update(gathered_y_hat, gathered_y)

                pbar.set_postfix(**self._progress_postfix(loss, self.val_metrics, split="val"))

                num_batches += 1

        if sanity_steps > 0:
            self.val_metrics.reset()
            return None

        avg_loss = (total_loss / num_batches).item()

        if final:
            if self.accelerator.is_main_process:
                log_dict = {"val_loss": avg_loss, **self._compute_and_reset(self.val_metrics)}
                log_dict = {f"final_{key}": value for key, value in log_dict.items()}
                self.logger.log_metrics(log_dict, step=epoch)
                (self.log_dir / "final_val_metrics.json").write_text(json.dumps(log_dict, indent=2))
            else:
                self.val_metrics.reset()
            if self.val_conf_mat is not None:
                self.val_conf_mat.reset()
            return None

        if self.accelerator.is_main_process:
            log_dict = {"val_loss": avg_loss, "epoch": epoch}
            metric_values = self._compute_and_reset(self.val_metrics)
            log_dict.update(metric_values)
            if self.val_conf_mat is not None:
                self.val_conf_mat.log_to_logger("val", self.logger, step=epoch)
                self.val_conf_mat.reset()
            if self.data.task == "Regression" and self.val_preds:
                self._log_scatter("Val", self.val_labels, self.val_preds, step=epoch)
                self.val_preds, self.val_labels = [], []
            self.logger.log_metrics(log_dict, step=epoch)
            self._append_local_metrics("val", log_dict)

            # Maintain top-K checkpoints by the selected validation metric.
            if self.cfg.enable_checkpointing and metric_values:
                metric_key = self._best_metric_key or next(iter(metric_values))
                if metric_key not in metric_values:
                    available = ", ".join(metric_values.keys())
                    raise KeyError(
                        f"training.best_checkpoint_metric={metric_key!r} was not found in "
                        f"the logged validation metrics. Available metrics: {available}"
                    )
                primary_val = metric_values[metric_key]
                higher_is_better = metric_key.casefold() not in {
                    "mse",
                    "mae",
                    "val_loss",
                    "loss",
                }
                if metric_key.casefold().endswith("_mse") or metric_key.casefold().endswith("_mae"):
                    higher_is_better = False
                if isinstance(primary_val, float):
                    improved = self._update_top_checkpoints(
                        model,
                        optimizer,
                        epoch,
                        primary_val,
                        higher_is_better=higher_is_better,
                        metric_key=metric_key,
                    )
        else:
            self.val_metrics.reset()

        # Broadcast improvement flag so all ranks agree on early stopping.
        flag = torch.tensor(
            [-1 if improved is None else (1 if improved else 0)],
            device=self.accelerator.device,
        )
        flag = self.accelerator.reduce(flag, reduction="max")
        value = int(flag.item())
        if value < 0:
            return None
        return value == 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _final_val(self, model: nn.Module, optimizer: torch.optim.Optimizer, val_loader) -> None:
        best_path = self.log_dir / "checkpoints" / "best.pt"
        self.accelerator.wait_for_everyone()
        if self.cfg.enable_checkpointing and best_path.exists():
            state = torch.load(best_path, map_location="cpu")
            self.accelerator.unwrap_model(model).load_state_dict(state["model"])
            self.accelerator.print(f"Final validation with {best_path} (epoch {state['epoch']}).")
        else:
            self.accelerator.print("No best.pt found; final validation uses the last model state.")
        self._val_epoch(model, optimizer, val_loader, epoch=self.cfg.epochs - 1, final=True)

    def _update_metrics(self, metrics, y_hat, y, conf_mat=None):
        metrics.update(y_hat, y)
        if conf_mat is not None:
            conf_mat.update(y_hat, y)

    def _append_local_metrics(self, split: str, metrics: dict[str, float | int]) -> None:
        payload: dict[str, float | int] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                payload[key] = int(value)
            elif isinstance(value, int):
                payload[key] = value
            else:
                payload[key] = float(value)
        self._metrics_history[split].append(payload)
        self.metrics_path.write_text(json.dumps(self._metrics_history, indent=2))

    def _get_model_aux_loss(self, model: nn.Module) -> torch.Tensor | None:
        aux_loss = getattr(model, "latest_aux_loss", None)
        if aux_loss is not None:
            return aux_loss
        wrapped = getattr(model, "module", None)
        if wrapped is not None:
            return getattr(wrapped, "latest_aux_loss", None)
        return None

    def _gather_for_metrics(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_hat = y_hat.detach()
        if self.data.task == "Classification" and y_hat.ndim == 2:
            if self.data.subtask == "multilabel":
                y_hat = y_hat.float().sigmoid()
            else:
                y_hat = y_hat.float().softmax(dim=1)
        gathered_y_hat = self.accelerator.gather_for_metrics(y_hat)
        gathered_y = self.accelerator.gather_for_metrics(y.detach())
        return gathered_y_hat, gathered_y

    def _compute_and_reset(self, metrics: MetricCollection) -> dict:
        result = {}
        raw = metrics.compute()
        for k, v in raw.items():
            if k.endswith("F1_per_class"):
                for i, val in enumerate(v):
                    result[k.replace("F1_per_class", f"F1_class_{i}")] = (
                        val.item() if not torch.isnan(val) else 0.0
                    )
            else:
                result[k] = v.item() if hasattr(v, "item") else float(v)
        metrics.reset()
        return result

    def _make_progress(
        self,
        loader,
        desc: str,
        rich_style: str | None = None,
        split: str | None = None,
        max_steps: int | None = None,
    ):
        iterable = loader if max_steps is None else islice(loader, max_steps)
        if not self.accelerator.is_local_main_process:
            return _NoProgress(iterable)
        total = len(loader) if max_steps is None else min(len(loader), max_steps)
        if self.cfg.cluster_progress_bar or not sys.stderr.isatty() or Progress is None:
            pbar = tqdm(iterable, desc=desc, disable=False, dynamic_ncols=True, total=total)
            if self.cfg.cluster_progress_bar:
                return _ThrottledTqdm(pbar, total=total)
            return pbar
        return _RichProgress(iterable, desc=desc, total=total, style=rich_style, split=split)

    def _progress_postfix(
        self,
        loss: torch.Tensor,
        metrics: MetricCollection,
        split: str,
        max_items: int = 3,
    ) -> dict[str, str]:
        if not self.accelerator.is_local_main_process:
            return {}
        postfix = {"split": split, "loss": f"{loss.item():.4f}"}
        try:
            raw = metrics.compute()
        except Exception:
            return postfix

        postfix["metric_1"] = ""
        postfix["metric_2"] = ""
        postfix["metric_3"] = ""
        count = 0
        for key, value in raw.items():
            if count >= max_items:
                break
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                scalar = value.item()
            else:
                scalar = float(value)
            short_key = key.replace("train_", "").replace("val_", "")
            postfix[f"metric_{count + 1}"] = f"{short_key}={scalar:.4f}"
            count += 1
        return postfix

    def _log_scatter(self, split: str, labels: list, preds: list, step: int | None = None):
        figure = plt.figure(figsize=(6, 6))
        ax = figure.add_subplot(111)
        ax.scatter(labels, preds, alpha=0.5, s=12)
        ax.set_title(f"{split} Scatterplot")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        self.logger.log_figure(f"{split}_Scatterplot", figure, step=step)
        plt.close(figure)

    def _loss_for_calibration(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.data.num_classes == 1:
            outputs = outputs.view(-1)
        target = targets.float() if self.data.subtask == "multilabel" else targets
        return self.criterion(outputs, target)

    def _maybe_update_gradual_unfreeze(self, model: nn.Module, epoch: int) -> None:
        if not self._gradual_unfreeze or self.peft_config is None:
            return
        unwrapped = self.accelerator.unwrap_model(model)
        prev = get_freeze_state(unwrapped)
        prev_unlocked = None if prev is None else prev.get("num_unlocked_from_end")
        state = update_for_epoch(
            unwrapped,
            epoch,
            epochs_per_unfreeze=getattr(self.peft_config, "epochs_per_unfreeze", None),
            units_per_unfreeze=getattr(self.peft_config, "units_per_unfreeze", None),
        )
        if state.get("num_unlocked_from_end") != prev_unlocked:
            unlocked = state.get("unlocked_unit_names") or []
            self.accelerator.print(
                f"[gradual_unfreeze] epoch={epoch}: "
                f"unlocked_from_end={state['num_unlocked_from_end']}/"
                f"{state['num_units']} "
                f"frozen={state['num_frozen_units']} "
                f"units={unlocked}"
            )
            if self.accelerator.is_main_process:
                self.logger.log_metrics(
                    {
                        "gradual_unfreeze/num_unlocked_from_end": state["num_unlocked_from_end"],
                        "gradual_unfreeze/num_frozen_units": state["num_frozen_units"],
                        "epoch": epoch,
                    },
                    step=epoch,
                )

    @staticmethod
    def _atomic_torch_save(state: dict, path: Path) -> None:
        """Write via a temp file then replace, avoiding corrupt zip checkpoints on NFS/full disks."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, tmp)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _trainable_param_stats(model: nn.Module) -> dict:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return {
            "trainable_params": int(trainable),
            "total_params": int(total),
            "trainable_fraction": float(trainable / total) if total else 0.0,
        }

    def _write_trainable_params_sidecar(
        self,
        ckpt_dir: Path,
        model: nn.Module,
        *,
        epoch: int,
        metric: float | None = None,
        metric_key: str | None = None,
    ) -> dict:
        """Write ``best_trainable_params.json`` next to ``best.pt``."""
        unwrapped = self.accelerator.unwrap_model(model)
        stats = self._trainable_param_stats(unwrapped)
        payload = {
            **stats,
            "epoch": int(epoch),
        }
        if metric is not None:
            payload["checkpoint_metric"] = float(metric)
        if metric_key is not None:
            payload["checkpoint_metric_key"] = metric_key
        out = Path(ckpt_dir) / "best_trainable_params.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return stats

    def _build_checkpoint_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        *,
        metric: float | None = None,
        metric_key: str | None = None,
        rank: int | None = None,
    ) -> dict:
        unwrapped_model = self.accelerator.unwrap_model(model)
        state = {
            "epoch": epoch,
            "model": unwrapped_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **self._trainable_param_stats(unwrapped_model),
        }
        if metric is not None:
            state["checkpoint_metric"] = float(metric)
        if metric_key is not None:
            state["checkpoint_metric_key"] = metric_key
        if rank is not None:
            state["checkpoint_rank"] = int(rank)
        adaptation_plan = getattr(getattr(unwrapped_model, "encoder", None), "adaptation_plan", None)
        if adaptation_plan is not None:
            state["nnssl_adaptation_plan"] = adaptation_plan
        freeze_state = get_freeze_state(unwrapped_model)
        if freeze_state is not None:
            freeze_state["epoch"] = epoch
            state["gradual_unfreeze"] = freeze_state
        return state

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        is_best: bool = False,
    ):
        """Always refresh last.pt. Top-K bests are handled in ``_update_top_checkpoints``."""
        del is_best  # kept for call-site compatibility
        ckpt_dir = self.log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = self._build_checkpoint_state(model, optimizer, epoch)
        self._atomic_torch_save(state, ckpt_dir / "last.pt")

    def _update_top_checkpoints(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric: float,
        *,
        higher_is_better: bool,
        metric_key: str,
    ) -> bool:
        """Keep up to ``_top_k_checkpoints`` best epochs on disk; refresh best.pt as rank-1.

        Returns True iff this epoch becomes the new #1 (used for early-stopping patience).
        """
        ckpt_dir = self.log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        prev_best = self._best_val_metric
        # Drop any prior entry for this epoch, then re-rank.
        candidates = [e for e in self._top_checkpoints if e["epoch"] != epoch]
        path = ckpt_dir / f"best_epoch_{epoch:04d}.pt"
        candidates.append({"epoch": epoch, "metric": float(metric), "path": path})
        # Primary: metric; tie-break: earliest epoch so best.pt stays the first best.
        if higher_is_better:
            candidates.sort(key=lambda e: (-float(e["metric"]), int(e["epoch"])))
        else:
            candidates.sort(key=lambda e: (float(e["metric"]), int(e["epoch"])))
        new_top = candidates[: self._top_k_checkpoints]
        kept_epochs = {e["epoch"] for e in new_top}

        # Delete checkpoints that fell out of the top-K.
        for old in self._top_checkpoints:
            if old["epoch"] not in kept_epochs:
                old_path = Path(old["path"])
                if old_path.exists():
                    old_path.unlink()

        became_best = False
        live_stats = self._trainable_param_stats(self.accelerator.unwrap_model(model))
        if any(e["epoch"] == epoch for e in new_top):
            # Assign ranks 1..K and (re)write files for the current top set.
            ranked = []
            for rank, entry in enumerate(new_top, start=1):
                entry = dict(entry)
                entry["rank"] = rank
                ranked.append(entry)
                if entry["epoch"] == epoch:
                    state = self._build_checkpoint_state(
                        model,
                        optimizer,
                        epoch,
                        metric=metric,
                        metric_key=metric_key,
                        rank=rank,
                    )
                    self._atomic_torch_save(state, Path(entry["path"]))
                    entry["trainable_params"] = int(state["trainable_params"])
                    entry["total_params"] = int(state["total_params"])
                    entry["trainable_fraction"] = float(state["trainable_fraction"])
            self._top_checkpoints = ranked
        else:
            self._top_checkpoints = [
                {**e, "rank": i} for i, e in enumerate(new_top, start=1)
            ]

        if self._top_checkpoints:
            top1 = self._top_checkpoints[0]
            self._best_val_metric = float(top1["metric"])
            # Stable alias for eval / early tooling (copy, don't reload multi-GB state).
            best_src = Path(top1["path"])
            best_dst = ckpt_dir / "best.pt"
            if best_src.exists() and best_src.resolve() != best_dst.resolve():
                tmp = best_dst.with_suffix(".pt.tmp")
                shutil.copy2(best_src, tmp)
                tmp.replace(best_dst)
            # Sidecar next to best.pt: prefer stats captured when that epoch was saved.
            if "trainable_params" in top1:
                stats = {
                    "trainable_params": int(top1["trainable_params"]),
                    "total_params": int(top1["total_params"]),
                    "trainable_fraction": float(
                        top1.get(
                            "trainable_fraction",
                            top1["trainable_params"] / top1["total_params"]
                            if top1["total_params"]
                            else 0.0,
                        )
                    ),
                }
                payload = {
                    **stats,
                    "epoch": int(top1["epoch"]),
                    "checkpoint_metric": float(top1["metric"]),
                    "checkpoint_metric_key": metric_key,
                }
                (ckpt_dir / "best_trainable_params.json").write_text(
                    json.dumps(payload, indent=2) + "\n"
                )
            else:
                stats = self._write_trainable_params_sidecar(
                    ckpt_dir,
                    model,
                    epoch=int(top1["epoch"]),
                    metric=float(top1["metric"]),
                    metric_key=metric_key,
                )
            became_best = prev_best is None or (
                (higher_is_better and self._best_val_metric > prev_best)
                or ((not higher_is_better) and self._best_val_metric < prev_best)
            )
            manifest = {
                "metric_key": metric_key,
                "higher_is_better": higher_is_better,
                "trainable_params": stats["trainable_params"],
                "total_params": stats["total_params"],
                "top": [
                    {
                        "rank": e["rank"],
                        "epoch": e["epoch"],
                        "metric": e["metric"],
                        "path": str(Path(e["path"]).name),
                        **(
                            {
                                "trainable_params": e["trainable_params"],
                                "total_params": e["total_params"],
                            }
                            if "trainable_params" in e
                            else {}
                        ),
                    }
                    for e in self._top_checkpoints
                ],
            }
            (ckpt_dir / "top_checkpoints.json").write_text(json.dumps(manifest, indent=2))
            if any(e["epoch"] == epoch for e in self._top_checkpoints):
                self.accelerator.print(
                    f"[checkpoint] epoch={epoch} {metric_key}={metric:.6f} "
                    f"trainable={live_stats['trainable_params']:,}/{live_stats['total_params']:,} "
                    f"kept in top-{self._top_k_checkpoints}: "
                    + ", ".join(
                        f"#{e['rank']} ep{e['epoch']}={e['metric']:.6f}"
                        for e in self._top_checkpoints
                    )
                )

        return became_best


# ---------------------------------------------------------------------------
# No-op progress wrapper for non-main processes
# ---------------------------------------------------------------------------

class _NoProgress:
    """Iterable with a no-op ``set_postfix`` for processes that do not show progress."""

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Cluster-friendly throttled tqdm wrapper
# ---------------------------------------------------------------------------

class _ThrottledTqdm:
    """Wraps a tqdm bar and throttles updates for SLURM/LSF log environments."""

    INITIAL = 5
    REMAINING = 10

    def __init__(self, pbar, total: int):
        self._pbar = pbar
        self._total = total
        self._step = 0

    def __iter__(self):
        for item in self._pbar:
            self._step += 1
            yield item

    def set_postfix(self, **kwargs):
        if self._should_update():
            self._pbar.set_postfix(**kwargs)

    def _should_update(self) -> bool:
        n = self._step
        total = self._total
        if n <= self.INITIAL:
            return True
        if total <= self.INITIAL:
            return n == total
        remaining_steps = total - self.INITIAL
        for i in range(1, self.REMAINING + 1):
            pos = self.INITIAL + round(i * remaining_steps / self.REMAINING)
            if n == pos:
                return True
        return False


class _RichProgress:
    """Rich-backed iterable with a tqdm-like ``set_postfix`` API."""

    def __init__(
        self,
        iterable,
        desc: str,
        total: int,
        style: str | None = None,
        split: str | None = None,
    ):
        self._iterable = iterable
        self._style = style
        self._split = split or ""
        rich_desc = f"[{style}]{desc}[/{style}]" if style else desc
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[loss]}", justify="left"),
            TextColumn("{task.fields[metric_1]}", justify="left"),
            TextColumn("{task.fields[metric_2]}", justify="left"),
            TextColumn("{task.fields[metric_3]}", justify="left"),
            transient=False,
        )
        self._task_id = self._progress.add_task(
            rich_desc,
            total=total,
            loss="",
            metric_1="",
            metric_2="",
            metric_3="",
        )

    def __iter__(self):
        with self._progress:
            for item in self._iterable:
                yield item
                self._progress.advance(self._task_id)

    def set_postfix(self, **kwargs):
        style = self._style

        def _fmt(text: str) -> str:
            if not text:
                return ""
            return f"[{style}]{text}[/{style}]" if style else text

        loss = kwargs.get("loss", "")
        self._progress.update(
            self._task_id,
            loss=_fmt(f"{self._split}_loss={loss}" if self._split else f"loss={loss}"),
            metric_1=_fmt(kwargs.get("metric_1", "")),
            metric_2=_fmt(kwargs.get("metric_2", "")),
            metric_3=_fmt(kwargs.get("metric_3", "")),
        )
