"""Bottleneck adapters parallel to the MLP of selected Primus/EVA blocks (pure PyTorch).

Used as ``BottleneckAdapter.__init__(model, ...)`` from the PEFT registry
(``self`` is the full classification model). The whole backbone is frozen;
only the adapters and the classification head are trained.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from glovita.models.peft.lora import (
    _is_head_param,
    _trainable_param_stats,
    get_transformer_blocks,
)

logger = logging.getLogger(__name__)


def _infer_width(mlp: nn.Module) -> int:
    """Input width of the MLP = in_features of its first Linear."""
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            return int(module.in_features)
    raise RuntimeError(f"Cannot infer width: no nn.Linear inside {type(mlp).__name__}.")


class MLPWithAdapter(nn.Module):
    """``mlp(h) + scale * up(dropout(gelu(down(h))))``; ``up`` starts at zero."""

    def __init__(
        self,
        mlp: nn.Module,
        width: int,
        bottleneck: int,
        dropout: float,
        scale: float,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.adapter_down = nn.Linear(width, bottleneck)
        self.adapter_up = nn.Linear(bottleneck, width)
        self.adapter_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = float(scale)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = self.adapter_up(self.adapter_drop(F.gelu(self.adapter_down(h))))
        return self.mlp(h) + self.scale * delta


class BottleneckAdapter:
    """Wrap ``blocks[i].mlp`` with a zero-init parallel adapter; train adapters + head."""

    def __init__(
        self,
        adapter_bottleneck: int = 32,
        adapter_layers: Sequence[int] | None = None,
        adapter_dropout: float = 0.1,
        adapter_scale: float = 1.0,
        adapter_lr_mult: float = 1.0,
    ) -> None:
        model: nn.Module = self  # type: ignore[assignment]

        layers = list(adapter_layers) if adapter_layers is not None else [39]
        blocks = get_transformer_blocks(model)
        n_blocks = len(blocks)
        for idx in layers:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(
                    f"adapter_layers index {idx} out of range for {n_blocks} transformer blocks."
                )

        for param in model.parameters():
            param.requires_grad = False

        for idx in layers:
            block = blocks[idx]
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                raise AttributeError(
                    f"Transformer block {idx} ({type(block).__name__}) has no '.mlp'."
                )
            width = _infer_width(mlp)
            block.mlp = MLPWithAdapter(
                mlp,
                width=width,
                bottleneck=int(adapter_bottleneck),
                dropout=float(adapter_dropout),
                scale=float(adapter_scale),
            )

        for name, param in model.named_parameters():
            if "adapter_down" in name or "adapter_up" in name:
                param.requires_grad = True
                param._lr_mult = float(adapter_lr_mult)
            elif _is_head_param(name):
                param.requires_grad = True

        trainable, total = _trainable_param_stats(model)
        pct = 100.0 * trainable / total if total else 0.0
        msg = (
            f"[adapter] layers={layers} bottleneck={adapter_bottleneck} "
            f"dropout={adapter_dropout} scale={adapter_scale} lr_mult={adapter_lr_mult} "
            f"trainable={trainable:,}/{total:,} ({pct:.3f}%)"
        )
        logger.info(msg)
        print(msg)
