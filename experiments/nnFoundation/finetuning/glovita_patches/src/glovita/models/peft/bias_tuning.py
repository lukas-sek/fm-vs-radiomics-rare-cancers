"""Bias-only tuning (BitFit-style) on selected Primus/EVA blocks (pure PyTorch).

Used as ``BiasTuning.__init__(model, ...)`` from the PEFT registry (``self`` is
the full classification model). All weights stay frozen; only bias terms
(Linear and LayerNorm biases, incl. ``q_bias``/``v_bias``-style parameters)
inside the selected blocks and the classification head are trained.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch.nn as nn

from glovita.models.peft.lora import (
    _is_head_param,
    _trainable_param_stats,
    get_transformer_blocks,
)

logger = logging.getLogger(__name__)


class BiasTuning:
    """Unfreeze only the biases of ``blocks[i]`` for i in ``bias_layers``; train head."""

    def __init__(
        self,
        bias_layers: Sequence[int] | None = None,
        bias_lr_mult: float = 1.0,
    ) -> None:
        model: nn.Module = self  # type: ignore[assignment]

        layers = list(bias_layers) if bias_layers is not None else [39]
        blocks = get_transformer_blocks(model)
        n_blocks = len(blocks)
        for idx in layers:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(
                    f"bias_layers index {idx} out of range for {n_blocks} transformer blocks."
                )

        for param in model.parameters():
            param.requires_grad = False

        n_bias_tensors = 0
        for idx in layers:
            n_in_block = 0
            for name, param in blocks[idx].named_parameters():
                if name.endswith("bias"):
                    param.requires_grad = True
                    param._lr_mult = float(bias_lr_mult)
                    n_in_block += 1
            if n_in_block == 0:
                raise RuntimeError(
                    f"No bias parameters found in transformer block {idx} "
                    f"({type(blocks[idx]).__name__})."
                )
            n_bias_tensors += n_in_block

        for name, param in model.named_parameters():
            if _is_head_param(name):
                param.requires_grad = True

        trainable, total = _trainable_param_stats(model)
        pct = 100.0 * trainable / total if total else 0.0
        msg = (
            f"[bias_tuning] layers={layers} bias_tensors={n_bias_tensors} "
            f"lr_mult={bias_lr_mult} trainable={trainable:,}/{total:,} ({pct:.3f}%)"
        )
        logger.info(msg)
        print(msg)
