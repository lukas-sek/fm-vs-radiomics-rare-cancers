"""Late-to-early gradual unfreezing for nnFoundation ResEnc and Primus backbones.

Units are ordered from late (unfrozen first) to early (unfrozen last):
  ResEnc: stages[-1] … stages[0], then stem
  Primus: last EVA block … first block, then down_projection (patch embed)

The classification head is always trainable. Start with ``num_unlocked_from_end=0``
(head only).
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

_HEAD_SUBSTRINGS = ("head", "classifier", "cls_head")
_STATE_ATTR = "_gradual_unfreeze_state"


def _is_head_param(name: str) -> bool:
    return any(sub in name for sub in _HEAD_SUBSTRINGS)


def _resolve_backbone(model: nn.Module) -> nn.Module:
    """Return the ResidualEncoderUNet / Primus module inside a ComposedModel."""
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "model"):
        return encoder.model
    backbone = getattr(model, "model", None)
    if backbone is not None:
        return backbone
    return model


def discover_units(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``[(name, module), ...]`` ordered late→early (unfreeze order)."""
    backbone = _resolve_backbone(model)

    # ResidualEncoderUNet → encoder.stages + encoder.stem
    residual = getattr(backbone, "encoder", None)
    if residual is not None and hasattr(residual, "stages"):
        stages = list(residual.stages)
        units: list[tuple[str, nn.Module]] = [
            (f"encoder.stages.{i}", stage) for i, stage in enumerate(stages)
        ]
        units.reverse()  # late stage first
        if hasattr(residual, "stem") and residual.stem is not None:
            units.append(("encoder.stem", residual.stem))
        return units

    # Primus / PrimusX: EVA blocks then patch embed
    eva = getattr(backbone, "eva", None)
    blocks = None
    if eva is not None:
        blocks = getattr(eva, "blocks", None) or getattr(eva, "layers", None)
    if blocks is None:
        blocks = getattr(backbone, "blocks", None)
    if blocks is not None:
        units = [(f"eva.blocks.{i}", block) for i, block in enumerate(list(blocks))]
        units.reverse()
        for attr in ("down_projection", "patch_embed"):
            module = getattr(backbone, attr, None)
            if module is not None:
                units.append((attr, module))
                break
        return units

    raise AttributeError(
        "Gradual unfreeze requires nnFoundation ResEnc (encoder.stages) or "
        "Primus (eva.blocks / blocks). "
        f"Got backbone type {type(backbone).__name__}."
    )


def apply_gradual_unfreeze(model: nn.Module, config) -> nn.Module:
    """Initial PEFT setup: freeze backbone, keep head trainable, store schedule state."""
    units = discover_units(model)
    state = {
        "epochs_per_unfreeze": int(config.epochs_per_unfreeze),
        "units_per_unfreeze": int(getattr(config, "units_per_unfreeze", 1)),
        "unit_names": [name for name, _ in units],
        "num_units": len(units),
        "num_unlocked_from_end": 0,
        "num_frozen_units": len(units),
        "epoch": -1,
    }
    setattr(model, _STATE_ATTR, state)
    set_unlocked_from_end(model, 0)
    return model


def set_unlocked_from_end(model: nn.Module, num_unlocked: int) -> dict[str, Any]:
    """Unfreeze the last ``num_unlocked`` backbone units (late→early); freeze the rest.

    Head parameters always remain trainable. Returns the updated freeze state dict.
    """
    units = discover_units(model)
    num_unlocked = max(0, min(int(num_unlocked), len(units)))

    if num_unlocked >= len(units):
        # All scheduled units unlocked → full model trainable.
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze everything first, then unfreeze head + selected units.
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if _is_head_param(name):
                param.requires_grad = True

        for name, module in units[:num_unlocked]:
            for param in module.parameters():
                param.requires_grad = True

    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        state = {
            "epochs_per_unfreeze": None,
            "unit_names": [n for n, _ in units],
            "num_units": len(units),
        }
        setattr(model, _STATE_ATTR, state)

    state["num_unlocked_from_end"] = num_unlocked
    state["num_frozen_units"] = len(units) - num_unlocked
    state["unit_names"] = [n for n, _ in units]
    state["num_units"] = len(units)
    state["unlocked_unit_names"] = [n for n, _ in units[:num_unlocked]]
    state["frozen_unit_names"] = [n for n, _ in units[num_unlocked:]]
    return dict(state)


def update_for_epoch(
    model: nn.Module,
    epoch: int,
    epochs_per_unfreeze: int | None = None,
    units_per_unfreeze: int | None = None,
) -> dict[str, Any]:
    """Apply the schedule for ``epoch`` (0-based) and return freeze state."""
    state = getattr(model, _STATE_ATTR, None)
    if epochs_per_unfreeze is None:
        if state is None or state.get("epochs_per_unfreeze") is None:
            raise ValueError("epochs_per_unfreeze is required when model has no gradual_unfreeze state.")
        epochs_per_unfreeze = int(state["epochs_per_unfreeze"])
    if units_per_unfreeze is None:
        units_per_unfreeze = int((state or {}).get("units_per_unfreeze") or 1)
    epochs_per_unfreeze = max(1, int(epochs_per_unfreeze))
    units_per_unfreeze = max(1, int(units_per_unfreeze))

    units = discover_units(model)
    # epoch 0..(E-1): 0 unlocked; each step unlocks ``units_per_unfreeze`` more.
    steps = epoch // epochs_per_unfreeze
    num_unlocked = min(steps * units_per_unfreeze, len(units))
    new_state = set_unlocked_from_end(model, num_unlocked)
    new_state["epoch"] = int(epoch)
    new_state["epochs_per_unfreeze"] = epochs_per_unfreeze
    new_state["units_per_unfreeze"] = units_per_unfreeze
    setattr(model, _STATE_ATTR, new_state)
    return dict(new_state)


def get_freeze_state(model: nn.Module) -> dict[str, Any] | None:
    """Return a JSON-serializable freeze snapshot, or None if not gradual-unfreeze."""
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return None
    return {
        "num_frozen_units": int(state.get("num_frozen_units", 0)),
        "num_unlocked_from_end": int(state.get("num_unlocked_from_end", 0)),
        "num_units": int(state.get("num_units", 0)),
        "unit_names": list(state.get("unit_names", [])),
        "unlocked_unit_names": list(state.get("unlocked_unit_names", [])),
        "frozen_unit_names": list(state.get("frozen_unit_names", [])),
        "epochs_per_unfreeze": state.get("epochs_per_unfreeze"),
        "units_per_unfreeze": state.get("units_per_unfreeze"),
        "epoch": state.get("epoch"),
    }
