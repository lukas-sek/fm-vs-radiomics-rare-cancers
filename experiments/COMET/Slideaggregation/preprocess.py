"""Select 3 interior lesion slices and prepare RGB batches for the COMET encoder.

Expects a processed volume with **full in-plane FOV** (no lesion bbox crop).
Finds the lesion Z extent, excludes the first and last foreground slices,
uniformly samples 3 interior indices, and builds a ``(3, 3, H, W)`` batch where
each whole slice is ImageNet-style RGB via ``repeat_channels`` + ``UnifySizes``.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import numpy as np
import torch
from mmm.transforms import UnifySizes
from mmm.volume3d import Tomo3DProcessor

NUM_SLICES = 3


def lesion_z_extent(label: torch.Tensor) -> tuple[int, int]:
    """Inclusive ``(z_min, z_max)`` of foreground voxels along the last axis."""
    if label.ndim != 3:
        raise ValueError(f"Expected label (X, Y, Z), got shape {tuple(label.shape)}")
    foreground = label.nonzero()
    if foreground.numel() == 0:
        raise ValueError("The mask contains no lesion voxels")
    z = foreground[:, 2]
    return int(z.min()), int(z.max())


def select_interior_slice_indices(z_min: int, z_max: int, n: int = NUM_SLICES) -> list[int]:
    """Uniformly sample ``n`` slice indices between ``z_min`` and ``z_max`` (exclusive ends).

    When the lesion is too thin for an exclusive interior, fall back to linspace
    over the inclusive span (duplicates allowed).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if z_max < z_min:
        raise ValueError(f"Invalid Z extent: z_min={z_min}, z_max={z_max}")
    depth = z_max - z_min + 1
    if depth < 3:
        indices = np.linspace(z_min, z_max, n)
    else:
        interior = list(range(z_min + 1, z_max))
        if len(interior) == 1:
            indices = np.full(n, interior[0], dtype=float)
        else:
            indices = np.linspace(interior[0], interior[-1], n)
    return [int(round(v)) for v in indices.tolist()]


def three_slice_rgb_stack(
    processor: Tomo3DProcessor,
    processed: dict[str, Any],
    max_edge_len: int = 512,
    divisable_by: int = 32,
) -> tuple[torch.Tensor, list[int]]:
    """Three interior lesion slices as ``(3, 3, H, W)`` RGB in ``[0, 1]``.

    Slice indices are chosen from the processed label (after augs), then each
    grayscale slice is expanded with ``repeat_channels`` and resized to match
    the COMET Swin backbone contract.
    """
    if "label" not in processed:
        raise ValueError("three_slice_rgb_stack requires a processed volume with a segmentation mask")
    slices = processor.extract_slices(processed)
    if not slices:
        raise ValueError("Processed patch has no slices")

    lesion_indices = [i for i, s in enumerate(slices) if bool(s["label"].any())]
    if not lesion_indices:
        raise ValueError("The mask contains no lesion voxels")
    z_min, z_max = lesion_indices[0], lesion_indices[-1]
    slice_indices = select_interior_slice_indices(z_min, z_max, n=NUM_SLICES)
    for index in slice_indices:
        if index < 0 or index >= len(slices):
            raise ValueError(f"Selected slice index {index} out of range for {len(slices)} slices")

    images = torch.stack(
        [processor.repeat_channels(slices[index]["image"]) for index in slice_indices]
    ).float()
    resized = UnifySizes(divisable_by=divisable_by, max_edge_len=max_edge_len)(
        [{"image": img} for img in images]
    )
    images = torch.stack([torch.as_tensor(case["image"]) for case in resized]).float()
    if images.ndim != 4 or images.shape[0] != NUM_SLICES or images.shape[1] != 3:
        raise ValueError(f"Expected ({NUM_SLICES}, 3, H, W), got {tuple(images.shape)}")
    return images.contiguous(), slice_indices
