"""Full in-plane WORC slices → three 1024×1024 RGB frames for MRI-CORE.

Selects three interior axial slices from a full-FOV Z-slab (``(C, X, Y, Z)``
image + ``(X, Y, Z)`` label), min–max normalizes each to ``[0, 1]``, repeats
grayscale to 3 channels, and bilinear-resizes to ``image_size`` (default 1024).

Volume loading / Z-slab clipping lives in COMET ``preprocess.load_worc_lesion_slices``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NUM_SLICES = 3
DEFAULT_IMAGE_SIZE = 1024


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


def normalize_slice_01(slice_2d: torch.Tensor) -> torch.Tensor:
    """Min–max normalize a 2D grayscale tensor to ``[0, 1]``."""
    slice_2d = slice_2d.float()
    lo = float(slice_2d.min())
    hi = float(slice_2d.max())
    if hi <= lo:
        return torch.zeros_like(slice_2d)
    return ((slice_2d - lo) / (hi - lo)).clamp(0.0, 1.0)


def resize_rgb_stack(images: torch.Tensor, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Bilinear-resize ``(S, 3, H, W)`` to ``(S, 3, image_size, image_size)``."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected (S, 3, H, W), got {tuple(images.shape)}")
    if image_size < 1:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if images.shape[-2:] == (image_size, image_size):
        return images.contiguous()
    return F.interpolate(
        images.float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).contiguous()


def augment_lesion_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproducible intensity jitter + axis flips on a full-FOV Z-slab."""
    rng = np.random.RandomState(seed)
    image = image.float().clone()
    label = label.clone()
    lo = float(image.min())
    hi = float(image.max())
    span = max(hi - lo, 1e-6)
    scale = float(rng.uniform(0.9, 1.1))
    shift = float(rng.uniform(-0.05, 0.05)) * span
    image = image * scale + shift
    # Spatial dims of image are (C, X, Y, Z); label is (X, Y, Z).
    for axis, image_dim, label_dim in ((0, 1, 0), (1, 2, 1), (2, 3, 2)):
        if rng.rand() < 0.5:
            image = torch.flip(image, dims=[image_dim])
            label = torch.flip(label, dims=[label_dim])
    return image.contiguous(), label.contiguous()


def three_slice_rgb_stack(
    image: torch.Tensor,
    label: torch.Tensor,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[torch.Tensor, list[int]]:
    """Three interior full-FOV lesion slices as ``(3, 3, image_size, image_size)`` in ``[0, 1]``.

    ``image`` is ``(C, X, Y, Z)`` (typically ``C=1``, full in-plane extent);
    ``label`` is ``(X, Y, Z)``.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected image (C, X, Y, Z), got {tuple(image.shape)}")
    if label.ndim != 3:
        raise ValueError(f"Expected label (X, Y, Z), got {tuple(label.shape)}")
    if image.shape[1:] != label.shape:
        raise ValueError(
            f"Image spatial shape {tuple(image.shape[1:])} != label {tuple(label.shape)}"
        )

    z_min, z_max = lesion_z_extent(label)
    slice_indices = select_interior_slice_indices(z_min, z_max, n=NUM_SLICES)
    depth = image.shape[-1]
    for index in slice_indices:
        if index < 0 or index >= depth:
            raise ValueError(f"Selected slice index {index} out of range for depth {depth}")

    # Average over input channels if multi-channel, then expand to RGB.
    channel_mean = image.float().mean(dim=0)  # (X, Y, Z)
    frames: list[torch.Tensor] = []
    for index in slice_indices:
        gray = normalize_slice_01(channel_mean[:, :, index])
        rgb = gray.unsqueeze(0).repeat(3, 1, 1)
        frames.append(rgb)
    stack = torch.stack(frames, dim=0)
    stack = resize_rgb_stack(stack, image_size=image_size)
    if stack.shape != (NUM_SLICES, 3, image_size, image_size):
        raise ValueError(
            f"Expected ({NUM_SLICES}, 3, {image_size}, {image_size}), got {tuple(stack.shape)}"
        )
    return stack, slice_indices


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "NUM_SLICES",
    "augment_lesion_patch",
    "lesion_z_extent",
    "normalize_slice_01",
    "resize_rgb_stack",
    "select_interior_slice_indices",
    "three_slice_rgb_stack",
]
