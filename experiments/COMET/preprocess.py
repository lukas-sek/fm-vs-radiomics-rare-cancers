"""WORC volumes → MMM ``Tomo3DProcessor`` input: load, clean, axis-order, 3D lesion crop.

Does not normalize, augment, slice, or run the encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.data.meta_tensor import MetaTensor

COMET_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = COMET_DIR.parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_nnFoundation_lesion_patches import get_clean_lesion_mask  # noqa: E402


def same_voxel_grid(image: sitk.Image, mask: sitk.Image) -> bool:
    return image.GetDimension() == mask.GetDimension() == 3 and image.GetSize() == mask.GetSize()


def load_worc_volume(image_path: Path, mask_path: Path) -> dict[str, torch.Tensor]:
    """Full volume as ``{image: (1, X, Y, Z), label: (X, Y, Z)}`` (MONAI axis order)."""
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got {image.GetDimension()}D: {image_path}")
    if not same_voxel_grid(image, mask):
        raise ValueError(f"Image/mask grid mismatch: {image_path} vs {mask_path}")
    clean, label = get_clean_lesion_mask(mask)
    image_xyz = np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0)))
    mask_xyz = np.ascontiguousarray(np.transpose(sitk.GetArrayFromImage(clean) == label, (2, 1, 0)))
    if image_xyz.shape != mask_xyz.shape:
        raise ValueError(f"Image {image_xyz.shape} and mask {mask_xyz.shape} shapes differ")
    return {
        "image": MetaTensor(torch.from_numpy(image_xyz)[None]),
        "label": torch.from_numpy(mask_xyz).long(),
    }


def crop_lesion_patch(
    image: torch.Tensor,
    label: torch.Tensor,
    margin: int = 16,
    z_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop ``(C, X, Y, Z)`` / ``(X, Y, Z)`` to the lesion bbox plus margins; pad at volume edges."""
    if margin < 0 or z_margin < 0:
        raise ValueError("margin and z_margin must be non-negative")
    foreground = label.nonzero()
    if foreground.numel() == 0:
        raise ValueError("The mask contains no lesion voxels")
    mins = foreground.min(0).values
    maxs = foreground.max(0).values
    starts = torch.stack([mins[0] - margin, mins[1] - margin, mins[2] - z_margin])
    ends = torch.stack([maxs[0] + margin, maxs[1] + margin, maxs[2] + z_margin])
    size = torch.tensor(label.shape, device=starts.device)
    pad_before = torch.clamp(-starts, min=0)
    pad_after = torch.clamp(ends - (size - 1), min=0)
    if bool(pad_before.any() or pad_after.any()):
        pad = (
            int(pad_before[2]),
            int(pad_after[2]),
            int(pad_before[1]),
            int(pad_after[1]),
            int(pad_before[0]),
            int(pad_after[0]),
        )
        image = F.pad(image.float(), pad, value=0)
        label = F.pad(label.unsqueeze(0).float(), pad, value=0).squeeze(0).long()
        starts = starts + pad_before
        ends = ends + pad_before
    x0, y0, z0 = (int(v) for v in starts.tolist())
    x1, y1, z1 = (int(v) + 1 for v in ends.tolist())
    return image[:, x0:x1, y0:y1, z0:z1].contiguous(), label[x0:x1, y0:y1, z0:z1].contiguous()


def load_worc_lesion_patch(
    image_path: Path,
    mask_path: Path,
    margin: int = 16,
    z_margin: int = 0,
) -> dict[str, torch.Tensor]:
    """Lesion-centered 3D patch ready for ``Tomo3DProcessor(..., with_segmask=True)``."""
    volume = load_worc_volume(image_path, mask_path)
    image, label = crop_lesion_patch(volume["image"], volume["label"], margin=margin, z_margin=z_margin)
    if isinstance(volume["image"], MetaTensor):
        image = MetaTensor(image)
    return {"image": image, "label": label}
