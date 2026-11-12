"""UMedPT helpers on top of the MedicalMultitaskModeling (``mmm``) library.

Importing this module accepts the mmm license (scientific non-commercial
research use, https://github.com/FraunhoferMEVIS/MedicalMultitaskModeling/blob/main/license.md)
through ``MMM_LICENSE_ACCEPTED``. Requires ``pip install MedicalMultitaskModeling m3-sdk``.
"""

from __future__ import annotations

import os

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

import monai.transforms as monai_transforms  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from monai.utils import set_determinism  # noqa: E402
from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS, M3Model  # noqa: E402
from mmm.mtl_modules.shared_blocks.Grouper import Grouper, MHAReducer  # noqa: E402
from mmm.transforms import RandomApply, UnifySizes  # noqa: E402
from mmm.utils import make_divisable_by  # noqa: E402
from mmm.volume3d import Tomo3DProcessor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "comet" / DEFAULT_MODEL


def load_umedpt(archive_path: Path = DEFAULT_ARCHIVE, device: str | torch.device = "cpu") -> M3Model:
    """The released UMedPT modules (``encoder``, ``squeezer``, ``imgnetclf``); downloads the archive if missing."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {M3_MODELS[DEFAULT_MODEL]} -> {archive_path}")
        torch.hub.download_url_to_file(M3_MODELS[DEFAULT_MODEL], str(archive_path))
    return M3Model(str(archive_path), device_identifier=str(device))


def build_grouper(num_heads: int, embedding_dim: int = 512) -> Grouper:
    """A new grouper with ``Grouper.Config()`` defaults; ``num_heads`` must divide ``embedding_dim``."""
    return Grouper(Grouper.Config(reducer=MHAReducer.Config(num_heads=num_heads)), embedding_dim=embedding_dim)


def build_processor(augment: bool, seed: int | None = None) -> Tomo3DProcessor:
    """
    ``Tomo3DProcessor`` with a mask; ``augment`` adds ``base_volume_augs``.

    With a ``seed`` the next call of the returned processor is reproducible, so call it right away:
    MONAI's ``Compose`` reseeds its transforms from the global MONAI seed, and ``RandomApply`` uses ``random``.
    """
    if not augment:
        augs = None
    elif seed is None:
        augs = Tomo3DProcessor.base_volume_augs
    else:
        set_determinism(seed=seed)
        augs = lambda keys: seed_transforms(Tomo3DProcessor.base_volume_augs(keys), seed)  # noqa: E731
    # mmm defaults to rotate_3d="ifcubical", whose 90 degree rotations can move the slice axis.
    return Tomo3DProcessor(Tomo3DProcessor.Config(rotate_3d="never"), augs, with_segmask=True)


def seed_transforms(transforms: list, seed: int) -> list:
    """Seed MONAI random transforms, including those wrapped in ``RandomApply`` that ``Compose`` does not reseed."""
    for offset, transform in enumerate(transforms):
        inner = transform.f if isinstance(transform, RandomApply) else transform
        if isinstance(inner, monai_transforms.Randomizable):
            inner.set_random_state(seed=(seed + offset) % (2**32 - 1))
    return transforms


def lesion_slice_stack(
    processor: Tomo3DProcessor,
    d: dict[str, Any],
    fov: Literal["full", "crop"] = "full",
    margin: int = 16,
    z_margin: int = 0,
    max_edge_len: int = 512,
    divisable_by: int = 32,
) -> tuple[torch.Tensor, list[int]]:
    """
    Slices of a processed volume (output of ``processor(...)`` with a mask) that cover the lesion.

    ``full`` keeps the whole slice, ``crop`` cuts one square box around the lesion that is shared by all
    slices so neighbouring slices stay aligned. Both are resized with ``UnifySizes`` like UMedPT inputs.
    Returns a ``(S, 3, H, W)`` float stack in [0, 1] and the slice indices along the last axis.
    """
    if "label" not in d:
        raise ValueError("lesion_slice_stack requires a processed volume with a segmentation mask")
    slices = processor.extract_slices(d)
    lesion_indices = [i for i, s in enumerate(slices) if bool(s["label"].any())]
    if not lesion_indices:
        raise ValueError("The mask contains no lesion voxels")
    first = max(0, lesion_indices[0] - z_margin)
    last = min(len(slices) - 1, lesion_indices[-1] + z_margin)
    chosen = slices[first : last + 1]
    images = torch.stack([processor.repeat_channels(s["image"]) for s in chosen]).float()

    if fov == "crop":
        foreground = torch.stack([s["label"] for s in chosen]).any(0).nonzero()
        (y0, x0), (y1, x1) = foreground.min(0).values.tolist(), foreground.max(0).values.tolist()
        side = make_divisable_by(max(y1 - y0, x1 - x0) + 1 + 2 * margin, divisable_by)
        center_y, center_x = (y0 + y1) // 2, (x0 + x1) // 2
        padded = F.pad(images, (side, side, side, side))
        top, left = center_y + side - side // 2, center_x + side - side // 2
        images = padded[..., top : top + side, left : left + side]
    elif fov != "full":
        raise ValueError(f"Unknown fov {fov!r}; choose 'full' or 'crop'")

    resized = UnifySizes(divisable_by=divisable_by, max_edge_len=max_edge_len)([{"image": img} for img in images])
    images = torch.stack([torch.as_tensor(case["image"]) for case in resized]).float()
    return images.contiguous(), list(range(first, last + 1))
