"""Cache frozen UMedPT slice tokens per WORC lesion for grouper fine-tuning.

Each lesion is cropped to a 3D patch (``preprocess.load_worc_lesion_patch``),
run through mmm ``Tomo3DProcessor``, and sliced. The frozen encoder + squeezer
produce one 512-d token per slice. ``augmentation_id=0`` is the original;
``--num-augmentations`` extra variants use ``Tomo3DProcessor.base_volume_augs``.

Example:
    python experiments/COMET/feature_extraction.py --dataset Liver --num-augmentations 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import zlib
from pathlib import Path
from typing import Any

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import monai.transforms as monai_transforms
import pandas as pd
import torch
from monai.utils import set_determinism
from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS, M3Model
from mmm.transforms import RandomApply, UnifySizes
from mmm.volume3d import Tomo3DProcessor

COMET_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMET_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (COMET_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_nnFoundation_lesion_patches import (  # noqa: E402
    DATASETS,
    WORC_ROOT,
    discover_lesion_masks,
    load_pinfo,
    resolve_image_path,
    resolve_label,
)
from prepare_worc_nnunet_dataset import safe_identifier  # noqa: E402
from preprocess import load_worc_lesion_patch  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT / "data" / "comet" / DEFAULT_MODEL
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "comet" / "cache"
DEFAULT_FEATURES_DIR = REPO_ROOT / "Results" / "COMET" / "Features"
MAX_AUGMENTATION_ATTEMPTS = 10
INDEX_COLUMNS = (
    "case_id",
    "patient_id",
    "lesion_id",
    "label",
    "mask_source",
    "augmentation_id",
    "augmentation_seed",
    "margin",
    "z_margin",
    "n_slices",
    "slice_start",
    "slice_end",
    "input_height",
    "input_width",
    "feature_dim",
    "file",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--margin", type=int, default=16, help="In-plane crop margin in voxels.")
    parser.add_argument("--z-margin", type=int, default=0, help="Slice-axis crop margin in voxels.")
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=10,
        help="Augmented variants per lesion in addition to the original.",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32, help="Slices per encoder forward pass.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast on CUDA.")
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Disable cuDNN (workaround for CUDNN_STATUS_NOT_INITIALIZED / driver mismatches).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHIVE, help="encoder-1.0.4.pt.zip")
    parser.add_argument("--worc-root", type=Path, default=WORC_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Default: data/comet/cache/<dataset>_tokens.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Feature index CSV. Default: Results/COMET/Features/"
            "<dataset>_features_comet.csv when --num-augmentations is 0, else "
            "<dataset>_augmented_features_comet.csv."
        ),
    )
    parser.add_argument("--patients", nargs="+", default=None, help="Only these patient IDs.")
    parser.add_argument("--max-patients", type=int, default=None, help="Only the first N patients.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cache files.")
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_output_csv(dataset: str, num_augmentations: int) -> Path:
    suffix = "augmented_features_comet" if num_augmentations > 0 else "features_comet"
    return DEFAULT_FEATURES_DIR / f"{dataset}_{suffix}.csv"


def variant_seed(base_seed: int, case_id: str, augmentation_id: int, attempt: int) -> int:
    case_offset = zlib.crc32(case_id.encode("utf-8"))
    return (base_seed * 1_000_003 + case_offset + augmentation_id * 1_009 + attempt) % (2**31 - 1)


def seed_transforms(transforms: list, seed: int) -> list:
    for offset, transform in enumerate(transforms):
        inner = transform.f if isinstance(transform, RandomApply) else transform
        if isinstance(inner, monai_transforms.Randomizable):
            inner.set_random_state(seed=(seed + offset) % (2**32 - 1))
    return transforms


def build_processor(seed: int | None) -> Tomo3DProcessor:
    if seed is None:
        augs = None
    else:
        set_determinism(seed=seed)
        augs = lambda keys: seed_transforms(Tomo3DProcessor.base_volume_augs(keys), seed)  # noqa: E731
    return Tomo3DProcessor(Tomo3DProcessor.Config(rotate_3d="never"), augs, with_segmask=True)


def patch_slice_stack(
    processor: Tomo3DProcessor,
    processed: dict[str, Any],
    max_edge_len: int = 512,
    divisable_by: int = 32,
) -> tuple[torch.Tensor, list[int]]:
    """All slices of a processed 3D lesion patch as ``(S, 3, H, W)`` in [0, 1]."""
    slices = processor.extract_slices(processed)
    if not slices:
        raise ValueError("Processed patch has no slices")
    images = torch.stack([processor.repeat_channels(s["image"]) for s in slices]).float()
    resized = UnifySizes(divisable_by=divisable_by, max_edge_len=max_edge_len)(
        [{"image": img} for img in images]
    )
    images = torch.stack([torch.as_tensor(case["image"]) for case in resized]).float()
    return images.contiguous(), list(range(len(slices)))


def ensure_checkpoint(path: Path) -> Path:
    path = resolve_cli_path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {M3_MODELS[DEFAULT_MODEL]} -> {path}")
        torch.hub.download_url_to_file(M3_MODELS[DEFAULT_MODEL], str(path))
    return path


@torch.inference_mode()
def encode_slice_tokens(
    encoder: torch.nn.Module,
    squeezer: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    tokens = []
    for chunk in images.split(batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _, pooled = squeezer(encoder(chunk.to(device, non_blocking=True)))
        flattened = pooled.flatten(1)
        if not torch.isfinite(flattened).all():
            raise FloatingPointError("Encoder/squeezer produced non-finite tokens")
        tokens.append(flattened.to(torch.float16).cpu())
    stacked = torch.cat(tokens)
    if stacked.ndim != 2:
        raise ValueError(f"Expected (S, D) tokens, got {tuple(stacked.shape)}")
    return stacked


def iter_lesions(dataset: str, worc_root: Path, patients: list[str] | None, max_patients: int | None):
    dataset_root = worc_root / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing WORC dataset directory: {dataset_root}")
    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if patients is not None:
        wanted = set(patients)
        patient_dirs = [path for path in patient_dirs if path.name in wanted]
        missing = wanted - {path.name for path in patient_dirs}
        if missing:
            raise FileNotFoundError(f"Unknown patients for {dataset}: {sorted(missing)}")
    if max_patients is not None:
        patient_dirs = patient_dirs[:max_patients]
    for patient_dir in patient_dirs:
        for lesion_number, (lesion_id, mask_path, mask_source) in enumerate(
            discover_lesion_masks(dataset, patient_dir), start=1
        ):
            case_id = f"{safe_identifier(patient_dir.name)}--{safe_identifier(lesion_id)}--{lesion_number}"
            yield patient_dir, case_id, lesion_id, mask_path, mask_source


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        # Warm the context before loading the Swin stem (avoids some lazy-init cuDNN failures).
        torch.empty(1, device=device).fill_(0)
        if args.no_cudnn:
            torch.backends.cudnn.enabled = False
            print("cuDNN disabled (--no-cudnn)")
        print(
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"device={torch.cuda.get_device_name(device)}"
        )
    amp = device.type == "cuda" and not args.no_amp
    worc_root = resolve_cli_path(args.worc_root)
    cache_dir = (
        resolve_cli_path(args.cache_dir)
        if args.cache_dir
        else DEFAULT_CACHE_ROOT / f"{args.dataset}_tokens"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_csv = (
        resolve_cli_path(args.output_csv)
        if args.output_csv is not None
        else default_output_csv(args.dataset, args.num_augmentations)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(
        {
            "cache_dir": str(cache_dir),
            "output_csv": str(output_csv),
            "amp": amp,
            "feature": "squeezer_token_512",
            "encoder_frozen": True,
            "squeezer_frozen": True,
        }
    )
    (cache_dir / "cache_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = M3Model(str(ensure_checkpoint(args.checkpoint)), device_identifier=str(device))
    encoder = model["encoder"]
    squeezer = model["squeezer"]
    encoder.eval().requires_grad_(False)
    squeezer.eval().requires_grad_(False)

    pinfo = load_pinfo(args.dataset)
    rows: list[dict] = []
    start = time.time()
    lesions = list(iter_lesions(args.dataset, worc_root, args.patients, args.max_patients))
    print(
        f"{args.dataset}: {len(lesions)} lesions, K={args.num_augmentations}, "
        f"margin={args.margin}, z_margin={args.z_margin}, device={device}"
    )

    for lesion_index, (patient_dir, case_id, lesion_id, mask_path, mask_source) in enumerate(lesions, start=1):
        patient_id = patient_dir.name
        label = int(resolve_label(args.dataset, patient_id, lesion_id, pinfo))
        patch = None

        for augmentation_id in range(args.num_augmentations + 1):
            out_path = cache_dir / f"{case_id}__aug{augmentation_id}.pt"
            if out_path.is_file() and not args.overwrite:
                entry = torch.load(out_path, map_location="cpu", weights_only=True)
            else:
                if patch is None:
                    patch = load_worc_lesion_patch(
                        resolve_image_path(patient_dir, lesion_id),
                        mask_path,
                        margin=args.margin,
                        z_margin=args.z_margin,
                    )
                entry = None
                for attempt in range(MAX_AUGMENTATION_ATTEMPTS if augmentation_id else 1):
                    seed = None if augmentation_id == 0 else variant_seed(
                        args.augmentation_seed, case_id, augmentation_id, attempt
                    )
                    try:
                        processor = build_processor(seed)
                        processed = processor(
                            {"image": patch["image"].clone(), "label": patch["label"].clone()}
                        )
                        stack, slice_indices = patch_slice_stack(processor, processed)
                    except ValueError as error:
                        if augmentation_id == 0:
                            raise ValueError(f"{case_id}: {error}") from error
                        continue
                    tokens = encode_slice_tokens(encoder, squeezer, stack, args.batch_size, device, amp)
                    entry = {
                        "tokens": tokens,
                        "slice_indices": torch.tensor(slice_indices, dtype=torch.long),
                        "augmentation_seed": -1 if seed is None else seed,
                        "input_shape": torch.tensor(stack.shape[-2:], dtype=torch.long),
                    }
                    torch.save(entry, out_path)
                    break
                if entry is None:
                    warnings.warn(
                        f"{case_id} aug{augmentation_id}: lesion lost in every attempt, skipped",
                        stacklevel=1,
                    )
                    continue

            tokens = entry["tokens"]
            slice_indices = entry["slice_indices"].tolist()
            rows.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "lesion_id": lesion_id,
                    "label": label,
                    "mask_source": mask_source or "",
                    "augmentation_id": augmentation_id,
                    "augmentation_seed": int(entry["augmentation_seed"]),
                    "margin": args.margin,
                    "z_margin": args.z_margin,
                    "n_slices": len(slice_indices),
                    "slice_start": slice_indices[0],
                    "slice_end": slice_indices[-1],
                    "input_height": int(entry["input_shape"][0]),
                    "input_width": int(entry["input_shape"][1]),
                    "feature_dim": int(tokens.shape[-1]),
                    "file": out_path.name,
                }
            )

        elapsed = time.time() - start
        print(f"  [{lesion_index}/{len(lesions)}] {case_id} label={label} ({elapsed / 60:.1f} min)")

    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    index.to_csv(output_csv, index=False)
    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
    print(f"Wrote {len(index)} variants of {index['case_id'].nunique()} lesions -> {output_csv}")
    print(f"Token cache ({total_bytes / 1024**3:.2f} GB) -> {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
