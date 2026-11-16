"""Cache frozen COMET 3-slice concatenated embeddings per WORC lesion.

Supports ``--fov full`` (default: full in-plane Z-slab) or ``--fov crop``
(lesion bounding-box crop with ``--margin``). Runs ``Tomo3DProcessor``, selects
3 interior axial slices (excluding first/last), encodes each as RGB through the
frozen encoder + squeezer, and concatenates the three 512-d tokens into one
1536-d embedding. ``augmentation_id=0`` is the original; ``--num-augmentations``
extra variants use ``Tomo3DProcessor.base_volume_augs``.

Example:
    python experiments/COMET/Slideaggregation/feature_extraction.py --dataset Liver --fov full
    python experiments/COMET/Slideaggregation/feature_extraction.py --dataset Liver --fov crop --margin 16
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
import zlib
from pathlib import Path

os.environ.setdefault("MMM_LICENSE_ACCEPTED", "i accept")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import monai.transforms as monai_transforms
import pandas as pd
import torch
from monai.utils import set_determinism
from mmm.api.M3Model import DEFAULT_MODEL, M3_MODELS, M3Model
from mmm.transforms import RandomApply
from mmm.volume3d import Tomo3DProcessor

SLIDEAGG_DIR = Path(__file__).resolve().parent
COMET_DIR = SLIDEAGG_DIR.parent
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# COMET and Slideaggregation both ship ``preprocess.py`` — load each by path.
_comet_preprocess = _load_module("comet_worc_preprocess", COMET_DIR / "preprocess.py")
for _name in ("load_worc_lesion_slices", "load_worc_lesion_patch"):
    if not hasattr(_comet_preprocess, _name):
        raise ImportError(
            f"{COMET_DIR / 'preprocess.py'} is missing {_name}. "
            "Sync the updated COMET preprocess.py."
        )
load_worc_lesion_slices = _comet_preprocess.load_worc_lesion_slices
load_worc_lesion_patch = _comet_preprocess.load_worc_lesion_patch

_slideagg_preprocess = _load_module("comet_slideagg_preprocess", SLIDEAGG_DIR / "preprocess.py")
NUM_SLICES = _slideagg_preprocess.NUM_SLICES
three_slice_rgb_stack = _slideagg_preprocess.three_slice_rgb_stack

DEFAULT_ARCHIVE = REPO_ROOT / "data" / "comet" / DEFAULT_MODEL
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "comet" / "cache"
DEFAULT_FEATURES_DIR = REPO_ROOT / "Results" / "COMET" / "Slideaggregation"
TOKEN_DIM = 512
EMBEDDING_DIM = TOKEN_DIM * NUM_SLICES
MAX_AUGMENTATION_ATTEMPTS = 10
FOV_CHOICES = ("full", "crop")
META_COLUMNS = (
    "case_id",
    "patient_id",
    "lesion_id",
    "label",
    "mask_source",
    "augmentation_id",
    "augmentation_seed",
    "fov",
    "margin",
    "z_margin",
    "slice_0",
    "slice_1",
    "slice_2",
    "input_height",
    "input_width",
    "feature_dim",
    "file",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--fov",
        choices=FOV_CHOICES,
        default="full",
        help="full = whole in-plane slices (Z-slab); crop = lesion bounding box.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=16,
        help="In-plane bbox margin in voxels when --fov crop (ignored for full).",
    )
    parser.add_argument(
        "--z-margin",
        type=int,
        default=0,
        help="Extra slices beyond the lesion Z extent.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=10,
        help="Augmented variants per lesion in addition to the original.",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=3, help="Slices per encoder forward pass.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast on CUDA.")
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Disable cuDNN (workaround for CUDNN_STATUS_NOT_INITIALIZED / driver mismatches).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARCHIVE, help="encoder-1.0.4.pt.zip")
    parser.add_argument("--worc-root", type=Path, default=WORC_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Default: data/comet/cache/<dataset>_slideagg_<fov>.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Feature CSV. Default: Results/COMET/Slideaggregation/"
            "<dataset>_slideagg_<fov>_features.csv (or *_augmented_features.csv)."
        ),
    )
    parser.add_argument("--patients", nargs="+", default=None, help="Only these patient IDs.")
    parser.add_argument("--max-patients", type=int, default=None, help="Only the first N patients.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cache files.")
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_output_csv(dataset: str, fov: str, num_augmentations: int) -> Path:
    suffix = (
        f"slideagg_{fov}_augmented_features"
        if num_augmentations > 0
        else f"slideagg_{fov}_features"
    )
    return DEFAULT_FEATURES_DIR / f"{dataset}_{suffix}.csv"


def default_cache_dir(dataset: str, fov: str) -> Path:
    return DEFAULT_CACHE_ROOT / f"{dataset}_slideagg_{fov}"


def load_lesion_volume(
    image_path: Path,
    mask_path: Path,
    fov: str,
    margin: int,
    z_margin: int,
) -> dict:
    if fov == "full":
        return load_worc_lesion_slices(image_path, mask_path, z_margin=z_margin)
    if fov == "crop":
        return load_worc_lesion_patch(
            image_path, mask_path, margin=margin, z_margin=z_margin
        )
    raise ValueError(f"Unknown fov {fov!r}; choose from {FOV_CHOICES}")


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


def embedding_to_feature_row(embedding: torch.Tensor) -> dict[str, float]:
    flat = embedding.detach().float().reshape(-1)
    if flat.numel() != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM}-d embedding, got {flat.numel()}")
    return {f"feature_{index}": float(flat[index]) for index in range(EMBEDDING_DIM)}


def main() -> None:
    args = parse_args()
    if args.margin < 0:
        raise SystemExit("--margin must be non-negative")
    if args.z_margin < 0:
        raise SystemExit("--z-margin must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
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
        else default_cache_dir(args.dataset, args.fov)
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_csv = (
        resolve_cli_path(args.output_csv)
        if args.output_csv is not None
        else default_output_csv(args.dataset, args.fov, args.num_augmentations)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    margin_value = int(args.margin) if args.fov == "crop" else 0

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(
        {
            "cache_dir": str(cache_dir),
            "output_csv": str(output_csv),
            "amp": amp,
            "feature": f"squeezer_token_{TOKEN_DIM}_x{NUM_SLICES}_concat",
            "embedding_dim": EMBEDDING_DIM,
            "fov": args.fov,
            "margin": margin_value,
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
        f"fov={args.fov}, margin={margin_value}, z_margin={args.z_margin}, "
        f"embedding_dim={EMBEDDING_DIM}, device={device}"
    )

    for lesion_index, (patient_dir, case_id, lesion_id, mask_path, mask_source) in enumerate(
        lesions, start=1
    ):
        patient_id = patient_dir.name
        label = int(resolve_label(args.dataset, patient_id, lesion_id, pinfo))
        volume = None

        for augmentation_id in range(args.num_augmentations + 1):
            out_path = cache_dir / f"{case_id}__aug{augmentation_id}.pt"
            if out_path.is_file() and not args.overwrite:
                entry = torch.load(out_path, map_location="cpu", weights_only=True)
            else:
                if volume is None:
                    volume = load_lesion_volume(
                        resolve_image_path(patient_dir, lesion_id),
                        mask_path,
                        fov=args.fov,
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
                            {"image": volume["image"].clone(), "label": volume["label"].clone()}
                        )
                        stack, slice_indices = three_slice_rgb_stack(processor, processed)
                    except ValueError as error:
                        if augmentation_id == 0:
                            raise ValueError(f"{case_id}: {error}") from error
                        continue
                    tokens = encode_slice_tokens(
                        encoder, squeezer, stack, args.batch_size, device, amp
                    )
                    if tokens.shape != (NUM_SLICES, TOKEN_DIM):
                        raise ValueError(
                            f"{case_id}: expected tokens ({NUM_SLICES}, {TOKEN_DIM}), "
                            f"got {tuple(tokens.shape)}"
                        )
                    embedding = tokens.reshape(-1).contiguous()
                    entry = {
                        "embedding": embedding,
                        "tokens": tokens,
                        "slice_indices": torch.tensor(slice_indices, dtype=torch.long),
                        "augmentation_seed": -1 if seed is None else seed,
                        "input_shape": torch.tensor(stack.shape[-2:], dtype=torch.long),
                        "fov": args.fov,
                    }
                    torch.save(entry, out_path)
                    break
                if entry is None:
                    warnings.warn(
                        f"{case_id} aug{augmentation_id}: lesion lost in every attempt, skipped",
                        stacklevel=1,
                    )
                    continue

            embedding = entry["embedding"]
            slice_indices = entry["slice_indices"].tolist()
            if len(slice_indices) != NUM_SLICES:
                raise ValueError(
                    f"{case_id}: expected {NUM_SLICES} slice indices, got {len(slice_indices)}"
                )
            row = {
                "case_id": case_id,
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "label": label,
                "mask_source": mask_source or "",
                "augmentation_id": augmentation_id,
                "augmentation_seed": int(entry["augmentation_seed"]),
                "fov": str(entry.get("fov", args.fov)),
                "margin": margin_value,
                "z_margin": args.z_margin,
                "slice_0": int(slice_indices[0]),
                "slice_1": int(slice_indices[1]),
                "slice_2": int(slice_indices[2]),
                "input_height": int(entry["input_shape"][0]),
                "input_width": int(entry["input_shape"][1]),
                "feature_dim": int(embedding.numel()),
                "file": out_path.name,
            }
            row.update(embedding_to_feature_row(embedding))
            rows.append(row)

        elapsed = time.time() - start
        print(f"  [{lesion_index}/{len(lesions)}] {case_id} label={label} ({elapsed / 60:.1f} min)")

    feature_columns = [f"feature_{index}" for index in range(EMBEDDING_DIM)]
    index = pd.DataFrame(rows, columns=[*META_COLUMNS, *feature_columns])
    index.to_csv(output_csv, index=False)
    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
    print(f"Wrote {len(index)} variants of {index['case_id'].nunique()} lesions -> {output_csv}")
    print(f"Embedding cache ({total_bytes / 1024**3:.2f} GB) -> {cache_dir}")
    print(f"FOV: {args.fov}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
