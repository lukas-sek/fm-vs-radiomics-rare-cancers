"""Cache frozen MRI-CORE 3-slice concatenated embeddings per WORC lesion.

Loads full in-plane slices over the lesion Z extent, selects 3 interior axial
slices, resizes to 1024×1024 RGB in ``[0, 1]``, runs the frozen SAM ViT-B
``image_encoder``, global-average-pools each slice embedding to 256-d, and
concatenates them into one 768-d vector.

Default is **no augmentations** (MRI-CORE linear-probing protocol). For
COMET-style MLP K-cached training, extract ``K`` intensity-jitter + flip
variants (COMET default ``K=10``) and pass ``--use-cached-augs`` to the MLP::

    python experiments/MRI-Core/feature_extraction.py --dataset Liver --num-augmentations 10
    python experiments/MRI-Core/MLP_classification.py --dataset Liver --use-cached-augs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import pandas as pd
import torch

MRI_CORE_DIR = Path(__file__).resolve().parent
# Supports both ``<repo>/experiments/MRI-Core`` and ``~/experiments/MRI-Core``.
EXPERIMENTS_DIR = MRI_CORE_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
COMET_DIR = EXPERIMENTS_DIR / "COMET"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _discover_mri_foundation() -> Path:
    """Locate vendored mazurowski-lab/mri_foundation (cfg + models.sam)."""
    candidates = (
        REPO_ROOT / "third_party" / "mri_foundation",
        EXPERIMENTS_DIR / "third_party" / "mri_foundation",
        MRI_CORE_DIR / "mri_foundation",
        Path.home() / "third_party" / "mri_foundation",
    )
    for candidate in candidates:
        if (candidate / "models" / "sam").is_dir() and (candidate / "cfg.py").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


THIRD_PARTY = _discover_mri_foundation()
# Prefer third_party (models.sam / cfg) then scripts; load preprocess modules by path.
for path in (SCRIPTS_DIR, THIRD_PARTY):
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

import importlib.util as _ilu  # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_local_preprocess = _load_module("mri_core_preprocess", MRI_CORE_DIR / "preprocess.py")
DEFAULT_IMAGE_SIZE = _local_preprocess.DEFAULT_IMAGE_SIZE
NUM_SLICES = _local_preprocess.NUM_SLICES
augment_lesion_patch = _local_preprocess.augment_lesion_patch
three_slice_rgb_stack = _local_preprocess.three_slice_rgb_stack

_comet_preprocess = _load_module("comet_worc_preprocess", COMET_DIR / "preprocess.py")
load_worc_lesion_slices = _comet_preprocess.load_worc_lesion_slices

DEFAULT_CHECKPOINT = REPO_ROOT / "data" / "MRI-Core" / "mri_foundation.pth"
DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "MRI-Core" / "cache"
DEFAULT_FEATURES_DIR = REPO_ROOT / "Results" / "MRI-Core" / "Features"
TOKEN_DIM = 256
EMBEDDING_DIM = TOKEN_DIM * NUM_SLICES
MAX_AUGMENTATION_ATTEMPTS = 10
META_COLUMNS = (
    "case_id",
    "patient_id",
    "lesion_id",
    "label",
    "mask_source",
    "augmentation_id",
    "augmentation_seed",
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
        "--z-margin",
        type=int,
        default=0,
        help="Extra slices kept beyond the lesion Z extent (full in-plane FOV).",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=0,
        help=(
            "Augmented variants per lesion in addition to the original "
            "(default: 0; MRI-CORE linear probing uses no augmentations). "
            "Use 10 to match COMET Slideaggregation K-cached MLP training."
        ),
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=3, help="Slices per encoder forward pass.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast on CUDA.")
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Disable cuDNN (workaround for driver mismatches).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MRI-CORE ViT-B checkpoint (.pth).",
    )
    parser.add_argument(
        "--mri-foundation-dir",
        type=Path,
        default=None,
        help=(
            "Path to cloned mazurowski-lab/mri_foundation (contains cfg.py and models/). "
            f"Default: auto-detect (currently {THIRD_PARTY})."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Encoder input resolution (must match checkpoint; default 1024).",
    )
    parser.add_argument("--worc-root", type=Path, default=WORC_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Default: data/MRI-Core/cache/<dataset>_slideagg.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Feature CSV. Default: Results/MRI-Core/Features/"
            "<dataset>_slideagg_features.csv when --num-augmentations is 0, else "
            "<dataset>_slideagg_augmented_features.csv."
        ),
    )
    parser.add_argument("--patients", nargs="+", default=None, help="Only these patient IDs.")
    parser.add_argument("--max-patients", type=int, default=None, help="Only the first N patients.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing cache files.")
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_output_csv(dataset: str, num_augmentations: int) -> Path:
    suffix = "slideagg_augmented_features" if num_augmentations > 0 else "slideagg_features"
    return DEFAULT_FEATURES_DIR / f"{dataset}_{suffix}.csv"


def variant_seed(base_seed: int, case_id: str, augmentation_id: int, attempt: int) -> int:
    case_offset = zlib.crc32(case_id.encode("utf-8"))
    return (base_seed * 1_000_003 + case_offset + augmentation_id * 1_009 + attempt) % (2**31 - 1)


def load_mri_core_encoder(
    checkpoint: Path,
    image_size: int,
    device: torch.device,
    mri_foundation_dir: Path | None = None,
) -> torch.nn.Module:
    """Load frozen MRI-CORE SAM ViT-B image encoder (adapters disabled)."""
    foundation_dir = (
        resolve_cli_path(mri_foundation_dir)
        if mri_foundation_dir is not None
        else THIRD_PARTY
    )
    if not (foundation_dir / "models" / "sam").is_dir() or not (foundation_dir / "cfg.py").is_file():
        raise FileNotFoundError(
            f"Missing MRI-CORE upstream code at {foundation_dir}. "
            "On the server run:\n"
            "  git clone --depth 1 https://github.com/mazurowski-lab/mri_foundation.git "
            f"{REPO_ROOT / 'third_party' / 'mri_foundation'}\n"
            "Or pass --mri-foundation-dir /path/to/mri_foundation."
        )
    if str(foundation_dir) not in sys.path:
        sys.path.insert(0, str(foundation_dir))

    checkpoint = resolve_cli_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MRI-CORE checkpoint not found: {checkpoint}")

    import cfg  # noqa: E402
    from models.sam import sam_model_registry  # noqa: E402

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        args = cfg.parse_args()
    finally:
        sys.argv = saved_argv

    args.if_encoder_adapter = False
    args.if_mask_decoder_adapter = False
    args.if_encoder_lora_layer = False
    args.if_decoder_lora_layer = False
    args.image_size = int(image_size)
    args.num_cls = int(getattr(args, "num_cls", 1))

    model = sam_model_registry["vit_b"](
        args,
        checkpoint=str(checkpoint),
        num_classes=args.num_cls,
        image_size=args.image_size,
        pretrained_sam=False,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_slice_embeddings(
    image_encoder: torch.nn.Module,
    images: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Encode ``(S, 3, H, W)`` → GAP-pooled ``(S, 256)`` float16 CPU tokens."""
    tokens = []
    for chunk in images.split(batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            emb = image_encoder(chunk.to(device, non_blocking=True))
        if emb.ndim != 4:
            raise ValueError(f"Expected image_encoder output (B, C, H, W), got {tuple(emb.shape)}")
        pooled = emb.float().mean(dim=(-2, -1))
        if pooled.ndim != 2 or pooled.shape[-1] != TOKEN_DIM:
            raise ValueError(f"Expected pooled (B, {TOKEN_DIM}), got {tuple(pooled.shape)}")
        if not torch.isfinite(pooled).all():
            raise FloatingPointError("Encoder produced non-finite embeddings")
        tokens.append(pooled.to(torch.float16).cpu())
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
    if args.num_augmentations < 0:
        raise SystemExit("--num-augmentations must not be negative")
    if args.image_size < 1:
        raise SystemExit("--image-size must be positive")
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
        else DEFAULT_CACHE_ROOT / f"{args.dataset}_slideagg"
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
            "feature": f"image_encoder_gap_{TOKEN_DIM}_x{NUM_SLICES}_concat",
            "embedding_dim": EMBEDDING_DIM,
            "fov": "full",
            "encoder_frozen": True,
            "third_party": str(THIRD_PARTY),
        }
    )
    (cache_dir / "cache_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = load_mri_core_encoder(
        args.checkpoint,
        args.image_size,
        device,
        mri_foundation_dir=args.mri_foundation_dir,
    )
    image_encoder = model.image_encoder

    pinfo = load_pinfo(args.dataset)
    rows: list[dict] = []
    start = time.time()
    lesions = list(iter_lesions(args.dataset, worc_root, args.patients, args.max_patients))
    print(
        f"{args.dataset}: {len(lesions)} lesions, K={args.num_augmentations}, "
        f"fov=full, z_margin={args.z_margin}, embedding_dim={EMBEDDING_DIM}, "
        f"image_size={args.image_size}, device={device}"
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
                    volume = load_worc_lesion_slices(
                        resolve_image_path(patient_dir, lesion_id),
                        mask_path,
                        z_margin=args.z_margin,
                    )
                entry = None
                for attempt in range(MAX_AUGMENTATION_ATTEMPTS if augmentation_id else 1):
                    seed = None if augmentation_id == 0 else variant_seed(
                        args.augmentation_seed, case_id, augmentation_id, attempt
                    )
                    try:
                        image = volume["image"].clone()
                        label_mask = volume["label"].clone()
                        if seed is not None:
                            image, label_mask = augment_lesion_patch(image, label_mask, seed)
                        stack, slice_indices = three_slice_rgb_stack(
                            image, label_mask, image_size=args.image_size
                        )
                    except ValueError as error:
                        if augmentation_id == 0:
                            raise ValueError(f"{case_id}: {error}") from error
                        continue
                    tokens = encode_slice_embeddings(
                        image_encoder, stack, args.batch_size, device, amp
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
    # When augmented rows are present, also write the original-only CSV the MLP expects.
    if args.num_augmentations > 0 and args.output_csv is None:
        orig_csv = default_output_csv(args.dataset, num_augmentations=0)
        orig_rows = index.loc[index["augmentation_id"] == 0].copy()
        orig_rows.to_csv(orig_csv, index=False)
        print(f"Wrote {len(orig_rows)} original embeddings -> {orig_csv}")
    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
    print(f"Wrote {len(index)} variants of {index['case_id'].nunique()} lesions -> {output_csv}")
    print(f"Embedding cache ({total_bytes / 1024**3:.2f} GB) -> {cache_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
