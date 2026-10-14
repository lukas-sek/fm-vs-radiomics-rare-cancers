"""Extract nnFoundationViT embeddings from nnU-Net-preprocessed lesion cases.

The input must have been produced by ``nnUNetv2_preprocess_like_nnssl`` with
the nnFoundationViT checkpoint. For each connected foreground component in the
preprocessed segmentation, this script crops the plan's patch size around the
component, runs the frozen Primus encoder (patch embed + EVA), and mean-pools
its tokens.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage


def discover_repo_root() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "scripts" / "build_nnFoundation_lesion_patches.py").is_file():
            return parent
    # Standard repository layout: experiments/nnFoundation/this_script.py.
    return script_path.parents[2]


REPO_ROOT = discover_repo_root()
WORC_DATASETS = ("Lipo", "Desmoid", "Liver", "GIST", "CRLM", "Melanoma")
DEFAULT_CHECKPOINT_SOURCE = "https://huggingface.co/MIC-DKFZ/nnFoundationViT"
DEFAULT_REPO_ID = "MIC-DKFZ/nnFoundationViT"
DEFAULT_FILENAME = "checkpoint_final.pth"
PLANS_GLOB = "ptPlans__nnFoundationViT____*.json"
PRETRAIN_MODEL_NAME = "nnFoundationViT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one mean-pooled nnFoundationViT embedding per lesion "
            "component from nnU-Net-preprocessed data."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help=(
            "WORC cohort name (for example Melanoma) to prepare from data/worc, "
            "or an existing nnU-Net dataset ID/name."
        ),
    )
    parser.add_argument(
        "-p",
        "--plans",
        default=None,
        help=(
            "Generated pretraining plans name (without .json) or an explicit "
            "plans JSON path. Optional for WORC cohort names."
        ),
    )
    parser.add_argument(
        "-c",
        "--configuration",
        default="3d_fullres",
        help="Plans configuration to use (default: 3d_fullres).",
    )
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=None,
        help=(
            "Override nnUNet_preprocessed. Otherwise the nnUNet_preprocessed "
            "environment variable is used."
        ),
    )
    parser.add_argument(
        "--worc-root",
        type=Path,
        default=REPO_ROOT / "data" / "worc",
        help="Source WORC root used for cohort names (default: data/worc).",
    )
    parser.add_argument(
        "--nnunet-raw-root",
        type=Path,
        default=None,
        help="Override nnUNet_raw (default: environment or data/nnUNet_raw).",
    )
    parser.add_argument(
        "--nnunet-results-root",
        type=Path,
        default=None,
        help=(
            "Override nnUNet_results (default: environment or "
            "data/nnUNet_results)."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        help="Override the built-in nnU-Net ID used for a WORC cohort.",
    )
    parser.add_argument(
        "--overwrite-preparation",
        action="store_true",
        help="Recreate the nnU-Net raw dataset from data/worc.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override pretrain_info.checkpoint_path from the generated plan.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Override the output CSV. By default, writes "
            "Results/nnFoundation/<DATASET>_nnvit_features.csv "
            "(or *_nnvit_augmented_features.csv when --num-augmentations > 0)."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda when available, otherwise cpu).",
    )
    parser.add_argument(
        "--min-component-voxels",
        type=int,
        default=1,
        help="Ignore smaller foreground components (default: 1).",
    )
    parser.add_argument(
        "--empty-mask",
        choices=("error", "skip", "center"),
        default="error",
        help=(
            "Action for cases without foreground: error, skip, or extract one "
            "image-centered patch (default: error)."
        ),
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed-precision inference.",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=0,
        help=(
            "Number of nnU-Net default training augmentations to extract per "
            "lesion in addition to the original (default: 0)."
        ),
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=42,
        help="Global seed for reproducible nnU-Net augmentations (default: 42).",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint-filename", default=DEFAULT_FILENAME)
    return parser.parse_args()


def resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def default_output_csv(
    dataset_name: str, num_augmentations: int = 0
) -> Path:
    cohort_name = (
        dataset_name.split("_", maxsplit=1)[1]
        if dataset_name.startswith("Dataset") and "_" in dataset_name
        else dataset_name
    )
    if cohort_name.startswith("WORC_"):
        cohort_name = cohort_name.removeprefix("WORC_")
    suffix = (
        "nnvit_augmented_features"
        if num_augmentations > 0
        else "nnvit_features"
    )
    return REPO_ROOT / "Results" / "nnFoundation" / f"{cohort_name}_{suffix}.csv"


def import_nnunet() -> dict[str, Any]:
    """Import nnU-Net lazily so utility functions remain independently testable."""
    try:
        from nnunetv2.training.dataloading.nnunet_dataset import (
            infer_dataset_class,
        )
        from nnunetv2.training.nnUNetTrainer.pretraining.pretrainedTrainer import (
            PretrainedTrainer,
        )
        from nnunetv2.utilities.dataset_name_id_conversion import (
            maybe_convert_to_dataset_name,
        )
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    except ImportError as error:
        raise RuntimeError(
            "This extractor requires the nnssl-enabled nnU-Net build pinned in "
            "requirements.txt. Install the project requirements in the same "
            "environment that runs this script. Underlying import error: "
            f"{error}"
        ) from error

    return {
        "PlansManager": PlansManager,
        "PretrainedTrainer": PretrainedTrainer,
        "infer_dataset_class": infer_dataset_class,
        "maybe_convert_to_dataset_name": maybe_convert_to_dataset_name,
    }


def resolve_dataset_name(
    dataset: str, maybe_convert_to_dataset_name: Any
) -> str:
    if dataset.startswith("Dataset"):
        return dataset
    try:
        return str(maybe_convert_to_dataset_name(int(dataset)))
    except ValueError as error:
        raise ValueError(
            "--dataset must be an integer ID or a DatasetXXX_Name"
        ) from error


def resolve_nnunet_root(
    override: Path | None,
    environment_name: str,
    fallback: Path,
    create: bool,
) -> Path:
    if override is not None:
        root = resolve_cli_path(override)
    else:
        configured = os.environ.get(environment_name)
        root = (
            Path(configured).expanduser().resolve()
            if configured
            else fallback.resolve()
        )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(f"{environment_name} directory not found: {root}")
    os.environ[environment_name] = str(root)
    return root


def resolve_plans_path(dataset_root: Path, plans: str) -> Path:
    candidate = Path(plans)
    explicit = (
        candidate.is_absolute()
        or candidate.suffix.lower() == ".json"
        or candidate.parent != Path(".")
    )
    path = resolve_cli_path(candidate) if explicit else dataset_root / f"{plans}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Plans file not found: {path}")
    return path


def find_generated_plans(dataset_root: Path) -> Path:
    candidates = sorted(dataset_root.glob(PLANS_GLOB))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one generated nnFoundationViT plan under "
            f"{dataset_root}, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def run_command(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Command {command[0]!r} was not found in the active environment"
        ) from error


def ensure_nnunet_preprocessing(
    dataset_name: str,
    dataset_root: Path,
    checkpoint: Path | None,
) -> Path:
    plans_candidates = sorted(dataset_root.glob(PLANS_GLOB))
    if plans_candidates:
        plans_path = find_generated_plans(dataset_root)
        plans = load_json(plans_path)
        data_identifier = plans["configurations"]["3d_fullres"]["data_identifier"]
        if (dataset_root / data_identifier).is_dir():
            return plans_path

    dataset_id = int(dataset_name[7:10])
    if not (dataset_root / "dataset_fingerprint.json").is_file():
        run_command(["nnUNetv2_extract_fingerprint", "-d", str(dataset_id)])
    if not (dataset_root / "nnUNetPlans.json").is_file():
        run_command(["nnUNetv2_plan_experiment", "-d", str(dataset_id)])

    checkpoint_source = (
        str(resolve_cli_path(checkpoint))
        if checkpoint is not None
        else DEFAULT_CHECKPOINT_SOURCE
    )
    if checkpoint is None and not os.environ.get("nnssl_pretrained_models"):
        checkpoint_cache = REPO_ROOT / "data" / "nnssl_pretrained_models"
        checkpoint_cache.mkdir(parents=True, exist_ok=True)
        os.environ["nnssl_pretrained_models"] = str(checkpoint_cache.resolve())
    run_command(
        [
            "nnUNetv2_preprocess_like_nnssl",
            "-d",
            str(dataset_id),
            "-n",
            PRETRAIN_MODEL_NAME,
            "-pc",
            checkpoint_source,
            "-am",
            "like_pretrained",
        ]
    )
    return find_generated_plans(dataset_root)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def component_centers(
    segmentation: np.ndarray,
    min_component_voxels: int = 1,
) -> Iterator[tuple[int, int, tuple[int, int, int], int]]:
    """Yield label, component ID, center and size for positive 3D components."""
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be at least 1")
    seg = np.asarray(segmentation)
    if seg.ndim == 4:
        if seg.shape[0] != 1:
            raise ValueError(
                f"Expected one segmentation channel, got shape {seg.shape}"
            )
        seg = seg[0]
    if seg.ndim != 3:
        raise ValueError(f"Expected a 3D segmentation, got shape {seg.shape}")

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    positive_labels = np.unique(seg[seg > 0])
    for raw_label in positive_labels:
        label_value = int(raw_label)
        components, count = ndimage.label(seg == raw_label, structure=structure)
        for component_id in range(1, count + 1):
            coordinates = np.argwhere(components == component_id)
            voxel_count = int(coordinates.shape[0])
            if voxel_count < min_component_voxels:
                continue
            center = tuple(
                int(np.rint(value)) for value in coordinates.mean(axis=0)
            )
            yield label_value, component_id, center, voxel_count


def crop_or_pad(
    data: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
    padding_value: float | int = 0,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    """Return a spatially centered crop with zero-padding when necessary."""
    array = np.asarray(data)
    if array.ndim != 4:
        raise ValueError(f"Expected data shaped (C, X, Y, Z), got {array.shape}")
    if len(center) != 3 or len(patch_size) != 3:
        raise ValueError("center and patch_size must contain three values")

    center = tuple(int(value) for value in center)
    patch_size = tuple(int(value) for value in patch_size)
    if any(size < 1 for size in patch_size):
        raise ValueError(f"Invalid patch size: {patch_size}")

    output = np.full(
        (array.shape[0], *patch_size),
        fill_value=padding_value,
        dtype=dtype,
    )
    source_slices: list[slice] = [slice(None)]
    target_slices: list[slice] = [slice(None)]
    for axis, (axis_center, requested, available) in enumerate(
        zip(center, patch_size, array.shape[1:])
    ):
        start = axis_center - requested // 2
        stop = start + requested
        source_start = max(start, 0)
        source_stop = min(stop, available)
        if source_start >= source_stop:
            raise ValueError(
                f"Center {center} lies outside data shape {array.shape[1:]}"
            )
        target_start = source_start - start
        target_stop = target_start + (source_stop - source_start)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))

    output[tuple(target_slices)] = array[tuple(source_slices)]
    if not np.isfinite(output).all():
        raise ValueError("Input patch contains non-finite values")
    return output


class _NnUNetAugmentationContext:
    """Minimal context required by nnU-Net's augmentation geometry method."""

    def __init__(self, configuration_manager: Any) -> None:
        self.configuration_manager = configuration_manager
        self.inference_allowed_mirroring_axes: tuple[int, ...] | None = None

    @staticmethod
    def print_to_log_file(*args: Any, **kwargs: Any) -> None:
        del args, kwargs


def build_default_nnunet_augmentation(
    configuration_manager: Any,
    label_manager: Any,
    nnunet: dict[str, Any],
) -> tuple[Any, tuple[int, int, int]]:
    """Build nnU-Net's own transform chain and enlarged input crop."""
    patch_size = np.asarray(configuration_manager.patch_size, dtype=int)
    if patch_size.shape != (3,):
        raise ValueError(
            "nnFoundationViT augmentation requires a 3D patch size, got "
            f"{tuple(patch_size)}"
        )

    trainer_class = nnunet["PretrainedTrainer"]
    geometry_context = _NnUNetAugmentationContext(configuration_manager)
    (
        rotation,
        do_dummy_2d,
        initial_patch_size,
        mirror_axes,
    ) = trainer_class.configure_rotation_dummyDA_mirroring_and_inital_patch_size(
        geometry_context
    )

    transform = trainer_class.get_training_transforms(
        patch_size=tuple(int(value) for value in patch_size),
        rotation_for_DA=rotation,
        deep_supervision_scales=None,
        mirror_axes=mirror_axes,
        do_dummy_2d_data_aug=do_dummy_2d,
        use_mask_for_norm=configuration_manager.use_mask_for_norm,
        is_cascaded=False,
        foreground_labels=label_manager.foreground_labels,
        regions=(
            label_manager.foreground_regions
            if label_manager.has_regions
            else None
        ),
        ignore_label=label_manager.ignore_label,
    )
    return transform, tuple(int(value) for value in initial_patch_size)


def variant_seed(
    global_seed: int,
    case_id: str,
    label_value: int,
    component_id: int,
    augmentation_id: int,
) -> int:
    identity = (
        f"{global_seed}\0{case_id}\0{label_value}\0"
        f"{component_id}\0{augmentation_id}"
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(identity, digest_size=8).digest(), "little"
    ) % (2**63 - 1)


def apply_nnunet_augmentation(
    data: np.ndarray,
    segmentation: np.ndarray,
    center: Sequence[int],
    initial_patch_size: Sequence[int],
    transform: Any,
    seed: int,
) -> np.ndarray:
    """Apply nnU-Net's stochastic training transforms reproducibly."""
    data_crop = crop_or_pad(data, center, initial_patch_size)
    segmentation_crop = crop_or_pad(
        segmentation,
        center,
        initial_patch_size,
        padding_value=-1,
        dtype=np.int16,
    )
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            transformed = transform(
                image=torch.from_numpy(data_crop).float(),
                segmentation=torch.from_numpy(segmentation_crop).to(torch.int16),
            )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)

    image = transformed["image"]
    if not isinstance(image, torch.Tensor):
        raise RuntimeError(
            f"nnU-Net transform returned unexpected image type {type(image)}"
        )
    result = image.float().cpu().numpy()
    if result.ndim != 4 or not np.isfinite(result).all():
        raise RuntimeError(
            f"nnU-Net augmentation produced invalid shape or values: {result.shape}"
        )
    return result


def _architecture_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    try:
        return checkpoint["nnssl_adaptation_plan"]["architecture_plans"][
            "arch_kwargs"
        ]
    except KeyError as error:
        raise RuntimeError(
            "Checkpoint does not contain nnssl_adaptation_plan."
            "architecture_plans.arch_kwargs"
        ) from error


def _matching_encoder_weights(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    model_state = model.state_dict()
    prefixes = ("down_projection.", "eva.")
    matching = {
        key: value
        for key, value in state_dict.items()
        if key.startswith(prefixes)
        and key in model_state
        and model_state[key].shape == value.shape
    }
    for prefix in prefixes:
        if not any(key.startswith(prefix) for key in matching):
            raise RuntimeError(
                f"No compatible checkpoint weights found for '{prefix[:-1]}'"
            )
    return matching


def resolve_checkpoint_path(
    plans: dict[str, Any],
    checkpoint_override: Path | None,
    repo_id: str,
    checkpoint_filename: str,
) -> Path:
    if checkpoint_override is not None:
        path = resolve_cli_path(checkpoint_override)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    pretrain_info = plans.get("pretrain_info")
    if isinstance(pretrain_info, dict):
        raw_checkpoint = pretrain_info.get("checkpoint_path")
        if raw_checkpoint:
            path = Path(str(raw_checkpoint)).expanduser().resolve()
            if path.is_file():
                return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required to download nnFoundationViT weights"
        ) from error
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=checkpoint_filename)
    )
    return downloaded.resolve()


def load_nnfoundation_vit(
    checkpoint_path: Path,
    device: str | torch.device = "cuda",
) -> Any:
    try:
        from dynamic_network_architectures.architectures.primus import Primus
    except ImportError as error:
        raise RuntimeError(
            "dynamic_network_architectures is required to load Primus/nnFoundationViT"
        ) from error

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    arch_kwargs = _architecture_kwargs(checkpoint)
    input_shape = tuple(int(value) for value in arch_kwargs["input_shape"])
    if len(input_shape) != 3:
        raise RuntimeError(f"Expected 3D Primus input_shape, got {input_shape}")

    model = Primus(
        input_channels=int(arch_kwargs["input_channels"]),
        embed_dim=int(arch_kwargs["embed_dim"]),
        patch_embed_size=tuple(arch_kwargs["patch_embed_size"]),
        num_classes=int(arch_kwargs["output_channels"]),
        eva_depth=int(arch_kwargs["encoder_eva_depth"]),
        eva_numheads=int(arch_kwargs["encoder_eva_numheads"]),
        input_shape=input_shape,
        drop_path_rate=0.0,
        patch_drop_rate=0.0,
        attn_drop_rate=float(arch_kwargs.get("attn_drop_rate", 0.0)),
        init_values=arch_kwargs.get("init_values"),
        scale_attn_inner=bool(arch_kwargs.get("scale_attn_inner", False)),
    )
    # Primus uses input_shape only to build RoPE/ref features and does not
    # retain it as an attribute; keep it for patch-size validation.
    model.input_shape = input_shape

    compatible_weights = _matching_encoder_weights(
        model, checkpoint["network_weights"]
    )
    load_result = model.load_state_dict(compatible_weights, strict=False)
    loaded_eva = sum(key.startswith("eva.") for key in compatible_weights)
    loaded_stem = sum(
        key.startswith("down_projection.") for key in compatible_weights
    )
    print(
        f"Loaded {loaded_eva} EVA and {loaded_stem} patch-embedding tensors "
        f"from {checkpoint_path}"
    )
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected compatible checkpoint keys: {load_result.unexpected_keys[:5]}"
        )

    del checkpoint, compatible_weights
    gc.collect()

    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def primus_input_shape(model: Any) -> tuple[int, int, int]:
    shape = getattr(model, "input_shape", None)
    if shape is None:
        raise RuntimeError("Primus model is missing input_shape")
    resolved = tuple(int(value) for value in shape)
    if len(resolved) != 3:
        raise RuntimeError(f"Expected 3D Primus input_shape, got {resolved}")
    return resolved


@torch.inference_mode()
def extract_embedding(
    model: Any,
    patch: np.ndarray,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    expected = primus_input_shape(model)
    spatial = tuple(int(size) for size in patch.shape[1:])
    if spatial != expected:
        raise ValueError(
            f"Patch spatial size {spatial} does not match Primus "
            f"input_shape {expected}"
        )

    tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
    tensor = tensor.to(device, non_blocking=device.type == "cuda")
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        embedded = model.down_projection(tensor)
        tokens = embedded.flatten(2).transpose(1, 2)
        tokens, _ = model.eva(tokens)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected (B, tokens, features), got {tokens.shape}"
            )
        embedding = tokens.mean(dim=1)
    result = embedding.squeeze(0).float().cpu().numpy()
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid embedding shape or values: {result.shape}")
    expected_dim = int(getattr(model, "embed_dim", result.size))
    if result.size != expected_dim:
        raise RuntimeError(
            f"Expected {expected_dim} features, got {result.size}"
        )
    return result


def resolve_plans_managers(
    plans: dict[str, Any],
    configuration: str,
    dataset_json: dict[str, Any],
    nnunet: dict[str, Any],
) -> tuple[Any, Any]:
    plans_manager = nnunet["PlansManager"](plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset_json)
    return configuration_manager, label_manager


def extract_dataset(
    model: Any,
    dataset: Any,
    patch_size: Sequence[int],
    device: torch.device,
    min_component_voxels: int,
    empty_mask: str,
    use_amp: bool,
    num_augmentations: int = 0,
    augmentation_seed: int = 42,
    augmentation_transform: Any | None = None,
    initial_patch_size: Sequence[int] | None = None,
) -> pd.DataFrame:
    if num_augmentations < 0:
        raise ValueError("num_augmentations must not be negative")
    if num_augmentations > 0 and (
        augmentation_transform is None or initial_patch_size is None
    ):
        raise ValueError(
            "Augmentation transform and initial patch size are required when "
            "num_augmentations is positive"
        )
    expected_shape = primus_input_shape(model)
    resolved_patch = tuple(int(value) for value in patch_size)
    if resolved_patch != expected_shape:
        raise ValueError(
            f"Plans patch size {resolved_patch} does not match Primus "
            f"input_shape {expected_shape}"
        )

    rows: list[dict[str, Any]] = []
    feature_count: int | None = None

    for case_number, case_id in enumerate(dataset.identifiers, start=1):
        data, segmentation, _ = dataset.load_case(case_id)
        data = np.asarray(data, dtype=np.float32)
        segmentation = np.asarray(segmentation)
        components = list(
            component_centers(segmentation, min_component_voxels)
        )
        if not components:
            if empty_mask == "error":
                raise ValueError(f"{case_id}: segmentation has no foreground")
            if empty_mask == "skip":
                print(f"Skipping {case_id}: segmentation has no foreground")
                continue
            image_center = tuple(int(size // 2) for size in data.shape[1:])
            components = [(0, 0, image_center, 0)]

        for label_value, component_id, center, voxel_count in components:
            variants: list[tuple[int, str, str, np.ndarray]] = [
                (
                    0,
                    "original",
                    "",
                    crop_or_pad(data, center, patch_size),
                )
            ]
            for augmentation_id in range(1, num_augmentations + 1):
                seed = variant_seed(
                    augmentation_seed,
                    str(case_id),
                    label_value,
                    component_id,
                    augmentation_id,
                )
                variants.append(
                    (
                        augmentation_id,
                        "nnunet_default",
                        str(seed),
                        apply_nnunet_augmentation(
                            data,
                            segmentation,
                            center,
                            initial_patch_size,
                            augmentation_transform,
                            seed,
                        ),
                    )
                )

            for augmentation_id, augmentation_type, seed, patch in variants:
                embedding = extract_embedding(
                    model, patch, device=device, use_amp=use_amp
                )
                if feature_count is None:
                    feature_count = int(embedding.size)
                elif embedding.size != feature_count:
                    raise RuntimeError(
                        "Feature dimensionality changed between lesion patches"
                    )
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "segmentation_label": label_value,
                    "component_id": component_id,
                    "component_voxels": voxel_count,
                    "center_x": center[0],
                    "center_y": center[1],
                    "center_z": center[2],
                    "augmentation_id": augmentation_id,
                    "augmentation_type": augmentation_type,
                    "augmentation_seed": seed,
                }
                row.update(
                    {
                        f"feature_{index}": float(value)
                        for index, value in enumerate(embedding)
                    }
                )
                rows.append(row)

        print(
            f"\rExtracted {case_number}/{len(dataset.identifiers)} cases",
            end="",
            flush=True,
        )

    print()
    if not rows:
        raise RuntimeError("No embeddings were extracted")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.min_component_voxels < 1:
        raise SystemExit("--min-component-voxels must be at least 1")
    if args.num_augmentations < 0:
        raise SystemExit("--num-augmentations must not be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    cohort_mode = args.dataset in WORC_DATASETS
    preprocessed_root = resolve_nnunet_root(
        args.preprocessed_root,
        "nnUNet_preprocessed",
        REPO_ROOT / "data" / "nnUNet_preprocessed",
        create=cohort_mode,
    )
    raw_root = resolve_nnunet_root(
        args.nnunet_raw_root,
        "nnUNet_raw",
        REPO_ROOT / "data" / "nnUNet_raw",
        create=True,
    )
    resolve_nnunet_root(
        args.nnunet_results_root,
        "nnUNet_results",
        REPO_ROOT / "data" / "nnUNet_results",
        create=True,
    )

    case_metadata: pd.DataFrame | None = None
    if cohort_mode:
        scripts_dir = REPO_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from prepare_worc_nnunet_dataset import (
            prepare_worc_nnunet_dataset,
        )

        dataset_name, case_metadata = prepare_worc_nnunet_dataset(
            args.dataset,
            resolve_cli_path(args.worc_root),
            raw_root,
            preprocessed_root,
            dataset_id=args.dataset_id,
            overwrite=args.overwrite_preparation,
        )
    else:
        dataset_name = ""

    nnunet = import_nnunet()
    if not cohort_mode:
        dataset_name = resolve_dataset_name(
            args.dataset, nnunet["maybe_convert_to_dataset_name"]
        )
    dataset_root = preprocessed_root / dataset_name
    if cohort_mode and args.plans is None:
        plans_path = ensure_nnunet_preprocessing(
            dataset_name,
            dataset_root,
            args.checkpoint,
        )
    elif args.plans is not None:
        plans_path = resolve_plans_path(dataset_root, args.plans)
    else:
        raise ValueError(
            "--plans is required when --dataset is an existing nnU-Net "
            "dataset ID/name"
        )

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Preprocessed dataset directory not found: {dataset_root}"
        )

    plans = load_json(plans_path)
    dataset_json_path = dataset_root / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")
    dataset_json = load_json(dataset_json_path)

    configuration_manager, label_manager = resolve_plans_managers(
        plans,
        args.configuration,
        dataset_json,
        nnunet,
    )
    checkpoint_path = resolve_checkpoint_path(
        plans,
        args.checkpoint,
        args.repo_id,
        args.checkpoint_filename,
    )
    model = load_nnfoundation_vit(checkpoint_path, device=device)

    data_folder = dataset_root / configuration_manager.data_identifier
    if not data_folder.is_dir():
        raise FileNotFoundError(
            f"Preprocessed configuration directory not found: {data_folder}"
        )
    dataset_class = nnunet["infer_dataset_class"](str(data_folder))
    dataset = dataset_class(str(data_folder))
    print(
        f"Reading {len(dataset.identifiers)} cases from {data_folder}; "
        f"patch size {tuple(configuration_manager.patch_size)}"
    )
    augmentation_transform = None
    initial_patch_size = None
    if args.num_augmentations > 0:
        augmentation_transform, initial_patch_size = (
            build_default_nnunet_augmentation(
                configuration_manager,
                label_manager,
                nnunet,
            )
        )
        print(
            f"Extracting original + {args.num_augmentations} default nnU-Net "
            f"augmentations per lesion; augmentation input crop "
            f"{initial_patch_size}"
        )

    features = extract_dataset(
        model,
        dataset,
        configuration_manager.patch_size,
        device,
        args.min_component_voxels,
        args.empty_mask,
        use_amp=not args.no_amp,
        num_augmentations=args.num_augmentations,
        augmentation_seed=args.augmentation_seed,
        augmentation_transform=augmentation_transform,
        initial_patch_size=initial_patch_size,
    )
    if case_metadata is None:
        metadata_path = dataset_root / "worc_case_metadata.csv"
        if metadata_path.is_file():
            case_metadata = pd.read_csv(metadata_path)
    if case_metadata is not None:
        features = features.merge(
            case_metadata,
            on="case_id",
            how="left",
            validate="many_to_one",
        )
    output_csv = (
        resolve_cli_path(args.output_csv)
        if args.output_csv is not None
        else default_output_csv(dataset_name, args.num_augmentations)
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)
    feature_columns = sum(
        column.startswith("feature_") for column in features.columns
    )
    print(
        f"Wrote {len(features)} lesion embeddings with {feature_columns} "
        f"features -> {output_csv}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
