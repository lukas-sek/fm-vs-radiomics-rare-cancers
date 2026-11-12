"""Entry point: GloViTa nnFoundation ViT + robust capped-unfreeze training."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from glovita.configs.data import Generic3dDatasetConfig
from glovita.configs.dataloading import DataloadingConfig
from glovita.configs.model import (
    ClassificationHeadConfig,
    ModelConfig,
    nnFoundationEncoderConfig,
)
from glovita.datasets.factory import build_dataloaders
from glovita.models.factory import build_model
from glovita.models.preprocessing import resolve_encoder_preprocessing_defaults

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vit_robust_ft import get_transformer_blocks, train_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust Primus/ViT fine-tune (capped unfreeze + LLRD + EMA).")
    p.add_argument("--exp_dir", type=Path, default=Path("Results/nnFoundation/finetune_vit_robust"))
    p.add_argument("--data_root_dir", type=Path, default=Path("data/glovita_worc/Liver"))
    p.add_argument("--fold", type=str, default="0")
    p.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("data/nnssl_pretrained_models/nnFoundationViT/checkpoint_final.pth"),
    )
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input_channels", type=int, default=1)
    p.add_argument("--head_init_std", type=float, default=0.01)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    data_cfg = Generic3dDatasetConfig(
        data_root_dir=args.data_root_dir,
        fold=args.fold,
    )
    dataloading = DataloadingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model_cfg = ModelConfig(
        encoder=nnFoundationEncoderConfig(
            checkpoint_path=args.checkpoint_path,
            input_channels=args.input_channels,
            pretrained=True,
        ),
        head=ClassificationHeadConfig(init_std=args.head_init_std),
    )

    encoder_preprocessing = resolve_encoder_preprocessing_defaults(model_cfg.encoder).as_kwargs()
    train_loader, val_loader, _ = build_dataloaders(
        data_cfg,
        dataloading,
        encoder_preprocessing=encoder_preprocessing,
    )

    num_classes = int(getattr(data_cfg, "num_classes", 2) or 2)
    model = build_model(model_cfg, output_dim=num_classes)
    n_blocks = len(get_transformer_blocks(model))
    print(f"Built model with {n_blocks} transformer blocks, num_classes={num_classes}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.exp_dir / f"{stamp}_vit_robust_fold{args.fold}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_root_dir": str(args.data_root_dir),
                "fold": args.fold,
                "checkpoint_path": str(args.checkpoint_path),
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "num_classes": num_classes,
                "schedule": {
                    "stage0_epochs": "0-9 head only, lr=1e-4",
                    "stage1_epochs": "10-19 block 39, lr=5e-5",
                    "stage2_epochs": "20+ blocks 38-39, lr=5e-5 (LLRD lower on 38)",
                    "decay_factor": 0.75,
                    "ema_alpha": 0.7,
                    "ema_metric": "val_AUROC",
                    "patience": 7,
                    "raw_patience": 7,
                    "label_smoothing": 0.1,
                },
            },
            indent=2,
        )
    )
    print(f"Run directory: {run_dir}")

    device = args.device if torch.cuda.is_available() else "cpu"
    train_pipeline(
        model,
        train_loader,
        val_loader,
        max_epochs=args.max_epochs,
        device=device,
        ckpt_dir=ckpt_dir,
    )
    print(f"Finished. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
