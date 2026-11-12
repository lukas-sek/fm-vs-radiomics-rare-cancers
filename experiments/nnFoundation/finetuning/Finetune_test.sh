#!/usr/bin/env bash
# Requires aimlab GloViTa with gradual-unfreeze patches applied.
# See: experiments/nnFoundation/finetuning/glovita_patches/README.md
set -euo pipefail

export MLFLOW_TRACKING_URI="sqlite:////home/lukas/experiments/mlflow.db"

accelerate launch --num_processes 1 -m glovita.cli.train \
  --exp_dir Results/nnFoundation/finetune_nnunet \
  --task.metrics acc balanced_acc f1 pr auroc \
  --optimizer.name AdamW \
  --optimizer.lr 1e-4 \
  --optimizer.weight_decay 0.01 \
  --optimizer.warmup_epochs 10 \
  --training.epochs 100 \
  --training.early_stopping_patience 20 \
  --training.early_stopping_min_epochs 10 \
  --training.precision fp16 \
  --training.gradient_accumulation_steps 8 \
  --training.train_steps_per_epoch 50 \
  --training.val_steps_per_epoch 100 \
  --training.best_checkpoint_metric val_AUROC \
  --dataloading.batch_size 1 \
  --dataloading.num_workers 8 \
  peft:gradual_unfreeze_config \
  --peft.epochs_per_unfreeze 5 \
  --peft.units_per_unfreeze 2 \
  data:generic3d_dataset_config \
  --data.data_root_dir data/glovita_worc_nnunet/Liver \
  --data.fold 0 \
  model.encoder:nnfoundation_encoder_config \
  --model.encoder.checkpoint_path data/nnssl_pretrained_models/nnFoundationViT/checkpoint_final.pth \
  --model.encoder.input_channels 1 \
  model.head:classification_head_config \
  --model.head.init_std 0.01 \
  logger:mlflow_logger_config \
  --logger.experiment_name test_experiment_vit \
  --logger.run_name test_run_vit
