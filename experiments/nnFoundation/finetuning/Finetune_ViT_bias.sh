#!/usr/bin/env bash
# Primus/ViT fine-tune with bias-only tuning (BitFit-style) on the last transformer
# block (39) + normally trained classification head. All weights stay frozen.
# Requires aimlab GloViTa with glovita_patches applied (incl. models/peft/bias_tuning.py).
# Trainable count is printed at startup: "[bias_tuning] layers=[39] ... trainable=...".
# LR: head 1e-4 (as in LoRA run), biases 1e-4 * bias_lr_mult = 1e-3.

set -euo pipefail

accelerate launch --num_processes 1 -m glovita.cli.train \
  --exp_dir Results/nnFoundation/finetune_vit_bias_nnunet \
  --task.metrics acc balanced_acc f1 pr auroc \
  --optimizer.name AdamW \
  --optimizer.lr 1e-4 \
  --optimizer.weight_decay 0.05 \
  --training.epochs 100 \
  --training.precision fp16 \
  --training.early_stopping_patience 10 \
  --training.early_stopping_min_epochs 10 \
  --training.best_checkpoint_metric val_AUROC \
  --dataloading.batch_size 1 \
  --dataloading.num_workers 8 \
  peft:bias_tuning_config \
  --peft.bias_layers 39 \
  --peft.bias_lr_mult 10 \
  data:generic3d_dataset_config \
  --data.data_root_dir data/glovita_worc_nnunet/Liver \
  --data.fold 2 \
  model.encoder:nnfoundation_encoder_config \
  --model.encoder.checkpoint_path data/nnssl_pretrained_models/nnFoundationViT/checkpoint_final.pth \
  --model.encoder.input_channels 1 \
  model.head:classification_head_config \
  --model.head.init_std 0.01 \
  logger:no_logger_config
