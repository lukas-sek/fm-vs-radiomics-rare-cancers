#!/usr/bin/env bash
# Primus/ViT fine-tune with a bottleneck adapter on the last transformer block (39)
# + normally trained classification head. Backbone weights stay frozen.
# Requires aimlab GloViTa with glovita_patches applied (incl. models/peft/bottleneck_adapter.py).
# Trainable: adapter 68,672 (bottleneck 32) + head 2,114 = 70,786 (LoRA run: 162,626).
# LR: head 1e-4, adapter 1e-4 * adapter_lr_mult = 5e-4.

set -euo pipefail

accelerate launch --num_processes 1 -m glovita.cli.train \
  --exp_dir Results/nnFoundation/finetune_vit_adapters_nnunet \
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
  peft:bottleneck_adapter_config \
  --peft.adapter_bottleneck 32 \
  --peft.adapter_layers 39 \
  --peft.adapter_dropout 0.1 \
  --peft.adapter_lr_mult 15 \
  data:generic3d_dataset_config \
  --data.data_root_dir data/glovita_worc_nnunet/Liver \
  --data.fold 2 \
  model.encoder:nnfoundation_encoder_config \
  --model.encoder.checkpoint_path data/nnssl_pretrained_models/nnFoundationViT/checkpoint_final.pth \
  --model.encoder.input_channels 1 \
  model.head:classification_head_config \
  --model.head.init_std 0.01 \
  logger:no_logger_config
