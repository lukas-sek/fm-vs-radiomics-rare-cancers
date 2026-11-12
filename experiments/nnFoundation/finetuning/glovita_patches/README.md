# GloViTa gradual-unfreeze / LoRA patches

Drop-in patches for aimlab’s `image_classification` / GloViTa install so fine-tuning can use late→early gradual unfreezing or **LoRA on selected Primus blocks** in a **single** train run.

## What these patches add

- `peft:gradual_unfreeze_config` with `--peft.epochs_per_unfreeze` / `--peft.units_per_unfreeze` (open-ended late→early)
- `peft:capped_unfreeze_config` (Primus/ViT): head → block 39 @10 → blocks 38–39 @20; LLRD; EMA + raw early stop on **val AUROC**
- `peft:lora_config`: HuggingFace LoRA on selected transformer blocks (default **block 39 only**) + trainable classification head — used by `Finetune_ViT.sh` to limit overfitting vs full late-block FT
- `peft:bottleneck_adapter_config`: zero-init bottleneck adapter parallel to the MLP of selected blocks (default **block 39 only**), backbone frozen, head trained normally — used by `Finetune_ViT_adapters.sh`
- `peft:bias_tuning_config`: bias-only tuning (BitFit-style) restricted to selected blocks (default **block 39 only**), all weights frozen, head trained normally — used by `Finetune_ViT_bias.sh`
- ResEnc (`nnFoundationCNN`): gradual unfreeze on `encoder.stages` + `stem`
- Primus (`nnFoundationViT`): gradual, capped, or LoRA block adaptation as above
- Optimizer registers **all** parameters up front so newly unlocked weights keep AdamW state (gradual/capped paths)
- Checkpoints: `last.pt` + top-3 by `best_checkpoint_metric` as `best_epoch_XXXX.pt` (plus `best.pt` = rank 1); each includes a `gradual_unfreeze` dict when applicable. No per-epoch dumps. Writing `best.pt` also writes `best_trainable_params.json` (trainable/total param counts at that freeze/LoRA state).
- `training.early_stopping_patience` stops on plateau of `best_checkpoint_metric` while still saving freeze metadata

## Apply on aimlab

From the thesis repo (paths assume aimlab checkout at `/home/lukas/experiments/image_classification`):

```bash
GLOVITA_SRC=/home/lukas/experiments/image_classification/src/glovita
PATCH_SRC=/path/to/fm-vs-radiomics-rare-cancers/experiments/nnFoundation/finetuning/glovita_patches/src/glovita

# Backup then copy patched modules
cp "$GLOVITA_SRC/configs/peft.py" "$GLOVITA_SRC/configs/peft.py.bak"
cp "$GLOVITA_SRC/configs/training.py" "$GLOVITA_SRC/configs/training.py.bak"
cp "$GLOVITA_SRC/models/peft/registry.py" "$GLOVITA_SRC/models/peft/registry.py.bak"
cp "$GLOVITA_SRC/training/trainer.py" "$GLOVITA_SRC/training/trainer.py.bak"
cp "$GLOVITA_SRC/training/optimizers.py" "$GLOVITA_SRC/training/optimizers.py.bak"
cp "$GLOVITA_SRC/cli/train.py" "$GLOVITA_SRC/cli/train.py.bak"

cp "$PATCH_SRC/configs/peft.py" "$GLOVITA_SRC/configs/peft.py"
cp "$PATCH_SRC/configs/training.py" "$GLOVITA_SRC/configs/training.py"
cp "$PATCH_SRC/models/peft/registry.py" "$GLOVITA_SRC/models/peft/registry.py"
cp "$PATCH_SRC/models/peft/gradual_unfreeze.py" "$GLOVITA_SRC/models/peft/gradual_unfreeze.py"
cp "$PATCH_SRC/models/peft/capped_unfreeze.py" "$GLOVITA_SRC/models/peft/capped_unfreeze.py"
cp "$PATCH_SRC/models/peft/lora.py" "$GLOVITA_SRC/models/peft/lora.py"
cp "$PATCH_SRC/models/peft/bottleneck_adapter.py" "$GLOVITA_SRC/models/peft/bottleneck_adapter.py"
cp "$PATCH_SRC/models/peft/bias_tuning.py" "$GLOVITA_SRC/models/peft/bias_tuning.py"
cp "$PATCH_SRC/training/trainer.py" "$GLOVITA_SRC/training/trainer.py"
cp "$PATCH_SRC/training/optimizers.py" "$GLOVITA_SRC/training/optimizers.py"
cp "$PATCH_SRC/cli/train.py" "$GLOVITA_SRC/cli/train.py"
```

Or sync the whole tree:

```bash
rsync -av \
  experiments/nnFoundation/finetuning/glovita_patches/src/glovita/ \
  /home/lukas/experiments/image_classification/src/glovita/
```

LoRA uses a **pure-PyTorch** adapter path (no HuggingFace `peft` package required).

## Train

### LoRA last-block (default `Finetune_ViT.sh`)

```bash
peft:lora_config \
  --peft.lora_rank 8 \
  --peft.lora_alpha 16 \
  --peft.lora_dropout 0.05 \
  --peft.lora_layers 39 \
```

Verify after startup:

- logs contain `[lora] layers=[39] rank=8 … trainable=…`
- only block-39 adapters + classification head have `requires_grad=True`

### Bottleneck adapter last-block (`Finetune_ViT_adapters.sh`)

```bash
peft:bottleneck_adapter_config \
  --peft.adapter_bottleneck 32 \
  --peft.adapter_layers 39 \
  --peft.adapter_dropout 0.1 \
  --peft.adapter_lr_mult 5 \
```

Adapter on block 39: `mlp(h) + scale * up(dropout(gelu(down(h))))`, with `up` zero-initialised so the model starts identical to the pretrained one. Optimizer as in the LoRA run (AdamW, LR 1e-4, wd 0.05) for the head; adapter weights get `lr * adapter_lr_mult` (5e-4) via a separate parameter group in `training/optimizers.py`.

Trainable parameter budget (width 1056, head 2,114):

| Method | Trainable |
|---|---|
| LoRA r=8, block 39 | 162,626 |
| Adapter b=32, block 39 | 70,786 |
| Adapter b=64, block 39 | 138,402 |

Verify after startup: logs contain `[adapter] layers=[39] bottleneck=32 … trainable=70,786/…`.

### Bias-only tuning last-block (`Finetune_ViT_bias.sh`)

```bash
peft:bias_tuning_config \
  --peft.bias_layers 39 \
  --peft.bias_lr_mult 10 \
```

Every parameter whose name ends in `bias` inside block 39 (Linear biases of attention / MLP and LayerNorm biases) plus the classification head is trained; all weights stay frozen. Biases get `lr * bias_lr_mult` (1e-3) via the same `_lr_mult` parameter group as the adapters; the head keeps 1e-4. Pass several indices (e.g. `--peft.bias_layers 36 37 38 39`) to widen the scope. The upstream `peft:bit_fit_config` remains available and trains *all* backbone biases.

Verify after startup: logs contain `[bias_tuning] layers=[39] bias_tensors=… trainable=…`.

### Gradual unfreeze (`Finetune_test.sh`)

```bash
peft:gradual_unfreeze_config \
  --peft.epochs_per_unfreeze 5 \
  --peft.units_per_unfreeze 2 \
```

Verify after a short run:

- logs contain `[gradual_unfreeze] epoch=… unlocked_from_end=…`
- `torch.load("…/checkpoints/best.pt")["gradual_unfreeze"]` is present (also on early stop via `last.pt`)
