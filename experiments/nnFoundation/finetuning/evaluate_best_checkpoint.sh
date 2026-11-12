#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRED_OUT="${PRED_OUT:-Results/nnFoundation/finetune/infer_Liver_adapters_nnunetPre_2.pt}"
EXP_DIR="${EXP_DIR:-Results/nnFoundation/finetune_vit_adapters_nnunet/generic_3d_dataset/20261104_103831_nnfoundationencoder_classificationhead_bottleneck_adapter_0db59a03}"
FOLD="${FOLD:-2}"
CHECKPOINT="${CHECKPOINT:-best}"
# Must match train cwd: Finetune_*.sh uses data/glovita_worc_nnunet/Liver.
DATA_ROOT="${DATA_ROOT:-data/glovita_worc_nnunet/Liver}"

glovita_infer \
  --exp_dir "$EXP_DIR" \
  --fold "$FOLD" \
  --checkpoint "$CHECKPOINT" \
  --dataloading.batch_size 1 \
  --metrics acc f1 \
  --pred_output "$PRED_OUT" \
  data:generic3d_dataset_config \
  --data.data_root_dir "$DATA_ROOT" \
  --data.fold "$FOLD" \
  --data.num_classes 2 \
  --data.subtask multiclass

python "$SCRIPT_DIR/pt_metrics.py" "$PRED_OUT" \
  --exp-dir "$EXP_DIR" \
  --fold "$FOLD" \
  --checkpoint "$CHECKPOINT"
