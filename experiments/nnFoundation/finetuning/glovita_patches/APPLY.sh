#!/usr/bin/env bash
# Copy GloViTa patches into an aimlab checkout.
# Usage: ./APPLY.sh /home/lukas/experiments/image_classification
set -euo pipefail

ROOT="${1:?Usage: $0 /path/to/image_classification}"
SRC="$(cd "$(dirname "$0")" && pwd)/src/glovita"
DST="${ROOT}/src/glovita"

if [[ ! -d "$DST" ]]; then
  echo "ERROR: expected GloViTa at $DST" >&2
  exit 1
fi

mkdir -p "$DST/models/peft" "$DST/configs" "$DST/training" "$DST/cli"

cp -v "$SRC/configs/peft.py" "$DST/configs/peft.py"
cp -v "$SRC/configs/training.py" "$DST/configs/training.py"
cp -v "$SRC/models/peft/registry.py" "$DST/models/peft/registry.py"
cp -v "$SRC/models/peft/gradual_unfreeze.py" "$DST/models/peft/gradual_unfreeze.py"
cp -v "$SRC/models/peft/capped_unfreeze.py" "$DST/models/peft/capped_unfreeze.py"
cp -v "$SRC/models/peft/lora.py" "$DST/models/peft/lora.py"
cp -v "$SRC/models/peft/bottleneck_adapter.py" "$DST/models/peft/bottleneck_adapter.py"
cp -v "$SRC/models/peft/bias_tuning.py" "$DST/models/peft/bias_tuning.py"
cp -v "$SRC/training/trainer.py" "$DST/training/trainer.py"
cp -v "$SRC/training/optimizers.py" "$DST/training/optimizers.py"
cp -v "$SRC/cli/train.py" "$DST/cli/train.py"
cp -v "$SRC/cli/infer.py" "$DST/cli/infer.py"

echo "Applied GloViTa patches (incl. LoRA, bottleneck adapter, bias tuning, infer) to $DST"
