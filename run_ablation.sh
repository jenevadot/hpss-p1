#!/bin/bash
# Run the remaining three ablation arms sequentially.
# The dual arm is expected to have been launched separately (it is the slowest
# at ~13.5 min/epoch; the single-stream arms are ~6.8 min/epoch each).
#
#   ./run_ablation.sh
#
# Runs sequentially on purpose: there is one MPS device, so concurrent runs
# would contend for the same command queues rather than overlap.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
SEED="${SEED:-42}"

for arm in raw harmonic percussive; do
  echo "=== $arm ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm "$arm" --epochs "$EPOCHS" --patience "$PATIENCE" \
      --seed "$SEED" --tag "${arm}_seed${SEED}" 2>&1 \
    | grep -vE "Warning|warn"
done

echo "=== ablation table ==="
.venv/bin/python -m src.analyze
