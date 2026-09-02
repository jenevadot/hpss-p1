#!/bin/bash
# Wait for the dual run to finish, then run the three single-stream arms.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

while pgrep -f "src.train --arm dual" >/dev/null 2>&1; do sleep 60; done
echo "dual finished at $(date +%H:%M); starting single-stream arms"

for arm in raw harmonic percussive; do
  echo "=== $arm ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm "$arm" --epochs 40 --patience 8 --seed 42 --tag "${arm}_seed42" 2>&1 \
    | grep -vE "Warning|warn"
done
echo "=== ABLATION TABLE ($(date +%H:%M)) ==="
.venv/bin/python -m src.analyze
