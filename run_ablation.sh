#!/bin/bash
# Full ablation ladder: 4 arms x N seeds, current config.
#
#   ./run_ablation.sh                          # seeds 42 43 44, 40 epochs
#   SEEDS="42" EPOCHS=10 ./run_ablation.sh     # quick single-seed pass
#   ARMS="dual raw" ./run_ablation.sh          # subset of arms
#
# Runs sequentially on purpose: there is one MPS device, so concurrent runs
# time-slice the same command queues and contend rather than overlap.
#
# Cost at STEM_POOL=2 (2.792 GFLOPs): dual ~4 min/epoch, single-stream ~2 min.
# 3 seeds x (1 dual + 3 single) x 40 epochs ~= 13 h.
#
# Seeds fix initial weights, batch order, SpecAugment draws and MixUp lambdas.
# They do NOT give bitwise reproducibility: MPS reduction order is not stable, so
# same-seed reruns land close but not identical. That is precisely why the table is
# reported as mean+-std over seeds.
#
# PATIENCE defaults to 20, not 8: cosine annealing needs to reach its floor, and
# engine.train() raises anything below epochs//2 anyway.
#
# Resumable -- an arm whose summary.json already exists is skipped, so an
# interrupted sweep can be rerun without repeating finished work.
set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-20}"
SEEDS="${SEEDS:-42 43 44}"
ARMS="${ARMS:-dual raw harmonic percussive}"

echo "=== ablation ladder: arms=[$ARMS] seeds=[$SEEDS] epochs=$EPOCHS ==="
date

for seed in $SEEDS; do
  for arm in $ARMS; do
    tag="${arm}_s${seed}_stem2"
    if [ -f "runs/${tag}/summary.json" ]; then
      echo "=== SKIP $tag (already complete) ==="
      continue
    fi
    echo "=== $arm seed=$seed ($(date +%H:%M)) ==="
    caffeinate -i .venv/bin/python -u -m src.train \
        --arm "$arm" --epochs "$EPOCHS" --patience "$PATIENCE" \
        --seed "$seed" --tag "$tag" 2>&1 \
      | grep -vE "Warning|warn"
  done
done

echo "=== ablation table ($(date +%H:%M)) ==="
.venv/bin/python -m src.analyze
