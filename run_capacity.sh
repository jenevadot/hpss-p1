#!/bin/bash
# CAPACITY CONTROL: is dual better because HPSS decomposition helps, or just
# because it is 2x bigger?
#
# ============================================================================
# THE CONFOUND
#   dual       15,824,344 params  ->  0.7633 +- 0.0029   (3-seed ladder)
#   raw         7,983,212 params  ->  0.7555 +- 0.0139   (+0.0078, p=0.437)
#
# X_h + X_p == X_raw exactly (soft-mask property, verified to 1.5e-05), so HPSS
# adds NO information. The paper's claim is about inductive bias, not information:
# each stream can specialise on horizontal (harmonic) or vertical (percussive)
# structure instead of one network learning filters for both. Doubling parameters
# is the stated cost of that specialisation.
#
# But dual's marginal +0.0078 is also exactly what you might expect from simply
# having 2x the capacity. Those two explanations are CONFOUNDED in every run so
# far, and no experiment to date separates them.
#
# THE TEST
#   raw at width=1.41 -> 15,691,199 params, within 0.8% of dual.
#
#   raw-wide >= dual  =>  HPSS decomposition contributes NOTHING on this task;
#                         dual's edge was capacity. The paper's mechanism does not
#                         transfer to anuran detection.
#   raw-wide <  dual  =>  the decomposition does something capacity alone cannot
#                         buy, and improving dual is worth the effort.
#
# SELECTION: dev_mAP, aggregate, 3 seeds. Pre-registered before running.
# Every seed is reported.
# ============================================================================
#
# Waits for the current tuning sweep to finish -- one MPS device, so concurrent
# runs time-slice the same command queues and contend rather than overlap.
#
# Cost: 3 seeds x ~40 epochs. raw-wide is ~2x raw's FLOPs, so ~200 s/epoch => ~7 h.
set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
WIDTH="${WIDTH:-1.41}"
EP="${EP:-40}"
SEEDS="${SEEDS:-42 43 44}"

echo "=== waiting for the current sweep to finish ==="
while pgrep -f "run_tuning.sh" >/dev/null 2>&1 || pgrep -f "src.train" >/dev/null 2>&1; do
  sleep 60
done
echo "=== clear at $(date +%H:%M); starting capacity control ==="

for seed in $SEEDS; do
  tag="rawwide_s${seed}"
  if [ -f "runs/${tag}/summary.json" ]; then
    echo "=== SKIP $tag (done) ==="
    continue
  fi
  echo "=== $tag width=$WIDTH ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm raw --width "$WIDTH" --epochs "$EP" --patience 20 \
      --seed "$seed" --tag "$tag" 2>&1 \
    | grep -vE "Warning|warn"
done

echo "=== capacity control complete ($(date +%H:%M)) ==="
.venv/bin/python -m src.analyze --tuning
echo
echo "Compare rawwide_* dev_mAP against the dual arm. Note the dual figures in the"
echo "3-seed ladder used the TWO-way split; for a like-for-like comparison use"
echo "tune_base (dev 0.7151), which is dual on the three-way split."
