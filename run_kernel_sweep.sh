#!/bin/bash
# HPSS KERNEL SWEEP — the last untested parameter in the paper's own contribution.
#
# ============================================================================
# THE ARGUMENT (physical units, not tuning intuition)
#
# HPSS separation is two median filters over |S| (513 x 130):
#   S_h_mag = median(|S|, along TIME,      k)   keeps what is LONG in time  -> harmonic
#   S_p_mag = median(|S|, along FREQUENCY, k)   keeps what is BROAD in freq -> percussive
#
# k is the only free parameter. At hop=512 / 22,050 Hz each frame is 23.2 ms and
# each STFT bin is 21.5 Hz, so k spans:
#     k=5  116 ms / 108 Hz      k=17  394 ms / 366 Hz   <- current default
#     k=9  209 ms / 194 Hz      k=25  580 ms / 538 Hz
#     k=13 302 ms / 280 Hz
#
# Anuran calls here are 200-500 ms events. So k=17 asks "does energy persist across
# 394 ms?" -- LONGER than many calls it must detect. A tonal 250 ms call fails that
# test and is routed to the PERCUSSIVE stream regardless of its real structure.
#
# k=17 was inherited from the paper's sweep on 10 s DCASE scene clips (5->69.3,
# 9->70.8, 13->71.6, 17->72.1, 21->71.8, 25->71.1) at STFT settings the paper never
# states. It has never been validated in this domain.
#
# This is the SAME failure mode already caught once: SpecAugment's time mask of 40
# frames was right for 431-frame clips and wrong for 130-frame ones. A value
# transferred as an integer when it should be transferred as a physical duration.
#
# CORROBORATING EVIDENCE
#   1. percussive-only (0.7373) beats harmonic-only (0.7109), p=0.015 -- expected if
#      the percussive stream is receiving content that belongs to the harmonic one.
#   2. measured harmonic energy share rises monotonically with k
#      (k=5: 0.383, k=9: 0.394, k=13: 0.410, k=17: 0.425, k=25: 0.456), so at the
#      default only ~42% of energy reaches the harmonic stream.
#   3. per-class alpha spread is modest (0.352-0.713, std 0.091). Cleanly separated
#      streams should produce SHARPER per-species preferences.
#
# PREDICTION: a shorter kernel (9 or 13) separates frog calls better than 17, and the
# sensitivity curve runs OPPOSITE to the paper's.
#
# SELECTION: dev_mAP, aggregate, single seed for the screen. Pre-registered.
# Every kernel is reported. A winner must clear seed noise (0.0078) to count, and
# then be confirmed with 3 seeds before adoption.
# ============================================================================
#
# Arm: percussive. Cheapest (7.98M params, ~110 s/epoch) and it is the stream the
# model currently leans on. NOTE the sweep only changes X_h/X_p -- X_raw is identical
# in every file, so a `raw` arm would be a pure no-op control (useful as a sanity
# check that nothing else drifted between files).
#
# Cost per kernel: ~16 min extraction + ~1.2 h training. 4 new kernels (17 already
# exists as data/features.h5) => ~6 h. Disk: 6.2 GB per file, ~25 GB total.
#
# k=5 (116 ms) is included at the user's request and is deliberately past the
# expected optimum -- it brackets the minimum instead of only approaching it from
# above, which is what makes a curve rather than a ranking. Its harmonic energy
# share is already characterised at 0.383.
set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
KERNELS="${KERNELS:-5 9 13 25}"   # 17 is the existing data/features.h5
ARM="${ARM:-percussive}"
EP="${EP:-40}"
SEED="${SEED:-42}"

# Ordering: this script is FOURTH in the queue, behind run_tuning.sh,
# run_capacity.sh and run_confirm.sh. Waiting on process absence alone would
# race run_confirm.sh -- both would see the same clear window and start together
# on one MPS device. So wait for confirm's ARTIFACTS, then for an idle device.
CONF_SEEDS="${CONF_SEEDS:-43 44}"
echo "=== waiting for run_confirm.sh to produce all conf_* summaries ==="
while :; do
  missing=""
  for s in $CONF_SEEDS; do
    for c in base "mix0.7" "tau0.3"; do
      [ -f "runs/conf_${c}_s${s}/summary.json" ] || missing="$missing ${c}_s${s}"
    done
  done
  [ -z "$missing" ] && break
  sleep 120
done
echo "=== confirmation complete at $(date +%H:%M); waiting for idle device ==="
while pgrep -f "run_tuning.sh" >/dev/null 2>&1 \
   || pgrep -f "run_capacity.sh" >/dev/null 2>&1 \
   || pgrep -f "run_confirm.sh" >/dev/null 2>&1 \
   || pgrep -f "src.train" >/dev/null 2>&1; do
  sleep 60
done
echo "=== clear at $(date +%H:%M) ==="

# Baseline at the current kernel, same arm/split/settings, for a like-for-like row.
if [ ! -f "runs/k17_${ARM}/summary.json" ]; then
  echo "=== k=17 (existing features) ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm "$ARM" --epochs "$EP" --patience 20 --seed "$SEED" \
      --tag "k17_${ARM}" 2>&1 | grep -vE "Warning|warn"
fi

for k in $KERNELS; do
  h5="data/features_k${k}.h5"
  tag="k${k}_${ARM}"

  if [ -f "runs/${tag}/summary.json" ]; then
    echo "=== SKIP $tag (done) ==="
    continue
  fi

  # Extraction is resumable via the `done` mask, so re-running is safe.
  echo "=== extracting k=$k -> $h5 ($(date +%H:%M)) ==="
  .venv/bin/python -u -m src.precompute --kernel "$k" --out "$h5" 2>&1 \
    | grep -vE "Warning|warn"

  echo "=== training $tag ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm "$ARM" --h5 "$h5" --epochs "$EP" --patience 20 --seed "$SEED" \
      --tag "$tag" 2>&1 | grep -vE "Warning|warn"
done

echo "=== KERNEL SWEEP RESULTS ($(date +%H:%M)) ==="
.venv/bin/python - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("runs/k*_*/summary.json")):
    s = json.load(open(p))
    rows.append((os.path.basename(os.path.dirname(p)), s.get("hpss_kernel"),
                 s.get("dev_mAP"), s["val_mAP"], s["best_epoch"]))
if not rows:
    print("no kernel runs found")
else:
    rows.sort(key=lambda r: -(r[2] or 0))
    print(f"{'run':18s} {'kernel':>7s} {'time_ms':>8s} {'dev_mAP':>8s} "
          f"{'val_mAP':>8s} {'ep':>4s}")
    for run, k, dev, val, ep in rows:
        try:
            ms = f"{int(k) * 23.2:.0f}"
        except (TypeError, ValueError):
            ms = "-"
        d = f"{dev:8.4f}" if dev is not None else "       -"
        print(f"{run:18s} {str(k):>7s} {ms:>8s} {d} {val:8.4f} {ep:>4d}")
    print("\nA winner must beat k=17 by more than the seed noise of 0.0078,")
    print("then be confirmed across 3 seeds before adoption.")
PY
