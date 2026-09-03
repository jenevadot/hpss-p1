#!/bin/bash
# PRE-REGISTERED tuning sweep. Read this before looking at any result.
#
# ============================================================================
# SELECTION RULE, fixed in advance:
#   Winner of each factor = highest **dev_mAP**, single aggregate number.
#   Per-class breakdowns must NOT be consulted to pick a winner.
#   val is not read during tuning at all (engine.py evaluates it once per run and
#   stores it, but selection uses dev_mAP only).
#   EVERY cell below is reported, including losers.
# ============================================================================
#
# Why pre-registration: with no labelled test set (test.7z has 31,187 wavs and no
# CSV), val is the only held-out estimate available. Choosing a configuration by
# inspecting val -- or by inspecting per-class dev detail and rationalising -- spends
# its credibility with nothing left to detect the damage. Writing the rule down
# first is what makes the eventual val number meaningful.
#
# One-factor-at-a-time, NOT a cross product: 4 factors x ~3 levels as a grid would
# be ~54 runs (~2 days). OFAT is 11 runs (~9 h) and is adequate because we are
# looking for main effects, not interactions.
#
# Baseline for every comparison: the current default config, dual arm, seed 42.
#
# Factors:
#   A  alpha_tau        {1.0, 0.3}          fusion sigmoid temperature
#   B  alpha_lr_mult    {1, 10}             dedicated lr for a_raw
#   C  mixup_p          {0.3, 0.5, 0.7}     augmentation strength
#   D  epochs           {40, 60}            schedule length
#   E  loss             {bce, asl}          declared in advance, reported either way
#
# A and B target a MEASURED optimisation fact, not a performance observation:
# 19-20 of 34 per-class alphas end within 0.05 of their 0.5 init (seed 42: 19/34,
# 43: 11/34, 44: 20/34), cross-class std ~0.083, yet cross-seed correlation is
# r=0.94. The preference is real and reproducible; the parameter simply cannot
# travel far enough in 40 epochs because a_raw sees gradient scaled by
# sigmoid'(0)=0.25 while competing with 15.8M parameters at one shared lr.
#
# C targets: dual has the LOWEST train loss (0.0130) of the four arms but not the
# best val -- a mild-overfit signature. Both directions are tested; 0.3 is not
# assumed better.
#
# D targets: raw peaked at epoch 36,37,16 of 40, i.e. two of three runs were still
# improving when the budget ended. Cosine anneals to its floor at exactly `epochs`,
# so a longer budget is a genuinely different schedule, not just more steps.
#
# Cost: 11 runs x ~40 epochs x ~200 s ~= 9 h (dual arm, dev split).
set -uo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
SEED="${SEED:-42}"
ARM="${ARM:-dual}"
EP="${EP:-40}"

run () {  # run <tag> <extra args...>
  local tag="tune_$1"; shift
  if [ -f "runs/${tag}/summary.json" ]; then
    echo "=== SKIP $tag (done) ==="
    return
  fi
  echo "=== $tag ($(date +%H:%M)) ==="
  caffeinate -i .venv/bin/python -u -m src.train \
      --arm "$ARM" --seed "$SEED" --patience 20 --tag "$tag" "$@" 2>&1 \
    | grep -vE "Warning|warn"
}

echo "=== PRE-REGISTERED TUNING SWEEP: arm=$ARM seed=$SEED ==="
date

# --- baseline (also serves as the A=1.0, B=1, C=0.5, D=40, E=bce cell) ---
run "base"            --epochs "$EP"

# --- A: fusion sigmoid temperature ---
run "tau0.3"          --epochs "$EP" --alpha-tau 0.3

# --- B: dedicated lr for a_raw ---
run "alr10"           --epochs "$EP" --alpha-lr-mult 10
# A x B, the one interaction worth checking: both target the same bottleneck, so
# they could easily be redundant rather than additive.
run "tau0.3_alr10"    --epochs "$EP" --alpha-tau 0.3 --alpha-lr-mult 10

# --- C: augmentation strength ---
run "mix0.3"          --epochs "$EP" --mixup-p 0.3
run "mix0.7"          --epochs "$EP" --mixup-p 0.7

# --- D: schedule length ---
run "ep60"            --epochs 60

# --- E: loss function ---
run "asl"             --epochs "$EP" --loss asl

# --- attribution runs: decompose the +0.11 over the legacy config ---
# Not tuning. These answer "which of the four simultaneous changes did the work?"
run "attr_scalar"     --epochs "$EP" --scalar-fusion
run "attr_stem1"      --epochs "$EP" --stem-pool 1
run "attr_flatlr"     --epochs "$EP" --lr-schedule none

echo "=== TUNING TABLE ($(date +%H:%M)) ==="
.venv/bin/python -m src.analyze --tuning
