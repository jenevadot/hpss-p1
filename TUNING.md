# Training Tuning Catalogue — HSPP on AnuraSet

Every training-side change made to the replication, with the reason it was made and
the concept behind it.

**Status legend:** ✅ adopted · 🔬 swept · ❌ reverted · ⏸ pending

**Companion docs:** `IMPLEMENTATION.md` (full technical/decision record) ·
`HANDOFF.md` (current state and next steps)

> **Reading the numbers.** Two split regimes are in play and they are **not
> comparable**. The 4-arm × 3-seed ablation ladder used the two-way split
> (train = 53,340) and reports `val_mAP`. The tuning sweep uses the three-way split
> (train = 44,408, −17% data) and reports `dev_mAP`. Within the sweep, `base`
> (dev 0.7151) is the only valid reference.
>
> **σ = 0.0078 IS OBSOLETE — use σ ≈ 0.017.** That figure came from the two-way
> ladder. Measured on the three-way split, dual's own 3-seed sd is **0.0172**
> (0.7151 / 0.7180 / 0.6869), 2.2× larger, because 17% less training data means
> more run-to-run variance. Every single-seed conclusion in this document is
> therefore *softer* than it was originally written to be. Where a verdict below
> rests on a sub-0.017 single-seed delta, it is now marked as such rather than
> silently rescored.
>
> **n=3 caps significance.** An exact paired sign-flip permutation test on 3 seeds
> has 8 assignments, so the minimum attainable two-sided p is **0.25**. Nothing in
> this project can reach p<0.05. Report "consistent across seeds" or "a tie".

---

## Summary table

| # | Change | Status | Effect |
|---|---|---|---|
| 1 | Adam → AdamW + no-decay groups | ✅ | correctness; BN γ/β and `a_raw` no longer decayed. **Never ablated** — no arm was ever run |
| 2 | Cosine LR + 2-epoch warmup | ✅ | killed the ±0.035 oscillation. `attr_flatlr` −0.0018 = **no measurable mAP gain** |
| 3 | Stem pool /2 | ✅ | 4.02× fewer FLOPs, 0 extra params; enabled 3-seed runs |
| 4 | MixUp probability sweep | ✅ | **ADOPTED `p=0.7`** — 3-seed +0.0174, 3/3 seeds, sd 0.0027 vs 0.0172 |
| 5 | Alpha temperature / LR multiplier | ❌ | **negative** — α spread ↑, mAP ↔/↓. `tau0.3` not adopted (2/3 seeds, p=0.750) |
| 6 | Longer schedule for `raw` | ⏸ | honesty check on the dual-vs-raw tie; **never run** |
| 7 | Batch size 32 → 64 | ✅ | free throughput + cleaner BN statistics |
| 8 | SpecAugment time mask 40 → 12 | ✅ | prevents erasing the labelled event |
| 9 | Gradient clipping | ❌ | never fired (norm 0.132), cost ~16%/step |
| 10 | AsymmetricLoss | ❌ | worse (0.7053 vs 0.7151), −0.0098. Rejected |
| 11 | Reproducible seeding | ✅ | decoupled RNG streams; enables mean±std |
| 12 | Three-way train/dev/val split | ✅ | exposed threshold optimism; protects val |
| 13 | Per-class fusion (attribution) | ✅ | fixed the *ordering*, not the absolute mAP (−0.0035) |
| 14 | **Capacity control (raw-wide)** | ✅ | **raw-wide ≥ dual → H2.** Decomposition adds nothing at matched params |
| 15 | **HPSS kernel sweep {5,9,13,25}** | ❌ | **null result.** Spread 0.0150 < σ 0.017; no trend. Kernel length does not matter |

---

## THE ADOPTED FINAL CONFIGURATION

What the reportable ablation (`run_final_ablation.sh`, tags `*_s{seed}_3way`) runs.
Verified against `runs/dual_s42_3way/summary.json`.

```
arm              dual | raw | harmonic | percussive   (4 arms x 3 seeds)
mixup_p          0.7          <- ADOPTED, the only confirmed tuning win
mixup_alpha      0.2
stem_pool        2
per_class_fusion True         <- dual only
alpha_tau        1.0          <- tau0.3 NOT adopted
alpha_lr_mult    1.0          <- alr10 rejected (hurt)
lr_schedule      cosine + 2-epoch warmup, floor LR*0.01
optimiser        AdamW, lr 1e-3, no-decay groups for BN gamma/beta and a_raw
loss             BCEWithLogits    <- ASL rejected (-0.0098)
grad_clip        0.0              <- measured norm 0.132, never fired
batch_size       64
epochs           40, patience 20
hpss_kernel      17, margin 1.0   <- sweep found no better value
spec_time_mask   12  (paper's 40 rescaled 431 -> 130 frames)
spec_freq_mask   8, n_masks 2
split            three-way grouped-by-recording, dev-selected, val read ONCE
params           15,824,344 dual / 7,983,212 single
```

**Honest summary of what is actually evidence-backed:** of the four changes credited
with the +0.109 over legacy, **three measure at zero or negative individually**
(per-class fusion −0.0035, cosine LR −0.0018, stem pool a single-epoch artefact) and
the fourth (AdamW) **was never ablated**. MixUp 0.7 is the single confirmed win in
the entire project. The rest of the config is defensible engineering, not measured
improvement, and should be described that way.

---

## 1. `Adam → AdamW` ✅

**What.** Swapped optimizer; split parameters into decay / no-decay groups.

**Purpose.** Two separate defects in one line of code.

### Concept — decoupled weight decay

L2 regularization and weight decay are identical under plain SGD but **not** under
adaptive optimizers. Adam adds `λ·w` to the gradient, which then passes through the
adaptive denominator:

```
Adam:   w ← w − lr · (ĝ + λw) / (√v̂ + ε)      ← decay divided by gradient scale
AdamW:  w ← w − lr · ĝ / (√v̂ + ε) − lr · λw   ← decay applied directly
```

Under Adam, a parameter with small historical gradients receives a *large* effective
decay and vice versa — regularization strength becomes an accident of gradient
history rather than a chosen hyperparameter. AdamW (Loshchilov & Hutter) decouples
them.

### The second, larger defect

`Adam(model.parameters(), weight_decay=1e-4)` decays **every** tensor, including
BatchNorm's γ and β. BN's entire purpose is learning a scale and shift; pulling γ
toward zero directly opposes it. Standard practice excludes all 1-D parameters.

Measured split:

```
decay:    24 tensors, 15,815,236 params
no_decay: 39 tensors,      9,108 params
          └ 32 BatchNorm γ/β + 6 biases + 1 a_raw
```

**Why excluding `a_raw` is correctness, not hygiene.** Decaying the fusion weight
biases it toward `sigmoid(0) = 0.5` — **confounding the exact quantity the ablation
measures.**

### Why Adam-family over SGD (the paper's choice, kept)

Per-parameter adaptive steps genuinely help long-tailed multi-label training, where
gradient magnitude reaching the 42 outputs spans ~3 orders of magnitude between
SPHSUR (13,258 positives) and LEPFLA (7). One global SGD learning rate would either
under-train the tail or destabilise the head.

---

## 2. Cosine LR schedule + warmup ✅

**What.** `LambdaLR` — 2-epoch linear warmup to `LR`, then cosine decay to
`LR × 0.01`, stepped **per batch**. The paper specifies no schedule.

### Purpose — the measured problem

Flat 1e-3 for 31 epochs produced:

```
ep 15  0.6281      ep 24  0.6541 ← "best", saved
ep 17  0.6059  ↓   ep 26  0.6050  ↓
ep 20  0.5855  ↓↓  ep 30  0.6527
ep 22  0.6142      ep 31  0.6471
```

A **±0.035 band with no descent**, while train loss sat flat at 0.053–0.055.

### Concept — orbiting vs entering the basin

A step size appropriate at epoch 1 is far too large at epoch 25. The optimizer
repeatedly overshoots the minimum, circling at a radius set by `lr × |∇|`. Annealing
shrinks that radius so late steps settle in rather than orbit.

### The worse harm — and the real reason this mattered

`best.pt` is selected by `val_mAP > best`. Under ±0.035 oscillation that saves
**whichever epoch got lucky**, not the best model — inflating every arm by an
unknown, arm-specific amount.

That is precisely why a 0.040 mAP spread across the top three arms was unreadable
against ±0.035 of noise. **The missing schedule was what made the original ablation
table inconclusive.** This change was about measurement trustworthiness first,
accuracy second.

### Why per-batch stepping

Per-epoch stepping gives 40 discrete LR values (a staircase); per-batch gives 33,320
and a genuinely smooth decay. It matters most late in the run, where a per-epoch
schedule holds a still-too-large LR for an entire 833-step epoch.

### Why warmup

BatchNorm running statistics are meaningless at initialisation, so the first few
hundred steps produce large, badly-scaled gradients. A 2-epoch ramp (1,666 steps)
avoids spending **the largest steps of the run on the worst gradient estimates**.

### A conflict worth knowing

Cosine anneals to its floor at exactly `epochs`; early stopping mid-anneal discards
the low-LR phase where convergence happens. The schedule and the stopper want
opposite things. Resolution: patience is raised to `max(patience, epochs//2)` when
cosine is active, and the stopper survives only as a runaway guard.

Measured curve: peak 1.00e-03 at step 1,665; floor 1.00e-05; monotonic anneal.

---

## 3. Stem pool `STEM_POOL=2` ✅

**What.** One `MaxPool2d(2)` before block 1.

```
STEM_POOL=1:  11.216 GFLOPs, block-4 map 16×16, 819 s/epoch
STEM_POOL=2:   2.792 GFLOPs, block-4 map  8×8,  ~230 s/epoch
```

**4.02× fewer FLOPs at zero parameter cost** — pooling has no weights, and global
average pooling absorbs the smaller map.

### Purpose — fidelity first, speed second

The paper reports **2.86 GFLOPs at 10 s**. With pooling only after each block as its
text describes, **block 1 alone at 128×431 costs ~6.9 G — already 2.4× the reported
total for the entire network.** Searching stem strides at 431 frames:

| pre-block-1 downsample | input to block 1 | dual GFLOPs |
|---|---|---:|
| none | 128×431 | 37.13 |
| /2 | 64×216 | 9.37 |
| **/4** | **32×108** | **2.32** ← nearest to 2.86 |
| /8 | 16×54 | 0.56 |

**The paper must downsample ~/4 before block 1, which it never describes.** So
running *with* a stem pool is arguably more faithful to its real architecture than
the literal reading of its prose.

### Concept — where FLOPs live vs where parameters live

```
block 4: 5,900,288 of 7,807,680 stream params = 75.6%   ← parameters back-loaded
block 1: runs at full input resolution                   ← FLOPs front-loaded
```

To cut parameters, touch block 4. To cut FLOPs, touch block 1. The stem pool halves
both spatial dimensions *before* the most expensive convolution.

### Why not `/4`

It would leave a 4×4 block-4 map — smaller than the 7×7 spatial-attention
convolution meant to gate it. `test_spatial_map_not_degenerate` enforces the 8×8
floor.

### The enabling effect

This is what made 3-seed runs affordable: the 12-run ladder dropped from ~53 h to
~13 h.

---

## 4. MixUp probability gate 🔬

**What.** `MIXUP_P`, previously 1.0 (implicit).

### Purpose

Every sample was SpecAugmented **and** MixUp'd — corrupted twice, 100% of the time.
Consistent with the measured signature: train loss plateaued at 0.053 with **no
train/val divergence across 31 epochs**, i.e. **augmentation-limited, not
capacity-limited**, despite 15.8M parameters on 53k samples.

### Concept

MixUp trains on convex combinations `λx_i + (1−λ)x_j` with matching soft labels,
enforcing locally-linear behaviour between examples. Valuable — but the model must
also see *clean* examples to learn what an uncorrupted call looks like. At `p=1.0`
combined with always-on SpecAugment, it never does.

`MIXUP_ALPHA=0.2` gives Beta(0.2, 0.2), which is U-shaped — mostly near-clean mixes
with occasional strong ones. Multi-label needs no change because BCE accepts soft
targets.

### Result — swept `{0.3, 0.5, 0.7}`, then CONFIRMED at 3 seeds

Screen (single seed, three-way split):

```
mix0.7   dev 0.7266   ← best of the whole sweep, +0.0115 over base
base     dev 0.7151   (p=0.5)
mix0.3   dev 0.7152   (+0.0001 — nothing)
```

Confirmation (`run_confirm.sh`, seeds 42/43/44, pre-registered bar +0.0078):

```
                 s42     s43     s44      mean     vs base   p      wins
mixup_p=0.7    0.7266  0.7212  0.7244   0.7241   +0.0174   0.250   3/3
mixup_p=0.5    0.7151  0.7180  0.6869   0.7067
per-seed diff  +0.0115 +0.0031 +0.0375
```

**ADOPTED.** `src/config.py: MIXUP_P = 0.7` (2026-09-04). Clears the bar, wins on
3/3 seeds, and p=0.250 is the *floor* at n=3 — it cannot do better. The strongest
part of the result is the variance: **sd 0.0027 vs base's 0.0172, a 6× reduction.**
It stabilises training as much as it improves it.

Two honest caveats. The mean is inflated by base's seed-44 run (0.6869, a low
outlier against its own 0.7151/0.7180) — the +0.0375 on that seed does most of the
work. And `mix0.3` doing nothing while `mix0.7` helps means the effect is not simply
monotone in augmentation strength; 0.7 may be near a local optimum that was not
bracketed above (0.9 untested).

### The surprise, worth flagging

I expected *less* augmentation to help — that was the overfit reading. **More
helped.** So the plateau was not over-augmentation: the model is genuinely capacity-
or data-limited and benefits from additional regularization. The reasoning that
motivated the test was half wrong; **testing both directions rather than only 0.3 is
what caught it.**

---

## 5. Alpha temperature + dedicated LR ❌ — negative result

**What.** `ALPHA_TAU` (`sigmoid(a_raw/τ)`) and `ALPHA_LR_MULT` (own AdamW group).

### Purpose — a gradient-magnitude fact, not a performance observation

19–20 of 34 per-class alphas finished within 0.05 of their 0.5 initialisation
(seed 42: 19/34, 43: 11/34, 44: 20/34), cross-class std ~0.083 — yet **cross-seed
correlation r = 0.939**. The direction is learned reliably; the magnitude cannot
travel.

This distinction matters for the split discipline: it is an *optimisation* fact
derived from parameter values, not from per-class performance, so acting on it does
not leak val information.

### Concept — why α is gradient-starved

At the neutral initialisation, `∂a/∂a_raw = sigmoid'(0) = 0.25` — the sigmoid's
flattest useful region. And `a_raw` is a single scalar per class competing against
15.8M parameters at one shared learning rate. Two independent remedies:

- **τ < 1** rescales the *map*: the same raw displacement yields a larger α change.
  Crucially `sigmoid(0/τ) = 0.5` for any τ, so the unbiased initialisation survives.
- **LR multiplier** rescales the *step*.

Neither tells the model which stream to prefer; both only make the destination
reachable within 40 epochs.

### Result — mechanically right, practically wrong

```
run             τ    alr   α range        α std   stuck<0.05   dev_mAP
base           1.0   1.0   0.352-0.713   0.0913     16/34      0.7151
tau0.3         0.3   1.0   0.275-0.820   0.1471     10/34      0.7265
alr10          1.0  10.0   0.145-0.924   0.1938     10/34      0.7012
tau0.3_alr10   0.3  10.0   0.080-0.934   0.2137      5/34      0.7037
```

α spread rises **monotonically** with both knobs; the stuck count falls 16 → 5. The
mechanism does exactly what it was designed to do. **But dev mAP does not follow** —
the most-spread configuration scores *below* baseline.

### `tau0.3` went to 3 seeds and was NOT adopted

`tau0.3` was the one α variant that looked promising on the screen (+0.0114), so it
went into `run_confirm.sh` alongside `mix0.7`:

```
                 s42     s43     s44      mean     vs base   p      wins
alpha_tau=0.3  0.7265  0.7004  0.7268   0.7179   +0.0113   0.750   2/3
base           0.7151  0.7180  0.6869   0.7067
per-seed diff  +0.0114 -0.0177 +0.0400
```

It nominally clears the +0.0078 bar, but: **2/3 seeds, p=0.750, and it lost seed 43
by −0.0177** — a larger miss than its own mean gain. The +0.0400 on seed 44 is
against base's low outlier, so the mean is not trustworthy. **NOT adopted;
`ALPHA_TAU` stays 1.0.**

⚠️ **Departure from the letter of the pre-registered rule, stated openly.** The rule
said that if both candidates cleared the bar and were within 0.0078 of *each other*,
that is a tie between them and the default is kept. `mix0.7` and `tau0.3` finished
0.0062 apart — inside that window — so read literally the clause would have rejected
*both*, including `mix0.7`. The clause was written to stop a coin-flip choice between
two **rival** settings of one knob. `mixup_p` and `alpha_tau` are **independent**
knobs (augmentation strength vs fusion temperature), so the tie-break does not apply
as intended, and each was judged on its own merits instead: `mix0.7` 3/3 seeds
adopted, `tau0.3` 2/3 seeds rejected. The rule was mis-specified, not the results;
future pre-registration should scope tie-breaks to levels of the same factor.

### Revised concept

α sitting near 0.5 was a **symptom the network tolerated, not the bottleneck.**
Freeing α lets it overcommit early on per-class gradients estimated from as few as
149 positives (DENELE, the rarest eval class). A classic bias-variance trade: less
bias in α, more variance, net loss.

### Consequence for the roadmap

This **kills dynamic/input-conditioned fusion**, previously the top-ranked next
step. A per-clip gate grants strictly *more* freedom — and more freedom just failed.

It **promotes the α-frozen-at-0.5 control** instead: if frozen matches learned, the
per-class mechanism is decorative and the real gain came from moving fusion to the
logit level with a shared head. Result #13 below reinforces this.

---

## 6. Longer schedule for `raw` ⏸

**What.** `--epochs 60` for the `raw` arm specifically.

**Purpose.** `raw` peaked at epochs **36, 37, 16 of 40** — two of three seeds were
still improving when the budget ended.

### Concept — why this is not merely "more steps"

Cosine anneals to its floor at exactly `epochs`, so `--epochs 60` is a **different
schedule**, not an extension: gentler decay, more time at moderate LR.

`ep60` on the dual arm gave dev 0.7151 — identical to base, no gain. But dual peaks
at epochs 14–25, so it had nothing to gain. `raw` is the arm that plausibly does.

### Why it matters for honesty, not performance

Dual vs raw is currently a tie (+0.0078, p=0.437). If a longer `raw` run improves,
the tie moves **against** dual and the headline changes. It is a check on our own
preferred result — which is exactly why it should be run.

---

## 7. Batch size 32 → 64 ✅

**What.** The paper uses 32.

**Purpose.** 32 underutilises MPS — per-step kernel-launch overhead dominates.
Measured throughput was flat from 32 to 64, so 64 is free.

### Concept — a second benefit

BatchNorm estimates μ and σ per batch. At 32 those estimates are noisier, which both
adds gradient noise and pollutes the running statistics used at evaluation time. 64
gives cleaner statistics at no throughput cost.

Above 64 the model becomes bandwidth-bound rather than launch-bound, so the gain
stops — which is why 64 and not 128.

---

## 8. SpecAugment time mask 40 → 12 ✅

**What.** Rescaled the paper's mask width.

### Purpose

The paper's 40 frames on **431-frame** clips is 9.3% per mask. Applied verbatim to
our **130-frame** clips it becomes **30.8% per mask, 61.5% with two masks.**

### Concept — transfer the ratio, not the integer

For *scene* classification, erasing 60% may survive: an "airport" is stationary
ambience, so the remaining 40% carries the same texture. For **event detection** it
is fatal — a frog call is a localised 200–500 ms event, and a 40-frame (928 ms) mask
can erase it entirely **while the label still says present.** That trains the model
to hallucinate the species from background alone.

12 frames = 278 ms, 18.5% worst case with two masks — preserving the *fraction* the
paper actually applied. `test_specaugment_preserves_shape_and_masks` encodes
`SPEC_TIME_MASK × SPEC_N_MASKS < 0.3 × N_FRAMES` so the reasoning survives future
edits.

### Subtlety worth knowing

Masks fill with `0.0`, and because masking runs *after* normalisation that value is
the train-split **mean** (−27.6 dB), not silence (≈ −3.87 in normalised units). This
is the original SpecAugment recommendation and is the better choice here: a
mean-filled patch reads as *uninformative*, whereas a silence-filled patch asserts
*confidently empty* — a stronger and more misleading claim for a detection task.

---

## 9. Gradient clipping ❌ — added, measured, reverted

**What.** `GRAD_CLIP=1.0` → `0.0`.

**Purpose.** Added as cheap insurance: ultra-rare classes (LEPFLA, 7 positives)
contribute large sparse gradients when they appear.

### Then measured

Mean pre-clip gradient norm: **0.132** — the threshold could never fire. Isolated
benchmark: **241 → 280 ms/step (+16%)** for a no-op. Across a 13-hour sweep, that is
real time spent on nothing.

### Concept — clipping is neither free nor neutral

It requires a full-model norm reduction every step. Worse, a threshold that fires
*constantly* silently rescales every update, changing the effective learning rate in
a way that appears in no log.

### How it was reverted — which is the point

Not deleted. The flag remains, and `grad_norm` is still logged (sampled every 50th
step, <1% cost) so `GRAD_CLIP=0` stays a **monitored** decision rather than an
unmonitored one. If `grad_norm` climbs toward 1, `--grad-clip 0.5` re-enables it.

*Correction recorded:* I initially attributed a 240 → 363 s/epoch slowdown to
clipping. The isolated benchmark showed +16%; the remainder was contention from a
concurrent pytest run.

---

## 10. AsymmetricLoss 🔬

**What.** `--loss asl` (Ridnik et al.), pre-registered as a declared comparison
rather than a result-driven choice.

**Purpose.** Purpose-built for the 26.8:1 negative:positive imbalance.

### Concept — two mechanisms

- **Asymmetric focusing** (`γ_neg=4.0, γ_pos=0.0`): easy negatives receive weight
  `p⁴ → 0` and vanish from the gradient; positives keep full weight. With 96.4% of
  the label matrix zero, most of BCE's gradient signal is easy negatives.
- **Probability shifting** (`clip=0.05`): negatives predicted below 0.05 contribute
  **exactly zero** — a hard floor, not a down-weight. This matters in bioacoustics,
  where faint distant calls are routinely missed by annotators, so some "negatives"
  are mislabelled positives.

Its advantage over `pos_weight` is requiring **no per-class constants** —
sidestepping rather than clamping the problem that inverse-frequency weighting would
hand LEPFLA a factor of ~8,883, letting a single example dominate a batch.

### Result

**dev 0.7053 vs base 0.7151 — worse.**

Its train loss (0.2178) is not comparable to BCE's; different objective. One oddity:
`asl` shows a *positive* dev→val gap (+0.0168) where most runs are negative. Not
interpretable at one seed, but worth watching if pursued.

---

## 11. Reproducible seeding ✅

**What.** Seeded the DataLoader generator and MixUp's RNG as separate streams.

**Purpose.** The DataLoader was **never seeded** — shuffle order came from the global
torch RNG, which `spec_augment` also consumes inside `__getitem__`.

### Concept — RNG stream coupling

Because both drew from one stream, batch order depended on **how many augmentation
draws had occurred**. Changing `SPEC_N_MASKS` would silently change batch order too,
coupling two things that must stay independent — and making any augmentation
ablation partly a batch-order ablation.

Now:

| Stream | Mechanism |
|---|---|
| Weight init | `torch.manual_seed(seed)` |
| Batch order | dedicated `torch.Generator` → `DataLoader(generator=...)` |
| MixUp λ + permutation | `np.random.default_rng(seed)`, own stream |
| SpecAugment | global torch RNG, seeded |
| Worker processes | `seed_worker()` re-seeds numpy/random (PyTorch seeds only torch) |

### What is not claimed: bitwise reproducibility

MPS parallel reductions do not guarantee summation order, and float addition is not
associative, so same-seed runs diverge — slowly, then materially once divergence
reaches `argmax`-based threshold selection and best-epoch choice. There is no MPS
equivalent of `use_deterministic_algorithms(True)`.

What seeding *does* buy: identical initialisation, batch order, and augmentation
draws — so seed-to-seed differences measure **seed sensitivity** rather than
unseeded RNG noise. That is the precondition for reporting mean±std.

---

## 12. Three-way train/dev/val split ✅

**What.** Carved dev from train via a nested grouped split; the val fold is
byte-identical to before.

```
train  44,408 clips / 767 recordings   gradients + normalisation statistics
dev     8,932 clips / 154 recordings   early stopping, checkpointing, thresholds
val     8,851 clips / 153 recordings   read ONCE, with dev-fitted thresholds
```

### Purpose

Val was doing three incompatible jobs: early stopping, per-class threshold fitting
(**42 fitted parameters**), and architecture selection. Normally the remedy is to
hold out the test set — but `test.7z` contains **31,187 wavs and no label CSV**, so
no second held-out set was available while these experiments ran. (Upstream labels
were located afterwards and test was scored once at the end; see `RESULTS_REPORT.md`
§7.4. No tuning decision here was informed by test.)

### Concept — every use of a holdout spends it

Threshold tuning fits 42 parameters; scoring them on the same data inflates
macro-F1. Measured:

```
dev macro-F1 tuned  0.5569   ← thresholds fitted on dev, scored on dev
val macro-F1 tuned  0.5133   ← same thresholds, scored on held-out val
```

Val lands **below** the dev-tuned figure. Part of the legacy "0.4568 → 0.6426 from
threshold tuning" was fitted-on-what-it-scores rather than real generalisation.

### Why val stayed byte-identical

12 completed ablation runs were scored on it; perturbing it would silently make them
incomparable. `test_val_fold_matches_completed_runs` enforces this against a stored
backup, and `src/splits.py` prints an explicit confirmation.

### Cost

Train shrinks 53,340 → 44,408 (−17%), which lowers every tuning number by
construction. `base` at dev 0.7151 is the only valid reference for tuning
comparisons — **never compare tuning runs against the ladder's 0.7633.**

---

## 13. Per-class fusion — the attribution result 🔬

**What.** `--scalar-fusion` reruns the paper's single global α under otherwise
identical current settings, isolating what per-class fusion actually contributed.

### Result

```
base          (per-class α, 42 weights)   dev 0.7151
attr_scalar   (paper's single scalar α)   dev 0.7116     Δ = −0.0035
```

**−0.0035 is well inside the measured seed noise of 0.0078.** In absolute mAP, on
this split and this seed, per-class fusion contributed **approximately nothing.**

### Why this does not invalidate the change

The two facts sit together and both matter:

1. **In absolute mAP:** no measurable gain (−0.0035, inside noise).
2. **In ordering:** under scalar fusion in the *original* configuration, dual scored
   **below its own better input stream** (0.6541 vs percussive 0.6886) — fusion was
   destroying information. With per-class fusion, dual beats both inputs with
   significance (+0.0260 over percussive, p=0.007).

So the honest claim is narrower than "per-class fusion improves the model": it
**removes a pathology** in which the fused model underperformed its own components.
That is a correctness fix, not a performance win.

### The corollary

Most of the +0.109 over the legacy configuration must come from the **other three**
simultaneous changes — cosine LR, AdamW, and the stem pool. The remaining
attribution runs (`attr_stem1`, `attr_flatlr`) will apportion it.

Note also `attr_scalar`'s α: 0.474, and it early-stopped at epoch 13. Combined with
finding #5 (more α freedom hurt), the picture is that **α is simply not where the
performance is** on this task — which strengthens the case for running the
α-frozen-at-0.5 control rather than building more elaborate fusion.

---

## 14. Capacity control — raw-wide vs dual ✅ (the decisive experiment)

**What.** `--width 1.41` widens a single raw-mel stream to 15,777,394 params, within
0.3% of dual's 15,824,344. Architecture becomes the only difference.

### Purpose — break a confound no earlier run could

Dual beat raw by +0.0078 (p=0.437) on the ladder while using **2× the parameters**.
Two explanations fit every experiment to that point: H1 the decomposition provides
better-structured input; H2 dual is simply bigger and HPSS is irrelevant.

This is unusually clean to test here because `margin=1.0` preserves
`X_h + X_p = X_raw` (verified to 1.5e-05), so **HPSS adds literally zero
information.** Dual cannot be winning by knowing more — only via inductive bias or
capacity.

### Result — 3 seeds, three-way split, dev-selected

```
              s42     s43     s44      mean     sd
raw-wide    0.7219  0.7126  0.7341   0.7229   0.0108
dual        0.7151  0.7180  0.6869   0.7067   0.0172
per-seed    +0.0068 -0.0054 +0.0473
                              mean diff +0.0162   p=0.500   raw-wide wins 2/3
```

**Verdict: H2.** By the pre-registered rule (`raw-wide ≥ dual`), the decomposition
contributes nothing at matched capacity — a single raw-mel stream *nominally beats*
the dual-stream HPSS model. **The paper's central mechanism does not transfer to
multi-label anuran detection.**

⚠️ **Report this as a tie, not a win.** p=0.500 on a paired sign-flip test. The
+0.0162 mean is driven almost entirely by one seed (+0.0473), and dual's seed-44 run
(0.6869) is a low outlier against its own 0.7151/0.7180. Remove that run and the
effect largely evaporates. The defensible claim is **"at matched capacity, HPSS
dual-stream is not better than a single raw-mel stream"** — which is still a negative
result for the paper, because the paper's claim requires dual to *win*.

### Side effect that reframed the whole document

Dual's three-way sd of **0.0172** is what retired the σ=0.0078 figure. See the
header. This retroactively vindicated distrusting `attr_stem1`'s +0.018.

### Caveat that survives either outcome

The paper's stated benefit is *cross-device robustness* on DCASE 2020 Task 1A.
AnuraSet uses a single recorder type, so the mechanism may be under test with its
primary advantage absent. Belongs in the writeup regardless.

---

## 15. HPSS kernel sweep ❌ — a well-motivated hypothesis, falsified

**What.** `--kernel {5, 9, 13, 25}` vs the inherited 17, each requiring its own
6 GB precomputed feature file. Arm: `percussive` (cheapest, and the stream the model
leans on). Single seed, pre-registered.

### Purpose — the last untested paper hyperparameter, on physical grounds

At `hop=512 / 22,050 Hz` each frame is 23.2 ms, so the median-filter span is:

```
k=5  116 ms      k=13  302 ms      k=25  580 ms
k=9  209 ms      k=17  394 ms  <- inherited default
```

Anuran calls here are **200–500 ms**. So k=17 asks "does energy persist across
394 ms?" — *longer than many calls it must separate*. A tonal 250 ms call fails that
test and is routed to the **percussive** stream regardless of its true structure.
Corroborating: harmonic energy share rises monotonically with k (0.383 / 0.394 /
0.410 / 0.425 / 0.456 for k = 5/9/13/17/25), so at the default only ~42% of energy
reaches the harmonic stream — consistent with percussive-only (0.7373) beating
harmonic-only (0.7109) at p=0.015.

**Prediction made in advance:** a shorter kernel (9 or 13) separates frog calls
better, and the sensitivity curve runs *opposite* to the paper's.

### Result — the prediction was wrong

```
k     span     dev_mAP   vs k17
25    580 ms   0.7288    +0.0090
5     116 ms   0.7219    +0.0021
17    394 ms   0.7198     —
9     209 ms   0.7192    -0.0006
13    302 ms   0.7138    -0.0060
```

**Null result.** Total spread across all five kernels is **0.0150 — below the 0.0172
seed sd on this split.** There is no monotone trend in either direction, and k=13
sits *below both its neighbours*, which is the signature of noise rather than a
curve. k=25 is nominally best, the opposite of the prediction, but +0.0090 on a
single seed is not a finding.

**Conclusion: HPSS kernel length does not measurably affect performance on this
task.** `HPSS_KERNEL` stays 17.

### Why this is still worth reporting

The physical argument was sound and the energy-share measurements confirmed the
mechanism *does* shift content between streams — yet the downstream model is
indifferent to it. Combined with #14 (decomposition adds nothing at matched
capacity), the coherent reading is that **the network is insensitive to how HPSS
apportions energy because it is not exploiting the decomposition in the first
place.** Two independent experiments pointing at the same conclusion.

Mechanics verified before trusting the numbers: each run received a distinct `--h5`,
and each file carries its own stamped `hpss_kernel` attr (5/9/13/25 confirmed).
`features.h5` (k=17) predates the stamping code, so it reports "unstamped" — cosmetic
only. `summary.json` does not record the `--h5` path, which is a real reporting gap
worth closing if the sweep is ever extended.

---

## Cross-cutting lessons

**The reverts and negative results carried more information than the adoptions.**
Gradient clipping *looked* free until measured (norm 0.132, cost 16%). The alpha
knobs worked exactly as designed and still made things worse — closing off dynamic
fusion, the direction I had been most confident about. MixUp moved the opposite way
from prediction. Per-class fusion, the headline contribution, turned out to be worth
~0.003 in absolute mAP.

**Measurement trustworthiness came before accuracy.** The LR schedule (#2), the
seeding fix (#11), and the dev split (#12) produced little or no direct mAP gain.
They mattered because without them the numbers could not be read: the checkpoint
lottery made the ablation table meaningless, coupled RNG streams made augmentation
ablations partly batch-order ablations, and threshold fitting on the reporting split
inflated macro-F1.

**Transferring hyperparameters requires transferring the ratio, not the integer.**
SpecAugment 40 → 12 (#8) is the clearest case: the same number meant 9.3% of a clip
in the paper and 30.8% here. `HPSS_KERNEL=17` was the other candidate — 394 ms of
median smoothing against 200–500 ms calls — and it was **swept and came back null**
(#15). The lesson holds for #8 but does not generalise: a hyperparameter can be
dimensionally wrong and still not matter, if the model is not exploiting the
mechanism it controls.

**Measure before optimising, and keep the measurement after deciding.** The pattern
used throughout: add the knob, measure whether it engages, revert if it does not —
but leave the diagnostic in place (`grad_norm` sampling, `frac_mixed`, `lr` per
epoch, per-class α in `summary.json`) so every decision stays monitored rather than
assumed.
