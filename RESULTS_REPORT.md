# RESULTS REPORT — HSPP replication on AnuraSet

**Generated:** 2026-09-05 20:15
**Source:** 12 runs, `runs/{dual,raw,harmonic,percussive}_s{42,43,44}_3way/summary.json`
**Config signature:** verified identical across all 12 runs (`mixup_p=0.7`,
`stem_pool=2`, `alpha_tau=1.0`, `alpha_lr_mult=1.0`, `grad_clip=0.0`,
`lr_schedule=cosine`, `selection_split=dev`, `width=1.0`)

---

## 1. HEADLINE

> On multi-label anuran detection, the paper's HPSS dual-stream architecture is
> **not better than a single-stream CNN on the undecomposed mel spectrogram**
> (dev mAP 0.7241 ± 0.0027 vs 0.7278 ± 0.0114; paired p=1.000 — a tie), while using
> **1.98× the parameters** (15,824,344 vs 7,983,212).
>
> **Confirmed on the upstream test set** (31,187 clips, 538 unseen recordings, labels
> from `AnuraSet_v1.0.0/metadata.csv`, scored once with dev-fitted thresholds):
> dual 0.6988 ± 0.0095 vs raw 0.6925 ± 0.0082, **p=0.750 — a tie**, with the sign
> flipped relative to dev. Two indistinguishable models across three evaluation sets.
>
> Five independent lines of evidence agree: the direct 4-arm ablation, the upstream
> test set, a parameter-matched capacity control (raw-wide ≥ dual, p=0.500), an HPSS
> kernel sweep spanning 116–580 ms (null, spread 0.0150 < σ 0.0172), and per-clip
> prediction agreement (dual vs raw **99.63%** of cells, r=0.979). Because
> `X_h + X_p = X_raw` exactly (verified to 1.5e-05), HPSS adds **zero information** —
> so there is no mechanism by which it could help other than inductive bias, and no
> evidence that it does.
>
> The paper's **single-scalar fusion is a genuine defect**: it caused the dual model
> to score *below its own better input stream* (0.6541 vs 0.6886). Per-class
> logit-level fusion (+41 parameters) removes that pathology and is reproducible
> across seeds (r=0.939) — but is worth **−0.0035 mAP**, inside noise. A correctness
> fix, not a performance gain.

---

## 2. THE ABLATION TABLE (the deliverable)

4 arms × 3 seeds, three-way grouped split, **dev-selected**, val read **once** with
dev-fitted thresholds.

| arm | s42 | s43 | s44 | **dev_mAP** | val_mAP | val macro-F1 | params |
|---|---|---|---|---|---|---|---|
| raw | 0.7253 | 0.7403 | 0.7178 | **0.7278 ± 0.0114** | 0.7081 ± 0.0109 | 0.6362 ± 0.0028 | 7,983,212 |
| dual | 0.7266 | 0.7212 | 0.7244 | **0.7241 ± 0.0027** | 0.7185 ± 0.0071 | 0.6409 ± 0.0134 | 15,824,344 |
| percussive | 0.7198 | 0.7073 | 0.7250 | **0.7174 ± 0.0091** | 0.6946 ± 0.0114 | 0.6273 ± 0.0093 | 7,983,212 |
| harmonic | 0.6605 | 0.6632 | 0.6701 | **0.6646 ± 0.0049** | 0.6687 ± 0.0015 | 0.5911 ± 0.0099 | 7,983,212 |

### Pairwise tests — exact paired sign-flip permutation on dev_mAP

| comparison | mean diff | p | seeds won | per-seed diffs |
|---|---|---|---|---|
| raw − dual | +0.0037 | **1.000** | **1/3** | −0.0013, +0.0191, −0.0066 |
| dual − percussive | +0.0067 | 0.500 | 2/3 | +0.0068, +0.0139, −0.0006 |
| raw − percussive | +0.0104 | 0.750 | 2/3 | +0.0055, +0.0330, −0.0072 |
| dual − harmonic | +0.0595 | **0.250** | **3/3** | +0.0662, +0.0579, +0.0543 |
| percussive − harmonic | +0.0528 | **0.250** | **3/3** | +0.0594, +0.0440, +0.0549 |
| raw − harmonic | +0.0632 | **0.250** | **3/3** | +0.0649, +0.0770, +0.0478 |

**With n=3 the permutation floor is p=0.250.** Nothing in this project can reach
p<0.05. Read p=0.250 as "consistent across all three seeds" — the strongest statement
available — and p≥0.500 as "a tie."

---

## 3. WHAT IS ESTABLISHED

### 3.1 dual vs raw is a tie, and the mean *favours raw* — but read it carefully

`raw − dual = +0.0037, p=1.000`. This is the maximum p the test can return.

**Important nuance that a mean alone hides: raw wins only 1 of 3 seeds.** Dual wins
seeds 42 and 44; raw wins seed 43 by a large +0.0191, which drags the mean positive.
So it is *not* correct to say "raw is better" — the per-seed record is 2–1 in dual's
favour while the mean tilts the other way. **The only defensible statement is that
the two are indistinguishable.**

Both framings must appear together or the result is misreported.

### 3.2 Fusion beats its own input streams — the per-class fix works

- dual − harmonic **+0.0595, 3/3 seeds, p=0.250**
- dual − percussive +0.0067, 2/3 seeds, p=0.500

Under the paper's scalar α, dual scored **below** percussive-only (0.6541 vs 0.6886).
With per-class α it is above both. **The fusion mechanism is now correct.** It simply
does not beat skipping HPSS altogether.

### 3.3 Harmonic is the weakest stream by a wide, consistent margin

harmonic 0.6646 is **−0.0632 below raw (3/3 seeds)** and −0.0528 below percussive
(3/3). Both p=0.250. This is the largest and most reproducible effect in the table.

Consistent with the measured energy split: at `HPSS_KERNEL=17` only **~42%** of
spectral energy reaches the harmonic stream. Anuran calls are largely
broadband/pulsatile, so the percussive component carries most of the discriminative
signal.

### 3.4 Dual's real advantage is variance, not accuracy

| arm | dev sd | dev→val gap |
|---|---|---|
| **dual** | **0.0027** | **−0.0056** |
| harmonic | 0.0049 | +0.0041 |
| percussive | 0.0091 | −0.0228 |
| raw | 0.0114 | −0.0198 |

Dual is **4.2× more stable across seeds than raw** (0.0027 vs 0.0114) and has the
smallest dev→val gap of the three strong arms (−0.0056 vs −0.0198). It also has the
best val_mAP (0.7185 vs raw 0.7081) and val macro-F1 (0.6409 vs 0.6362) despite the
lower dev mean.

**This is a legitimate secondary finding and must be labelled as such.** dev is the
pre-registered selection metric; val is reported once for transparency and was **not**
used to choose. Ranking arms by val_mAP after seeing it would be exactly the
selection error the three-way split exists to prevent. State it as *"dual is more
stable and shows a smaller generalisation gap,"* never as *"dual wins on val."*

---

## 4. CORROBORATING EXPERIMENTS

### 4.1 Capacity control — parameter-matched

Widening a single raw stream to dual's size (`--width 1.41` → 15,777,394 params,
within 0.3% of 15,824,344):

```
              s42     s43     s44      mean     sd
raw-wide    0.7219  0.7126  0.7341   0.7229   0.0108
dual        0.7151  0.7180  0.6869   0.7067   0.0172
                    mean diff +0.0162   p=0.500   raw-wide wins 2/3
```

**H2 confirmed:** the decomposition contributes nothing at matched capacity. Report
as a tie (p=0.500); the mean is inflated by one seed (+0.0473) against dual's low
outlier (0.6869).

*Note: these runs used `mixup_p=0.5` (pre-adoption), so they are internally
comparable but not directly comparable to §2.*

### 4.2 HPSS kernel sweep — null

```
k     span     dev_mAP   vs k17
25    580 ms   0.7288    +0.0090
5     116 ms   0.7219    +0.0021
17    394 ms   0.7198      —
9     209 ms   0.7192    -0.0006
13    302 ms   0.7138    -0.0060
```

Spread **0.0150 < σ 0.0172**; no monotone trend; k=13 sits below both neighbours.
The pre-registered prediction (shorter kernel wins, curve opposite to the paper's)
was **falsified** — k=25 is nominally best. Kernel length does not measurably matter.

The physical argument was nonetheless sound: harmonic energy share rises
monotonically with k (0.383/0.394/0.410/0.425/0.456 for k=5/9/13/17/25), so the
kernel *does* shift content between streams. The model is simply indifferent to it —
consistent with it not exploiting the decomposition at all.

### 4.3 Tuning — 1 of 11 interventions survived

| change | result | verdict |
|---|---|---|
| **MixUp p 0.5 → 0.7** | **+0.0174, 3/3 seeds, sd 0.0027 vs 0.0172** | **ADOPTED** |
| α temperature 0.3 | +0.0113, 2/3 seeds, p=0.750 | rejected |
| α LR ×10 | 0.7012 vs 0.7151 | rejected |
| AsymmetricLoss | 0.7053, −0.0098 | rejected |
| 60 epochs | ±0.000 | rejected |
| MixUp p 0.3 | +0.0001 | rejected |
| gradient clipping | norm 0.132 — never fired, cost 16%/step | reverted |

### 4.4 Attribution — the +0.109 over legacy is NOT decomposable

```
base          (all four changes)     dev 0.7151
attr_scalar   (no per-class fusion)  dev 0.7116   -0.0035
attr_flatlr   (no cosine LR)         dev 0.7133   -0.0018
attr_stem1    (no stem pool)         dev 0.7334   +0.018   <- NOT REAL
AdamW                                             never ablated
```

`attr_stem1`'s +0.018 is a **single-epoch spike at epoch 13**: the run never exceeded
that value across the following 13 epochs and spent much of them *below* base.
Selecting max-over-40-epochs is biased upward for the noisier sequence.

Every individual removal costs ~nothing, AdamW was never tested, and the legacy
comparison straddles both a split change (two-way → three-way) and a metric change
(val → dev). **MixUp 0.7 is the only confirmed win in the project.** Everything else
in the config is defensible engineering, not demonstrated gain.

---

## 5. METHOD

```
corpus      62,191 clips / 42 species / 1,074 recordings / 4 sites / 3.0 s @ 22.05 kHz
imbalance   3.59% positive cells, 26.8:1 neg:pos, 36.2% all-negative clips
            prevalence spans 4 orders of magnitude (SPHSUR 21.3% -> LEPFLA 0.011%)
eval        34 of 42 classes (>=100 positives); 2 zero-positive, 6 ultra-rare excluded

split       train 44,408 / dev 8,932 / val 8,851 clips
            767 / 154 / 153 recordings, GROUPED BY RECORDING (clips from one
            recording are near-duplicates; a random split leaks)
            dev: early stopping, checkpoint selection, threshold fitting
            val: read ONCE at the end with dev-fitted thresholds
```

The bundle provided held 31,187 test wavs with **no label CSV**, so val was the only
held-out estimate available while the experiments ran, and every decision was made on
dev. The test labels were located upstream *afterwards* (§7.4) and scored once. Its
538 recordings are group-disjoint from all 1,074 training recordings.

**Pre-registered before running:** selection on `dev_mAP` only; no architecture choice
justified by per-class AP inspection; every arm reported including losers.

**σ ≈ 0.017 on this split.** The often-quoted 0.0078 came from an earlier two-way
split with 17% more training data and understates variance by 2.2×.

---

## 6. LIMITATIONS

1. **n=3 seeds → permutation floor p=0.250.** No result here can reach p<0.05.
2. **Single recorder type.** The paper's stated benefit is *cross-device robustness*
   on DCASE 2020 Task 1A. AnuraSet uses one recorder model, so the mechanism may be
   under test with its primary advantage absent. This bounds the conclusion; it does
   not rescue the claim, since the claim was never restricted to multi-device data.
3. **One seed does heavy lifting in two places.** raw's seed-43 (+0.0191) drives the
   raw−dual mean; dual's seed-44 (0.6869) inflates both the capacity-control and
   MixUp margins.
4. **Test labels were located late** (upstream `metadata.csv`), so dev and val drove
   every decision and test was scored once at the end. That is the correct order, but
   it means test played no role in model selection — by design.
5. **Four classes are sub-chance** (DENCRU AP 0.0096 at 602 positives). Cause is
   **recording diversity, not support**: mean AP 0.350 with ≤2 val recordings vs
   0.764 with more.
6. **`tail_mAP` is NaN, and that is correct.** All 6 ultra-rare species live in 1–3
   recordings, so a group-disjoint split leaves zero val positives. Reporting 0.0
   would be fabrication.
7. **MPS is not bitwise reproducible** (non-deterministic reduction order), which is
   why everything is reported as mean ± sd rather than single numbers.
8. **The paper's own figures are internally inconsistent.** 2.86 GFLOPs is
   arithmetically unreachable from its stated architecture — our faithful
   implementation gives 11.216 GFLOPs, implying an undocumented ~/4 early downsample.

---

## 7. TEST-SET INFERENCE AND SCORING (upstream labels obtained)

All 12 checkpoints were run over the held-out `test.7z` clips.

```
features   data/features_test.h5  -- 31,187 clips, 2.90 GB, 91 s extraction
           zero NaN across X_h / X_p / X_raw (the 40 fp32 overflow warnings in the
           mel matmul are benign and recover in log space; they occur in the
           training features too)
outputs    preds/{arm}_s{seed}_3way.csv          per-class sigmoid probabilities
           preds/{arm}_s{seed}_3way_binary.csv   dev-fitted thresholds applied
           preds/ENSEMBLE_{arm}.csv              3-seed mean probabilities
normalisation  each run's own norm.json (TRAIN-split stats). Recomputing on test
               would be test-time leakage.
thresholds     thresholds.npy per run (dev-fitted). Never re-tuned on test.
```

Sections 7.1–7.3 were computed **before** labels were available and are pure
consistency checks. §7.4 has the actual scores, obtained after the upstream labels
were located — and it confirms the pre-registered prediction.

### 7.1 Calibration transfers to unseen recordings

Predicted positive rate on 538 recordings disjoint from all 1,074 training
recordings, against the training prior of **0.0359**:

| arm | ensemble positive rate | vs prior |
|---|---|---|
| dual | 0.0321 | −0.0038 |
| raw | 0.0322 | −0.0037 |
| percussive | 0.0317 | −0.0042 |
| harmonic | 0.0318 | −0.0041 |

All four land within 0.004 of the prior on completely unseen recordings — mild
under-prediction, consistent across arms. Single-seed rates spanned 0.0353–0.0394.

### 7.2 Seed stability confirms the dev-side variance ordering

Mean pairwise probability correlation between seeds within each arm:

```
dual        0.9641   <- most stable, matching its dev sd of 0.0027
percussive  0.9554
raw         0.9544
harmonic    0.9486
```

**dual is the most self-consistent arm on test as well as on dev.** This is
independent corroboration of §3.4 from data that played no part in selection — the
one genuinely new piece of evidence in this section.

### 7.3 The arms make nearly the same predictions

Cross-arm agreement on binarised predictions over 1,309,854 cells:

```
                  dual      raw   percussive  harmonic
dual            1.0000   0.9792     0.9729    0.9691     <- probability correlation
raw             0.9792   1.0000     0.9764    0.9634
percussive      0.9729   0.9764     1.0000    0.9525
harmonic        0.9691   0.9634     0.9525    1.0000

binarised agreement:  dual vs raw 99.63%   dual vs percussive 99.56%
                      raw vs percussive 99.58%   worst pair 99.34%
```

**dual and raw agree on 99.63% of cells and correlate at r=0.979.** The two
architectures are not merely scoring alike in aggregate — they are making the *same
per-clip decisions*.

This is the strongest single piece of evidence in the report that HPSS decomposition
is not doing independent work. A tie in mAP (§3.1) could in principle hide two models
with different strengths that cancel out. It does not: they are near-duplicates
clip-by-clip. Combined with `X_h + X_p = X_raw` (HPSS adds zero information), the
coherent reading is that **both networks converge on substantially the same function
regardless of whether the input is decomposed.**

Note that even `harmonic` — 0.0632 below raw in dev mAP, the largest gap in the
table — still agrees with dual on 99.52% of cells. At a 3.59% positive rate most
cells are easy negatives, so high agreement is partly structural. The *ordering* of
the correlations is the signal, not their absolute level.

### 7.4 TEST LABELS OBTAINED AND SCORED — the pre-registered prediction held

`AnuraSet_v1.0.0/metadata.csv` was downloaded and **contains the test labels.**

**Provenance verified before any score was computed:**

```
metadata.csv       93,378 rows, subset column: train=62,191 / test=31,187
                   EXACT match to our local split, both sides
species columns    identical to C.SPECIES, in order
train cross-check  all 62,191 rows vs our train.csv -> 0 label mismatches
join key           "{fname}_{min_t}_{max_t}.wav" -> 31,187/31,187, 0 unmatched
test prior         0.0360 positive rate (train 0.0359); 36.6% all-negative
coverage           all 34 EVAL_CLASSES have test positives (min 10)
                   4 species have zero (LEPELE, RHISCI, LEPFLA, SCIRIZ) -- none
                   is an eval class, so the 34-class metric is unaffected
```

**This corrects the earlier provenance hypothesis.** The 2:1
per-site partition is **defined upstream by the dataset authors**, not constructed
locally: `metadata.csv`'s own `subset` column reproduces our train/test split
exactly, on both sides. The course bundle simply shipped `train.csv` filtered to
`subset=='train'` and omitted the test labels. Nothing was hand-cut.

**Scoring rule, pre-registered in `src/score_test.py` before running:** all 12
checkpoints scored once; mAP over the same 34 eval classes; **dev-fitted thresholds,
never re-tuned on test**; mean ± sd over seeds; paired sign-flip permutation tests.

```
arm             s42      s43      s44     test_mAP mean+-sd   test_F1 mean+-sd
dual          0.6903   0.6971   0.7090   0.6988 +- 0.0095    0.6668 +- 0.0096
raw           0.7012   0.6916   0.6848   0.6925 +- 0.0082    0.6535 +- 0.0142
percussive    0.6842   0.6753   0.6932   0.6843 +- 0.0090    0.6384 +- 0.0077
harmonic      0.6500   0.6501   0.6475   0.6492 +- 0.0015    0.5897 +- 0.0088
```

```
comparison               mean diff     p    wins   per-seed
dual - raw                 +0.0063  0.750   2/3   -0.0109 +0.0055 +0.0242
dual - percussive          +0.0145  0.250   3/3   +0.0061 +0.0218 +0.0158
raw - percussive           +0.0083  0.500   2/3   +0.0169 +0.0163 -0.0084
dual - harmonic            +0.0496  0.250   3/3   +0.0403 +0.0470 +0.0616
percussive - harmonic      +0.0351  0.250   3/3   +0.0342 +0.0253 +0.0457
raw - harmonic             +0.0433  0.250   3/3   +0.0511 +0.0416 +0.0373
```

**The pre-registered prediction was a tie, and test delivered a tie.**
`dual − raw = +0.0063 at p=0.750`, 2/3 seeds, with raw winning seed 42 by −0.0109.
Dev had raw nominally ahead (+0.0037, p=1.000); test has dual nominally ahead
(+0.0063, p=0.750). **Both are ties, and the sign flips between them** — which is
exactly what two indistinguishable models look like across two evaluation sets.

Do not report this as "dual wins on test." The direction reversed from dev, neither
result approaches the p=0.250 floor, and the gap is well inside the ~0.009 test sd.

**What test confirms cleanly (3/3 seeds, p=0.250 — the floor):**
- dual > percussive (+0.0145), raw > percussive (+0.0083 at 2/3)
- **all three strong arms > harmonic** (+0.0351 to +0.0496)

So the ordering established on dev reproduces on a genuinely untouched set. The
harmonic stream being far weakest, and fusion beating its own inputs, are the robust
findings; **dual vs raw is a tie on all three of dev, val, and test.**

**Absolute level.** Test mAP (~0.69–0.70) sits below dev (~0.72–0.73) and val
(~0.71) for every arm. Expected: test is 3.5× larger than val (31,187 vs 8,851
clips), spans 538 unseen recordings, and uses thresholds fitted on dev. The
*ordering* is what transfers, not the level.

**Status of this estimate.** Test was scored **once**, after the config was frozen
and after dev/val had settled every decision. Thresholds came from dev. It is the
cleanest number in the project — and it agrees with dev and val.

### 7.5 Two findings that only the test set could reveal

**(a) The tie is not uniform — it is offsetting per-class differences.**

Per-class test AP, dual vs raw (3-seed ensembles, 34 eval classes):

```
raw better on 21/34 classes, dual better on 13/34
mean |per-class diff| 0.0287     median 0.0176

dual's biggest wins            raw's biggest wins
  PHYMAR  +0.1148 (n=117)        ADEDIP  -0.0921 (n=223)
  ELABIC  +0.1111 (n=491)        RHIICT  -0.0593 (n=180)
  PHYNAT  +0.0575 (n=65)         SCIFUV  -0.0535 (n=1395)
  ELAMAT  +0.0500 (n=293)        DENELE  -0.0485 (n=17)
  PHYDIS  +0.0246 (n=324)        SCIALT  -0.0447 (n=10)
```

So the aggregate tie (+0.0063) hides per-class swings up to ±0.11 that **cancel out**.
Raw wins more classes; dual wins by larger margins where it wins. This is a genuine
qualification of §7.3: the two models agree on 99.63% of *binary decisions* but their
*rankings* differ meaningfully for individual species.

**This does not rescue the decomposition, and must not be presented as if it does.**
Nothing here identifies *which* species benefit in advance — the per-class pattern is
only visible after scoring test, and using it to justify an architecture would be
exactly the val/test-fitting the split discipline forbids. It is an observation about
why aggregate ties can be misleading, not a route to a better model.

**(b) Seed-ensembling beats every architectural difference in the project.**

Averaging the 3 seeds' probabilities per arm, scored on test:

```
arm          3-seed ensemble   single-seed mean    gain
raw               0.7240           0.6925        +0.0315
harmonic          0.6785           0.6492        +0.0293
percussive        0.7122           0.6843        +0.0279
dual              0.7204           0.6988        +0.0216

dual + raw fused  0.7329       (best single arm 0.6988)   +0.0341
```

**Every ensembling gain (+0.022 to +0.032) is larger than every architectural
difference measured anywhere in this project** — larger than dual−raw (+0.0063),
larger than MixUp 0.7 (+0.0174), larger than dual−percussive (+0.0145). Fusing dual
and raw reaches 0.7329, beating the best single model by +0.0341.

The practical implication is blunt: on this task, **three seeds of the cheaper
single-stream model, averaged, outperform any single model of either architecture** —
at 3 × 7.98M params of training cost but only single-stream inference cost per member,
versus dual's 15.8M. If the goal is accuracy per unit of engineering effort, seed
averaging dominates architectural choice here.

Caveat: ensembles were scored on test with dev-fitted thresholds, and the ensembling
comparison was **not** pre-registered — it was computed after the primary result was
settled. Treat it as a well-supported observation for future work, not as a
hypothesis this project tested.

---

## 8. CONCLUSION

The replication is faithful (architecture unchanged: layer types, block ordering,
channel widths, asymmetric 1×n / n×1 pairs) and the negative result is robust across
three independent tests.

**What the paper's mechanism does not buy here:** accuracy. dual ties raw on dev
(p=1.000), on val, and on the upstream test set (p=0.750, sign reversed) at 1.98x the
parameters; ties it again when raw is widened to match; is insensitive to the HPSS
kernel that controls the decomposition; and — on 31,187 unseen test clips — **makes
the same binary decision as raw on 99.63% of cells (r=0.979).** The two architectures
converge on substantially the same function.

**What test DOES confirm, 3/3 seeds at the p=0.250 floor:** all three strong arms beat
harmonic (+0.0351 to +0.0496), and dual beats percussive (+0.0145). The ordering
established on dev reproduces on an untouched set -- so the negative result is about
dual vs raw specifically, not a failure to measure anything at all.

**What it does buy:** stability. dual's cross-seed sd is 4.2× smaller than raw's with
the smallest dev→val gap among the strong arms, and it is also the most
self-consistent arm across seeds on test (r=0.964 vs raw 0.954) — corroborated on
data that played no part in selection. A secondary observation, not a
selection-metric win.

**Methodological contributions:** (a) per-class logit-level fusion, which removes a
real defect in the published design; (b) a reusable capacity-control design for any
two-stream claim whose streams sum to the original input.

**What should not be claimed:** that HPSS works on this task, that per-class fusion
improves accuracy, or that any difference here is statistically significant.
