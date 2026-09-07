# HSPP on AnuraSet — Implementation & Decision Record

Technical documentation for the replication of Liu & Fan, *"Structure-aware acoustic
scene classification: a feature decoupling framework using HPSS and asymmetric
convolutions"* (Scientific Reports, 2026), retargeted from single-label acoustic
scene classification to multi-label anuran species detection on AnuraSet.

This document records **what the paper specifies, what this codebase does, and why
each difference exists**. It is a decision record, not a tutorial: every deviation
is stated with the reasoning and the measurement that motivated it.

Status at time of writing: 4-arm × 3-seed ablation ladder in progress.
All numbers marked *measured* were produced on this machine (M4 Pro, 24 GB, MPS).

---

## 0. Task translation

The paper and this work solve structurally different problems. Everything downstream
follows from this table.

| | Paper (DCASE 2020) | Here (AnuraSet) |
|---|---|---|
| Task | acoustic **scene** classification | anuran **species** detection |
| Labels | 1 of 10, mutually exclusive | 0–8 of 42, co-occurring |
| Clip length | 10 s (431 frames) | 3.0 s (130 frames) |
| Output | softmax over 10 | 42 independent sigmoids |
| Loss | cross-entropy | `BCEWithLogitsLoss` |
| Metric | accuracy | mAP / macro-F1 over 34 classes |
| Corpus | balanced by design | 26.8:1 neg:pos, 4 orders of magnitude of class skew |

The architecture transfers. The head, loss, metrics, split protocol, and evaluation
discipline do not, and are re-derived here.

### Why softmax cannot be kept

Softmax enforces `Σ p_i = 1` — mutually exclusive classes. Measured label
distribution over 62,191 clips:

```
labels/clip:  0 → 22,504 (36.2%)    4 → 5,032
              1 → 12,886            5 → 1,685
              2 → 10,591            6 →   664
              3 →  8,636            7 →   182 ,  8 → 11
mean 1.51 labels/clip, max 8
```

39.8% of clips carry ≥2 species. Softmax would force the model to *choose* between
frogs calling simultaneously, and — more fundamentally — it can never represent the
36.2% all-negative clips, because it always sums to 1 and cannot say "nothing here."
42 independent sigmoids is the only formulation that admits both.

---

## 1. Signal processing

The paper's contribution is the **feature decoupling** front end. The CNN behind it
is conventional. This section is therefore the load-bearing one.

### 1.1 The pipeline

```
wav (66,150 samples @ 22,050 Hz, 3.000 s, mono)
  │
  ├─ librosa.stft(n_fft=1024, hop_length=512)        → S ∈ ℂ^(513 × 130)
  │
  ├─ librosa.decompose.hpss(S, kernel_size=17, margin=1.0)
  │        → S_h, S_p                        ← ON THE COMPLEX LINEAR STFT
  │
  ├─ for each of S_h, S_p:
  │        power = |S|²                              (513 × 130)
  │        mel   = FB_(128×513) @ power              (128 × 130)
  │        db    = power_to_db(mel, ref=1.0, top_db=80)
  │
  └─ X_h, X_p ∈ ℝ^(128 × 130)   [+ X_raw, the no-HPSS control]
```

Implemented once in `src/features.py`, imported by both `src/precompute.py` and
`src/serve.py`. Never reimplemented — see §7.1.

### 1.2 HPSS mechanism

HPSS (Fitzgerald 2010) exploits a geometric asymmetry in the time-frequency plane:

- **Harmonic** content — a sustained tonal call — forms **horizontal ridges**:
  stable in frequency, extended in time.
- **Percussive** content — a click, knock, or call onset — forms **vertical
  ridges**: broadband, instantaneous.

```
S_h_mag = median_filter(|S|, along TIME,      kernel 17)   # smears time → keeps horizontals
S_p_mag = median_filter(|S|, along FREQUENCY, kernel 17)   # smears freq → keeps verticals

M_h = S_h_mag^p / (S_h_mag^p + S_p_mag^p)                  # soft Wiener-style masks
M_p = S_p_mag^p / (S_h_mag^p + S_p_mag^p)
S_h = S · M_h ,   S_p = S · M_p
```

### 1.3 Three decisions in the feature path

#### (a) HPSS runs on the linear STFT, never on the mel spectrogram

`src/features.py:7-12`. This is the single most consequential ordering choice in the
project.

The mel filterbank is a non-uniform warp: above ~1 kHz, bin width grows
logarithmically. Harmonic partials of a 400 Hz call sit at 400/800/1200/1600 Hz — on
a linear grid, evenly spaced horizontal lines with clean gaps between them. After mel
projection, the 4th/5th/6th partials collapse into the *same* mel bin. The
frequency-axis median filter then sees a smooth blob instead of a comb, and
harmonic/percussive discrimination degrades toward noise.

Running HPSS after the mel projection silently destroys the paper's core mechanism —
no error, no crash, just a decomposition that no longer decomposes.

*Verified:* the soft-mask partition property `S_h + S_p == S` holds to **1.5e-05**,
and harmonic components are measurably smoother along time than percussive ones in
**12/12** clips tested (harmonic freq/time gradient ratio ~1.3 vs percussive ~0.4).

#### (b) `margin=1.0`, preserving the partition

With `margin > 1`, librosa switches from soft masks to **hard masks with a rejection
band**: energy that is not confidently harmonic *or* confidently percussive is
discarded into neither stream. At `margin=1.0` the partition `S_h + S_p = S` holds
exactly.

Since the architecture's premise is that the two streams *jointly* cover the signal
and are then fused, discarding the ambiguous residual would remove precisely the
information the fusion exists to recombine.

#### (c) Fixed dB reference, `ref=1.0`, not `ref=np.max`

`librosa.power_to_db(..., ref=np.max)` is the common tutorial idiom and normalises
each clip so its loudest bin becomes 0 dB. For music that is harmless. For
bioacoustic **detection** it is destructive on two counts:

1. It erases absolute level — the cue distinguishing a loud nearby frog from a faint
   distant one.
2. It **amplifies pure-noise clips to full dynamic range**, so an all-negative clip
   acquires the same apparent contrast as a confident detection. With 36.2% of clips
   all-negative, this is not an edge case.

A fixed reference preserves calibrated, SPL-proportional level across the corpus.

### 1.4 Time-frequency resolution: a documented conflict

`src/config.py:24-27` names a tension usually left implicit:

- `n_fft=1024` at 22,050 Hz → **21.5 Hz/bin**, **46.4 ms** window
- `hop=512` → 23.2 ms frame step → **130 frames** for 3 s

HPSS wants **fine frequency** resolution, to resolve harmonic combs (larger `n_fft`).
The 5×3 frequency-extended convolutions want the same. The 3×5 time-extended
convolutions want **fine time** resolution (smaller `n_fft`). Gabor's uncertainty
principle forbids both.

The chosen compromise additionally yields a near-square **128 × 130** input, so the
paper's 4-block ÷2 pooling schedule transfers unchanged
(128→64→32→16, 130→65→32→16). The DSP hyperparameter and the architectural
hyperparameter were co-designed rather than chosen independently.

### 1.5 Feature hyperparameters and provenance

| Parameter | Value | Source | Assumption |
|---|---|---|---|
| `SR` | 22050 | measured from wavs | corpus verified uniform; `load_audio` guards fail loud otherwise |
| `CLIP_SECONDS` | 3.0 | AnuraSet segmentation | a 3 s window contains ≥1 complete call for all 42 species |
| `N_FFT` / `HOP_LENGTH` | 1024 / 512 | **chosen here** | §1.4 |
| `N_MELS` | 128 | paper | |
| `N_FRAMES` | 130 | derived | pinned by `_fix_frames` so every stored feature aligns |
| `FMIN` / `FMAX` | 50.0 / 11025.0 | **chosen here** | no species energy below 50 Hz; Nyquist uncut, since several *Scinax*/*Boana* calls exceed 5 kHz |
| `DB_REF` / `DB_TOP` | 1.0 / 80.0 | **chosen here** | §1.3c |
| `HPSS_KERNEL` | 17 | paper's sweep | **transferred, not re-validated — see below** |
| `HPSS_MARGIN` | 1.0 | **chosen here** | §1.3b |

`HPSS_KERNEL=17` comes from the paper's sensitivity table (5→69.3, 9→70.8, 13→71.6,
**17→72.1**, 21→71.8, 25→71.1 % accuracy). It is a **transferred** hyperparameter:
tuned on 10 s DCASE scene clips at unstated STFT settings. At our 130 frames, 17
frames = **394 ms** of temporal median smoothing. Plausible for frog calls, but
untested in this domain — the highest-value cheap ablation still outstanding (§9).

### 1.6 Why precompute

Median filtering does not vectorise like convolution — it is a sorting problem per
window, with no BLAS path. *Measured:* **29.8 ms/clip**.

On-the-fly that is 62,191 × 29.8 ms ≈ **31 min of pure CPU per epoch**; over 100
epochs, 51 hours of feature extraction alone, with the CPU rather than the GPU as
the bottleneck. Precompute costs ~16 min once (10 processes).

| Decision | Rationale |
|---|---|
| `mp.Pool(10)` of 14 cores | leaves 4 for OS + writer process |
| workers **return** arrays; parent writes | h5py is **not fork-safe for concurrent writers**; parallel writes corrupt silently |
| `imap_unordered(chunksize=32)` | order irrelevant (index travels with payload); chunking amortises IPC |
| `done` boolean mask | resumable — a crash at clip 50,000 costs minutes, not the run |
| **fp16** storage | 3 × 128 × 130 × 2 B × 62,191 = 5.8 GB (6.2 GB actual). dB-mels span ~[−80, +4]; *measured roundtrip error 0.03 dB* |
| **no gzip** | gains ~15% but costs decompression CPU on *every batch read of every epoch* — would move the bottleneck to the loader |
| cast fp32 at load | MPS is fp32-native; Metal has no fp64 at all |

---

## 2. Architecture

### 2.1 Asymmetric convolution block

```python
Conv2d(c_in,  c_out, (3,5), padding=(1,2), bias=False) → BN → ReLU
Conv2d(c_out, c_out, (5,3), padding=(2,1), bias=False) → BN → ReLU
```

Tensor layout is `(B, C, n_mels, n_frames)`, so **dim 2 = frequency, dim 3 = time**.
Therefore:

- **(3,5)** — 3 along frequency, 5 along time → **time-extended**, sees
  onset-sustain-decay dynamics.
- **(5,3)** — 5 along frequency, 3 along time → **frequency-extended**, sees
  harmonic overtone spacing.

Cascaded, they cover an effective **7×7** receptive field at **30 MACs/position
instead of 49** (−39%), plus an extra nonlinearity between them.

Same factorisation logic as Inception-v3's 7×7 → 1×7 + 7×1, but only **partially**
factorised (3×5, not 1×5). This is deliberate: full separability discards the joint
T-F sensitivity that diagonal frequency sweeps require, and frog calls contain them.

`bias=False` throughout — BatchNorm immediately follows and carries its own
learnable β, so a conv bias is redundant and adds a degenerate direction to the loss
surface.

### 2.2 Stream, with measured shape and parameter walk

*Measured* at `STEM_POOL=2`:

```
layer                       out shape        params
input                    (1, 128, 130)            0
MaxPool2d  (stem /2)       (1, 64, 65)            0
AsymConvBlock  1→64       (64, 64, 65)       62,656
MaxPool2d                 (64, 32, 32)            0
AsymConvBlock 64→128     (128, 32, 32)      369,152
MaxPool2d                (128, 16, 16)            0
AsymConvBlock 128→256    (256, 16, 16)    1,475,584
MaxPool2d                  (256, 8, 8)            0
AsymConvBlock 256→512      (512, 8, 8)    5,900,288
────────────────────────────────────────────────────
stream total                              7,807,680
```

Two properties worth internalising:

- **Parameters are back-loaded.** Block 4 alone holds 5,900,288 of 7,807,680
  (**75.6%**).
- **FLOPs are front-loaded.** Block 1 runs at the highest spatial resolution.

Consequence: *to cut parameters, touch block 4; to cut FLOPs, touch block 1.* This is
exactly what the stem pool does (§2.6).

**Pooling after the first three blocks only.** Block 4 deliberately retains its
spatial map so spatial attention has a non-degenerate plane to act on.
`tests/test_model.py::test_spatial_map_not_degenerate` asserts `min(H,W) ≥ 8`,
protecting the design intent rather than a shape.

### 2.3 Dual attention — resolving an incoherent equation

CBAM (Woo et al. 2018), applied in a specific order:

```python
def _branch(self, x, stream, spatial, channel):
    m = stream(x)              # (B, 512, H, W)
    m = m * spatial(m)         # spatial gate ON THE FEATURE MAP
    f = m.mean(dim=(2, 3))     # GAP → (B, 512)
    return f * channel(m)      # channel gate ON THE POOLED VECTOR
```

- **SpatialAttention** — concat channel-wise mean and max → (B,2,H,W) → 7×7 conv →
  sigmoid → (B,1,H,W). Asks *where in time-frequency is the evidence?* The mean+max
  concat is the CBAM trick: mean captures diffuse energy, max captures peaks, and a
  frog call is a peak event on diffuse background.
- **ChannelAttention** — GAP and GMP each through a **shared** MLP
  (512→32→512, reduction 16), summed, sigmoid. Asks *which learned filters matter for
  this clip?* Weight sharing is what makes this CBAM rather than two independent SE
  blocks, and it halves parameters while forcing both statistics into one
  representation.

**The paper's Eq. 2 is not well-formed as written.** It fuses 512-d post-GAP vectors
and calls one factor "spatial attention" — but a post-GAP vector has no spatial
extent; there is no plane for `w_s` to act on. `src/model.py:7-11` records the
resolution: the only coherent reading is **spatial attention before pooling, channel
attention after**, which is what is implemented.

This is the general policy for paper ambiguity in this project: pick the reading that
is mathematically well-formed, document that a choice was made, and state why.

### 2.4 Activation functions

| Location | Activation | Justification |
|---|---|---|
| Conv blocks | **ReLU** (`inplace=True`) | Paper's choice. Non-saturating → no vanishing gradient through 8 conv layers; sparse activations; dying-ReLU is a non-issue because BN keeps pre-activations centred. `inplace` matters at 64×512×8×8. |
| Attention gates | **Sigmoid** | A multiplicative gate must be bounded and must be able to fully suppress (→0) and fully pass (→1). |
| Fusion weight | **Sigmoid** | Constrains α ∈ (0,1) with a live gradient everywhere — see §2.5. |
| Head hidden | **ReLU** | Consistency. |
| **Output** | **none — raw logits** | Non-negotiable with `BCEWithLogitsLoss`. |

The output case is the classic silent bug. `BCEWithLogitsLoss` fuses the sigmoid with
the log via log-sum-exp:

```
loss = max(z,0) − z·y + log(1 + exp(−|z|))
```

numerically stable for |z| up to ~50. Applying sigmoid in `forward` and *then*
feeding `BCEWithLogitsLoss` yields a **double sigmoid**: outputs compressed into
(0.5, 0.73), gradients throttled ~4×, and a model that **still trains to a
plausible-looking but degraded result**. No error is raised.

`tests/test_model.py::test_no_sigmoid_in_forward` catches this directly rather than
by proxy: it forces the final bias to 8.0 and asserts `out.max() > 1.5`. A
double-sigmoided output cannot exceed 1.0.

### 2.5 Fusion — the paper's scalar, and why it was replaced

#### The paper's formulation (retained as `--scalar-fusion`)

```python
self.a_raw = nn.Parameter(torch.zeros(1))   # sigmoid(0) = 0.5, unbiased init
a = torch.sigmoid(self.a_raw)
f = a * f_h + (1.0 - a) * f_p               # fused BEFORE the head
logits = self.head(f)
```

The sigmoid reparameterisation is chosen over the naive alternative for a specific
reason recorded at `src/model.py`: **`clamp` yields exactly zero gradient at the
boundary**, so a clamped parameter that reaches 0 or 1 is permanently dead. Sigmoid
saturates but never fully dies.

#### Measured failure of the scalar form

Seed-42 ladder, `STEM_POOL=1`, flat LR, Adam, scalar fusion:

```
raw          0.6938   7.98M    ← wins, no HPSS at all
percussive   0.6886   7.98M
dual         0.6541  15.82M    ← the paper's method, 3rd of 4, 2× the parameters
harmonic     0.6433   7.98M
```

**The dual arm scored below percussive-only.** A fusion that cannot match the better
of its own two inputs is destroying information, not combining it.

The learned scalar decayed monotonically 0.470 → 0.087 over 31 epochs without
plateauing — 91.3% weight on the percussive stream, approximately the corpus-average
optimum. Per-class APs show why one number cannot suffice:

| species | support | harmonic | percussive | prefers | dual got | Δ |
|---|---:|---:|---:|---|---:|---:|
| BOALUN | 2,060 | 0.4503 | **0.9414** | percussive | 0.5387 | **−0.4026** |
| BOAPRA | 480 | 0.4985 | **0.7451** | percussive | 0.1520 | **−0.5931** |
| SCIPER | 3,791 | **0.8609** | 0.7368 | harmonic | 0.7324 | −0.1285 |
| LEPNOT | 1,062 | **0.6318** | 0.4432 | harmonic | 0.3671 | −0.2647 |

Arm wins across the 34 eval classes: percussive 12, raw 10, dual 7, harmonic 5.
**Harmonic-only genuinely wins 5 classes**, including SCIPER at 3,791 positives — so
the decomposition carries real signal. The scalar fusion simply cannot express *"use
percussive for BOALUN, harmonic for SCIPER."*

A second concern: α is **degenerate up to a rescaling**, because `head[0]` can absorb
any scale factor applied to the fused vector. Monotonic drift toward a boundary is
the expected symptom of exactly that degeneracy.

#### Per-class fusion (`PER_CLASS_FUSION = True`, default)

```python
n_a = n_classes if self.per_class_fusion else 1
self.a_raw = nn.Parameter(torch.zeros(n_a))         # (42,)
...
a = torch.sigmoid(self.a_raw)
return a * self.head(f_h) + (1.0 - a) * self.head(f_p)   # broadcasts over logits
```

Two structural consequences:

1. **Fusion moves to the logit level.** A per-class weight *cannot* act on the
   pre-head 512-d vector — the class dimension does not exist there yet. This is
   forced, not stylistic.
2. **The head is shared, not duplicated.** Two heads would add ~142k parameters and
   let each stream drift to a different logit scale, making α uninterpretable as a
   mixing ratio.

Cost: **+41 parameters** (15,824,303 → 15,824,344), exactly `N_CLASSES − 1`.
*Measured.*

`tests/test_model.py::test_per_class_fusion_has_one_weight_per_class` asserts shape
`(42,)`, the exact +41 delta, and — critically — that **each class receives its own
gradient** (`g.abs().std() > 0`). Without that last assertion, weights accidentally
broadcast from a scalar would pass silently and the fix would be cosmetic.

### 2.6 Stem pool — reverse-engineered from the paper's own FLOPs

`STEM_POOL = 2` inserts one `MaxPool2d(2)` before block 1.

*Measured* (params shown with the default per-class fusion):

| | GFLOPs | block-4 map | params |
|---|---:|---|---:|
| `STEM_POOL=1` (literal reading) | 11.216 | 16×16 | 15,824,344 |
| **`STEM_POOL=2` (default)** | **2.792** | **8×8** | 15,824,344 |

**4.02× FLOP reduction at zero parameter cost** — pooling has no weights, and GAP
absorbs the smaller map. Epoch time *measured* 819 s → ~228–246 s.

The justification is not merely speed; it is **fidelity**. See §3.2: the paper's
reported 2.86 GFLOPs is arithmetically unreachable without an undocumented early
downsample, and 2.792 G is essentially that figure.

`STEM_POOL=4` is deliberately not offered: it would leave a 4×4 map, smaller than
the 7×7 spatial-attention convolution meant to gate it. The 8×8 floor is what
`test_spatial_map_not_degenerate` enforces.

### 2.7 Parameter distribution (measured, dual + per-class fusion)

```
stream_h        7,807,680   49.34%
stream_p        7,807,680   49.34%
head              142,122    0.90%
channel_h          33,312    0.21%
channel_p          33,312    0.21%
spatial_h              98    0.00%
spatial_p              98    0.00%
a_raw                  42    0.00%
──────────────────────────────────
TOTAL          15,824,344
```

98.7% of capacity is in the two streams; the entire attention and fusion apparatus is
0.4%. Worth knowing when reasoning about what the model can and cannot learn — the
fusion mechanism is a 42-parameter decision on top of a 15.8M-parameter encoder.

---

## 3. Two internal inconsistencies in the paper

Both are reported **as measured**, with no architectural tuning to reconcile them.

### 3.1 Parameter count

Paper reports **12.4M**. Its stated channel progression is `64→64→128→256→512` —
**five numbers for four blocks**. Every reading was enumerated:

| Reading | Params |
|---|---:|
| literal (implemented): 1→64, 64→128, 128→256, 256→512 | **15.82M** |
| ending at 256 channels | 4.16M |
| five blocks | 16.07M |
| `c_in→c_out` on both convs within a block | 10.54M |

None yields 12.4M; the nearest brackets are 10.54M and 15.82M. The literal reading is
implemented and 15.82M reported.

`test_param_count_in_expected_band` asserts `12e6 < n < 17e6` — a **band, not an
equality**. It catches accidental architecture drift while explicitly declining to
assert a number the source does not support.

### 3.2 FLOPs — arithmetically decisive

Paper reports **2.86 GFLOPs at 10 s** (431 frames). With pooling only after each
block as the text describes, **block 1 alone at 128×431 costs ~6.9 G — already 2.4×
the reported total for the entire network.**

Searching stem strides at 431 frames:

| pre-block-1 downsample | input to block 1 | dual GFLOPs |
|---|---|---:|
| none | 128×431 | 37.13 |
| /2 | 64×216 | 9.37 |
| **/4** | **32×108** | **2.32** ← nearest to 2.86 |
| /8 | 16×54 | 0.56 |

**Conclusion: the paper must downsample by roughly /4 before block 1**, which its
text does not describe. This is not cosmetic — a /4 stem changes the receptive-field
geometry of every subsequent block, and the paper's headline efficiency comparison
(12.4M vs AST's 86M; 24.1 ms vs 68.3 ms) rests on it.

This finding is what motivated `STEM_POOL` (§2.6): running with the stem pool is
arguably *more* faithful to the paper's real architecture than the literal reading of
its prose.

---

## 4. Training

### 4.1 Hyperparameters and provenance

| Parameter | Value | Source | Note |
|---|---|---|---|
| `BATCH_SIZE` | **64** | paper: 32 | 32 underutilises MPS; throughput flat 32→64. BN statistics also improve at 64 |
| `LR` | 1e-3 | paper | |
| `WEIGHT_DECAY` | 1e-4 | paper | applied to weights only — §4.3 |
| Optimizer | **AdamW** | paper: Adam | §4.3 |
| `LR_SCHEDULE` | **cosine** | paper: none | §4.4 |
| `WARMUP_EPOCHS` | 2 | **chosen here** | §4.4 |
| `LR_MIN_FACTOR` | 0.01 | **chosen here** | cosine floor = 1e-5 |
| `MAX_EPOCHS` | 100 | paper | early stop dominates; sweeps use 40 |
| `EARLY_STOP_PATIENCE` | 10 → raised to `epochs//2` | **chosen here** | §4.4 |
| `dropout` | 0.5 | paper | head only; BN already regularises the conv stack |
| `SPEC_TIME_MASK` | **12** | paper: 40 | §4.2 |
| `SPEC_FREQ_MASK` | 8 | paper | 8/128 mels = 6.3% |
| `SPEC_N_MASKS` | 2 | paper | |
| `MIXUP_ALPHA` | 0.2 | paper | Beta(0.2,0.2) is U-shaped → mostly near-clean, occasional strong mixes |
| `MIXUP_P` | **0.7** | **chosen here** | §4.5 — swept {0.3,0.5,0.7}, then confirmed at 3 seeds. **Adopted 2026-09-05** |
| `GRAD_CLIP` | **0.0** | **chosen here** | §4.6 |
| `ALPHA_TAU` | 1.0 | **chosen here** | `0.3` swept and rejected (2/3 seeds, p=0.750) |
| `ALPHA_LR_MULT` | 1.0 | **chosen here** | `10` rejected — measurably hurt |
| `HPSS_KERNEL` | 17 | paper | swept {5,9,13,25}; **null result**, see `TUNING.md` #15 |
| `SEED` | 42 | | §6 |

### 4.1a The adopted configuration, and what actually backs it

Consolidated view of the config the reportable ablation runs (verified against
`runs/dual_s42_3way/summary.json`). Full derivations in `TUNING.md`.

```
mixup_p          0.7      ADOPTED — +0.0174 at 3 seeds, 3/3 wins, sd 0.0027 vs 0.0172
mixup_alpha      0.2
stem_pool        2        4.02x fewer FLOPs; stem_pool=1's +0.018 was a 1-epoch artefact
per_class_fusion True     dual only; removes the scalar-fusion pathology (worth -0.0035)
alpha_tau        1.0      tau0.3 rejected: 2/3 seeds, p=0.750, lost s43 by -0.0177
alpha_lr_mult    1.0      alr10 rejected: hurt (0.7012 vs 0.7151)
lr_schedule      cosine + 2-epoch warmup, floor LR*0.01
optimiser        AdamW, lr 1e-3, no-decay groups for BN gamma/beta and a_raw
loss             BCEWithLogits    ASL rejected (-0.0098)
grad_clip        0.0              measured norm 0.132 -- never fired, cost ~16%/step
batch_size       64
epochs           40, patience 20
hpss_kernel      17, margin 1.0   swept 116-580 ms, null
spec_time_mask   12               paper's 40 rescaled 431 -> 130 frames
spec_freq_mask   8, n_masks 2
split            three-way grouped-by-recording, dev-selected, val read ONCE
params           15,824,344 dual / 7,983,212 single / 15,777,394 raw-wide
```

**The architecture is the paper's, untouched** — layer types, block ordering, channel
widths and the asymmetric 1×n / n×1 pairs are all as published. Every deviation is
optimizer, augmentation, or input-side. That boundary is deliberate: it keeps this a
replication of the paper's ablation rather than an evaluation of a different model.

**What is evidence-backed, stated plainly.** Of the four changes once credited with
the +0.109 over the legacy config:

| Change | Isolated effect | Status |
|---|---|---|
| Per-class fusion | −0.0035 | correctness fix; no measurable gain |
| Cosine LR | −0.0018 (`attr_flatlr`) | stability; no measurable gain |
| Stem pool /2 | single-epoch artefact | 4.02× speedup; no measurable gain |
| AdamW | **never ablated** | untested — no arm was ever run |

So **the +0.109 is not decomposable into these four factors.** It also straddles a
split change (two-way → three-way) and a metric change (val → dev), so part of it is
a measurement artefact rather than an improvement. `MIXUP_P=0.7` is the only
confirmed win in the project. Anything else described as an "improvement" is
overstating the evidence.

### 4.2 SpecAugment rescaling — transfer the ratio, not the integer

The paper uses `time_mask=40` on **431-frame** clips = 9.3% per mask, 18.6% with two.
Transferring 40 verbatim to **130-frame** clips gives 30.8% per mask, **61.5% with
two**.

For scene classification that may survive — an "airport" is stationary ambience, so
erasing 60% leaves 40% of the same texture. For **event detection** it is fatal: a
frog call is a localised 200–500 ms event, and a 40-frame (928 ms) mask can erase it
entirely *while the label still says present*. That trains the model to hallucinate
the species from background alone.

`SPEC_TIME_MASK=12` (278 ms, 18.5% worst case with two masks) preserves the
**fraction** the paper actually applied.
`test_specaugment_preserves_shape_and_masks` encodes the invariant
`SPEC_TIME_MASK * SPEC_N_MASKS < 0.3 * N_FRAMES` so the reasoning survives edits.

**Masking value.** Masks are filled with `0.0`, and because masking runs *after*
normalisation this is the train-split **mean** (−27.6 dB), not silence (which would be
≈ −3.87 in normalised units). This is the original SpecAugment recommendation and is
the better choice here: a mean-filled patch reads as *uninformative*, whereas a
silence-filled patch asserts *confidently empty* — a stronger and more misleading
claim for a detection task. Documented at `src/dataset.py`.

### 4.3 Weight decay — two corrections

**(a) Exclude 1-D parameters.** `Adam(model.parameters(), weight_decay=1e-4)` decays
*everything*, including BatchNorm γ/β. Decaying BN γ toward zero directly fights the
network's ability to scale features — BN's entire purpose.

Now split by dimensionality. *Measured:*

```
decay:    24 tensors, 15,815,236 params
no_decay: 39 tensors,      9,108 params
          └ 32 BatchNorm γ/β  +  6 biases  +  1 a_raw
```

`a_raw` exclusion matters beyond hygiene: decaying the fusion weight would bias it
toward `sigmoid(0) = 0.5` and **confound the very quantity the ablation measures**.

**(b) AdamW, not Adam.** Adam's `weight_decay` adds L2 to the *gradient*, which is
then divided by the per-parameter adaptive denominator — so effective decay ends up
inversely proportional to gradient magnitude. AdamW decouples the two and applies
decay directly to the weight.

**On Adam vs SGD** (paper's choice, kept): Adam's per-parameter adaptive steps are
genuinely valuable for long-tailed multi-label training, where gradient magnitude
reaching the 42 outputs differs by ~3 orders of magnitude between SPHSUR (13,258
positives) and LEPFLA (7). A single global SGD learning rate would either under-train
the tail or destabilise the head.

### 4.4 Learning-rate schedule

The paper specifies none. *Measured* consequence, seed-42 dual run, flat 1e-3:

```
ep 15  0.6281      ep 24  0.6541 ← "best", saved
ep 16  0.6407      ep 25  0.6480
ep 17  0.6059  ↓   ep 26  0.6050  ↓
ep 19  0.6333      ep 28  0.6397
ep 20  0.5855  ↓↓  ep 30  0.6527
ep 22  0.6142      ep 31  0.6471
```

A **±0.035 band with no descent**, while train loss sat flat at 0.053–0.055. That is
the signature of a step size too large for the current basin: the optimiser orbits
the minimum instead of entering it.

Two costs, the second worse than the first:

1. Accuracy left unclaimed (typically +0.01–0.03 mAP once it can settle).
2. **Corrupted model selection.** `best.pt` is chosen by `val_mAP > best`, so under
   oscillation it saves *whichever epoch got lucky*. Every arm's headline is inflated
   by an unknown, arm-specific amount — which is precisely why a 0.040 mAP spread
   across the top three arms was unreadable against ±0.035 of noise.

#### Implementation

`build_scheduler()` returns `(scheduler, granularity)` with three modes:

- **`cosine`** (default) — linear warmup then cosine to `LR × 0.01`, as a single
  `LambdaLR`. Warmup is folded into the same object so there is no hand-off bug
  between two schedulers.
- **`plateau`** — `ReduceLROnPlateau(mode="max")` on val mAP. `mode="max"` is
  essential: the default `"min"` would cut the LR every time the model *improved*.
- **`none`** — reproduces the paper's flat LR exactly.

**Cosine steps per batch, not per epoch.** Per-epoch would give only 40 discrete
values (a staircase); per batch gives 33,320 and a genuinely smooth decay. The
difference matters most late in the run, where a per-epoch schedule would hold a
still-too-large LR for an entire 833-step epoch.

`scheduler.step()` is called **after** `optimiser.step()`, per PyTorch's documented
order.

*Measured curve* (40 epochs × 833 batches):

```
epoch      lr
    1  6.00e-07   (warmup start)
    3  1.00e-03   ← peak, at end of 2-epoch warmup
   11  8.96e-04
   21  5.46e-04
   31  1.70e-04
   40  1.00e-05   (floor = LR × 0.01)
```

**Why warmup.** BatchNorm running statistics are meaningless at initialisation, so
the first few hundred steps produce large, poorly-scaled gradients. A 2-epoch linear
ramp (1,666 steps) avoids spending the largest steps of the run on the worst gradient
estimates.

**Interaction with early stopping — a genuine conflict.** Cosine anneals to its floor
at exactly `epochs`; stopping early discards the low-LR phase where convergence
actually happens. The schedule and the stopper want opposite things. Resolution:
patience is raised to `max(patience, epochs//2)` when cosine is active, and the
stopper is retained only as a runaway guard. Logged when it fires.

`test_cosine_schedule_warms_up_then_anneals_to_floor` walks the entire curve and
asserts shape at every landmark — warmup linearity, exact peak at `LR`, monotonic
anneal, and arrival at the floor. A schedule that silently does nothing is the
failure mode: training would look normal and the oscillation would persist
unexplained.

### 4.5 MixUp gate

```python
if use_mixup and mixup_rng.random() < mixup_p:
    batch = mixup(batch, rng=mixup_rng)
```

Previously MixUp ran on **100% of batches**, on top of SpecAugment on 100% of
samples — every sample the model ever saw was corrupted twice. That is consistent
with the measured plateau: train loss flat at 0.053 with **no train/val divergence
across 31 epochs**, i.e. the model was *augmentation-limited*, not
capacity-limited, despite 15.8M parameters on 53k samples.

`MIXUP_P = 0.7` (adopted 2026-09-05; was 0.5 during the sweep). `frac_mixed` is
logged per epoch to confirm the gate fires at the configured rate (*measured:* 0.50
at `p=0.5`, 0.70 at `p=0.7`). Confirmation: 3-seed mean 0.7241 vs 0.7067 at `p=0.5`,
+0.0174, winning 3/3 seeds, with cross-seed sd falling from 0.0172 to 0.0027.
Notably `p=0.3` did **nothing** (+0.0001), so the effect is not monotone in
augmentation strength and 0.7 may sit near a local optimum that was never bracketed
from above (0.9 untested).

MixUp uses **one λ per batch** with a permutation, matching the original paper. BCE
accepts soft targets, so mixing labels with the same λ requires no other change.

### 4.6 Gradient clipping — added, measured, disabled

Added at `1.0`, then the first epoch reported `grad_norm = 0.132`. The threshold could
never fire. An isolated benchmark:

```
clip=off   241.4 ms/step  → 3.4 min/epoch
clip=1.0   279.5 ms/step  → 3.9 min/epoch      (+16%)
```

Paying ~16% per step for a no-op across a 13-hour sweep. Set `GRAD_CLIP = 0.0`.

The flag is **kept, not removed**, and `grad_norm` is still logged — sampled every
50th step when clipping is off (<1% cost) — so this remains a *monitored* decision
rather than an unmonitored one. If `grad_norm` climbs toward 1, `--grad-clip 0.5`
re-enables it.

*Measured in the live sweep:* grad_norm 0.132 → 0.070 over epochs 1–2, i.e. moving
away from any useful threshold.

### 4.9 Fusion-weight optimisation: `ALPHA_TAU` and `ALPHA_LR_MULT`

Both default to the identity (1.0), so the pre-registered sweep (§8B) **measures**
their effect rather than assuming it.

**The problem, measured.** Across the 3-seed ladder, 19–20 of 34 per-class alphas
finish training within 0.05 of their 0.5 initialisation (seed 42: 19/34, seed 43:
11/34, seed 44: 20/34), with a cross-class standard deviation of only ~0.083.

**Why this is an optimisation fact, not a performance observation.** Cross-seed
correlation of the per-class alphas is **r = 0.939** — the model reliably learns
*which* stream each species prefers. What it cannot do is travel far enough:
`a_raw` receives gradient scaled by `sigmoid'(0) = 0.25` at the neutral
initialisation, and is a single scalar per class competing against 15.8M other
parameters under one shared learning rate.

Two independent remedies, neither of which tells the model which stream to prefer:

```python
ALPHA_TAU      # a = sigmoid(a_raw / tau);  tau < 1 sharpens the map, so the same
               # raw displacement yields a larger change in alpha
ALPHA_LR_MULT  # put a_raw in its own AdamW group at lr * mult
```

`tau` does not shift the neutral point — `sigmoid(0/tau) = 0.5` for any `tau` — so
the unbiased initialisation is preserved. Asserted by
`test_alpha_temperature_sharpens_and_stays_consistent`, which also checks that
`forward()`, `fusion_weight` and `fusion_weights()` all route through the single
`_alpha()` definition. Without that, the logged alpha could silently be a different
quantity from the one used to fuse, and every alpha interpretation in this document
would be wrong.

### 4.7 Loop-level decisions

| Decision | Rationale |
|---|---|
| Select on **val mAP**, not val loss | mAP is threshold-free and the actual quantity of interest; BCE loss is dominated by the 96.4% negative cells. The two diverge — in the seed-42 run, epoch 28 had the lowest train loss but not the best mAP |
| `drop_last=True` on train, absent on val | dropping the ragged batch stabilises BN running-statistic updates (a batch of 12 gives a noisy μ/σ that pollutes the running average); on val every sample must be scored or the denominator differs from what is reported |
| Threshold tuning **after** reloading `best.pt` | tuning per-epoch would cost 42 × 99 × 8,851 F1 evaluations per epoch and would corrupt model selection by letting threshold search compensate for worse ranking |
| `history.json` written every epoch | a 7-hour run that dies at hour 6 still yields the full curve |
| `zero_grad(set_to_none=True)` | frees gradient buffers rather than zero-filling; turns a forgotten `zero_grad` into an explicit `None` error instead of silent accumulation |
| `norm.json` persisted with checkpoint | serving reads it rather than recomputing — §7.1 |

### 4.8 Loaders

```
DataLoader (num_workers=0)   ~2,500 samples/s
Model fwd+bwd (MPS)              ~66 samples/s   (STEM_POOL=1)
                                 ─────────────
                                 38× headroom
```

`num_workers=0` is **measured-best**, not a compromise. Features are precomputed, so
a sample is a ~66 KB contiguous read from a page-cached HDF5 file plus a fp16→fp32
cast. Spawning workers on macOS (no fork) would add process startup, per-worker HDF5
reopen, and tensor IPC to buy throughput that is already 38× surplus.

`features.h5` is 6.2 GB on a 24 GB machine, so after the first epoch the entire
feature set is resident in the page cache — which is why steady-state epoch times are
stable to ±3 s.

---

## 5. Class imbalance

### 5.1 Measured distribution

```
clips                        62,191
species                          42
positive cells               93,875 of 2,612,022  =  3.59%
global negative:positive                            26.8 : 1

per-class prevalence spans FOUR orders of magnitude:
  SPHSUR  13,258  (21.318%)   ← head
  BOABIS  10,888  (17.507%)
  ...
  DENELE     149  ( 0.240%)   ← EVAL_CLASSES floor
  SCIRIZ      73  ( 0.117%)  ┐
  AMEPIC      68  ( 0.109%)  │
  LEPELE      34  ( 0.055%)  ├ ULTRA_RARE
  RHIORN      21  ( 0.034%)  │
  RHISCI      11  ( 0.018%)  │
  LEPFLA       7  ( 0.011%)  ┘  ← 1,894:1 per-class
  SCINAS       0             ┐ ZERO_POSITIVE — AP undefined
  SCIFUS       0             ┘
```

LEPFLA has **7 positives in 62,191 clips**. There is no learning theory under which
6 training examples of a 128×130 input support a 15.8M-parameter model.

### 5.2 Mechanism 1 — metrics that make the wrong answer impossible

`src/metrics.py` states *"Accuracy is deliberately absent"* and does not implement it.
For the trivial always-negative predictor:

| metric | score |
|---|---|
| element-wise accuracy | **96.4%** |
| subset / exact-match accuracy | **36.2%** |
| mAP over 34 classes | **~0.036** ← correctly near-zero |

Making the misleading metric *impossible to compute* is a stronger guarantee than a
comment discouraging it.

### 5.3 Mechanism 2 — three-tier class stratification

| Tier | Classes | Treatment |
|---|---|---|
| `ZERO_POSITIVE` | SCIFUS, SCINAS (0 pos) | **Excluded.** AP on an all-zero column is undefined. Scoring them 0.0 would drag macro-mAP down by 2/42 = 4.8% — a **pure artifact** lowering every arm by the same constant and obscuring real differences |
| `ULTRA_RARE` | LEPFLA(7), RHISCI(11), RHIORN(21), LEPELE(34), AMEPIC(68), SCIRIZ(73) | Reported separately as `tail_mAP`. AP on 1–10 val positives has a confidence interval wider than [0,1] |
| `EVAL_CLASSES` | remaining **34** (≥100 positives) | Headline mAP / macro-F1 / micro-F1 |

Two properties that make this legitimate rather than cherry-picking:

1. **Pre-registered in config**, not chosen after seeing results, with a round
   principled criterion (≥100 positives) rather than a threshold reverse-engineered
   to look good.
2. **All 42 logits are retained in the head.** Indices stay aligned with `train.csv`
   column order, eliminating the most common source of silently-shuffled per-class
   metrics. Exclusion happens at *metric* time via `EVAL_IDX`, not at *architecture*
   time.

### 5.4 Mechanism 3 — per-class threshold tuning

A global 0.5 threshold is badly wrong under imbalance: a class at 0.24% prevalence
has its optimal F1 threshold far below 0.5, because a correctly-calibrated model
rarely exceeds 0.5 for it.

Grid search over `linspace(0.01, 0.99, 99)` maximising per-class F1, **on val only**.

***Measured impact: macro-F1 0.4568 → 0.6426 — +41% relative, from a post-hoc
calibration step with zero additional training.*** The largest single improvement in
the project, and it costs seconds. Under imbalance, the operating point is part of
the deliverable, not an afterthought.

Classes with <5 val positives fall back to 0.5 and are **returned in a `fell_back`
list** that is logged and persisted. *Measured:*
`[SCIFUS, AMEPIC, LEPELE, RHISCI, SCINAS, RHIORN, LEPFLA, SCIRIZ]`. A threshold
fitted on 1–4 examples is noise dressed as calibration; refusing to fit it *and
recording the refusal* is the correct behaviour.

### 5.5 Mechanism 4 — rarity-biased split stratification

See §6.2. The split itself is an imbalance mechanism.

### 5.6 Implemented but deliberately not enabled

`src/losses.py` provides two further mechanisms, both off by default
(`loss_name="bce"`):

**`clamped_pos_weight`** — inverse-frequency weights clamped at 20.0. Unclamped,
LEPFLA would receive `(62191−7)/7 ≈ 8,883×`; a single LEPFLA positive in a batch
would contribute ~8,883× a normal sample's gradient, spiking the norm and taking one
enormous Adam step in a direction set by one example.

**`AsymmetricLoss`** (Ridnik et al. 2021) — purpose-built for this regime:
- *asymmetric focusing* (`γ_neg=4.0, γ_pos=0.0`): easy negatives get weight
  `p⁴ → 0` and vanish from the gradient; positives keep full weight.
- *probability shifting* (`clip=0.05`): negatives below 0.05 contribute **exactly
  zero** — a hard floor, which matters in bioacoustics where faint distant calls are
  easily missed by annotators.

Its advantage over `pos_weight` is needing **no per-class constants**, sidestepping
the 8,883× problem rather than clamping it.

**Shipping BCE first is deliberate**: you cannot attribute a gain to ASL without a
BCE baseline to attribute it against, and the current ablation is about *HPSS*. Adding
a loss dimension simultaneously would confound both.

### 5.7 Not done, with reasons

- **No oversampling / `WeightedRandomSampler`.** In multi-label, oversampling is
  ill-defined: a clip with LEPFLA *and* SPHSUR, oversampled for LEPFLA, also
  oversamples SPHSUR. No per-sample weight fixes 42 marginals simultaneously.
- **No focal loss.** ASL supersedes it (focal is the symmetric `γ_pos = γ_neg` case).

---

## 6. Data leakage and reproducibility

### 6.1 The threat

AnuraSet clips are 3 s segments cut from continuous field recordings:

```
INCT20955_20190909_050000_0_3.wav
└──site──┘ └─date─┘ └time┘ └seg┘
```

*Measured:* 62,191 clips / 1,074 recordings ≈ **58 segments per recording**. Those 58
segments share the same individual frog, the same background insects and stream, the
same microphone and gain, and are seconds apart. **They are near-duplicates, not
independent samples.**

A random 85/15 row split places ~49 of every 58 segments in train and 9 in val. The
model then need not learn vocalisations at all — it can memorise *background texture*
and retrieve the label by recognising the recording. That yields macro-F1 above 0.85.

`src/splits.py` states the failure mode with the right emphasis: *"It fails silently
AND flatteringly, which is the worst combination."* A crash costs an hour; a leaked
split produces a beautiful number, a confident write-up, and a model that collapses
on genuinely new recordings — with nothing in the logs hinting at it.

### 6.2 Protocol

**Step 1 — group key.** `"_".join(filename.split("_")[:3])`. The docstring records a
robustness finding: **58 filenames carry an extra 6th field**
(`..._041500_000_1_4.wav`). A strict 5-field regex would throw or mis-parse; taking
the first 3 fields is invariant to trailing structure. → **1,074 groups**.

**Step 2 — rarity-biased stratum per group.** `StratifiedGroupKFold` needs one label
per group, but groups are multi-label. Each group is reduced to **the rarest species
present anywhere in it**, or `"NEG"` if it has none:

```python
rarity = {species: rank for rank, species in enumerate(support.sort_values().index)}
strata.append(min(present, key=lambda s: rarity[s]) if present else "NEG")
```

A deliberate bias: it makes the stratifier work hardest for the classes least able to
absorb maldistribution. A group with both SPHSUR (13,258) and DENELE (149) is
stratified as DENELE, because SPHSUR will be well-represented on both sides by
volume regardless. `"NEG"` groups balance the 36.2% all-negative population too.

**Step 3 —** `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`, first
fold as val (~14.3%).

**Step 4 — three hard assertions**, executed *inside* `make_split` before it returns:

```python
assert not (tr & va)                    # no filename on both sides
assert len(tr) + len(va) == len(df)     # exact partition
assert not (tr_groups & va_groups)      # no recording spans the split
```

You cannot obtain an unverified split from this module. A separate verification
script would be skippable.

*Measured:* the two-way split gave **53,340 train / 8,851 val** across **921 / 153**
recordings, 40 / 34 species with ≥1 positive. It has since been replaced by a
three-way **44,408 / 8,932 / 8,851** train/dev/val split — see §8A. The val fold is
identical in both.

### 6.3 The other four channels

| Channel | Mechanism | Location |
|---|---|---|
| **Normalisation statistics** | mean/std from **train filenames only**, applied to val. Computing over train+val leaks the val distribution's first two moments into every training sample. *Measured:* mean −27.561, std 13.552 | `dataset.py`, `engine.py` |
| **Threshold fitting** | **dev only** (§8A). Per-class F1-argmax is a 42-parameter fit — small but nonzero capacity, so it must not touch the reporting set. Originally fitted on val, which inflated the reported macro-F1; measured optimism 0.5569 dev-tuned vs 0.5133 on held-out val | `metrics.py`, `engine.py` |
| **Model selection** | early stopping and `best.pt` on **val mAP**. `test.7z` (~31,260 clips) is **never opened by any code path in `src/`** — verified | `engine.py` |
| **Train/serve skew** | `serve.py` **imports** `features_from_path` from the same module `precompute` uses, and reads `norm.json` from the run directory. Not "the same algorithm" — the same function object and constants | `serve.py`, §7.1 |

**Norm-stats sampling caveat.** `compute_norm_stats` subsamples via
`rows[np.linspace(0, len(rows)-1, 4000)]` — **systematic** over sorted (chronological)
HDF5 rows, not random. For a single global scalar over 66.6M values this is fine and
arguably better stratified across the recording timeline than a random draw. It would
**not** be safe for a per-mel-bin statistic, where periodicity in row order could
alias with the sampling stride. Documented at `src/dataset.py`.

### 6.4 Site generalisation — stating the limit

*Measured:* only **5 of 42** species (`BOAFAB, DENMIN, LEPLAT, PHYCUV, PITAZU`) appear
at more than one site. **35 of 42 are single-site**, and none appear at all four.

Implication: **val mAP measures generalisation to new recordings within known sites,
not to new sites.** That is a meaningfully weaker claim than "generalises," it is the
strongest claim the data supports, and it is stated rather than blurred.

A leave-one-site-out protocol would be structurally unable to predict 35 species it
never observed, so it is not run — and the reason is documented rather than the
experiment being deferred as future work.

### 6.5 Reproducibility

Seeding covers every RNG the training path touches:

| Stream | Mechanism |
|---|---|
| Weight init | `torch.manual_seed(seed)` |
| **Batch order** | dedicated `torch.Generator` passed to `DataLoader(generator=...)` |
| **MixUp λ + permutation** | `np.random.default_rng(seed)`, its own stream |
| SpecAugment | global torch RNG, seeded |
| Worker processes | `seed_worker()` re-seeds numpy/random per worker |

**Why the DataLoader needs its own generator.** Without one, shuffle order draws from
the global torch RNG — which `spec_augment` also consumes inside `__getitem__`. Batch
order would then depend on how many augmentation draws had occurred, so changing
`SPEC_N_MASKS` would silently change the batch order too, coupling two things that
must stay independent.

**Why `seed_worker` is needed.** PyTorch seeds each worker's *torch* RNG
automatically but not numpy's or Python's `random`. Since macOS spawns workers, each
would otherwise start with a fresh entropy-seeded numpy RNG.

**What is not claimed: bitwise reproducibility.** MPS parallel reduction kernels do
not guarantee a fixed summation order, and float addition is not associative, so
identically-seeded runs diverge — slowly at first, then materially once divergence
reaches `argmax`-based threshold selection and best-epoch choice. There is no MPS
equivalent of `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`.

What seeding *does* buy: identical initial weights, identical batch order, identical
augmentation draws. That removes every run-to-run variance source **except** MPS
reduction order, which is what makes seed-to-seed differences measure *seed
sensitivity* rather than unseeded RNG noise — the precondition for reporting
mean±std.

`test_same_seed_gives_identical_init_and_batch_order` asserts same-seed → identical
weights *and* batch order, and different-seed → different both.

---

## 7. Engineering practices

### 7.1 One feature path, imported not reimplemented

`src/config.py` holds every feature and training constant; `precompute`, `train`, and
`serve` all import from it. `serve.py` imports `features_from_path` from the same
`src.features` that `precompute` uses, and reads `norm.json` from the run directory
rather than recomputing.

This is the difference between *intending* train/serve consistency and *guaranteeing*
it. The most common production ML failure is a serving preprocessor reimplemented "to
match" training that does not.

### 7.2 Comments that record rejected alternatives, with measurements

The codebase consistently answers *"why not X"*, not only *"what"*:

| Rejected | Reason recorded |
|---|---|
| `channels_last` | tested; **fails on MPS backward** (`view size is not compatible...`) |
| `ref=np.max` | normalises per clip, destroys absolute level |
| gzip on HDF5 | gains ~15%, costs CPU on every batch read of every epoch |
| `clamp` on fusion weight | zero gradient at boundary; the parameter dies |
| `num_workers > 0` | loader already 38× faster than the model |
| `pin_memory=True` | CUDA async-DMA concept; unified memory has no such copy |
| `GRAD_CLIP=1.0` | measured norm 0.132 — never fires, costs 16%/step |
| Ray (Data/Train/Tune/Serve) | each rejected with the specific value proposition that one machine removes; Ray Train is dispositive — `torch.distributed` **has no MPS backend** |

This is the highest-value documentation form, because the failure it prevents is a
future maintainer "improving" the code by reintroducing a tested-and-rejected option.

### 7.3 Test suite — 18 tests, protecting decisions not shapes

| Test | What it protects |
|---|---|
| **`test_overfit_tiny_batch`** | **The primary correctness gate.** 32 samples, no aug, no dropout, 250 steps. *Measured 0.6926 → 0.000200.* Jointly verifies end-to-end differentiability, gradients reaching every parameter, loss wired to the right targets, label/logit alignment, and optimizer stepping. If it fails, no hyperparameter tuning will save the real run — and finding out in 30 s rather than 7 h is an 850× feedback improvement |
| `test_no_sigmoid_in_forward` | catches the double-sigmoid bug that produces a *plausible working model*, not a crash (§2.4) |
| `test_spatial_map_not_degenerate` | the "don't pool after block 4" decision; the 8×8 floor that caps `STEM_POOL` at 2 |
| `test_per_class_fusion_has_one_weight_per_class` | shape `(42,)`, exact +41 param delta, **and per-class gradient independence** |
| `test_cosine_schedule_warms_up_then_anneals_to_floor` | walks the whole LR curve — a silently-flat schedule is the failure mode |
| `test_same_seed_gives_identical_init_and_batch_order` | seeding actually fixes weights *and* batch order |
| `test_stem_pool_reduces_map_and_preserves_shape` | map halves per factor; logits unchanged at both settings |
| `test_mixup_rng_is_reproducible_and_independent` | seeded mixup reproducible **and** does not consume the global numpy stream |
| `test_param_count_in_expected_band` | a **band**, not an equality — catches drift while declining to assert an unsupported number |
| `test_specaugment_preserves_shape_and_masks` | the mask-fraction invariant from §4.2 |
| `test_alpha_temperature_sharpens_and_stays_consistent` | `tau<1` moves α further from 0.5; neutral init preserved at any `tau`; `forward`/`fusion_weight`/`fusion_weights` all agree (§4.9) |
| `test_split_is_three_way_and_fully_disjoint` | train/dev/val pairwise disjoint by filename **and** by parent recording (§8A) |
| `test_val_fold_matches_completed_runs` | the val fold did not move when dev was carved out — otherwise 12 completed runs silently stop being comparable |

### 7.4 A bug worth recording

`Stream.__init__(self, ..., stem_pool: int = C.STEM_POOL)` binds the config value
**once, at import time**. `train.py` assigning `C.STEM_POOL = args.stem_pool` was
therefore silently ignored, and `--stem-pool 1` would have run *with* the pool
anyway.

Caught because measured GFLOPs were identical at both settings — 2.792 for each, when
they should have differed 4×. Fixed with a `None` sentinel resolved at call time, in
both `Stream` and `HSPPNet`. This is Python's mutable-default trap in a
config-injection disguise, and it would have quietly invalidated the entire stem
ablation.

### 7.5 Resumability

Precompute carries a `done` mask; training writes `history.json` every epoch;
`run_ablation.sh` skips any arm whose `summary.json` exists. A 16-minute job, a
2.5-hour run, and a 13-hour sweep all survive interruption without repeating work.

---

## 8. Ablation design

### 8.1 The four arms

Ablation removes one component at a time and measures the damage; the logic is
causal. The discipline is that **exactly one thing changes** — split, seed, optimizer,
augmentation, epochs, patience, and metric must be identical, or differences cannot be
attributed.

| Arm | Input | Params | Question |
|---|---|---|---|
| `raw` | `X_raw` (undecomposed log-mel) | 7.98M | **The control.** Does HPSS help at all? |
| `harmonic` | `X_h` | 7.98M | Is the harmonic half sufficient alone? |
| `percussive` | `X_p` | 7.98M | Is the percussive half sufficient alone? |
| `dual` | `X_h` + `X_p` + attention fusion | 15.82M | **The paper's method.** Does fusing beat either half? |

Selection is a 4-line dict mapping arm → HDF5 keys. All three streams were
precomputed in one pass, so **arms read the same bytes from the same file** — no
re-extraction, no possibility of inter-arm feature drift.

**`raw` is the arm that matters most**: it is the only one that can falsify the paper.
`harmonic`/`percussive` are internal decompositions; `raw` asks whether the entire
HPSS front end earns its keep.

### 8.2 Results under the paper's configuration

Seed 42, `STEM_POOL=1`, flat LR, Adam, scalar fusion:

```
configuration                          mAP  macroF1  microF1   ep      α   params    paper%
single stream, raw mel (no HPSS)    0.6938   0.6690   0.7720   28      -   7.98M      69.4
percussive stream only              0.6886   0.6743   0.7637   30      -   7.98M      67.5
dual-stream + dual attention        0.6541   0.6426   0.7413   24  0.087  15.82M      72.1
harmonic stream only                0.6433   0.6261   0.7170   28      -   7.98M      68.2
```

Paper's ordering: `dual > raw > harmonic > percussive`.
Measured ordering: `raw > percussive > dual > harmonic`.

**The paper's method placed third of four**, losing to a no-HPSS baseline at half the
parameters and half the FLOPs. The only ordinal agreement is that harmonic-only is
weak. This is a failed replication of the paper's central claim on this task — and
the ladder was constructed carefully enough to say so with confidence.

### 8.3 Why that table was not yet conclusive

The top three arms span **0.040 mAP**; within-run epoch-to-epoch oscillation was
**±0.035**. Same magnitude. Consequently:

- **Supported:** dual does not beat the no-HPSS baseline; the per-class collapses
  (BOAPRA −0.59, BOALUN −0.40) are an order of magnitude above noise; harmonic-only
  is weakest on all three metrics.
- **Not supported:** the raw-vs-percussive ordering (0.005 apart — pure noise); any
  precise effect size for "HPSS hurts."

The per-class evidence is what rescues this from an inconclusive four-way tie. The
aggregate ordering needs seeds and a converged LR; the *mechanism* finding does not.

### 8.4 What the current sweep changes

`ablation_table()` now:

1. **Groups runs by config**, so incompatible runs are never averaged. The four
   original `*_seed42` runs appear in a quarantined
   `legacy (pre-stem-pool, flat lr, Adam, scalar fusion)` block.
2. **Reports mean±std over seeds** when multiple exist.
3. **Prints an explicit verdict:**
   `between-arm spread X vs mean seed std Y → ORDERING IS RESOLVED / NOT RESOLVED`.

That last line is the whole point. **NOT RESOLVED** means the honest conclusion is
"no detectable difference" — *not* whichever arm printed highest.

Running: 4 arms × 3 seeds (42, 43, 44) × 40 epochs, `STEM_POOL=2`, per-class fusion,
cosine LR, AdamW. Affordable only because the stem pool cut cost 4×.

### 8.5 Final ablation result — 3 seeds, complete

*Measured*, `STEM_POOL=2`, per-class fusion, cosine LR, AdamW, 40 epochs:

```
configuration                           mAP mean±std          macroF1   n      a
dual-stream + dual attention         0.7633 ± 0.0029  0.7458 ± 0.0100   3  0.508
single stream, raw mel (no HPSS)     0.7555 ± 0.0139  0.7514 ± 0.0182   3      -
percussive stream only               0.7373 ± 0.0059  0.7276 ± 0.0107   3      -
harmonic stream only                 0.7109 ± 0.0086  0.6932 ± 0.0098   3      -

between-arm spread 0.0524 vs mean seed std 0.0078 -> ORDERING IS RESOLVED
```

Per seed, with Welch t-tests:

```
arm             s42     s43     s44 |    mean     std   best_ep
raw          0.7715  0.7458  0.7494 |  0.7555  0.0139   36,37,16
harmonic     0.7171  0.7146  0.7011 |  0.7109  0.0086    7,20,11
percussive   0.7387  0.7309  0.7425 |  0.7373  0.0059   19,13,17
dual         0.7628  0.7664  0.7607 |  0.7633  0.0029   14,25,11

dual vs raw        : +0.0078  p=0.437  NOT significant
dual vs percussive : +0.0260  p=0.007  significant
percussive vs harm : +0.0264  p=0.015  significant
raw vs percussive  : +0.0182  p=0.139  NOT significant
```

#### What is established

**1. The scalar-fusion defect is fixed.** Dual now beats **both** of its own input
streams with statistical significance (+0.0260 over percussive, p=0.007). Under the
paper's scalar fusion it scored *below* percussive-only (0.6541 < 0.6886) — the
evidence that fusion was destroying information. That failure mode is gone in all
three seeds.

**2. The per-class fusion weights are reproducible.** Cross-seed correlation of the
34 per-class alphas is **r = 0.939** (pairwise 0.941 / 0.962 / 0.913). Three
independent initialisations rediscover the same per-species stream preference, with
tight per-class standard deviations:

```
harmonic-leaning:    PHYALB 0.725 (sd 0.015)   ELABIC 0.659 (sd 0.014)
percussive-leaning:  DENMIN 0.375 (sd 0.010)   DENNAN 0.436 (sd 0.005)
```

This is the strongest evidence in the replication that the HPSS decomposition
carries real, species-specific information — it is a stable learned property, not
fitting noise.

**3. Harmonic-only is the weakest arm** (p=0.015 vs percussive).

**4. Dual is the most stable arm** — σ=0.0029 vs raw's 0.0139, 4.8× tighter. Fusing
two streams averages away seed variance that a single stream is exposed to.

#### What is NOT established

**Dual vs raw is a statistical tie: +0.0078, p=0.437**, with dual winning 2 of 3
seeds. Given σ_raw = 0.0139, three seeds cannot separate a 0.0078 gap.

**So the paper's central claim — that HPSS decomposition beats the undecomposed
baseline — remains unproven on this task.** It is no longer *refuted* (the legacy
configuration had dual placing 3rd of 4, losing outright), but "not significantly
different from doing nothing" is not a vindication, and dual pays 2× the parameters
for it.

Two further cautions:

- **`raw` may be under-trained.** Its best epochs were 36, 37, 16 of 40 — two of
  three runs were still improving when the budget ended. A longer schedule could
  move it further, which would work against dual.
- **The +0.109 over legacy is not attributed.** Four things changed simultaneously
  (stem pool, per-class fusion, cosine LR, AdamW). The `attr_*` runs in
  `run_tuning.sh` decompose it.

#### Remaining headroom inside the fusion

From the seed-42 per-class breakdown: dual beats all three singles on only **5 of
34** classes and loses (>0.01) on **18**. If it matched the best single stream on
every class it would score **0.8135** rather than 0.7628 — **~0.05 mAP of headroom
inside the fusion mechanism itself**, larger than the entire dual-vs-raw gap.

Diagnosis of why: 19–20 of 34 alphas end within 0.05 of their 0.5 initialisation
(seed 42: 19/34, 43: 11/34, 44: 20/34), cross-class std ~0.083. Combined with
r=0.94 reproducibility, the reading is that the *direction* is learned correctly but
the *magnitude* cannot travel far enough in 40 epochs — `a_raw` receives gradient
scaled by `sigmoid'(0) = 0.25` while competing against 15.8M parameters at one
shared learning rate. This is what `ALPHA_TAU` and `ALPHA_LR_MULT` address (§4.9).

---

## 8A. Split discipline: the three-way split

### Why it exists

Through the 3-seed ladder, **val was doing three incompatible jobs**: early
stopping, per-class threshold fitting (42 fitted parameters), and — once tuning
began — architecture selection. Each use spends some of val's credibility as an
unbiased estimate.

Normally the remedy is to hold out the test set for a single final read. **That is
not available *during the experiments*:** `test.7z` ships **no label CSV**. (The
labels were located upstream afterwards -- see `RESULTS_REPORT.md` §7.4 -- and test
was scored once at the end. Selection was never affected.)
There is no second held-out set to detect overfitting to val.

### Protocol

```
train  44,408 clips / 767 recordings   gradients + normalisation statistics
dev     8,932 clips / 154 recordings   early stopping, checkpoint selection,
                                       per-class threshold fitting
val     8,851 clips / 153 recordings   read ONCE at the end, dev-fitted thresholds
```

Implementation is a **nested** grouped split (`src/splits.py`):

1. Hold out the val fold **exactly as before** — same `StratifiedGroupKFold`, same
   seed, same fold. This is deliberate: 12 completed runs were scored on it, and
   perturbing it would silently make them incomparable.
2. Carve dev from the **remaining train groups only**, via a second
   `StratifiedGroupKFold(n_splits=6, random_state=seed+1)`, re-stratified on the
   train subset so dev inherits the same rarity-protecting bias (§6.2).

`verify_split` now asserts pairwise disjointness across **all three** splits, by
filename *and* by parent recording, and that they partition the frame exactly.
`src/splits.py` prints an explicit confirmation that the val fold is unchanged;
`test_val_fold_matches_completed_runs` enforces it against a stored backup.

Cost: train shrinks from 53,340 to 44,408 clips (−17%). That is the price of an
honest val.

### Measured effect of fixing the threshold leak

From a 2-epoch smoke run on the dev path:

```
dev macro-F1 @0.5   0.4101
dev macro-F1 tuned  0.5569   <- thresholds fitted on dev, scored on dev
val macro-F1 tuned  0.5133   <- same thresholds, scored on held-out val
dev -> val mAP gap  +0.0107
```

Val macro-F1 lands **below** the dev-tuned figure. That difference is precisely the
optimism the old protocol was hiding: in the legacy runs, threshold tuning appeared
to gain 0.4568 → 0.6426, but a portion of that was fitted-on-what-it-scores rather
than real generalisation.

`summary.json` records `selection_split`, `dev_mAP`, `dev_macro_f1_tuned` and
`dev_to_val_gap`, so tuning decisions can be compared **without reading val at
all**. Runs with `selection_split == "val"` are the legacy path and are excluded
from tuning comparisons by `analyze.py --tuning`.

---

## 8B. Pre-registered tuning sweep

Because val is now the only held-out estimate and there is no test set behind it,
**the selection rule is written down before the runs execute** (`run_tuning.sh`):

> Winner of each factor = highest **`dev_mAP`**, single aggregate number. Per-class
> breakdowns must not be consulted to pick a winner. Every cell is reported,
> including losers.

This matters because the diagnosis in §8.5 came from inspecting per-class val APs.
Any architecture choice justified by that inspection — for example warm-starting
alpha from the single-stream per-class results, or adding a third `raw` stream
because raw wins on BOALUN and PHYDIS — would bake val information into the model.
Those options were considered and **dropped** for exactly this reason.

Design: **one factor at a time**, not a cross product. A full grid over 4 factors ×
~3 levels is ~54 runs (~2 days); OFAT is 11 runs (~9 h) and suffices for main
effects.

| Factor | Levels | Justified by |
|---|---|---|
| `alpha_tau` | 1.0, 0.3 | 19–20/34 alphas stuck near init — a **gradient-magnitude** fact |
| `alpha_lr_mult` | 1, 10 | same bottleneck, independent remedy |
| `mixup_p` | 0.3, 0.5, 0.7 | dual has the **lowest train loss** (0.0130) but not the best val — mild-overfit signature. Both directions tested |
| `epochs` | 40, 60 | `raw` peaked at 36/37 of 40 — two of three runs still improving |
| `loss` | bce, asl | declared in advance, reported either way |

Plus three **attribution** runs (`--scalar-fusion`, `--stem-pool 1`,
`--lr-schedule none`) to decompose the +0.109 over legacy. These are not tuning.

Reported by `python -m src.analyze --tuning`, ranked by `dev_mAP`, with val shown
as transparency-only. The table also warns that a single-seed dev gain below
~0.008 sits inside the seed noise measured on the 3-seed ladder.

### 8B.1 Result: the alpha diagnosis was mechanically right and practically wrong

*Measured*, 8 of 11 runs complete, dual arm, seed 42, three-way split:

```
run             tau  alr_mult   alpha range    alpha std   stuck<0.05   dev_mAP
base            1.0       1.0   0.352-0.713      0.0913      16/34      0.7151
tau0.3          0.3       1.0   0.275-0.820      0.1471      10/34      0.7265
alr10           1.0      10.0   0.145-0.924      0.1938      10/34      0.7012
tau0.3_alr10    0.3      10.0   0.080-0.934      0.2137       5/34      0.7037
```

The α spread rises **monotonically** with both knobs, and the count of near-neutral
weights falls from 16/34 to 5/34. The mechanism works exactly as designed: α *was*
gradient-starved, and both remedies free it.

**But dev mAP does not follow the spread.** The most-spread configuration
(`tau0.3_alr10`) scores *below* baseline. Mild sharpening (`tau0.3`, +0.0114) helps
marginally; aggressive freedom (`alr10`, −0.0139) hurts.

**Revised interpretation.** α sitting near 0.5 was **not** the performance
bottleneck — it was a symptom the network tolerated. Allowing α to overcommit early,
on per-class gradients estimated from as few as 149 positives (DENELE) among the 34
eval classes, trades a small bias for a larger variance and loses.

**Consequence for the roadmap.** This removes the motivation for
**dynamic/input-conditioned fusion**, which was previously the top-ranked next step
(§11). A per-clip gate grants strictly *more* freedom than a per-class scalar, and
more freedom is what just failed. It should not be built on the old rationale.

It also promotes the **α-frozen-at-0.5 control** from a minor check to the more
informative experiment: if frozen α matches learned α, the per-class mechanism is
decorative and the real gain came from moving fusion to the logit level with a
shared head — a different and simpler claim.

Full tuning table, ranked by the pre-registered metric:

```
run               dev_mAP  dev_F1 |  val_mAP  val_F1     gap   ep      a
mix0.7             0.7266  0.6800 |   0.7184  0.6464 -0.0082   13  0.505
tau0.3             0.7265  0.6795 |   0.7216  0.6659 -0.0049   24  0.504
mix0.3             0.7152  0.6706 |   0.7101  0.6354 -0.0051   13  0.507
ep60               0.7151  0.6704 |   0.7060  0.6461 -0.0092   24  0.499
base               0.7151  0.6741 |   0.7100  0.6561 -0.0051   22  0.501
asl                0.7053  0.6788 |   0.7221  0.6467 +0.0168   33  0.488
tau0.3_alr10       0.7037  0.6641 |   0.7165  0.6378 +0.0128   14  0.488
alr10              0.7012  0.6687 |   0.7183  0.6748 +0.0171   32  0.491
```

**These numbers are not comparable to §8.5.** The ladder used the two-way split
(train = 53,340); every tuning run uses the three-way split (train = 44,408). The
17% reduction in training data lowers all of them by construction. `base` at
dev 0.7151 is the only valid reference for tuning comparisons.

Both leaders (`mix0.7` +0.0115, `tau0.3` +0.0114) sit at roughly 1.5× the seed noise
of 0.0078 measured on the 3-seed ladder, on a **single seed**. Neither is confirmed;
both need 3 seeds before adoption. Note also that `asl` and `alr10` show *positive*
dev→val gaps (+0.017) where every other run is negative — with one seed that is not
interpretable, but it is worth watching if either is pursued.

### 8B.2 Attribution: per-class fusion was worth ~nothing in absolute mAP

`attr_scalar` reruns the paper's single global α under otherwise identical current
settings, isolating what the per-class change contributed:

```
base          (per-class α, 42 weights)   dev 0.7151
attr_scalar   (paper's single scalar α)   dev 0.7116     Δ = -0.0035
```

**−0.0035 sits well inside the measured seed noise — σ ≈ 0.017 on this split** (the
0.0078 figure often quoted came from the two-way ladder and understates variance by
2.2×). On this split and seed, per-class fusion contributed approximately **nothing**
to absolute performance.

Both facts hold simultaneously and must be reported together:

1. **In absolute mAP:** no measurable gain.
2. **In ordering:** under scalar fusion in the *original* configuration, dual scored
   **below its own better input stream** (0.6541 vs percussive 0.6886) — fusion was
   destroying information. With per-class α, dual beats both inputs with significance
   (+0.0260 over percussive, p=0.007).

So the defensible claim is narrower than "per-class fusion improves the model": it
**removes a pathology** in which the fused model underperformed its own components.
That is a correctness fix, not a performance win, and §8.5 and §10 should be read
with that qualification attached.

**Corollary.** Most of the +0.109 over the legacy configuration must therefore come
from the **other three** simultaneous changes — cosine LR, AdamW, and the stem pool.
`attr_stem1` and `attr_flatlr` apportion the remainder.

Note `attr_scalar`'s α settled at 0.474 and it early-stopped at epoch 13. Combined
with §8B.1 (more α freedom hurt), the consistent reading is that **α is not where
the performance is** on this task — which favours running the α-frozen-at-0.5
control over building any more elaborate fusion mechanism.

---

## 8C. Full tuning catalogue

A companion document, **`TUNING.md`**, catalogues all 13 training-side changes —
Adam→AdamW, stem pool, cosine LR, MixUp probability, α temperature/LR, batch size,
SpecAugment rescaling, gradient clipping, AsymmetricLoss, seeding, the dev split, and
the per-class fusion attribution — each with its purpose, the underlying concept, and
its measured outcome. It is organised by status (adopted / swept / reverted /
pending) rather than by pipeline stage, and is the fastest way to see what was tried
and why.


---

## 9. Findings that are data properties, not model defects

`python -m src.analyze --recordings` was added to settle two open questions. Both
resolve to **data limits that no architectural change can move.**

### 9.1 The four sub-chance classes

Four `EVAL_CLASSES` scored at or below chance while ADEDIP reached 0.9997 on only 390
positives. Support does not explain it; **recording diversity does**:

```
mean AP with <=2 val recordings (n=9):  0.3500
mean AP with  >2 val recordings (n=25): 0.7636
```

All four sub-chance classes have exactly **2 val recordings**:

| species | positives | AP | val positives / recordings |
|---|---:|---:|---|
| DENCRU | 602 | 0.0096 | 35 in 2 recordings |
| RHIICT | 310 | 0.0247 | 62 in 2 recordings |
| PHYMAR | 200 | 0.0264 | 26 in 2 recordings |
| ADEMAR | 520 | 0.0762 | 70 in 2 recordings |

DENCRU has **602 positives — more than ADEDIP's 390 at AP 0.9997.** Clip count was
never the issue.

With ≤2 recordings, "learn the species" and "learn that recording's background" are
indistinguishable on train, and the grouped split deliberately creates the
acoustic-condition gap that exposes the difference. **This is the split working as
designed** — revealing genuine non-generalisation rather than hiding it. But it means
these APs are properties of the data, and they should be reported with their
recording count attached.

### 9.2 Why `tail_mAP` is NaN rather than low

*Measured:*

```
LEPFLA   7 positives across 2 recording(s), 0 in val
RHISCI  11 positives across 1 recording(s), 0 in val
RHIORN  21 positives across 2 recording(s), 0 in val
LEPELE  34 positives across 2 recording(s), 0 in val
AMEPIC  68 positives across 2 recording(s), 0 in val
SCIRIZ  73 positives across 3 recording(s), 0 in val
```

Every ultra-rare species is confined to 1–3 recordings, so a group-disjoint split
places all of them on one side — here, all in train. With zero val positives, AP is
undefined.

**The tail is currently unmeasurable, not measured-as-bad.** Reporting NaN is
correct; reporting 0.0 would be a fabrication. Measuring it would need a
tail-specific split, and even then 1–3 recordings cannot support a generalisation
claim.

---

## 10. Deviations from the paper — consolidated

| # | Paper | Here | Reason |
|---|---|---|---|
| 1 | softmax + CE, 10 classes | sigmoid + BCE, 42 logits | multi-label: 0–8 species/clip, 36.2% all-negative (§0) |
| 2 | accuracy | mAP / macro-F1 over 34 classes | accuracy is 96.4% for an always-negative predictor (§5.2) |
| 3 | `Softmax` output | raw logits | `BCEWithLogitsLoss` fuses the sigmoid stably; double-sigmoid is silent (§2.4) |
| 4 | SpecAugment time mask 40 | **12** | 40/130 frames = 31%; two masks erase 60% of a call. Ratio transferred, not integer (§4.2) |
| 5 | batch size 32 | **64** | 32 underutilises MPS; throughput flat 32→64 (§4.1) |
| 6 | Adam | **AdamW**, no-decay on 1-D params | decoupled decay; BN γ/β and `a_raw` must not be decayed (§4.3) |
| 7 | no LR schedule | **cosine + 2-epoch warmup** | flat LR caused ±0.035 oscillation and a best-epoch lottery (§4.4) |
| 8 | MixUp always on | **p = 0.5** | augmentation-limited plateau; every sample was corrupted twice (§4.5) |
| 9 | pooling after each block only | **`STEM_POOL=2`** | paper's own 2.86 GFLOPs is unreachable without a ~/4 stem (§3.2) |
| 10 | single scalar fusion weight | **per-class (42,), logit-level** | scalar fused *worse than its own better input*; species have opposing preferences (§2.5) |
| 11 | 12.4M params / 2.86 GFLOPs | **15.82M / 2.792 G measured** | reported as measured; neither reverse-engineered into the architecture (§3) |

---

## 11. Outstanding work, sequenced

**Done since the 3-seed ladder:**

- ✅ 4-arm × 3-seed ablation with mean±std and a resolved-ordering verdict (§8.5)
- ✅ Three-way train/dev/val split, closing the threshold-fitting leak (§8A)
- ✅ `ALPHA_TAU` / `ALPHA_LR_MULT` knobs for the stuck-alpha problem (§4.9)
- ✅ Pre-registered OFAT tuning sweep, **running** (§8B)

**Running now** (`run_tuning.sh`, ~9 h, 11 runs):

1. **Tuning factors** — `alpha_tau`, `alpha_lr_mult`, `mixup_p`, `epochs`, `loss`.
   Selected on `dev_mAP` only, per the pre-registered rule.
2. **Attribution** — `--scalar-fusion`, `--stem-pool 1`, `--lr-schedule none`
   decompose the +0.109 over legacy into its four simultaneous causes.

**Conditional on those results:**

3. **Dynamic / input-conditioned fusion** — **DEPRIORITIZED by §8B.1.** The original
   rationale was that α could not travel far enough. The sweep showed α spread rises
   monotonically with `tau`/`alr_mult` while dev mAP does *not* follow — more freedom
   made things worse. A per-clip gate is strictly more freedom, so this needs a new
   justification before being built.
4. **α frozen at 0.5 control** — **PROMOTED by §8B.1**, now the more informative
   experiment. If frozen α matches learned α, the per-class mechanism is decorative
   and the gain came from logit-level fusion with a shared head instead.
5. **Confirm `mix0.7` / `tau0.3` with 3 seeds** — both lead on a single seed by
   ~1.5× seed noise. Not adoptable until confirmed.
6. **Longer `raw` run (60 epochs, 3 seeds)** — `raw` peaked at epoch 36/37 of 40 in
   two of three seeds, so it may be under-trained. If it improves, the dual-vs-raw
   tie moves *against* dual. An honesty check, not an optimisation.
7. **Independently-trained then frozen encoders** — distinguishes "fusion is badly
   parameterised" from "joint training starves both streams." Live hypothesis, since
   dual early-stops much sooner than `raw` (14/25/11 vs 36/37/16).
8. **Re-run the 4-arm × 3-seed ladder** with whatever is confirmed, on the dev split,
   and report val once.

**Now the best remaining idea:**

9. **`HPSS_KERNEL` sweep** {9, 13, 17, 25} — the one paper hyperparameter
   transferred without domain validation (§1.5). At 130 frames, 17 frames = 394 ms
   of temporal median smoothing against 200–500 ms calls, so it may be smoothing
   across the entire call. Requires re-running precompute per value (~16 min each)
   into separate `.h5` files. Run on the `percussive` arm — cheapest, and the stream
   the model leans on. With the α avenue now closed, this is the largest untapped
   gain remaining.

**Rejected, and why** (recorded so they are not revisited):

- **Warm-starting α from the single-stream per-class APs** — would bake val-derived
  results into initialisation. With no labelled test set there is nothing left to
  detect the resulting overfit.
- **Adding a third `raw` stream because raw wins on BOALUN/PHYDIS** — the
  justification is val-derived class-level reasoning. Viable as an architecture idea;
  not on that evidence.
- **Oversampling / `WeightedRandomSampler`** — ill-defined in multi-label (§5.7).
- **Focal loss** — ASL supersedes it.

**Known-unfixable with current data:** the four sub-chance classes (§9.1) and the
tail (§9.2). Both are recording-diversity limits, not model defects.

**Structural limit:** `test.7z` has 31,187 wavs and **no labels**, so val is the
only held-out estimate available while the experiments ran. Every additional read of
it costs credibility. That is why selection runs on dev and why the tuning rule was
pre-registered.

---

## Appendix A — Commands

```bash
cd /Users/jent/UTEC/ciclo5/deepLearning/paper1

# one-time
.venv/bin/python -m src.splits          # three-way grouped split + leakage asserts
.venv/bin/python -m src.splits --no-dev # reproduce the original two-way split
.venv/bin/python -m src.precompute      # → data/features.h5 (6.2 GB, ~16 min)
.venv/bin/python -m pytest tests/ -q    # 18 verification tests

# training
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
  .venv/bin/python -u -m src.train --arm dual --epochs 40 --seed 42 --tag dual_s42_stem2

nohup ./run_ablation.sh > runs_ladder.log 2>&1 &    # 4 arms × 3 seeds, resumable
nohup ./run_tuning.sh  > runs_tuning.log 2>&1 &     # pre-registered OFAT sweep
./watch_ladder.sh                                    # live progress, any runs

# reproduce the paper's configuration exactly
.venv/bin/python -m src.train --arm dual --stem-pool 1 --scalar-fusion \
    --lr-schedule none --mixup-p 1.0 --alpha-tau 1.0 --tag dual_paper

# analysis
.venv/bin/python -m src.analyze                       # ladder: mean±std + verdict
.venv/bin/python -m src.analyze --tuning              # tuning, ranked by dev_mAP
.venv/bin/python -m src.analyze --complexity          # params / FLOPs vs Table 5
.venv/bin/python -m src.analyze --per-class RUN
.venv/bin/python -m src.analyze --recordings RUN      # clip vs recording support

# serving
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/uvicorn src.serve:app --port 8000
```

## Appendix B — MPS platform notes

MPS is a backend inside standard PyTorch (`torch.device("mps")`, same
`pip install torch`), not a separate library. Practical consequences:

| Finding | Reason |
|---|---|
| `pin_memory=False` | pinned memory serves async DMA over PCIe; unified memory has no such copy |
| fp32 only | Metal has **no fp64**; autocast/bf16 immature |
| `channels_last` dropped | tested; fails on MPS **backward** (`view size is not compatible...`) |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | unimplemented ops fall back to CPU rather than crashing mid-run |
| lazy per-worker h5py handle | macOS **spawns** workers → dataset is pickled → an open h5py handle raises. `__getstate__` drops it; `_handle()` reopens with `swmr=True` |
| no `torch.distributed` | Ray Train's `TorchTrainer` cannot run here at all |
| sequential runs only | one device; concurrent runs time-slice the same command queues and contend |
| non-deterministic reductions | report mean±std over seeds; no bitwise reproducibility (§6.5) |

Roughly 3–6× slower than the paper's RTX 3090 for this model: M4 Pro GPU ~7 TFLOPS
fp32 vs 3090 ~35, discounted by MPS kernel maturity for asymmetric (3×5, 5×3) shapes,
offset upward by unified memory eliminating host↔device transfer.

---

*Numbers marked "measured" were produced on M4 Pro / 24 GB / MPS, PyTorch with MPS
backend, Python 3.11. Python 3.11 is pinned deliberately: 3.14 has no reliable torch
wheels, and librosa requires numba, which lags new CPython releases.*
