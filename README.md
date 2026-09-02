# HSPP replication on AnuraSet

Replication of the dual-stream HPSS + asymmetric-convolution CNN from
*"Structure-aware acoustic scene classification: a feature decoupling framework
using HPSS and asymmetric convolutions"* (Liu & Fan, Sci Reports 2026), adapted
from single-label acoustic scene classification to **multi-label anuran species
detection** on AnuraSet.

Note the paper does not propose an AST/Transformer method. It proposes a
lightweight CNN and positions it *against* AST (12.4M vs 86M params, 24.1 ms vs
68.3 ms inference, 72.1% vs 73.1% accuracy). AST is one of its baselines.

## Setup

```bash
uv venv --python ~/.local/bin/python3.11 .venv
VIRTUAL_ENV=.venv uv pip install "torch>=2.2" torchaudio "librosa>=0.10" soundfile \
    "numpy<2.1" pandas scikit-learn h5py tqdm tensorboard thop matplotlib seaborn \
    fastapi uvicorn python-multipart pytest

tar -xf train.7z -C data/          # 62,191 clips, 7.7 GB
tar -xf test.7z  -C data/          # ~31,260 clips
```

Python 3.11 is pinned deliberately: 3.14 has no reliable torch wheels, and
librosa needs numba, which lags new CPython releases.

## Pipeline

```bash
.venv/bin/python -m src.splits                    # grouped split + leakage asserts
.venv/bin/python -m src.precompute                # -> data/features.h5 (5.2 GB, ~16 min)
.venv/bin/python -m pytest tests/ -q -s           # 8 verification tests
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i \
  .venv/bin/python -u -m src.train --arm dual     # one ablation arm
.venv/bin/python -m src.analyze                   # ablation table
.venv/bin/python -m src.analyze --complexity      # params / FLOPs
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/uvicorn src.serve:app --port 8000
```

## Measured on this machine (M4 Pro, 24 GB, MPS)

| | value |
|---|---|
| HPSS feature extraction | 29.8 ms/clip → ~16 min for 62,191 clips on 10 procs |
| features.h5 | 5.2 GB (3 streams: X_h, X_p, X_raw), fp16 |
| dual arm throughput | 66 samples/s → **13.5 min/epoch** |
| single-stream arms | ~132 samples/s → **6.8 min/epoch** |
| DataLoader (num_workers=0) | 2,500 samples/s — 38× faster than the model, so not the bottleneck |
| params (dual / single) | 15,824,303 / 7,983,212 |

`num_workers=0` is the measured-best setting: features are precomputed, so each
sample is a small HDF5 read and worker spawn overhead is pure cost.

## Deviations from the paper, and why

| # | Paper | Here | Reason |
|---|---|---|---|
| 1 | Softmax + cross-entropy, 10 classes | Sigmoid + BCE, 42 logits | Task is multi-label: 0–8 species per clip |
| 2 | Accuracy | mAP / macro-F1 over 34 classes | 36% of clips are all-negative, so accuracy is meaningless |
| 3 | SpecAugment time mask 40 | **12** | 40 frames is 31% of a 130-frame clip; 2 masks would erase 60% of a call |
| 4 | Batch size 32 | 64 | 32 underutilises MPS; throughput is flat 32→64 so 64 costs nothing |
| 5 | `Softmax` output | raw logits | `BCEWithLogitsLoss` applies the sigmoid internally and stably |

## Two internal inconsistencies found in the paper

**1. Parameter count.** The paper reports 12.4M. Its stated channel progression
(`64→64→128→256→512` — five numbers for four blocks) is ambiguous and no reading
reproduces 12.4M:

| reading | params |
|---|---|
| literal (ours): 1→64, 64→128, 128→256, 256→512 | **15.82M** |
| ending at 256 channels | 4.16M |
| five blocks | 16.07M |
| `ci→co` on both convs in a block | 10.54M |

**2. FLOPs — the more serious one.** The paper reports 2.86 GFLOPs at 10 s input.
That is arithmetically unreachable with pooling placed after each block as the
text describes: block 1 alone at 128×431 costs ~6.9 G, already exceeding the
reported total. Searching stem strides at 431 frames:

| pre-block-1 downsample | dual-stream GFLOPs |
|---|---|
| none (128×431) | 37.13 |
| /2 (64×216) | 9.37 |
| **/4 (32×108)** | **2.32** ← closest to the reported 2.86 |
| /8 (16×54) | 0.56 |

So the paper must reduce resolution well before block 1. Our literal schedule
measures 11.2 G at 3 s. Adding a /2 stem pool would give 2.78 G — essentially the
paper's figure at ~4× less compute — and is the natural next experiment.

Neither number was reverse-engineered into the architecture; both are reported as
measured.

## Data discipline

**Splits are grouped by parent recording.** Adjacent 3 s segments come from the
same ~58-segment recording and are near-duplicates (same individual, same
background, seconds apart). A random row split leaks val into train and produces
flattering, meaningless numbers. `src/splits.py` groups on `site_date_time`
(1,074 groups) and asserts group and filename disjointness.

Split: 53,340 train / 8,851 val across 921 / 153 recordings.

**Rare classes.** The head has all 42 logits so indices stay aligned with
`train.csv`, but headline metrics cover only the **34 species with ≥100
positives**:

- `SCIFUS`, `SCINAS` have **zero** positives — AP is undefined, and including
  them would drag macro-mAP down by 4.8% for no reason.
- `LEPFLA` (7), `RHISCI` (11), `RHIORN` (21), `LEPELE` (34), `AMEPIC` (68),
  `SCIRIZ` (73) are reported in a separate table as statistically meaningless.

**Leave-one-site-out is a 5-class experiment, not 42.** Only `BOAFAB`, `DENMIN`,
`LEPLAT`, `PHYCUV`, `PITAZU` appear at more than one site; 35 of 42 are
single-site and none appear at all four. A site-held-out model cannot predict
species it never saw.

**Thresholds** are tuned per class on val only, never test. Classes with <5 val
positives fall back to 0.5 and are flagged.

## Why not Ray

Considered and rejected for every component, because one machine with one
non-partitionable GPU removes each value proposition:

- **Ray Data/Core** for precompute — it is a 16-minute one-shot job;
  `multiprocessing.Pool` plus a `done` mask gives parallelism and resumability
  without the object-store serialisation of 66 KB arrays.
- **Ray Train** — `TorchTrainer` wraps `torch.distributed`, which has **no MPS
  backend**. Multiple workers on one MPS device time-slice the same command
  queues and contend rather than speed up.
- **Ray Tune** — the 4 ablation arms are not a hyperparameter search; all four
  get reported, so early-stopping them is wrong. Trials would also be forced to
  concurrency 1.
- **Ray Serve** — strongest case, since CPU preprocessing (~30 ms HPSS)
  dominates the forward pass and separating CPU/GPU deployments is a real Serve
  strength. But `FastAPI` + `ProcessPoolExecutor` captures that locally without
  running a cluster for one model.

## MPS notes

MPS is not a separate library — it is a backend inside standard PyTorch
(`torch.device("mps")`, same `pip install torch`). Practical consequences:

- `pin_memory=False`: it exists for async DMA over PCIe, which unified memory
  does not have.
- fp32 only. Metal has no fp64, and autocast/bf16 are immature.
- `channels_last` **fails on MPS backward** here (`view size is not compatible…`)
  — tested and dropped.
- Reduction order is non-deterministic, so report mean±std over seeds rather
  than claiming bitwise reproducibility.
- macOS spawns DataLoader workers, so `h5py` handles cannot be pickled;
  `AnuraFeatures.__getstate__` drops the handle and each worker reopens lazily.
- Roughly 3–6× slower than the paper's RTX 3090 for this model.

## Verification (`tests/`, all 8 passing)

The load-bearing one is **overfit-32**: 32 samples, no augmentation, no dropout,
250 steps. Loss went 0.6925 → 0.000098, confirming the architecture and loss
wiring are correct. Also checked: output shapes across all arms, param count in
band, spatial map non-degenerate before attention, fusion weight starts at 0.5
and receives gradient, no sigmoid in `forward`, SpecAugment bounds, MixUp target
range.

Sanity checks on the feature path: HPSS soft-mask property `S_h + S_p == S`
holds to 1.5e-05, and harmonic components are smoother along time than percussive
ones in **12/12** clips tested (harmonic freq/time gradient ratio ~1.3 vs
percussive ~0.4), confirming the decomposition separates horizontal from vertical
structure as intended.
