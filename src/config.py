"""Central configuration for the HSPP replication on AnuraSet.

All feature/training hyperparameters live here so that precompute, training and
serving cannot drift apart (train/serve feature skew is a silent killer).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO_TRAIN = DATA / "train"
AUDIO_TEST = DATA / "test"
FEATURES_H5 = DATA / "features.h5"
SPLITS_DIR = DATA / "splits"
RUNS_DIR = ROOT / "runs"
TRAIN_CSV = ROOT / "train.csv"

# ---------------------------------------------------------------- audio / features
# Verified by probing the extracted wavs: 22050 Hz, mono, int16, exactly 3.000 s.
SR = 22050
CLIP_SECONDS = 3.0
N_SAMPLES = int(SR * CLIP_SECONDS)  # 66150

# n_fft=1024 / hop=512 gives T=130 frames for 3 s, making the input nearly
# square (128x130) so the paper's 4-block downsampling schedule transfers
# unchanged. Tension worth knowing: HPSS wants fine frequency resolution while
# the 5x3 temporal convs want fine time resolution; this is the compromise.
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
N_FRAMES = 130
FMIN = 50.0     # cuts wind/DC rumble; lowest anuran fundamentals here are ~150-200 Hz
FMAX = SR / 2   # 11025 Hz, comfortably covers anuran call bands

# power_to_db with a FIXED reference (not ref=np.max, which normalises per clip
# and destroys absolute level, making a loud near call look like a faint far one).
DB_REF = 1.0
DB_TOP = 80.0

# HPSS median filter size. Paper's sensitivity table: 5->69.3, 9->70.8, 13->71.6,
# 17->72.1 (best), 21->71.8, 25->71.1.
HPSS_KERNEL = 17
# margin=1.0 keeps the soft mask property S_h + S_p == S. Higher margins discard
# residual energy into neither stream, losing information the two streams are
# meant to jointly cover.
HPSS_MARGIN = 1.0

# ---------------------------------------------------------------- training
BATCH_SIZE = 64      # paper used 32; 32 underutilises MPS and per-step launch overhead dominates
LR = 1e-3            # paper
WEIGHT_DECAY = 1e-4  # paper
MAX_EPOCHS = 100     # paper
EARLY_STOP_PATIENCE = 10
SEED = 42

# ---------------------------------------------------------------- LR schedule
# The paper specifies no schedule; a flat 1e-3 was measured to be the single
# largest source of wasted headroom. From the seed-42 dual run (STEM_POOL=1),
# epochs 15-31 with train loss flat at 0.053-0.055:
#     ep15 0.6281  ep16 0.6407  ep17 0.6059  ep19 0.6333  ep20 0.5855
#     ep24 0.6541  ep26 0.6050  ep28 0.6397  ep30 0.6527  ep31 0.6471
# A +-0.035 band with no descent: the step size is too large for the basin, so
# the optimiser orbits the minimum instead of entering it. Two costs:
#   1. accuracy left on the table (typically +0.01-0.03 mAP once it can settle)
#   2. WORSE -- best.pt is selected by whichever epoch got lucky on the
#      oscillation, so every arm's headline is inflated by an unknown,
#      arm-specific amount. That is why the top three arms spanning 0.040 mAP
#      is unreadable against +-0.035 of noise.
#
# Cosine annealing decays lr smoothly from LR to LR*LR_MIN_FACTOR over
# T_max=epochs, so late epochs take small steps and converge.
#
# WARMUP_EPOCHS: BatchNorm running statistics are meaningless at init, so the
# first few hundred steps produce large, poorly-scaled gradients. A short linear
# warmup avoids spending the largest steps of the run on the worst gradient
# estimates. 2 epochs at 833 steps each is ~1,666 warmup steps.
LR_SCHEDULE = "cosine"   # "cosine" | "plateau" | "none"
LR_MIN_FACTOR = 0.01     # cosine floor: lr ends at LR * this (1e-5)
WARMUP_EPOCHS = 2        # linear ramp 0 -> LR; 0 disables

# ReduceLROnPlateau alternative. PLATEAU_PATIENCE must stay well below
# EARLY_STOP_PATIENCE or the run dies before the LR is ever cut.
PLATEAU_FACTOR = 0.3
PLATEAU_PATIENCE = 3

# SpecAugment. The paper's time mask width of 40 was tuned for 10 s / 431-frame
# clips; on our 130-frame clip that is 31% of the audio and two masks can erase
# 60% of a frog call. Scaled down deliberately -- documented deviation.
SPEC_TIME_MASK = 12
SPEC_FREQ_MASK = 8
SPEC_N_MASKS = 2
MIXUP_ALPHA = 0.2

# Probability that a given batch gets mixed. The seed-42 runs applied MixUp to
# 100% of batches on top of SpecAugment on 100% of samples, so every sample the
# model ever saw was corrupted twice. That is consistent with what was measured:
# train loss plateaued at 0.053 with NO train/val divergence across 31 epochs, i.e.
# the model was augmentation-limited rather than capacity-limited. A gate at 0.5
# lets the model also see clean-ish samples. Set to 1.0 to recover the old
# behaviour.
MIXUP_P = 0.5

# Gradient-norm clipping. Cheap insurance rather than a fix for an observed
# explosion: with BN everywhere and Adam the gradients are well-behaved, but the
# ultra-rare classes (LEPFLA has 7 positives in 62,191 clips) contribute large,
# sparse gradients when they do appear. 0 disables.
GRAD_CLIP = 1.0

# ---------------------------------------------------------------- architecture
# Stem downsample factor applied BEFORE block 1 (1 = the literal reading of the
# paper's text, i.e. no stem pool).
#
# Reason for offering /2: the paper reports 2.86 GFLOPs at 10 s / 431 frames, which
# is arithmetically unreachable with pooling placed only after each block -- block 1
# alone at 128x431 costs ~6.9 G, already exceeding the reported total. Measured
# dual-stream GFLOPs vs stem stride at 431 frames:
#     none (128x431) 37.13 | /2 (64x216) 9.37 | /4 (32x108) 2.32 | /8 (16x54) 0.56
# So the paper must reduce resolution well before block 1, which its text omits.
#
# At our 3 s / 130 frames: STEM_POOL=1 -> 11.2 G, STEM_POOL=2 -> 2.78 G, essentially
# the paper's headline figure at ~4x less compute. This makes multi-seed runs
# affordable (dual drops ~13.6 -> ~4 min/epoch), which is what the ablation needs:
# the top three arms span 0.040 mAP against a +-0.035 within-run oscillation.
#
# Kept as a config knob rather than hardcoded so STEM_POOL=1 vs 2 is itself an
# ablation. Note /4 would leave block 4 with an 8x8 map -- the floor that
# test_spatial_map_not_degenerate enforces for spatial attention to be meaningful.
STEM_POOL = 2

# Per-class fusion weights instead of one global scalar.
#
# The scalar version failed measurably. Measured seed-42 ablation (STEM_POOL=1):
#     raw 0.6938 | percussive 0.6886 | dual 0.6541 | harmonic 0.6433
# The dual arm scored BELOW percussive-only, i.e. the fusion did worse than the
# better of the two streams it fuses -- it destroyed information rather than
# combining it. The scalar settled at a=0.087 (91% percussive, the corpus-average
# optimum) and collapsed the classes with the opposite preference:
#     BOAPRA  perc 0.7451 -> dual 0.1520   (-0.59)
#     BOALUN  perc 0.9414 -> dual 0.5387   (-0.40)
#     SCIPER  harm 0.8609 -> dual 0.7324   (-0.13)  <- this one wants HARMONIC
# One number cannot serve 42 species with opposing stream preferences.
#
# With PER_CLASS_FUSION the weight becomes a (42,) vector and fusion moves to the
# LOGIT level (a per-class weight cannot act on the pre-head 512-d vector -- the
# class dimension does not exist there yet). Cost: +41 parameters.
#
# Set False to recover the paper's exact scalar formulation for comparison.
PER_CLASS_FUSION = True

# ---------------------------------------------------------------- labels
# Column order in train.csv. Keep all 42 so label indices stay in sync with the CSV.
SPECIES = [
    "SPHSUR", "BOABIS", "SCIPER", "DENNAH", "LEPLAT", "RHIICT", "BOALEP", "BOAFAB",
    "PHYCUV", "DENMIN", "ELABIC", "BOAPRA", "DENCRU", "BOALUN", "BOAALB", "PHYMAR",
    "PITAZU", "PHYSAU", "LEPFUS", "DENNAN", "PHYALB", "LEPLAB", "SCIFUS", "BOARAN",
    "SCIFUV", "AMEPIC", "LEPPOD", "ADEDIP", "ELAMAT", "PHYNAT", "LEPELE", "RHISCI",
    "SCINAS", "LEPNOT", "ADEMAR", "BOAALM", "PHYDIS", "RHIORN", "LEPFLA", "SCIRIZ",
    "DENELE", "SCIALT",
]
N_CLASSES = len(SPECIES)

# SCIFUS and SCINAS have ZERO positives in train.csv -- their AP is undefined and
# would silently drag macro-mAP down by 4.8% for no reason.
ZERO_POSITIVE = ["SCIFUS", "SCINAS"]
# 7-73 positives total: statistically meaningless, reported in a separate table.
ULTRA_RARE = ["LEPFLA", "RHISCI", "RHIORN", "LEPELE", "AMEPIC", "SCIRIZ"]
# Headline metrics are computed over these 34 species (>=100 positives).
EVAL_CLASSES = [s for s in SPECIES if s not in ZERO_POSITIVE and s not in ULTRA_RARE]
EVAL_IDX = [SPECIES.index(s) for s in EVAL_CLASSES]

# Only these 5 species appear at more than one site, so leave-one-site-out can
# only ever be reported as a 5-class experiment.
MULTI_SITE_SPECIES = ["BOAFAB", "DENMIN", "LEPLAT", "PHYCUV", "PITAZU"]

# ---------------------------------------------------------------- splits
VAL_FRACTION = 0.15
N_SPLIT_FOLDS = 7  # StratifiedGroupKFold: 1 fold val (~1/7 ~= 14%), rest train


def group_key(filename: str) -> str:
    """Parent recording id: ``INCT20955_20190909_050000``.

    Adjacent 3 s segments from the same parent recording are near-duplicates
    (same individual, same background, seconds apart), so splits MUST group on
    this. Taking the first 3 fields is robust to the 58 filenames that carry an
    extra 6th field (``..._041500_000_1_4.wav``); a strict 5-field regex is not.
    """
    return "_".join(filename.split("_")[:3])
