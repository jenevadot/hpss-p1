"""Dataset over the precomputed HDF5 features, with SpecAugment and MixUp.

MPS/macOS specifics handled here:
  - macOS uses spawn, not fork, so an open h5py.File handle cannot be pickled to
    DataLoader workers. The handle is opened LAZILY inside each worker on first
    access, never in __init__.
  - features are stored fp16 and cast to float32 on load (MPS handles fp32 best;
    fp64 is unsupported by Metal entirely).
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C

# Which HDF5 dataset each ablation arm reads.
ARM_STREAMS = {
    "dual": ("X_h", "X_p"),
    "harmonic": ("X_h",),
    "percussive": ("X_p",),
    "raw": ("X_raw",),
}


class AnuraFeatures(Dataset):
    def __init__(self, filenames, labels, arm="dual", train=False,
                 h5_path=None, norm=None):
        self.arm = arm
        self.streams = ARM_STREAMS[arm]
        self.train = train
        self.h5_path = str(h5_path or C.FEATURES_H5)
        self.norm = norm  # (mean, std) computed on the TRAIN split only
        self._h5 = None

        with h5py.File(self.h5_path, "r") as h5:
            all_names = [n.decode() for n in h5["filenames"][:]]
        index_of = {n: i for i, n in enumerate(all_names)}
        self.rows = np.array([index_of[f] for f in filenames], dtype=np.int64)
        self.labels = labels.astype(np.float32)
        self.filenames = list(filenames)

    def _handle(self):
        # Lazy per-worker open: a pickled/inherited HDF5 handle across spawn is a
        # classic source of silent corruption and hangs.
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __getstate__(self):
        # macOS DataLoader workers are spawned, so the dataset is pickled. An open
        # h5py handle raises "h5py objects cannot be pickled", so drop it here and
        # let each worker reopen its own via _handle().
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        h5 = self._handle()
        row = self.rows[i]
        xs = []
        for key in self.streams:
            x = h5[key][row].astype(np.float32)
            if self.norm is not None:
                x = (x - self.norm[0]) / self.norm[1]
            x = torch.from_numpy(x).unsqueeze(0)  # (1, n_mels, n_frames)
            if self.train:
                x = spec_augment(x)
            xs.append(x)
        return (*xs, torch.from_numpy(self.labels[i]))


def spec_augment(x: torch.Tensor) -> torch.Tensor:
    """Random time and frequency masking, in place on a copy.

    Widths are deliberately smaller than the paper's (time 40, freq 8): at 130
    frames a 40-frame mask is 31% of the clip and two of them can erase 60% of a
    frog call. See config.SPEC_TIME_MASK.

    Masks are filled with 0.0, which -- because this runs AFTER normalisation --
    is the train-split MEAN energy (mean=-27.6 dB), not silence. Silence would be
    about (-80 - -27.6)/13.6 = -3.87. Mean-filling is the original SpecAugment
    recommendation and is the better choice here: a mean-filled patch reads as
    "uninformative", whereas a silence-filled patch asserts "confidently empty",
    which is a stronger and more misleading claim for a detection task.
    """
    x = x.clone()
    _, n_mels, n_frames = x.shape
    for _ in range(C.SPEC_N_MASKS):
        w = int(torch.randint(0, C.SPEC_FREQ_MASK + 1, (1,)))
        if w:
            f0 = int(torch.randint(0, max(n_mels - w, 1), (1,)))
            x[:, f0:f0 + w, :] = 0.0
        w = int(torch.randint(0, C.SPEC_TIME_MASK + 1, (1,)))
        if w:
            t0 = int(torch.randint(0, max(n_frames - w, 1), (1,)))
            x[:, :, t0:t0 + w] = 0.0
    return x


def mixup(batch, alpha: float = C.MIXUP_ALPHA, rng=None):
    """Multi-label MixUp: mix inputs and targets with the SAME lambda.

    BCE accepts soft targets, so this needs no other change. Applied after
    SpecAugment; must be disabled at eval.

    `rng` is a np.random.Generator so lambda draws are reproducible and do not
    depend on unrelated global numpy state. Falls back to the legacy global RNG
    when None, which keeps older call sites working.

    One lambda per batch (not per sample), matching the original MixUp. The
    permutation shares that lambda, so a batch is mixed with a shuffle of itself.
    """
    *xs, y = batch
    if rng is None:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(y.size(0))
    else:
        lam = float(rng.beta(alpha, alpha))
        # Derive the permutation from the same stream so a single seed fixes both.
        perm = torch.from_numpy(rng.permutation(y.size(0))).long()
    xs = [lam * x + (1.0 - lam) * x[perm] for x in xs]
    y = lam * y + (1.0 - lam) * y[perm]
    return (*xs, y)


def compute_norm_stats(filenames, arm="dual", h5_path=None, sample=4000):
    """Global mean/std over a sample of the TRAIN split only (never val/test).

    The np.linspace subsample is SYSTEMATIC over sorted (i.e. chronological, by
    site/date) HDF5 rows, not random. For a single global scalar over 4,000 x 128 x
    130 = 66.6M values that is fine and arguably better stratified across the
    recording timeline than a random draw. It would NOT be safe for a per-mel-bin
    statistic: any periodicity in row order could alias with the sampling stride.
    """
    h5_path = str(h5_path or C.FEATURES_H5)
    with h5py.File(h5_path, "r") as h5:
        all_names = [n.decode() for n in h5["filenames"][:]]
        index_of = {n: i for i, n in enumerate(all_names)}
        rows = np.array(sorted(index_of[f] for f in filenames))
        if len(rows) > sample:
            rows = rows[np.linspace(0, len(rows) - 1, sample).astype(int)]
        vals = []
        for key in ARM_STREAMS[arm]:
            vals.append(h5[key][rows].astype(np.float32))
    stacked = np.concatenate(vals, axis=0)
    return float(stacked.mean()), float(stacked.std() + 1e-6)


def build_datasets(arm="dual", h5_path=None):
    """Returns (train_ds, val_ds, norm) using the saved grouped split."""
    from .splits import load_labels, load_split

    df = load_labels().set_index("filename")
    split = load_split()

    with h5py.File(str(h5_path or C.FEATURES_H5), "r") as h5:
        available = {n.decode() for n in h5["filenames"][:]}
        done = h5["done"][:]
        names = [n.decode() for n in h5["filenames"][:]]
        ready = {n for n, d in zip(names, done) if d}

    out = {}
    for name in ("train", "val"):
        keep = [f for f in split[name] if f in available and f in ready]
        dropped = len(split[name]) - len(keep)
        if dropped:
            print(f"  {name}: skipping {dropped} clips absent from features.h5")
        out[name] = (keep, df.loc[keep, C.SPECIES].to_numpy())

    norm = compute_norm_stats(out["train"][0], arm=arm, h5_path=h5_path)
    train_ds = AnuraFeatures(*out["train"], arm=arm, train=True, h5_path=h5_path, norm=norm)
    val_ds = AnuraFeatures(*out["val"], arm=arm, train=False, h5_path=h5_path, norm=norm)
    return train_ds, val_ds, norm
