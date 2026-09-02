"""Precompute HPSS harmonic/percussive log-mels once into a single HDF5 file.

Why precompute: librosa.decompose.hpss runs two 2-D median filters over a
513x130 complex STFT. Median filtering does not vectorise like convolution and
measures ~30 ms/clip here. On the fly that is ~30 min of pure CPU per epoch --
disqualifying over 100 epochs. Precomputing costs a few minutes once.

Storage: 62,191 x 3 streams x 128 x 130 x 2 bytes (fp16) = 5.8 GB. fp16 is safe
for dB-scale mels (range ~[-80,+4]; measured roundtrip error 0.03 dB) and is cast
to float32 at load time, since MPS handles fp32 best.

Resumable via a `done` mask so a crash at clip 50,000 does not cost the run.
h5py is not fork-safe for concurrent writers, so workers return arrays and the
parent process performs all writes sequentially.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as C
from .features import hpss_mel, load_audio, raw_mel

N_WORKERS = 10  # leave 4 of 14 cores for the OS and the writer process


def _one(task):
    """Worker: returns (index, X_h, X_p, X_raw) as fp16, or (index, None...) on error."""
    idx, path = task
    try:
        y = load_audio(path)
        x_h, x_p = hpss_mel(y)
        x_r = raw_mel(y)
        return idx, x_h.astype(np.float16), x_p.astype(np.float16), x_r.astype(np.float16)
    except Exception as exc:  # noqa: BLE001 - record and continue; reported at the end
        return idx, None, None, str(exc)


def build(csv_path, audio_dir, out_path, limit=None):
    df = pd.read_csv(csv_path)
    names = df["filename"].tolist()
    if limit:
        names = names[:limit]
    n = len(names)
    shape = (n, C.N_MELS, C.N_FRAMES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "a") as h5:
        # No chunk compression: gzip on fp16 mels gains ~15% and costs CPU on
        # every batch read, which would become the training bottleneck.
        for key in ("X_h", "X_p", "X_raw"):
            if key not in h5:
                h5.create_dataset(key, shape=shape, dtype="float16")
        if "done" not in h5:
            h5.create_dataset("done", shape=(n,), dtype=bool, data=np.zeros(n, bool))
        if "filenames" not in h5:
            h5.create_dataset("filenames", data=np.array(names, dtype="S64"))

        done = h5["done"][:]
        todo = [(i, audio_dir / names[i]) for i in range(n) if not done[i]]
        print(f"{n:,} clips total, {len(todo):,} to compute, {int(done.sum()):,} already done")
        if not todo:
            return

        errors = []
        with mp.get_context("spawn").Pool(N_WORKERS) as pool:
            it = pool.imap_unordered(_one, todo, chunksize=32)
            for idx, x_h, x_p, x_r in tqdm(it, total=len(todo), unit="clip"):
                if x_h is None:
                    errors.append((names[idx], x_r))
                    continue
                h5["X_h"][idx] = x_h
                h5["X_p"][idx] = x_p
                h5["X_raw"][idx] = x_r
                h5["done"][idx] = True

    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 2**30:.2f} GB)")
    if errors:
        print(f"{len(errors)} FAILED:")
        for name, msg in errors[:10]:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(C.TRAIN_CSV))
    ap.add_argument("--audio-dir", default=str(C.AUDIO_TRAIN))
    ap.add_argument("--out", default=str(C.FEATURES_H5))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from pathlib import Path
    build(Path(args.csv), Path(args.audio_dir), Path(args.out), args.limit)
