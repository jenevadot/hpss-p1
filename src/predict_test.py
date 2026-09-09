"""Batch inference on the unlabelled test set.

Emits per-class sigmoid probabilities for every clip, plus binarised predictions
using the DEV-FITTED thresholds stored in each run's summary.json.

IMPORTANT — what this can and cannot tell you:
  * `test.7z` ships NO label CSV (verified: zero filename overlap with train.csv),
    so NOTHING here can be scored locally. These are predictions, not results.
  * The test recordings ARE group-disjoint from all 1,074 training recordings
    (538 test recordings, zero overlap), so if labels are ever obtained this is a
    genuine held-out estimate.
  * Normalisation stats come from the run's own norm.json, i.e. TRAIN-split
    statistics. Recomputing them on test would be test-time leakage.

Usage:
    python -m src.predict_test --run dual_s42_3way --out preds/dual_s42.csv
    python -m src.predict_test --run raw_s42_3way  --out preds/raw_s42.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from . import config as C
from .engine import pick_device
from .model import HSPPNet

ARM_STREAMS = {"dual": ("X_h", "X_p"), "harmonic": ("X_h",),
               "percussive": ("X_p",), "raw": ("X_raw",)}


def load_run(run_dir: Path):
    summary = json.load(open(run_dir / "summary.json"))
    norm = json.load(open(run_dir / "norm.json"))
    arm = summary["arm"]

    # STEM_POOL is read from config at call time inside Stream.__init__, so it must
    # be set on the module before constructing the model -- HSPPNet takes no
    # stem_pool argument. Restore it afterwards so importing this does not mutate
    # global state for anything else in the process.
    prev_stem = C.STEM_POOL
    C.STEM_POOL = summary.get("stem_pool", C.STEM_POOL)
    try:
        model = HSPPNet(
            arm=arm,
            per_class_fusion=summary.get("per_class_fusion"),
            alpha_tau=summary.get("alpha_tau"),
            width=summary.get("width") or 1.0,
        )
    finally:
        C.STEM_POOL = prev_stem

    state = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Thresholds live in thresholds.npy (dev-fitted), NOT in summary.json.
    thr_path = run_dir / "thresholds.npy"
    thr = np.load(thr_path) if thr_path.exists() else None
    return model, arm, norm, summary, thr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under runs/")
    ap.add_argument("--h5", default="data/features_test.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run
    model, arm, norm, summary, thr = load_run(run_dir)
    device = pick_device()
    model = model.to(device)

    streams = ARM_STREAMS[arm]
    mean, std = norm
    print(f"run={args.run} arm={arm} streams={streams} "
          f"norm=({mean:.4f}, {std:.4f}) device={device}")

    with h5py.File(args.h5, "r") as h5:
        names = [n.decode() for n in h5["filenames"][:]]
        n = len(names)
        assert int(h5["done"][:].sum()) == n, "features file is incomplete"
        kernel = h5.attrs.get("hpss_kernel")
        print(f"{n:,} clips, hpss_kernel={kernel}")

        out = np.zeros((n, C.N_CLASSES), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                j = min(i + args.batch_size, n)
                xs = []
                for key in streams:
                    x = np.asarray(h5[key][i:j], dtype=np.float32)
                    x = (x - mean) / std
                    xs.append(torch.from_numpy(x).unsqueeze(1).to(device))
                out[i:j] = torch.sigmoid(model(*xs)).cpu().numpy()
                if (i // args.batch_size) % 20 == 0:
                    print(f"  {j:,}/{n:,}", flush=True)

    # Dev-fitted thresholds from thresholds.npy; 0.5 only if the file is absent.
    if thr is None:
        print("WARNING: no thresholds.npy in run dir; using 0.5 for all classes")
        thr = np.full(C.N_CLASSES, 0.5, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    assert thr.shape == (C.N_CLASSES,), f"threshold shape {thr.shape}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "filename," + ",".join(C.SPECIES) + "\n"
    with open(out_path, "w") as fh:
        fh.write(header)
        for name, row in zip(names, out):
            fh.write(name + "," + ",".join("%.6f" % v for v in row) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size / 2**20:.1f} MB)")

    binp = out_path.with_name(out_path.stem + "_binary.csv")
    with open(binp, "w") as fh:
        fh.write(header)
        for name, row in zip(names, out):
            fh.write(name + "," + ",".join(str(int(v)) for v in (row >= thr)) + "\n")
    print(f"wrote {binp}")

    pos = (out >= thr).sum()
    print(f"\npositive rate: {pos / out.size:.4f} "
          f"({pos:,} of {out.size:,} cells)")
    print(f"train-set positive rate for reference: 0.0359")
    print("\nNOTE: unscoreable locally -- test.7z ships no labels.")


if __name__ == "__main__":
    main()
