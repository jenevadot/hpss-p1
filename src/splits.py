"""Grouped, multi-label-stratified train/val split with hard leakage assertions.

This is the highest-risk component in the project. Adjacent 3 s segments come
from the same ~58-segment parent recording and are near-duplicates: same
individual frog, same background, seconds apart. A random row split leaks val
into train, reports macro-F1 around 0.85+, and means nothing. It fails silently
AND flatteringly, which is the worst combination.

Protocol:
  - group on parent recording (site_date_time), 1,074 groups
  - reduce each group to one stratum label = its RAREST species (globally), or
    "NEG" if the group has no positives. This deliberately biases the stratifier
    toward protecting tail classes.
  - StratifiedGroupKFold, one fold held out as val
  - assert group and filename disjointness before returning
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C


def load_labels(csv_path=None) -> pd.DataFrame:
    df = pd.read_csv(csv_path or C.TRAIN_CSV)
    missing = [s for s in C.SPECIES if s not in df.columns]
    if missing:
        raise ValueError(f"train.csv is missing species columns: {missing}")
    df["group"] = df["filename"].map(C.group_key)
    df["site"] = df["filename"].str.split("_").str[0]
    return df


def group_strata(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One stratum label per group: the rarest species present anywhere in it."""
    support = df[C.SPECIES].sum(axis=0)
    # Rank species by global rarity; rarer species win when a group has several.
    rarity = {s: r for r, s in enumerate(support.sort_values().index)}

    groups, strata = [], []
    for gid, sub in df.groupby("group", sort=True):
        present = [s for s in C.SPECIES if sub[s].to_numpy().any()]
        groups.append(gid)
        strata.append(min(present, key=lambda s: rarity[s]) if present else "NEG")
    return np.asarray(groups), np.asarray(strata)


def make_split(df: pd.DataFrame, seed: int = C.SEED,
               with_dev: bool = True) -> dict[str, list[str]]:
    """Split train.csv rows into train/dev/val by parent recording.

    Three-way, because val was doing three incompatible jobs at once: early
    stopping, per-class threshold fitting (42 fitted parameters), and architecture
    selection. With no labelled test set available (test.7z has 31,187 wavs and no
    CSV), there is nothing left to detect the resulting overfit.

    Protocol:
      1. Hold out the val fold exactly as before -- the SAME fold, from the same
         StratifiedGroupKFold with the same seed. This is deliberate: 12 completed
         ablation runs were scored on it, and changing it would invalidate them.
      2. Carve dev out of the REMAINING train groups only, again group-disjoint.

    Consequence: train shrinks by ~1/6 of itself. That cost buys an honest val.

    Use dev for: threshold fitting, early stopping, and every tuning decision.
    Use val for: the final reported number, read as few times as possible.
    """
    groups, strata = group_strata(df)

    # StratifiedGroupKFold needs per-sample arrays; here each "sample" is a group.
    sgkf = StratifiedGroupKFold(n_splits=C.N_SPLIT_FOLDS, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(groups, strata, groups=groups))
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    assert not (train_groups & val_groups), "group leakage between train and val"

    if not with_dev:
        is_val = df["group"].isin(val_groups)
        split = {
            "train": df.loc[~is_val, "filename"].tolist(),
            "val": df.loc[is_val, "filename"].tolist(),
        }
        verify_split(df, split)
        return split

    # Second, independent grouped split over the train groups only. Re-stratify on
    # the train subset so dev inherits the same rarity-protecting bias.
    sub = df[df["group"].isin(train_groups)]
    sub_groups, sub_strata = group_strata(sub)
    sgkf2 = StratifiedGroupKFold(n_splits=C.N_DEV_FOLDS, shuffle=True,
                                 random_state=seed + 1)
    keep_idx, dev_idx = next(sgkf2.split(sub_groups, sub_strata, groups=sub_groups))
    dev_groups = set(sub_groups[dev_idx])
    keep_groups = set(sub_groups[keep_idx])

    assert not (dev_groups & val_groups), "dev overlaps val"
    assert not (dev_groups & keep_groups), "dev overlaps train"

    split = {
        "train": df.loc[df["group"].isin(keep_groups), "filename"].tolist(),
        "dev": df.loc[df["group"].isin(dev_groups), "filename"].tolist(),
        "val": df.loc[df["group"].isin(val_groups), "filename"].tolist(),
    }
    verify_split(df, split)
    return split


def verify_split(df: pd.DataFrame, split: dict[str, list[str]]) -> None:
    """Hard assertions over every pair of splits present.

    If val metrics look spectacular, suspect these first.
    """
    names = [n for n in ("train", "dev", "val") if n in split]
    sets = {n: set(split[n]) for n in names}
    g = df.set_index("filename")["group"]
    group_sets = {n: {g[f] for f in sets[n]} for n in names}

    total = 0
    for n in names:
        total += len(sets[n])
        assert len(sets[n]) == len(split[n]), f"{n} contains duplicate filenames"
    assert total == len(df), (
        f"splits cover {total} rows but the frame has {len(df)}")

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sets[a] & sets[b]
            assert not overlap, f"{len(overlap)} filenames appear in both {a} and {b}"
            g_overlap = group_sets[a] & group_sets[b]
            assert not g_overlap, (
                f"{len(g_overlap)} parent recording(s) span {a} and {b}: "
                f"{sorted(g_overlap)[:3]}")


def split_report(df: pd.DataFrame, split: dict[str, list[str]]) -> str:
    g = df.set_index("filename")["group"]
    names = [n for n in ("train", "dev", "val") if n in split]
    lines = [f"{'split':6s} {'clips':>7s} {'groups':>7s} {'species>=1pos':>14s}"]
    for name in names:
        sub = df[df["filename"].isin(set(split[name]))]
        n_species = int((sub[C.SPECIES].sum(axis=0) > 0).sum())
        n_groups = len({g[f] for f in split[name]})
        lines.append(f"{name:6s} {len(sub):>7,} {n_groups:>7,} {n_species:>14d}")

    # Per-class support -- classes with <5 positives cannot have a meaningfully
    # tuned threshold, so flag them here rather than discovering it later.
    for name in names:
        if name == "train":
            continue
        sub = df[df["filename"].isin(set(split[name]))]
        thin = [s for s in C.EVAL_CLASSES if sub[s].sum() < 5]
        if thin:
            lines.append(f"\nEVAL_CLASSES with <5 {name} positives "
                         f"(threshold falls back to 0.5): {thin}")
    return "\n".join(lines)


def save_split(split: dict[str, list[str]], path=None) -> None:
    path = path or (C.SPLITS_DIR / "split_grouped.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=0))


def load_split(path=None) -> dict[str, list[str]]:
    """Load the split. Callers must tolerate a missing 'dev' key (older files)."""
    path = path or (C.SPLITS_DIR / "split_grouped.json")
    return json.loads(path.read_text())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dev", action="store_true",
                    help="reproduce the original two-way train/val split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frame = load_labels()
    print(f"rows={len(frame):,}  groups={frame['group'].nunique():,}  "
          f"sites={frame['site'].nunique()}")
    result = make_split(frame, with_dev=not args.no_dev)
    print(split_report(frame, result))

    # The val fold must stay byte-identical to the one the completed ablation runs
    # were scored on, or those 12 results become incomparable. Check it explicitly.
    existing = C.SPLITS_DIR / "split_grouped.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if set(old.get("val", [])) == set(result["val"]):
            print("\nval fold UNCHANGED vs the existing split file (runs stay comparable)")
        else:
            print("\n*** WARNING: val fold CHANGED -- completed runs are no longer "
                  "comparable to new ones ***")

    out = Path(args.out) if args.out else None
    save_split(result, out)
    print(f"wrote {out or (C.SPLITS_DIR / 'split_grouped.json')}")
