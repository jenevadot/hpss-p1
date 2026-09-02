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


def make_split(df: pd.DataFrame, seed: int = C.SEED) -> dict[str, list[str]]:
    """Split train.csv rows into train/val by parent recording."""
    groups, strata = group_strata(df)

    # StratifiedGroupKFold needs per-sample arrays; here each "sample" is a group.
    sgkf = StratifiedGroupKFold(n_splits=C.N_SPLIT_FOLDS, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(groups, strata, groups=groups))
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    assert not (train_groups & val_groups), "group leakage between train and val"

    is_val = df["group"].isin(val_groups)
    split = {
        "train": df.loc[~is_val, "filename"].tolist(),
        "val": df.loc[is_val, "filename"].tolist(),
    }
    verify_split(df, split)
    return split


def verify_split(df: pd.DataFrame, split: dict[str, list[str]]) -> None:
    """Hard assertions. If val metrics look spectacular, suspect these first."""
    tr, va = set(split["train"]), set(split["val"])
    assert not (tr & va), f"{len(tr & va)} filenames appear in both train and val"
    assert len(tr) + len(va) == len(df), "split does not cover every row exactly once"

    g = df.set_index("filename")["group"]
    tr_groups = {g[f] for f in tr}
    va_groups = {g[f] for f in va}
    assert not (tr_groups & va_groups), "parent recording spans train and val"


def split_report(df: pd.DataFrame, split: dict[str, list[str]]) -> str:
    g = df.set_index("filename")["group"]
    lines = [
        f"{'split':6s} {'clips':>7s} {'groups':>7s} {'species>=1pos':>14s}",
    ]
    for name in ("train", "val"):
        sub = df[df["filename"].isin(set(split[name]))]
        n_species = int((sub[C.SPECIES].sum(axis=0) > 0).sum())
        n_groups = len({g[f] for f in split[name]})
        lines.append(f"{name:6s} {len(sub):>7,} {n_groups:>7,} {n_species:>14d}")

    # Per-class support in val -- classes with <5 val positives cannot have a
    # meaningfully tuned threshold, so flag them here rather than discovering later.
    val = df[df["filename"].isin(set(split["val"]))]
    thin = [s for s in C.EVAL_CLASSES if val[s].sum() < 5]
    if thin:
        lines.append(f"\nEVAL_CLASSES with <5 val positives (threshold will fall back to 0.5): {thin}")
    return "\n".join(lines)


def save_split(split: dict[str, list[str]], path=None) -> None:
    path = path or (C.SPLITS_DIR / "split_grouped.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=0))


def load_split(path=None) -> dict[str, list[str]]:
    path = path or (C.SPLITS_DIR / "split_grouped.json")
    return json.loads(path.read_text())


if __name__ == "__main__":
    frame = load_labels()
    print(f"rows={len(frame):,}  groups={frame['group'].nunique():,}  sites={frame['site'].nunique()}")
    result = make_split(frame)
    print(split_report(frame, result))
    save_split(result)
    print(f"\nwrote {C.SPLITS_DIR / 'split_grouped.json'}")
