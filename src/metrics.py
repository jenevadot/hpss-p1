"""Multi-label metrics. Accuracy is deliberately absent.

36% of AnuraSet clips carry zero labels, so a model predicting "nothing, ever"
scores extremely well on accuracy and on subset/exact-match. Those metrics are
actively misleading here and are not implemented.

Headline numbers are macro-averaged over EVAL_CLASSES (the 34 species with >=100
positives). SCIFUS and SCINAS have no positives at all, so their AP is undefined;
including them would silently drag macro-mAP down by 4.8% for no reason.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from . import config as C


def per_class_ap(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AP per species; NaN where the class has no positives in y_true."""
    out = {}
    for i, name in enumerate(C.SPECIES):
        col = y_true[:, i]
        out[name] = (float(average_precision_score(col, y_score[:, i]))
                     if col.sum() > 0 else float("nan"))
    return out


def summarise(y_true: np.ndarray, y_score: np.ndarray,
              thresholds: np.ndarray | None = None) -> dict:
    """Headline metrics over EVAL_CLASSES, plus the tail reported separately."""
    aps = per_class_ap(y_true, y_score)
    eval_aps = [aps[s] for s in C.EVAL_CLASSES if not np.isnan(aps[s])]

    if thresholds is None:
        thresholds = np.full(C.N_CLASSES, 0.5)
    y_pred = (y_score >= thresholds[None, :]).astype(int)

    idx = C.EVAL_IDX
    res = {
        "mAP": float(np.mean(eval_aps)) if eval_aps else float("nan"),
        "macro_f1": float(f1_score(y_true[:, idx], y_pred[:, idx],
                                   average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true[:, idx], y_pred[:, idx],
                                   average="micro", zero_division=0)),
        "n_eval_classes": len(eval_aps),
        "per_class_ap": aps,
    }
    tail = [aps[s] for s in C.ULTRA_RARE if not np.isnan(aps[s])]
    res["tail_mAP"] = float(np.mean(tail)) if tail else float("nan")
    return res


def tune_thresholds(y_true: np.ndarray, y_score: np.ndarray,
                    min_positives: int = 5) -> tuple[np.ndarray, list[str]]:
    """Per-class F1-maximising thresholds, fitted on VAL only.

    Tuning on test would be leakage. Classes with fewer than `min_positives`
    validation positives fall back to 0.5, since a threshold fitted on 1-4
    examples is noise; those names are returned so they can be flagged.
    """
    thresholds = np.full(C.N_CLASSES, 0.5)
    fell_back = []
    grid = np.linspace(0.01, 0.99, 99)
    for i, name in enumerate(C.SPECIES):
        col = y_true[:, i]
        if col.sum() < min_positives:
            fell_back.append(name)
            continue
        scores = [f1_score(col, (y_score[:, i] >= t).astype(int), zero_division=0)
                  for t in grid]
        thresholds[i] = float(grid[int(np.argmax(scores))])
    return thresholds, fell_back


def format_report(res: dict, support: dict[str, int] | None = None) -> str:
    lines = [
        f"mAP (macro over {res['n_eval_classes']} eval classes) : {res['mAP']:.4f}",
        f"macro-F1                                : {res['macro_f1']:.4f}",
        f"micro-F1                                : {res['micro_f1']:.4f}",
        f"tail mAP (ultra-rare, <100 pos)         : {res['tail_mAP']:.4f}",
        "",
        f"{'species':10s} {'support':>8s} {'AP':>7s}",
    ]
    ranked = sorted(
        C.SPECIES,
        key=lambda s: -(support.get(s, 0) if support else 0),
    )
    for s in ranked:
        ap = res["per_class_ap"][s]
        sup = support.get(s, 0) if support else 0
        flag = ""
        if s in C.ZERO_POSITIVE:
            flag = "  (zero positives - excluded)"
        elif s in C.ULTRA_RARE:
            flag = "  (ultra-rare - excluded)"
        lines.append(f"{s:10s} {sup:>8,} {ap:>7.4f}{flag}"
                     if not np.isnan(ap) else
                     f"{s:10s} {sup:>8,} {'n/a':>7s}{flag}")
    return "\n".join(lines)
