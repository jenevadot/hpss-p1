"""Assemble the ablation table, per-class AP report, and complexity comparison.

  python -m src.analyze                 # ablation table across runs/
  python -m src.analyze --complexity    # params/FLOPs vs the paper's Table 5
  python -m src.analyze --per-class dual_seed42
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .metrics import format_report
from .model import HSPPNet, count_parameters

ARM_LABELS = {
    "raw": "single stream, raw mel (no HPSS)",
    "harmonic": "harmonic stream only",
    "percussive": "percussive stream only",
    "dual": "dual-stream + dual attention",
}
# The paper's Table 1 accuracies on DCASE 2020, for shape comparison only. Our
# task and metric differ (multi-label mAP vs single-label accuracy), so only the
# ORDERING is expected to transfer, not the values.
PAPER_ACC = {"raw": 69.4, "harmonic": 68.2, "percussive": 67.5, "dual": 72.1}


def collect_runs() -> pd.DataFrame:
    rows = []
    for run in sorted(C.RUNS_DIR.glob("*/summary.json")):
        s = json.loads(run.read_text())
        rows.append({
            "run": run.parent.name,
            "arm": s["arm"],
            "configuration": ARM_LABELS.get(s["arm"], s["arm"]),
            "val_mAP": s["val_mAP"],
            "macro_f1_tuned": s["val_macro_f1_tuned"],
            "micro_f1_tuned": s["val_micro_f1_tuned"],
            "tail_mAP": s.get("tail_mAP", float("nan")),
            "best_epoch": s["best_epoch"],
            "fusion_a": s.get("fusion_a", float("nan")),
            "paper_acc_pct": PAPER_ACC.get(s["arm"]),
        })
    return pd.DataFrame(rows)


def ablation_table() -> str:
    df = collect_runs()
    if df.empty:
        return "no completed runs found under runs/*/summary.json"

    order = ["raw", "harmonic", "percussive", "dual"]
    df["_o"] = df["arm"].map({a: i for i, a in enumerate(order)}).fillna(99)
    df = df.sort_values(["_o", "run"]).drop(columns="_o")

    lines = [
        "Ablation ladder (AnuraSet, multi-label; headline metrics over "
        f"{len(C.EVAL_CLASSES)} species with >=100 positives)",
        "",
        f"{'configuration':34s} {'mAP':>7s} {'macroF1':>8s} {'microF1':>8s} "
        f"{'tailmAP':>8s} {'ep':>4s} {'a':>6s} {'paper%':>7s}",
    ]
    for _, r in df.iterrows():
        a = f"{r['fusion_a']:.3f}" if not np.isnan(r["fusion_a"]) else "-"
        paper = f"{r['paper_acc_pct']:.1f}" if r["paper_acc_pct"] else "-"
        lines.append(
            f"{r['configuration']:34s} {r['val_mAP']:7.4f} {r['macro_f1_tuned']:8.4f} "
            f"{r['micro_f1_tuned']:8.4f} {r['tail_mAP']:8.4f} {int(r['best_epoch']):4d} "
            f"{a:>6s} {paper:>7s}"
        )
    lines += [
        "",
        "paper% is the paper's DCASE-2020 single-label ACCURACY, shown for ordering",
        "comparison only -- it is not comparable in value to multi-label mAP.",
    ]
    return "\n".join(lines)


def complexity_table() -> str:
    """Params and FLOPs per arm, cross-checked against the paper's Table 5.

    Finding: the paper's 2.86 GFLOPs is arithmetically UNREACHABLE with its stated
    architecture unless it downsamples aggressively before the first block. With
    128 mels x 431 frames (10 s) and pooling after each block as described, the
    first block alone costs ~6.9 G -- already over the reported total. Searching
    stem strides at 431 frames:

        no stem downsample (128x431)  37.13 G
        stem /2            (64x216)    9.37 G
        stem /4            (32x108)    2.32 G   <- closest to the reported 2.86 G
        stem /8            (16x54)     0.56 G

    So the paper must downsample by ~/4 before block 1. Our schedule pools AFTER
    each block instead, which is the literal reading of the text, and costs 13x
    more. Reported here rather than silently reconciled.
    """
    lines = [f"{'arm':12s} {'params':>12s} {'GFLOPs':>9s}"]
    try:
        from thop import profile
    except ImportError:
        return "thop not installed; run: uv pip install thop"

    import torch
    for arm in HSPPNet.ARMS:
        model = HSPPNet(arm=arm)
        x = torch.randn(1, 1, C.N_MELS, C.N_FRAMES)
        inputs = (x, x) if arm == "dual" else (x,)
        flops, _ = profile(model, inputs=inputs, verbose=False)
        lines.append(f"{arm:12s} {count_parameters(model):>12,} {flops / 1e9:>9.3f}")

    lines += [
        "",
        "Two discrepancies with the paper's Table 5 (12.4M params / 2.86 GFLOPs):",
        "",
        "1. PARAMS. Our dual arm measures ~15.8M. The stated channel progression",
        "   (64->64->128->256->512 -- five numbers for four blocks) does not",
        "   reconcile with 12.4M under any reading tried: 256-end=4.2M,",
        "   5 blocks=16.1M, ci->co on both convs=10.5M.",
        "",
        "2. FLOPS. 2.86 G at 10 s input is unreachable with pooling placed after",
        "   each block: block 1 alone at 128x431 costs ~6.9 G. A ~/4 stem",
        "   downsample before block 1 gives 2.32 G, the closest match, so the",
        "   paper must reduce resolution earlier than its text describes.",
        f"   Ours at 3 s / {C.N_FRAMES} frames: 11.2 G. Adding a /2 stem pool would",
        "   give 2.78 G, essentially the paper's figure at ~4x less compute.",
        "",
        "We report measured values rather than tuning the architecture to match",
        "numbers that are not internally consistent.",
    ]
    return "\n".join(lines)


def per_class(run_name: str) -> str:
    summary = json.loads((C.RUNS_DIR / run_name / "summary.json").read_text())
    df = pd.read_csv(C.TRAIN_CSV)
    support = {s: int(df[s].sum()) for s in C.SPECIES}
    res = {
        "mAP": summary["val_mAP"],
        "macro_f1": summary["val_macro_f1_tuned"],
        "micro_f1": summary["val_micro_f1_tuned"],
        "tail_mAP": summary.get("tail_mAP", float("nan")),
        "n_eval_classes": len(C.EVAL_CLASSES),
        "per_class_ap": summary["per_class_ap"],
    }
    return format_report(res, support)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", action="store_true")
    ap.add_argument("--per-class", metavar="RUN")
    args = ap.parse_args()

    if args.complexity:
        print(complexity_table())
    elif args.per_class:
        print(per_class(args.per_class))
    else:
        print(ablation_table())
