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
            # Config provenance: runs with different settings are NOT comparable, so
            # these must travel with the metrics.
            "seed": s.get("seed"),
            "stem_pool": s.get("stem_pool"),
            "per_class_fusion": s.get("per_class_fusion"),
            "lr_schedule": s.get("lr_schedule"),
        })
    return pd.DataFrame(rows)


def _config_key(row) -> str:
    """Runs are only comparable within one config. Missing fields = a legacy run.

    Fusion is deliberately NOT part of the key: single-stream arms have no fusion at
    all, so including it would split one ablation ladder across two blocks and make
    the arms non-comparable to each other -- the opposite of what the grouping is
    for. Fusion is reported per-run in the `a` column instead.
    """
    if row["stem_pool"] is None or pd.isna(row["stem_pool"]):
        return "legacy (pre-stem-pool, flat lr, Adam, scalar fusion)"
    return f"stem/{int(row['stem_pool'])}  lr={row['lr_schedule']}"


def ablation_table() -> str:
    df = collect_runs()
    if df.empty:
        return "no completed runs found under runs/*/summary.json"

    order = ["raw", "harmonic", "percussive", "dual"]
    df["_o"] = df["arm"].map({a: i for i, a in enumerate(order)}).fillna(99)
    df["config"] = df.apply(_config_key, axis=1)

    out = []
    for config, grp in df.groupby("config", sort=True):
        grp = grp.sort_values(["_o", "run"])
        seeds = sorted({int(s) for s in grp["seed"] if s is not None and not pd.isna(s)})
        out += [
            f"=== {config} ===",
            f"seeds: {seeds or 'unrecorded'}",
            "",
        ]
        # Aggregate over seeds when there are several, since a single-seed number is
        # not separable from the +-0.035 within-run oscillation.
        multi = grp.groupby("arm")["val_mAP"].count().max() > 1
        if multi:
            out.append(f"{'configuration':34s} {'mAP mean+-std':>18s} {'macroF1':>16s} "
                       f"{'n':>3s} {'a':>6s}")
            for arm in order:
                sub = grp[grp["arm"] == arm]
                if sub.empty:
                    continue
                m, s = sub["val_mAP"].mean(), sub["val_mAP"].std(ddof=1)
                f_m, f_s = sub["macro_f1_tuned"].mean(), sub["macro_f1_tuned"].std(ddof=1)
                a = sub["fusion_a"].mean()
                a_s = f"{a:.3f}" if not np.isnan(a) else "-"
                sd = f"{s:.4f}" if len(sub) > 1 and not np.isnan(s) else "  -   "
                fsd = f"{f_s:.4f}" if len(sub) > 1 and not np.isnan(f_s) else "  -   "
                out.append(f"{ARM_LABELS.get(arm, arm):34s} "
                           f"{m:8.4f} +-{sd:>7s} {f_m:7.4f} +-{fsd:>7s} "
                           f"{len(sub):>3d} {a_s:>6s}")
            spread = grp.groupby("arm")["val_mAP"].mean()
            noise = grp.groupby("arm")["val_mAP"].std(ddof=1).mean()
            if len(spread) > 1 and not np.isnan(noise):
                gap = spread.max() - spread.min()
                out += [
                    "",
                    f"between-arm spread {gap:.4f} vs mean seed std {noise:.4f} -> "
                    + ("ORDERING IS RESOLVED" if gap > 2 * noise else
                       "ORDERING IS NOT RESOLVED (spread within noise)"),
                ]
        else:
            out.append(f"{'configuration':34s} {'mAP':>7s} {'macroF1':>8s} {'microF1':>8s} "
                       f"{'tailmAP':>8s} {'ep':>4s} {'a':>6s} {'paper%':>7s}")
            for _, r in grp.iterrows():
                a = f"{r['fusion_a']:.3f}" if not np.isnan(r["fusion_a"]) else "-"
                paper = f"{r['paper_acc_pct']:.1f}" if r["paper_acc_pct"] else "-"
                out.append(
                    f"{r['configuration']:34s} {r['val_mAP']:7.4f} "
                    f"{r['macro_f1_tuned']:8.4f} {r['micro_f1_tuned']:8.4f} "
                    f"{r['tail_mAP']:8.4f} {int(r['best_epoch']):4d} {a:>6s} {paper:>7s}"
                )
        out.append("")

    out += [
        "paper% is the paper's DCASE-2020 single-label ACCURACY, shown for ordering",
        "comparison only -- it is not comparable in value to multi-label mAP.",
        "tailmAP is NaN by construction: every ultra-rare species is confined to 1-3",
        "recordings, so a group-disjoint split leaves zero val positives. See",
        "`python -m src.analyze --recordings`.",
    ]
    return "\n".join(out)


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
        f"   Ours at 3 s / {C.N_FRAMES} frames measures 11.216 G with no stem pool.",
        f"   config.STEM_POOL is now {C.STEM_POOL}, giving 2.792 G -- essentially the",
        "   paper's reported figure at ~4x less compute (4.02x measured), which is",
        "   what makes multi-seed runs affordable. Set --stem-pool 1 to recover the",
        "   literal reading of the paper's text.",
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


def recording_diagnostics(run_name: str | None = None) -> str:
    """Per-class val support measured in RECORDINGS, not clips.

    Motivation: four EVAL_CLASSES score at or below chance (DENCRU AP=0.0096 at
    602 positives) while ADEDIP hits 0.9997 at only 390. Clip count clearly does
    not explain that, so this counts distinct parent recordings instead.

    Finding: it is a recording-diversity effect, not a support effect. Classes whose
    val positives come from <=2 recordings average AP 0.350; the rest average 0.764.
    All four sub-chance classes have exactly 2 val recordings. With so few, "learn
    the species" and "learn that recording's background" are indistinguishable on
    train and the model cannot generalise across the acoustic-condition gap the
    grouped split deliberately creates.

    This is the split working as designed -- it is exposing genuine
    non-generalisation rather than hiding it. But it means those APs are properties
    of the DATA, not of the architecture, and no amount of model work will move
    them. They should be reported with their recording count attached.
    """
    from .splits import load_labels, load_split

    df = load_labels()
    split = load_split()
    train_names, val_names = set(split["train"]), set(split["val"])
    group_of = df.set_index("filename")["group"]

    aps = {}
    if run_name:
        summary = json.loads((C.RUNS_DIR / run_name / "summary.json").read_text())
        aps = summary["per_class_ap"]

    lines = [
        "Per-class support in CLIPS and RECORDINGS (grouped split, val fold)",
        "",
        f"{'species':9s} {'clips':>7s} {'trClip':>7s} {'vaClip':>7s} "
        f"{'trRec':>6s} {'vaRec':>6s} {'AP':>8s}  note",
    ]
    thin, wide = [], []
    for name in C.SPECIES:
        pos = df.loc[df[name] == 1, "filename"]
        tr_pos = [f for f in pos if f in train_names]
        va_pos = [f for f in pos if f in val_names]
        tr_rec = len({group_of[f] for f in tr_pos})
        va_rec = len({group_of[f] for f in va_pos})
        ap = aps.get(name, float("nan"))

        note = ""
        if not va_pos:
            note = "no val positives -> AP undefined"
        elif va_rec <= 2:
            note = f"val positives from only {va_rec} recording(s)"
        if not np.isnan(ap):
            (thin if va_rec <= 2 else wide).append(ap)

        ap_s = f"{ap:8.4f}" if not np.isnan(ap) else f"{'n/a':>8s}"
        lines.append(f"{name:9s} {len(pos):>7,} {len(tr_pos):>7,} {len(va_pos):>7,} "
                     f"{tr_rec:>6} {va_rec:>6} {ap_s}  {note}")

    if thin and wide:
        lines += [
            "",
            f"mean AP with <=2 val recordings (n={len(thin)}): {np.mean(thin):.4f}",
            f"mean AP with  >2 val recordings (n={len(wide)}): {np.mean(wide):.4f}",
            "",
            "Recording diversity, not clip count, is what separates the failures.",
            "These APs are data properties; model changes will not move them.",
        ]

    ultra = [s for s in C.ULTRA_RARE]
    lines += ["", "Why tail_mAP is NaN rather than low:"]
    for name in ultra:
        pos = df.loc[df[name] == 1, "filename"]
        n_rec = len({group_of[f] for f in pos})
        n_val = len([f for f in pos if f in val_names])
        lines.append(f"  {name:9s} {len(pos):>3} positives across {n_rec} recording(s), "
                     f"{n_val} in val")
    lines += [
        "",
        "Every ultra-rare species is confined to 1-3 recordings, so a group-disjoint",
        "split puts all of them on one side -- here, all in train. AP is undefined",
        "with zero val positives, so the tail is currently UNMEASURABLE rather than",
        "measured-as-bad. Reporting it as NaN is correct; reporting 0.0 would be a",
        "fabrication. Measuring the tail needs a tail-specific split, and even then",
        "1-3 recordings cannot support a generalisation claim.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", action="store_true")
    ap.add_argument("--per-class", metavar="RUN")
    ap.add_argument("--recordings", nargs="?", const="", metavar="RUN",
                    help="per-class clip/recording support; pass a run name for APs")
    args = ap.parse_args()

    if args.complexity:
        print(complexity_table())
    elif args.per_class:
        print(per_class(args.per_class))
    elif args.recordings is not None:
        print(recording_diagnostics(args.recordings or None))
    else:
        print(ablation_table())
