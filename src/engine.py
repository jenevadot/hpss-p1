"""Training / evaluation loop for the HSPP replication on MPS."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .dataset import ARM_STREAMS, mixup
from .metrics import summarise, tune_thresholds


def pick_device() -> torch.device:
    """MPS is a backend inside standard PyTorch, not a separate package."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(train_ds, val_ds, batch_size=C.BATCH_SIZE, num_workers=0):
    """num_workers=0 is the measured-best default here.

    Features are precomputed, so a sample is just a small HDF5 read: single-process
    loading measures ~2,500 samples/s against the model's ~66 samples/s on MPS, so
    the loader is 38x faster than the GPU and workers add only spawn overhead.
    """
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        # pin_memory is a CUDA host-memory concept for async DMA over PCIe. On
        # Apple unified memory there is no such copy, so it is pure overhead.
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def _forward(model, batch, device, arm):
    *xs, y = batch
    xs = [x.to(device, non_blocking=False) for x in xs]
    y = y.to(device)
    logits = model(*xs) if arm == "dual" else model(xs[0])
    return logits, y


@torch.no_grad()
def evaluate(model, loader, device, arm, thresholds=None):
    model.eval()
    scores, targets = [], []
    for batch in loader:
        logits, y = _forward(model, batch, device, arm)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.cpu().numpy())
    y_score = np.concatenate(scores)
    y_true = np.concatenate(targets)
    return summarise(y_true, y_score, thresholds), y_true, y_score


def build_scheduler(optimiser, kind, epochs, steps_per_epoch, log=print):
    """Return (scheduler, step_granularity) where granularity is "batch" or "epoch".

    Cosine steps PER BATCH, not per epoch. Stepping per epoch would give only
    `epochs` discrete lr values (40 staircase drops at --epochs 40); per batch gives
    epochs*steps_per_epoch (33,320 at 833 batches/epoch) and a genuinely smooth
    decay. This matters most at the end of the run, where the per-epoch version
    would hold a still-too-large lr for a whole 833-step epoch.

    Warmup is folded into the same LambdaLR so there is one schedule object and no
    hand-off bug between two of them.
    """
    if kind == "none":
        log("lr schedule: none (flat lr -- expect the +-0.035 oscillation)")
        return None, None

    if kind == "plateau":
        # Steps per EPOCH on val mAP, and needs the metric passed in. mode="max"
        # because mAP is better-when-larger; the default "min" would cut the lr
        # every time the model IMPROVED.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", factor=C.PLATEAU_FACTOR,
            patience=C.PLATEAU_PATIENCE,
        )
        log(f"lr schedule: plateau (factor={C.PLATEAU_FACTOR}, "
            f"patience={C.PLATEAU_PATIENCE}, on val_mAP)")
        return sched, "epoch"

    if kind != "cosine":
        raise ValueError(f"unknown lr schedule {kind!r}; expected cosine/plateau/none")

    total_steps = epochs * steps_per_epoch
    warmup_steps = C.WARMUP_EPOCHS * steps_per_epoch
    floor = C.LR_MIN_FACTOR

    def lr_lambda(step: int) -> float:
        """Multiplier on the base lr. Linear warmup, then cosine to `floor`."""
        if warmup_steps and step < warmup_steps:
            # step+1 so the very first step is not exactly 0.0 (a zero-lr step is
            # a wasted forward/backward pass).
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)  # clamp: early stop can leave steps unused
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    sched = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    log(f"lr schedule: cosine {C.LR:g} -> {C.LR * floor:g} over {total_steps:,} steps "
        f"({epochs} epochs x {steps_per_epoch} batches), "
        f"{C.WARMUP_EPOCHS}-epoch warmup ({warmup_steps:,} steps)")
    return sched, "batch"


def train(model, train_ds, val_ds, arm="dual", *, epochs=C.MAX_EPOCHS,
          lr=C.LR, weight_decay=C.WEIGHT_DECAY, batch_size=C.BATCH_SIZE,
          num_workers=0, patience=C.EARLY_STOP_PATIENCE, loss_name="bce",
          use_mixup=True, lr_schedule=C.LR_SCHEDULE, out_dir=None, log=print):
    device = pick_device()
    model = model.to(device)
    out_dir = Path(out_dir or (C.RUNS_DIR / arm))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the train-split normalisation alongside the checkpoint. Serving must
    # apply the identical transform; recomputing it there would risk drift.
    if getattr(train_ds, "norm", None) is not None:
        (out_dir / "norm.json").write_text(json.dumps(list(train_ds.norm)))

    from .losses import build_loss
    criterion = build_loss(loss_name)

    # Weight decay applies to conv/linear WEIGHTS only, never to 1-D parameters.
    # BatchNorm's gamma/beta and every bias are excluded: decaying BN gamma toward
    # zero directly fights the network's ability to scale features, which is BN's
    # entire job. The fusion weight a_raw is also 1-D and must not be decayed --
    # decay would bias it toward sigmoid(0)=0.5 and confound the very quantity the
    # ablation measures.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    # AdamW, not Adam: Adam's `weight_decay` adds L2 to the gradient, which then
    # gets divided by the per-parameter adaptive denominator -- so the effective
    # decay ends up inversely proportional to gradient magnitude. AdamW decouples
    # the two and applies decay directly to the weight.
    optimiser = torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr)
    log(f"weight decay {weight_decay} on {len(decay)} tensors, "
        f"0.0 on {len(no_decay)} (BN/bias/fusion)")

    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size, num_workers)

    scheduler, sched_step = build_scheduler(
        optimiser, lr_schedule, epochs, len(train_loader), log=log,
    )

    log(f"arm={arm}  device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    log(f"train={len(train_ds):,} clips  val={len(val_ds):,} clips  "
        f"batch={batch_size}  loss={loss_name}  mixup={use_mixup}")

    # Cosine anneals to its floor at exactly `epochs`, so early stopping mid-cosine
    # discards the low-lr phase where convergence actually happens -- the schedule
    # and the stopper want opposite things. Patience is therefore relaxed to at
    # least half the run so a transient dip cannot abort the anneal. The stopper is
    # kept (not removed) purely as a runaway guard.
    if sched_step == "batch" and patience < epochs // 2:
        log(f"patience {patience} -> {epochs // 2} (cosine needs to reach its floor)")
        patience = epochs // 2

    best = {"mAP": -1.0, "epoch": -1}
    history = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, n_batches = 0.0, 0

        for batch in train_loader:
            if use_mixup:
                batch = mixup(batch)
            logits, y = _forward(model, batch, device, arm)
            loss = criterion(logits, y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            # AFTER optimiser.step(), per PyTorch's documented order. Calling it
            # before would apply epoch N+1's lr to epoch N's final update and emit
            # a warning.
            if sched_step == "batch":
                scheduler.step()
            running += float(loss.detach())
            n_batches += 1

        # Read the lr the epoch actually finished on, before any epoch-level step.
        current_lr = optimiser.param_groups[0]["lr"]

        train_time = time.perf_counter() - t0
        res, _, _ = evaluate(model, val_loader, device, arm)
        epoch_time = time.perf_counter() - t0

        # Plateau needs the metric it is monitoring; cosine already stepped per batch.
        if sched_step == "epoch":
            scheduler.step(res["mAP"])

        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_mAP": res["mAP"],
            "val_macro_f1": res["macro_f1"],
            "fusion_a": model.fusion_weight,
            "lr": current_lr,
            "train_s": round(train_time, 1),
            "epoch_s": round(epoch_time, 1),
        }
        history.append(row)
        log(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val_mAP {res['mAP']:.4f}  "
            f"macro_f1 {res['macro_f1']:.4f}  a={row['fusion_a']:.3f}  "
            f"lr={current_lr:.2e}  {epoch_time:.0f}s")

        if res["mAP"] > best["mAP"]:
            best = {"mAP": res["mAP"], "epoch": epoch}
            stale = 0
            torch.save({"model": model.state_dict(), "arm": arm, "epoch": epoch,
                        "val_mAP": res["mAP"]}, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                log(f"early stop at epoch {epoch} (best mAP {best['mAP']:.4f} @ {best['epoch']})")
                break

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Reload best weights, then fit per-class thresholds on val.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    res, y_true, y_score = evaluate(model, val_loader, device, arm)
    thresholds, fell_back = tune_thresholds(y_true, y_score)
    tuned, _, _ = evaluate(model, val_loader, device, arm, thresholds)

    log(f"\nbest epoch {ckpt['epoch']}  val mAP {res['mAP']:.4f}")
    log(f"after per-class threshold tuning: macro_f1 {res['macro_f1']:.4f} "
        f"-> {tuned['macro_f1']:.4f}")
    if fell_back:
        log(f"thresholds left at 0.5 (<5 val positives): {fell_back}")

    np.save(out_dir / "thresholds.npy", thresholds)

    # Per-class fusion weights, keyed by species. This is the object the per-class
    # fusion change exists to produce: it says which stream each species actually
    # relies on, and is directly comparable against the per-class APs above.
    a_vec = model.fusion_weights()
    fusion_per_class = (dict(zip(C.SPECIES, a_vec)) if len(a_vec) == len(C.SPECIES)
                        else None)
    if fusion_per_class:
        ranked = sorted(fusion_per_class.items(), key=lambda kv: kv[1])
        log("\nfusion a per class (low = percussive-leaning, high = harmonic-leaning)")
        log("  most percussive: " + ", ".join(f"{s} {v:.3f}" for s, v in ranked[:5]))
        log("  most harmonic  : " + ", ".join(f"{s} {v:.3f}" for s, v in ranked[-5:]))

    (out_dir / "summary.json").write_text(json.dumps({
        "arm": arm, "best_epoch": ckpt["epoch"], "val_mAP": res["mAP"],
        "val_macro_f1_at_0.5": res["macro_f1"],
        "val_macro_f1_tuned": tuned["macro_f1"],
        "val_micro_f1_tuned": tuned["micro_f1"],
        "tail_mAP": res["tail_mAP"],
        "fusion_a": model.fusion_weight,
        "fusion_a_per_class": fusion_per_class,
        "stem_pool": C.STEM_POOL,
        "per_class_fusion": getattr(model, "per_class_fusion", False),
        "lr_schedule": lr_schedule,
        "lr_final": history[-1]["lr"] if history else None,
        "epochs_run": len(history),
        "per_class_ap": res["per_class_ap"],
        "threshold_fallbacks": fell_back,
    }, indent=2))
    return model, history, tuned
