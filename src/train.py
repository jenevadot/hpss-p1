"""Train one ablation arm.

  python -m src.train --arm dual
  python -m src.train --arm raw --epochs 30
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from . import config as C
from .dataset import build_datasets
from .engine import train
from .model import HSPPNet, count_parameters


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches.

    Covered here:
      random        -- MixUp's lambda draw goes through np.random, but SpecAugment
                       and any stdlib shuffle use this one.
      numpy         -- mixup() calls np.random.beta.
      torch (CPU)   -- weight init, SpecAugment's torch.randint, DataLoader shuffle
                       when no explicit generator is passed.
      torch (MPS)   -- manual_seed already covers all devices; torch.mps has no
                       separate seed function.

    NOT achievable on MPS, and deliberately not claimed: bitwise reproducibility.
    Metal's parallel reduction kernels do not guarantee a fixed summation order and
    float addition is not associative, so two identically-seeded runs diverge
    slowly, then materially once the divergence reaches argmax-based threshold
    selection and best-epoch choice. There is no MPS equivalent of
    torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG.

    What seeding DOES buy: identical initial weights, identical batch order, and
    identical augmentation draws. That removes every source of run-to-run variance
    except MPS reduction order, which makes seed-to-seed comparison meaningful and
    keeps within-seed reruns close (not identical). Report mean+-std over seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # No-op on current PyTorch (manual_seed already covers MPS) but explicit,
        # and correct if a device-specific generator is added later.
        torch.mps.manual_seed(seed)


def seeded_generator(seed: int) -> torch.Generator:
    """Dedicated generator for DataLoader shuffling.

    Without this the loader draws from the global torch RNG, which is also consumed
    by SpecAugment inside __getitem__. Batch order would then depend on how many
    augmentation draws happened, coupling two things that should be independent --
    so changing SPEC_N_MASKS would silently change the batch order too.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dual", choices=HSPPNet.ARMS)
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--loss", default="bce", choices=["bce", "asl"])
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=C.EARLY_STOP_PATIENCE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--tag", default=None, help="run directory suffix")
    ap.add_argument("--stem-pool", type=int, default=C.STEM_POOL,
                    help="downsample factor before block 1 (1 = paper's literal reading)")
    ap.add_argument("--scalar-fusion", action="store_true",
                    help="use the paper's single global fusion weight instead of per-class")
    ap.add_argument("--lr-schedule", default=C.LR_SCHEDULE,
                    choices=["cosine", "plateau", "none"],
                    help="'none' reproduces the paper's flat lr")
    args = ap.parse_args()

    set_seed(args.seed)
    # Applied to the config module so Stream() picks it up at construction time.
    C.STEM_POOL = args.stem_pool

    print(f"building datasets for arm={args.arm} ...")
    train_ds, val_ds, norm = build_datasets(arm=args.arm)
    print(f"norm (train-split only): mean={norm[0]:.3f} std={norm[1]:.3f}")

    model = HSPPNet(arm=args.arm, per_class_fusion=not args.scalar_fusion)
    fusion = "scalar" if args.scalar_fusion else "per-class"
    print(f"model params: {count_parameters(model):,}  "
          f"stem_pool={args.stem_pool}  fusion={fusion}")

    out_dir = C.RUNS_DIR / (args.tag or f"{args.arm}_seed{args.seed}")
    train(
        model, train_ds, val_ds, arm=args.arm,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        num_workers=args.num_workers, patience=args.patience,
        loss_name=args.loss, use_mixup=not args.no_mixup,
        lr_schedule=args.lr_schedule, out_dir=out_dir,
    )
    print(f"\nartifacts in {out_dir}")


if __name__ == "__main__":
    main()
