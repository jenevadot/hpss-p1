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
      torch (CUDA)  -- manual_seed also covers CUDA, but torch.cuda.manual_seed_all
                       is called explicitly below to seed every visible GPU, not
                       just the current one -- relevant on a multi-GPU box even
                       though this project only ever targets a single device.

    NOT achievable on MPS, and deliberately not claimed: bitwise reproducibility.
    Metal's parallel reduction kernels do not guarantee a fixed summation order and
    float addition is not associative, so two identically-seeded runs diverge
    slowly, then materially once the divergence reaches argmax-based threshold
    selection and best-epoch choice. There is no MPS equivalent of
    torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG.

    CUDA does not have this limitation -- torch.use_deterministic_algorithms(True)
    plus CUBLAS_WORKSPACE_CONFIG=:4096:8 gets bitwise-reproducible runs on an
    NVIDIA GPU, but that determinism mode is NOT enabled here to keep behaviour
    identical across devices for this replication (same seeding contract,
    same "report mean+-std over seeds" discipline, regardless of accelerator).

    What seeding DOES buy: identical initial weights, identical batch order, and
    identical augmentation draws. That removes every source of run-to-run variance
    except reduction order on the accelerator, which makes seed-to-seed comparison
    meaningful and keeps within-seed reruns close (not identical). Report
    mean+-std over seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        # No-op on current PyTorch (manual_seed already covers MPS) but explicit,
        # and correct if a device-specific generator is added later.
        torch.mps.manual_seed(seed)


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
    ap.add_argument("--mixup-p", type=float, default=C.MIXUP_P,
                    help="fraction of batches to mix (1.0 = the old always-on behaviour)")
    ap.add_argument("--grad-clip", type=float, default=C.GRAD_CLIP,
                    help="max gradient norm; 0 disables")
    ap.add_argument("--alpha-tau", type=float, default=C.ALPHA_TAU,
                    help="fusion sigmoid temperature; <1 sharpens toward 0/1")
    ap.add_argument("--alpha-lr-mult", type=float, default=C.ALPHA_LR_MULT,
                    help="lr multiplier for the fusion weight only")
    ap.add_argument("--width", type=float, default=1.0,
                    help="channel multiplier; 1.41 makes a single stream match "
                         "dual's parameter count (capacity control)")
    ap.add_argument("--h5", default=None,
                    help="features file; use per-kernel files for the HPSS sweep")
    args = ap.parse_args()

    set_seed(args.seed)
    # Applied to the config module so Stream() picks it up at construction time.
    C.STEM_POOL = args.stem_pool

    print(f"building datasets for arm={args.arm} ...")
    train_ds, dev_ds, val_ds, norm = build_datasets(arm=args.arm, h5_path=args.h5)
    print(f"norm (train-split only): mean={norm[0]:.3f} std={norm[1]:.3f}")

    # Report the separation the features were actually built with. Training on a
    # per-kernel file while believing it is the default is the failure mode the
    # HPSS sweep is most exposed to, so surface it rather than trusting the filename.
    import h5py
    with h5py.File(str(args.h5 or C.FEATURES_H5), "r") as h5:
        feat_kernel = h5.attrs.get("hpss_kernel", "unstamped (pre-sweep file)")
    print(f"features: {args.h5 or C.FEATURES_H5}  hpss_kernel={feat_kernel}")

    if dev_ds is None:
        print("WARNING: split file has no 'dev' key -- falling back to selecting on "
              "val, which makes val metrics optimistically biased. "
              "Run `python -m src.splits` to create the three-way split.")

    model = HSPPNet(arm=args.arm, per_class_fusion=not args.scalar_fusion,
                    alpha_tau=args.alpha_tau, width=args.width)
    fusion = "scalar" if args.scalar_fusion else "per-class"
    print(f"model params: {count_parameters(model):,}  "
          f"stem_pool={args.stem_pool}  fusion={fusion}  "
          f"alpha_tau={args.alpha_tau}  width={args.width}")

    out_dir = C.RUNS_DIR / (args.tag or f"{args.arm}_seed{args.seed}")
    train(
        model, train_ds, val_ds, arm=args.arm, dev_ds=dev_ds,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        num_workers=args.num_workers, patience=args.patience,
        loss_name=args.loss, use_mixup=not args.no_mixup,
        mixup_p=args.mixup_p, grad_clip=args.grad_clip,
        alpha_lr_mult=args.alpha_lr_mult,
        lr_schedule=args.lr_schedule, seed=args.seed, out_dir=out_dir,
        hpss_kernel=feat_kernel,
    )
    print(f"\nartifacts in {out_dir}")


if __name__ == "__main__":
    main()
