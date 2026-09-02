"""Verification tests. Run these BEFORE any real training run.

  pytest tests/ -s          (or: .venv/bin/python -m pytest tests/ -s)
"""
from __future__ import annotations

import numpy as np
import torch

from src import config as C
from src.model import HSPPNet, count_parameters


def test_shapes_all_arms():
    """Every arm maps a 128x130 input to 42 logits."""
    for arm in HSPPNet.ARMS:
        model = HSPPNet(arm=arm)
        x_h = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
        x_p = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
        out = model(x_h, x_p) if arm == "dual" else model(x_h)
        assert out.shape == (2, C.N_CLASSES), f"{arm}: got {out.shape}"


def test_param_count_in_expected_band():
    """~15.8M for the dual arm.

    The paper reports 12.4M, which does not reconcile with its stated channel
    progression (64->64->128->256->512 is five numbers for four blocks). The
    literal reading gives 15.8M; we report ours rather than tuning the
    architecture to hit a possibly-misreported number.
    """
    n = count_parameters(HSPPNet(arm="dual"))
    assert 12e6 < n < 17e6, f"dual-arm params {n:,} outside the 12-17M band"


def test_spatial_map_not_degenerate():
    """Block 4 must keep a non-trivial T-F plane for spatial attention to mean anything.

    The 8x8 floor is what caps STEM_POOL at 2: a /4 stem would leave 4x4, at which
    point a 7x7 spatial-attention conv is larger than the map it gates.
    """
    model = HSPPNet(arm="harmonic")
    feats = model.stream_h(torch.randn(1, 1, C.N_MELS, C.N_FRAMES))
    assert feats.shape[1] == 512, feats.shape
    assert min(feats.shape[2], feats.shape[3]) >= 8, f"spatial map too small: {feats.shape}"


def test_stem_pool_reduces_map_and_preserves_shape():
    """STEM_POOL halves the block-4 map per factor of 2 without changing the output."""
    from src.model import Stream

    plain = Stream(stem_pool=1)(torch.randn(1, 1, C.N_MELS, C.N_FRAMES))
    pooled = Stream(stem_pool=2)(torch.randn(1, 1, C.N_MELS, C.N_FRAMES))
    assert pooled.shape[2] == plain.shape[2] // 2, (plain.shape, pooled.shape)
    assert pooled.shape[3] == plain.shape[3] // 2, (plain.shape, pooled.shape)
    # Logits are unaffected: GAP collapses whatever spatial extent remains.
    for stem in (1, 2):
        C_stem = C.STEM_POOL
        try:
            C.STEM_POOL = stem
            out = HSPPNet(arm="dual")(torch.randn(2, 1, C.N_MELS, C.N_FRAMES),
                                      torch.randn(2, 1, C.N_MELS, C.N_FRAMES))
            assert out.shape == (2, C.N_CLASSES), f"stem={stem}: {out.shape}"
        finally:
            C.STEM_POOL = C_stem


def test_fusion_weight_starts_neutral_and_is_learnable():
    for per_class in (False, True):
        model = HSPPNet(arm="dual", per_class_fusion=per_class)
        assert abs(model.fusion_weight - 0.5) < 1e-6, f"per_class={per_class}"
        x = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
        model(x, x).sum().backward()
        assert model.a_raw.grad is not None
        assert model.a_raw.grad.abs().sum().item() > 0, f"per_class={per_class}"


def test_per_class_fusion_has_one_weight_per_class():
    """The whole point: 42 independent mixing ratios, not one shared scalar.

    Also asserts every class receives its OWN gradient. If the weights were
    accidentally broadcast from a scalar, gradients would be identical across
    classes and the fix would be cosmetic.
    """
    scalar = HSPPNet(arm="dual", per_class_fusion=False)
    per_class = HSPPNet(arm="dual", per_class_fusion=True)
    assert scalar.a_raw.shape == (1,)
    assert per_class.a_raw.shape == (C.N_CLASSES,)
    assert len(per_class.fusion_weights()) == C.N_CLASSES
    # +41 params for 42 classes, and nothing else changed.
    assert count_parameters(per_class) - count_parameters(scalar) == C.N_CLASSES - 1

    x = torch.randn(4, 1, C.N_MELS, C.N_FRAMES)
    # Weight one class heavily so its gradient must differ from the others'.
    target = torch.zeros(4, C.N_CLASSES)
    target[:, 3] = 1.0
    (per_class(x, x) * target).sum().backward()
    g = per_class.a_raw.grad
    assert g[3].abs().item() > 0, "the weighted class got no gradient"
    assert g.abs().std().item() > 0, "all classes share one gradient -- not per-class"


def test_no_sigmoid_in_forward():
    """Logits must be unbounded; a sigmoid here plus BCEWithLogitsLoss is a
    silent double-sigmoid bug that still trains to a plausible-looking model."""
    model = HSPPNet(arm="dual")
    with torch.no_grad():
        model.head[-1].bias.fill_(8.0)
        out = model(torch.randn(2, 1, C.N_MELS, C.N_FRAMES),
                    torch.randn(2, 1, C.N_MELS, C.N_FRAMES))
    assert out.max().item() > 1.5, f"output looks squashed: max={out.max().item()}"


def test_overfit_tiny_batch():
    """THE highest-value test: 32 samples, no aug, no dropout, ~250 steps.

    If this cannot drive loss to ~0 the architecture or loss wiring is broken,
    and no amount of hyperparameter tuning will save the real run.
    """
    torch.manual_seed(0)
    model = HSPPNet(arm="dual", dropout=0.0)
    x_h = torch.randn(32, 1, C.N_MELS, C.N_FRAMES)
    x_p = torch.randn(32, 1, C.N_MELS, C.N_FRAMES)
    y = (torch.rand(32, C.N_CLASSES) > 0.85).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = None
    for step in range(250):
        loss = criterion(model(x_h, x_p), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    final = loss.item()
    print(f"\n  overfit-32: loss {first:.4f} -> {final:.6f}")
    assert final < 0.02, f"could not overfit 32 samples (loss {final:.4f})"


def test_specaugment_preserves_shape_and_masks():
    from src.dataset import spec_augment
    x = torch.ones(1, C.N_MELS, C.N_FRAMES)
    out = spec_augment(x)
    assert out.shape == x.shape
    # Masks are random-width and may be zero, so only assert the plan's bound:
    # never enough to erase most of a 130-frame clip.
    assert C.SPEC_TIME_MASK * C.SPEC_N_MASKS < 0.3 * C.N_FRAMES


def test_mixup_keeps_targets_in_range():
    from src.dataset import mixup
    x = torch.randn(8, 1, C.N_MELS, C.N_FRAMES)
    y = (torch.rand(8, C.N_CLASSES) > 0.8).float()
    xa, xb, ym = mixup((x, x.clone(), y))
    assert xa.shape == x.shape and xb.shape == x.shape
    assert ym.min() >= 0.0 and ym.max() <= 1.0


def test_cosine_schedule_warms_up_then_anneals_to_floor():
    """The lr curve must actually be a warmup-then-cosine, not a flat line.

    A schedule that silently does nothing is the failure mode here: training would
    look normal and the +-0.035 oscillation would persist unexplained. This walks
    the whole curve and asserts its shape at every landmark.
    """
    from src.engine import build_scheduler

    epochs, steps = 10, 100
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=C.LR)
    sched, granularity = build_scheduler(opt, "cosine", epochs, steps, log=lambda *a: None)
    assert granularity == "batch"

    warmup_steps = C.WARMUP_EPOCHS * steps
    lrs = []
    for _ in range(epochs * steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    # Warmup: starts near zero, ramps linearly, peaks at exactly LR.
    assert lrs[0] < C.LR / 10, f"warmup should start small, got {lrs[0]:.2e}"
    assert lrs[0] > 0, "a zero-lr first step is a wasted pass"
    if warmup_steps:
        assert abs(max(lrs) - C.LR) < 1e-9, f"peak lr {max(lrs):.2e} != base {C.LR:.2e}"
        assert abs(lrs[warmup_steps - 1] - C.LR) < 1e-9, "peak should land at warmup end"
        mid = warmup_steps // 2
        assert abs(lrs[mid] - C.LR * (mid + 1) / warmup_steps) < 1e-9, "warmup not linear"

    # Anneal: monotonically decreasing after the peak, ending at the floor.
    post = lrs[warmup_steps:]
    assert all(b <= a + 1e-12 for a, b in zip(post, post[1:])), "anneal not monotonic"
    expected_floor = C.LR * C.LR_MIN_FACTOR
    assert lrs[-1] < C.LR / 50, f"final lr {lrs[-1]:.2e} did not decay"
    assert lrs[-1] >= expected_floor * 0.99, f"final lr {lrs[-1]:.2e} below floor"

    # Cosine is symmetric: halfway through the anneal sits near the midpoint.
    half = warmup_steps + (len(lrs) - warmup_steps) // 2
    assert abs(lrs[half] - (C.LR + expected_floor) / 2) < C.LR * 0.05


def test_schedule_none_is_flat():
    """--lr-schedule none must reproduce the paper's constant lr exactly."""
    from src.engine import build_scheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=C.LR)
    sched, granularity = build_scheduler(opt, "none", 10, 100, log=lambda *a: None)
    assert sched is None and granularity is None
    assert opt.param_groups[0]["lr"] == C.LR
