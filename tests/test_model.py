"""Verification tests. Run these BEFORE any real training run.

  pytest tests/ -s          (or: .venv/bin/python -m pytest tests/ -s)
"""
from __future__ import annotations

import math

import numpy as np
import pytest
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


def test_same_seed_gives_identical_init_and_batch_order():
    """Seeding must fix BOTH the weights and the batch order.

    These are the two things a seed can actually guarantee on MPS (reduction order
    is still non-deterministic, so this runs on CPU tensors only). If either drifts,
    seed-to-seed comparison is meaningless and the mean+-std over seeds would be
    measuring RNG noise rather than seed sensitivity.
    """
    from src.train import set_seed

    def first_weights_and_order(seed):
        set_seed(seed)
        model = HSPPNet(arm="dual")
        w = model.stream_h.body[1].body[0].weight.detach().clone()
        gen = torch.Generator()
        gen.manual_seed(seed)
        order = list(torch.utils.data.DataLoader(
            torch.arange(500), batch_size=32, shuffle=True, generator=gen,
        ))[0]
        return w, order

    w1, o1 = first_weights_and_order(42)
    w2, o2 = first_weights_and_order(42)
    w3, o3 = first_weights_and_order(43)

    assert torch.equal(w1, w2), "same seed produced different initial weights"
    assert torch.equal(o1, o2), "same seed produced a different batch order"
    assert not torch.equal(w1, w3), "different seeds produced identical weights"
    assert not torch.equal(o1, o3), "different seeds produced identical batch order"


def test_mixup_rng_is_reproducible_and_independent():
    """A seeded rng must give identical mixes, and must not touch global numpy state."""
    from src.dataset import mixup

    def mixed(seed):
        rng = np.random.default_rng(seed)
        x = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 1, 4, 1)
        y = torch.eye(8, C.N_CLASSES)
        return mixup((x, y), rng=rng)

    a_x, a_y = mixed(0)
    b_x, b_y = mixed(0)
    c_x, _ = mixed(1)
    assert torch.equal(a_x, b_x) and torch.equal(a_y, b_y), "seeded mixup not reproducible"
    assert not torch.equal(a_x, c_x), "different seeds gave the same mix"

    # Global numpy state must be untouched by the generator path.
    np.random.seed(7)
    before = np.random.random()
    np.random.seed(7)
    mixed(123)
    assert np.random.random() == before, "mixup consumed the global numpy stream"


def test_grad_clip_bounds_the_norm():
    """Clipping must cap the norm when enabled.

    Uses an explicit threshold rather than C.GRAD_CLIP: the default is 0 (off,
    because the measured norm is ~0.13 so 1.0 never fired), but the mechanism still
    has to be correct for when --grad-clip is passed.
    """
    limit = 0.5
    model = HSPPNet(arm="harmonic")
    x = torch.randn(2, 1, C.N_MELS, C.N_FRAMES)
    # Large targets -> large loss -> large gradients, so clipping has to bite.
    loss = torch.nn.functional.mse_loss(model(x), torch.full((2, C.N_CLASSES), 1e4))
    loss.backward()

    pre = torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
    post = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                          if p.grad is not None))
    assert pre > limit, f"test is vacuous: pre-clip norm {pre:.3f} already below {limit}"
    assert post <= limit * 1.01, f"post-clip norm {post:.3f} > {limit}"


def test_alpha_temperature_sharpens_and_stays_consistent():
    """tau<1 must move alpha further from 0.5 for the same raw value.

    Also asserts forward() and fusion_weights() agree: if they applied different
    temperatures, the logged alpha would be a different quantity from the one
    actually used to fuse, and every alpha interpretation would be wrong.
    """
    raw = 0.4
    a_at = {}
    for tau in (1.0, 0.3):
        m = HSPPNet(arm="dual", alpha_tau=tau)
        with torch.no_grad():
            m.a_raw.fill_(raw)
        a_at[tau] = m.fusion_weights()[0]
        # consistency: the property, the vector, and the maths must match
        assert abs(m.fusion_weight - a_at[tau]) < 1e-6
        expected = 1.0 / (1.0 + math.exp(-raw / tau))
        assert abs(a_at[tau] - expected) < 1e-6, f"tau={tau}: {a_at[tau]} != {expected}"

    assert a_at[0.3] > a_at[1.0], "tau<1 should push alpha further from 0.5"
    assert abs(a_at[0.3] - 0.5) > abs(a_at[1.0] - 0.5)

    # A neutral raw value must still map to exactly 0.5 at any temperature.
    m = HSPPNet(arm="dual", alpha_tau=0.3)
    assert abs(m.fusion_weight - 0.5) < 1e-6, "tau must not shift the neutral init"

    with pytest.raises(ValueError):
        HSPPNet(arm="dual", alpha_tau=0.0)


def test_width_scales_capacity_and_matches_dual():
    """width=1.41 must bring a single stream within ~1% of dual's parameter count.

    This is the capacity control: dual has 2x a single stream's parameters, so
    "dual wins because HPSS decomposition helps" is confounded with "dual wins
    because it is bigger" until a single-stream model is given matching capacity.
    If this drifts, the control stops controlling and the comparison is void.
    """
    dual = count_parameters(HSPPNet(arm="dual"))
    narrow = count_parameters(HSPPNet(arm="raw", width=1.0))
    wide = count_parameters(HSPPNet(arm="raw", width=1.41))

    assert narrow < dual, "single stream should be smaller at width=1.0"
    assert abs(wide - dual) / dual < 0.01, (
        f"width=1.41 gives {wide:,} vs dual {dual:,} "
        f"({abs(wide - dual) / dual * 100:.2f}% off, want <1%)")

    # Width must propagate to the attention and head dims, not just the streams,
    # or the model would fail to build at all.
    m = HSPPNet(arm="dual", width=1.41)
    out = m(torch.randn(2, 1, C.N_MELS, C.N_FRAMES),
            torch.randn(2, 1, C.N_MELS, C.N_FRAMES))
    assert out.shape == (2, C.N_CLASSES)


def test_split_is_three_way_and_fully_disjoint():
    """train/dev/val must be pairwise disjoint by FILENAME and by RECORDING.

    Recording-level disjointness is the load-bearing one: adjacent 3 s segments from
    the same parent recording are near-duplicates, so a shared recording leaks
    regardless of filenames differing.
    """
    from src.splits import load_labels, load_split, verify_split

    df = load_labels()
    split = load_split()
    assert "dev" in split, "run `python -m src.splits` to build the three-way split"

    # verify_split raises on any overlap; call it as the primary assertion.
    verify_split(df, split)

    group_of = df.set_index("filename")["group"]
    recs = {k: {group_of[f] for f in split[k]} for k in ("train", "dev", "val")}
    assert not (recs["train"] & recs["dev"])
    assert not (recs["train"] & recs["val"])
    assert not (recs["dev"] & recs["val"])
    assert sum(len(split[k]) for k in ("train", "dev", "val")) == len(df)
    # dev must be large enough to fit 42 thresholds on.
    assert len(split["dev"]) > 5000, f"dev has only {len(split['dev'])} clips"


def test_val_fold_matches_completed_runs():
    """The val fold must not have changed when dev was carved out.

    12 ablation runs were scored on the original val fold. If adding dev perturbed
    it, those results silently stop being comparable to anything new -- the kind of
    failure that produces a confident, wrong table.
    """
    import json
    from pathlib import Path

    from src.splits import load_split

    backup = Path("data/splits/split_grouped_2way_backup.json")
    if not backup.exists():
        pytest.skip("no pre-dev backup to compare against")
    old = json.loads(backup.read_text())
    assert set(old["val"]) == set(load_split()["val"]), (
        "val fold changed -- completed runs are no longer comparable")
