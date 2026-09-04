"""Dual-stream asymmetric-convolution CNN with dual-attention fusion (HSPP).

Faithful to Liu & Fan (Sci Reports 2026) up to the classifier head, which is
converted from softmax/CE (single-label, 10 scenes) to raw logits + BCE
(multi-label, 42 species).

Interpretation note on the dual attention (plan section 2c): the paper's Eq. 2
fuses 512-d post-GAP vectors, but "spatial attention" can only act on a feature
map -- w_s cannot apply to f_h. The coherent reading, implemented here, is
spatial attention on the block-4 map BEFORE global pooling, and channel
attention on the pooled vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config as C


class AsymConvBlock(nn.Module):
    """Cascaded 3x5 then 5x3 convolution, each followed by BN + ReLU.

    Tensor layout is (B, C, n_mels, n_frames), so dim 2 is FREQUENCY and dim 3 is
    TIME. Therefore:
      (3, 5) -- 3 along frequency, 5 along time: a TIME-extended kernel that sees
                onset-sustain-decay dynamics.
      (5, 3) -- 5 along frequency, 3 along time: a FREQUENCY-extended kernel that
                sees harmonic overtone spacing.
    Together they cover an effective 7x7 field using 30 multiply-accumulates per
    position instead of 49, plus an extra nonlinearity in between.

    Only partially factorised (3x5, not 1x5) on purpose: full separability would
    lose the joint T-F sensitivity that diagonal frequency sweeps need.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, (3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class Stream(nn.Module):
    """One branch: optional stem pool, then 4 asymmetric blocks, 64 -> 512 channels.

    MaxPool after the first three blocks only. Block 4 deliberately keeps its
    spatial map so spatial attention has a non-degenerate plane to work on.

    The stem pool (config.STEM_POOL) downsamples BEFORE block 1, which is where
    FLOPs are concentrated -- block 1 runs at full input resolution. With
    STEM_POOL=2 the dual arm costs 2.78 G instead of 11.2 G. See config.STEM_POOL
    for why the paper's reported FLOPs imply it does something equivalent.

    Shapes at 128x130 input:
      STEM_POOL=1:  block1 128x130 -> ... -> block4 map 16x16
      STEM_POOL=2:  block1  64x65  -> ... -> block4 map  8x8
    """

    def __init__(self, channels=(64, 128, 256, 512), stem_pool: int | None = None):
        super().__init__()
        # Resolved at CALL time, not import time. A `stem_pool=C.STEM_POOL` default
        # would bind the value once when this module is first imported, so
        # `--stem-pool` (which assigns to C.STEM_POOL in train.py) would be silently
        # ignored and every run would use the import-time value.
        if stem_pool is None:
            stem_pool = C.STEM_POOL
        layers, c_in = [], 1
        if stem_pool > 1:
            layers.append(nn.MaxPool2d(stem_pool))
        for i, c_out in enumerate(channels):
            layers.append(AsymConvBlock(c_in, c_out))
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            c_in = c_out
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class SpatialAttention(nn.Module):
    """Channel-wise avg+max pooling -> 7x7 conv -> sigmoid gate over the T-F plane."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


class ChannelAttention(nn.Module):
    """GAP + GMP through a SHARED MLP -> sigmoid gate over channels (CBAM-style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels),
        )

    def forward(self, feat_map):
        avg = self.mlp(feat_map.mean(dim=(2, 3)))
        mx = self.mlp(feat_map.amax(dim=(2, 3)))
        return torch.sigmoid(avg + mx)


class HSPPNet(nn.Module):
    """The full framework.

    arm:
      "dual"       -- both streams + dual-attention fusion (the paper's method)
      "harmonic"   -- harmonic stream only
      "percussive" -- percussive stream only
      "raw"        -- single stream on the un-decomposed mel (no HPSS baseline)

    Fusion (dual arm only) has two forms, selected by config.PER_CLASS_FUSION:

      scalar (the paper)   f = a*f_h + (1-a)*f_p, then head(f)
      per-class (default)  a*head(f_h) + (1-a)*head(f_p), a of shape (n_classes,)

    The per-class form exists because the scalar one measurably failed: dual scored
    below percussive-only, and species with opposite stream preferences (BOALUN
    wants percussive 0.94/0.45, SCIPER wants harmonic 0.86/0.74) cannot share one
    mixing ratio. See config.PER_CLASS_FUSION for the numbers.
    """

    ARMS = ("dual", "harmonic", "percussive", "raw")

    def __init__(self, n_classes: int = C.N_CLASSES, arm: str = "dual",
                 dropout: float = 0.5, per_class_fusion: bool | None = None,
                 alpha_tau: float | None = None, width: float = 1.0):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"arm must be one of {self.ARMS}, got {arm!r}")
        # Resolved at call time -- see the note in Stream.__init__ about import-time
        # default binding making config overrides silently ineffective.
        if per_class_fusion is None:
            per_class_fusion = C.PER_CLASS_FUSION
        if alpha_tau is None:
            alpha_tau = C.ALPHA_TAU
        if alpha_tau <= 0:
            raise ValueError(f"alpha_tau must be > 0, got {alpha_tau}")
        self.alpha_tau = float(alpha_tau)
        self.width = float(width)
        self.arm = arm
        self.dual = arm == "dual"
        self.n_classes = n_classes
        self.per_class_fusion = per_class_fusion and self.dual

        # Channel schedule, scaled by `width`. width=1.0 is the paper's
        # 64->128->256->512. Used by the capacity control (§ raw-wide): the dual arm
        # has 2x the parameters of a single stream, so "is dual better because of the
        # HPSS decomposition, or just because it is bigger?" is confounded until a
        # single-stream model is given matching capacity.
        channels = tuple(int(round(c * self.width)) for c in (64, 128, 256, 512))
        c_last = channels[-1]

        # Identical structure, independent parameters -- the paper is explicit that
        # weights are NOT shared between the harmonic and percussive streams.
        self.stream_h = Stream(channels=channels)
        self.spatial_h = SpatialAttention()
        self.channel_h = ChannelAttention(c_last)

        if self.dual:
            self.stream_p = Stream(channels=channels)
            self.spatial_p = SpatialAttention()
            self.channel_p = ChannelAttention(c_last)
            # Learnable fusion weight a in [0,1], stored raw and squashed with
            # sigmoid. Init 0 -> a=0.5 (unbiased). Never clamp: clamping gives zero
            # gradient at the boundary and the parameter dies.
            #
            # Shape (n_classes,) when per_class_fusion, else (1,). Per-class is the
            # fix for the measured failure of the scalar version -- see the class
            # docstring and config.PER_CLASS_FUSION.
            n_a = n_classes if self.per_class_fusion else 1
            self.a_raw = nn.Parameter(torch.zeros(n_a))

        self.head = nn.Sequential(
            nn.Linear(c_last, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def _alpha(self) -> torch.Tensor:
        """Fusion weight(s) in (0,1). Single definition, used everywhere.

        Temperature is applied here so `forward`, `fusion_weight` and
        `fusion_weights()` cannot disagree -- a mismatch would make the logged alpha
        a different quantity from the one actually used to fuse.
        """
        return torch.sigmoid(self.a_raw / self.alpha_tau)

    @property
    def fusion_weight(self) -> float:
        """Mean value of a -- kept scalar so history.json stays comparable.

        Tells you whether the task leans harmonic (tonal calls) or percussive.
        With per-class fusion this is the average over classes; use
        fusion_weights() for the full per-class vector, which is the interesting
        object (different species genuinely prefer different streams).
        """
        if not self.dual:
            return float("nan")
        return self._alpha().mean().item()

    def fusion_weights(self) -> list[float]:
        """Per-class a values, aligned with config.SPECIES. Length 1 if scalar."""
        if not self.dual:
            return []
        return self._alpha().detach().cpu().flatten().tolist()

    def _branch(self, x, stream, spatial, channel):
        m = stream(x)              # (B, 512, H, W)
        m = m * spatial(m)         # spatial gate on the feature map
        f = m.mean(dim=(2, 3))     # GAP -> (B, 512)
        return f * channel(m)      # channel gate on the pooled vector

    def forward(self, x_h, x_p=None):
        """Returns raw logits. NO sigmoid here -- BCEWithLogitsLoss applies it."""
        f = self._branch(x_h, self.stream_h, self.spatial_h, self.channel_h)
        if not self.dual:
            return self.head(f)

        if x_p is None:
            raise ValueError("dual arm requires the percussive input x_p")
        f_p = self._branch(x_p, self.stream_p, self.spatial_p, self.channel_p)
        a = self._alpha()

        if self.per_class_fusion:
            # LOGIT-level fusion. A per-class weight cannot act on the pre-head
            # 512-d vector because the class dimension does not exist yet, so the
            # SHARED head is applied to each stream and a (n_classes,) weight
            # broadcasts over the (B, n_classes) logits.
            #
            # The head is shared, not duplicated: two heads would add ~11k params
            # and let each stream drift to a different logit scale, which would
            # make a uninterpretable as a mixing ratio.
            return a * self.head(f) + (1.0 - a) * self.head(f_p)

        # Scalar fusion: the paper's formulation, fused before the head.
        return self.head(a * f + (1.0 - a) * f_p)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
