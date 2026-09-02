"""Losses for long-tailed multi-label classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric loss (Ridnik et al.) for long-tailed multi-label problems.

    Designed for exactly this regime. The negative-side focusing term plus
    probability clipping discards easy negatives without needing per-class
    weights -- which matters here because inverse-frequency pos_weight would give
    LEPFLA a factor of ~8,900 and destabilise training.

    Note on BCE as the default: start with plain BCEWithLogitsLoss and only reach
    for this once a correct baseline exists.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_neg = p
        if self.clip > 0:
            # Shift negative probabilities down; fully-easy negatives contribute nothing.
            p_neg = (p - self.clip).clamp(min=0)

        loss_pos = targets * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps))

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.no_grad():
                w_pos = (1 - p) ** self.gamma_pos
                w_neg = p_neg ** self.gamma_neg
            loss_pos = loss_pos * w_pos
            loss_neg = loss_neg * w_neg

        return -(loss_pos + loss_neg).sum(dim=1).mean()


def build_loss(name: str = "bce", pos_weight: torch.Tensor | None = None) -> nn.Module:
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    raise ValueError(f"unknown loss {name!r}; expected 'bce' or 'asl'")


def clamped_pos_weight(support: torch.Tensor, n_total: int,
                       max_weight: float = 20.0) -> torch.Tensor:
    """Inverse-frequency weights, clamped.

    Unclamped, the rarest class here would get ~8,900x and blow up training.
    """
    support = support.clamp(min=1)
    w = (n_total - support) / support
    return w.clamp(max=max_weight)
