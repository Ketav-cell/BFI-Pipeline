"""
model/losses.py — Combined loss function for BFI training.

L = CrossEntropy(y_pred, y_true, weight=class_weights)
  + lambda * Σ_k BCE(aux_k, pattern_k_active)

Auxiliary pattern labels (pattern_k_active) are derived from the feature
space using simple threshold heuristics:
  • Pattern 1 active if mean alpha coherence > median of training set
  • Pattern 2 active if mean hemispheric asymmetry > threshold
  • Pattern 3 active if frontal theta/alpha ratio > threshold

In practice the thresholds are estimated per-batch for simplicity; they can
be calibrated from the training set in the Trainer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class BFILoss(nn.Module):
    """
    Combined cross-entropy + auxiliary BCE loss.

    Parameters
    ----------
    class_weights : (3,) tensor of class weights for cross-entropy
    aux_lambda    : weight for auxiliary losses
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        aux_lambda:    float = cfg.AUX_LAMBDA,
    ):
        super().__init__()
        self.aux_lambda    = aux_lambda
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        class_probs: torch.Tensor,   # (batch, 3) — softmax output
        y_true:      torch.Tensor,   # (batch,)   — integer class labels
        aux_scores:  torch.Tensor,   # (batch, 3) — sigmoid auxiliary scores
        aux_labels:  torch.Tensor,   # (batch, 3) — binary auxiliary labels
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss : scalar
        ce_loss    : cross-entropy component
        aux_loss   : auxiliary BCE component
        """
        # ── Cross-entropy on logit-equivalent (log-probs) ─────────────────
        # class_probs are already softmax → use NLLLoss on log
        log_probs = torch.log(class_probs + 1e-30)
        ce_loss = F.nll_loss(
            log_probs,
            y_true.long(),
            weight=self.class_weights.to(class_probs.device)
                   if self.class_weights is not None else None,
        )

        # ── Auxiliary BCE losses ───────────────────────────────────────────
        aux_loss = F.binary_cross_entropy(
            aux_scores,
            aux_labels.float(),
            reduction="mean",
        )

        total = ce_loss + self.aux_lambda * aux_loss
        return total, ce_loss, aux_loss


def compute_class_weights(labels: torch.Tensor, n_classes: int = cfg.NUM_CLASSES) -> torch.Tensor:
    """
    Inverse-frequency class weights.

    Parameters
    ----------
    labels    : (N,) integer tensor
    n_classes : number of classes

    Returns
    -------
    weights : (n_classes,) float tensor
    """
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        counts[c] = (labels == c).sum().float()
    # Avoid division by zero
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes   # normalize to sum to n_classes
    return weights.float()


def derive_auxiliary_labels(
    p1: torch.Tensor,   # (batch, L, D1)
    p2: torch.Tensor,   # (batch, L, D2)
    p3: torch.Tensor,   # (batch, L, D3)
    thresholds: Optional[dict] = None,
) -> torch.Tensor:
    """
    Derive binary auxiliary labels from feature values.

    Heuristics (using last time-step features):
      aux[0] (Pattern 1 active) : mean(p1_last) > median(p1_last)
      aux[1] (Pattern 2 active) : mean(p2_last) > median(p2_last)
      aux[2] (Pattern 3 active) : mean(p3_last) > median(p3_last)

    Parameters
    ----------
    p1, p2, p3  : feature tensors (batch, L, Dk)
    thresholds  : optional dict with keys "p1", "p2", "p3" → float threshold

    Returns
    -------
    aux_labels : (batch, 3) binary float tensor
    """
    def _active(feat: torch.Tensor, key: str) -> torch.Tensor:
        last = feat[:, -1, :]          # (batch, Dk)
        m    = last.mean(dim=-1)       # (batch,)
        if thresholds and key in thresholds:
            thresh = thresholds[key]
        else:
            thresh = float(m.median().item())
        return (m > thresh).float()

    a1 = _active(p1, "p1")
    a2 = _active(p2, "p2")
    a3 = _active(p3, "p3")
    return torch.stack([a1, a2, a3], dim=-1)   # (batch, 3)
