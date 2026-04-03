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

    def __init__(self, class_weights: Optional[torch.Tensor]=None, aux_lambda: float=cfg.AUX_LAMBDA):
        super().__init__()
        self.aux_lambda = aux_lambda
        self.register_buffer('class_weights', class_weights)

    def forward(self, class_probs: torch.Tensor, y_true: torch.Tensor, aux_scores: torch.Tensor, aux_labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = torch.log(class_probs + 1e-30)
        ce_loss = F.nll_loss(log_probs, y_true.long(), weight=self.class_weights.to(class_probs.device) if self.class_weights is not None else None)
        aux_loss = F.binary_cross_entropy(aux_scores, aux_labels.float(), reduction='mean')
        total = ce_loss + self.aux_lambda * aux_loss
        return (total, ce_loss, aux_loss)

def compute_class_weights(labels: torch.Tensor, n_classes: int=cfg.NUM_CLASSES) -> torch.Tensor:
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return weights.float()

def derive_auxiliary_labels(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, thresholds: Optional[dict]=None) -> torch.Tensor:

    def _active(feat: torch.Tensor, key: str) -> torch.Tensor:
        last = feat[:, -1, :]
        m = last.mean(dim=-1)
        if thresholds and key in thresholds:
            thresh = thresholds[key]
        else:
            thresh = float(m.median().item())
        return (m > thresh).float()
    a1 = _active(p1, 'p1')
    a2 = _active(p2, 'p2')
    a3 = _active(p3, 'p3')
    return torch.stack([a1, a2, a3], dim=-1)