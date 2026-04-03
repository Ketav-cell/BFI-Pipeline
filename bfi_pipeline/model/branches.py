"""
model/branches.py — Pattern-specific and global encoder branches.

Each branch: Linear(d_in → d_proj) → ReLU → Dropout(branch_dropout)

Four branches:
  • PatternEncoder(d1) → (batch, L, d_proj)   for Pattern 1
  • PatternEncoder(d2) → (batch, L, d_proj)   for Pattern 2
  • PatternEncoder(d3) → (batch, L, d_proj)   for Pattern 3
  • PatternEncoder(d4) → (batch, L, d_proj)   for Global
"""

from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class PatternEncoder(nn.Module):
    """
    Single branch encoder: FC → LayerNorm → ReLU → Dropout.

    Parameters
    ----------
    d_in      : input feature dimension
    d_proj    : output projection dimension
    dropout   : dropout probability
    """

    def __init__(self, d_in: int, d_proj: int = cfg.D_PROJ, dropout: float = cfg.BRANCH_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_proj),
            nn.LayerNorm(d_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, L, d_in)

        Returns
        -------
        (batch, L, d_proj)
        """
        return self.net(x)


class BranchEncoders(nn.Module):
    """
    Four encoders (p1, p2, p3, global) returning a concatenated embedding.

    Parameters
    ----------
    d_p1, d_p2, d_p3, d_global : input dimensions per pattern channel
    d_proj                      : output dim per branch
    dropout                     : branch dropout
    """

    def __init__(
        self,
        d_p1:    int,
        d_p2:    int,
        d_p3:    int,
        d_global: int,
        d_proj:  int   = cfg.D_PROJ,
        dropout: float = cfg.BRANCH_DROPOUT,
    ):
        super().__init__()
        self.enc_p1  = PatternEncoder(d_p1,    d_proj, dropout)
        self.enc_p2  = PatternEncoder(d_p2,    d_proj, dropout)
        self.enc_p3  = PatternEncoder(d_p3,    d_proj, dropout)
        self.enc_gl  = PatternEncoder(d_global, d_proj, dropout)

        self.d_concat = 4 * d_proj  # total concatenated dimension

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        gl: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        p1, p2, p3, gl : each (batch, L, d_pX)

        Returns
        -------
        (batch, L, 4 * d_proj)
        """
        z1 = self.enc_p1(p1)
        z2 = self.enc_p2(p2)
        z3 = self.enc_p3(p3)
        zg = self.enc_gl(gl)
        return torch.cat([z1, z2, z3, zg], dim=-1)   # (batch, L, 4*d_proj)
