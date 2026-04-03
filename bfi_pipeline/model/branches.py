from __future__ import annotations
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class PatternEncoder(nn.Module):

    def __init__(self, d_in: int, d_proj: int=cfg.D_PROJ, dropout: float=cfg.BRANCH_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_proj), nn.LayerNorm(d_proj), nn.ReLU(inplace=True), nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BranchEncoders(nn.Module):

    def __init__(self, d_p1: int, d_p2: int, d_p3: int, d_global: int, d_proj: int=cfg.D_PROJ, dropout: float=cfg.BRANCH_DROPOUT):
        super().__init__()
        self.enc_p1 = PatternEncoder(d_p1, d_proj, dropout)
        self.enc_p2 = PatternEncoder(d_p2, d_proj, dropout)
        self.enc_p3 = PatternEncoder(d_p3, d_proj, dropout)
        self.enc_gl = PatternEncoder(d_global, d_proj, dropout)
        self.d_concat = 4 * d_proj

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, gl: torch.Tensor) -> torch.Tensor:
        z1 = self.enc_p1(p1)
        z2 = self.enc_p2(p2)
        z3 = self.enc_p3(p3)
        zg = self.enc_gl(gl)
        return torch.cat([z1, z2, z3, zg], dim=-1)