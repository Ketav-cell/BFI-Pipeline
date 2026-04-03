from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from model.branches import BranchEncoders
from model.bilstm import StackedBiLSTM
from model.attention import PatternConditionedAttention, MeanPooling

class BFIModel(nn.Module):

    def __init__(self, d_p1: int, d_p2: int, d_p3: int, d_global: int, d_proj: int=cfg.D_PROJ, d_hidden: int=cfg.D_HIDDEN, n_layers: int=cfg.N_LAYERS, dropout: float=cfg.DROPOUT, use_attention: bool=True):
        super().__init__()
        self.branches = BranchEncoders(d_p1, d_p2, d_p3, d_global, d_proj=d_proj, dropout=cfg.BRANCH_DROPOUT)
        d_lstm_in = self.branches.d_concat
        self.bilstm = StackedBiLSTM(d_lstm_in, d_hidden, n_layers, dropout)
        d_lstm_out = self.bilstm.d_out
        if use_attention:
            self.attention = PatternConditionedAttention(d_lstm_out, n_heads=3)
        else:
            self.attention = MeanPooling(d_lstm_out, n_heads=3)
        d_context = self.attention.d_context
        self.classifier = nn.Sequential(nn.LayerNorm(d_context), nn.Dropout(dropout), nn.Linear(d_context, cfg.NUM_CLASSES))
        d_branch = d_proj
        self.aux_heads = nn.ModuleList([nn.Linear(d_branch, 1) for _ in range(3)])
        self._register_buffers(d_lstm_out)

    def _register_buffers(self, d_lstm_out: int) -> None:
        self.register_buffer('_bfi_weights', torch.tensor([0.0, cfg.BFI_PRE_WEIGHT, cfg.BFI_IMM_WEIGHT]))

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, gl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.branches(p1, p2, p3, gl)
        h = self.bilstm(z)
        context, attn_w = self.attention(h)
        logits = self.classifier(context)
        class_probs = F.softmax(logits, dim=-1)
        with torch.no_grad():
            z1 = self.branches.enc_p1(p1[:, -1, :])
            z2 = self.branches.enc_p2(p2[:, -1, :])
            z3 = self.branches.enc_p3(p3[:, -1, :])
        aux_branches = [z1, z2, z3]
        aux_scores = torch.cat([torch.sigmoid(self.aux_heads[k](aux_branches[k])) for k in range(3)], dim=-1)
        return (class_probs, aux_scores, attn_w)

    def compute_bfi(self, class_probs: torch.Tensor) -> torch.Tensor:
        w = self._bfi_weights.to(class_probs.device)
        bfi = 100.0 * (class_probs * w).sum(dim=-1)
        return bfi.clamp(0.0, 100.0)

    @torch.no_grad()
    def predict_bfi(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, gl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        class_probs, _, _ = self.forward(p1, p2, p3, gl)
        bfi = self.compute_bfi(class_probs)
        return (bfi, class_probs)

def build_model(d_p1: int, d_p2: int, d_p3: int, d_global: int, **kwargs) -> BFIModel:
    return BFIModel(d_p1=d_p1, d_p2=d_p2, d_p3=d_p3, d_global=d_global, d_proj=kwargs.get('d_proj', cfg.D_PROJ), d_hidden=kwargs.get('d_hidden', cfg.D_HIDDEN), n_layers=kwargs.get('n_layers', cfg.N_LAYERS), dropout=kwargs.get('dropout', cfg.DROPOUT), use_attention=kwargs.get('use_attention', True))