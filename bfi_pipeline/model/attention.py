"""
model/attention.py — Pattern-conditioned attention mechanism.

Three independent attention heads, one per pattern.
Each head has its own learned query vector q_k ∈ R^{d_lstm}.

For head k:
  e_{t,k} = q_k · h_t                   (dot-product scoring)
  α_{t,k} = softmax_t(e_{t,k})          (attention weights over time)
  c_k     = Σ_t α_{t,k} · h_t          (context vector)

The three context vectors are concatenated:
  c = [c_1 || c_2 || c_3]  ∈ R^{3 * d_lstm}

Inputs
------
lstm_out : (batch, L, d_lstm)  — output of StackedBiLSTM

Returns
-------
context     : (batch, 3 * d_lstm)
attn_weights: (batch, 3, L)     — for interpretability / logging
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class PatternConditionedAttention(nn.Module):
    """
    Three pattern-specific additive attention heads.

    Parameters
    ----------
    d_lstm   : dimension of LSTM output (= 2 * d_hidden)
    n_heads  : number of patterns (default 3)
    """

    def __init__(self, d_lstm: int, n_heads: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.d_lstm  = d_lstm

        # One query vector per pattern head
        self.queries = nn.ParameterList([
            nn.Parameter(torch.randn(d_lstm) * 0.01)
            for _ in range(n_heads)
        ])
        self.d_context = n_heads * d_lstm

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        lstm_out : (batch, L, d_lstm)

        Returns
        -------
        context      : (batch, n_heads * d_lstm)
        attn_weights : (batch, n_heads, L)
        """
        B, L, D = lstm_out.shape

        contexts: list[torch.Tensor] = []
        weights:  list[torch.Tensor] = []

        for k in range(self.n_heads):
            q = self.queries[k]                          # (D,)
            # Scores: (batch, L)
            scores = torch.einsum("bld,d->bl", lstm_out, q)
            alpha  = F.softmax(scores, dim=1)            # (batch, L)
            # Context: (batch, D)
            c = torch.einsum("bl,bld->bd", alpha, lstm_out)
            contexts.append(c)
            weights.append(alpha)

        context      = torch.cat(contexts, dim=-1)       # (batch, n_heads * D)
        attn_weights = torch.stack(weights, dim=1)       # (batch, n_heads, L)

        return context, attn_weights


class MeanPooling(nn.Module):
    """
    Drop-in replacement for PatternConditionedAttention that uses mean
    pooling (used in ablation study).
    """

    def __init__(self, d_lstm: int, n_heads: int = 3):
        super().__init__()
        self.n_heads   = n_heads
        self.d_lstm    = d_lstm
        self.d_context = n_heads * d_lstm

    def forward(
        self, lstm_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = lstm_out.shape
        c   = lstm_out.mean(dim=1)                       # (batch, D)
        ctx = c.unsqueeze(1).expand(-1, self.n_heads, -1).reshape(B, -1)
        uniform = torch.ones(B, self.n_heads, L, device=lstm_out.device) / L
        return ctx, uniform
