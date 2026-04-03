from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class PatternConditionedAttention(nn.Module):

    def __init__(self, d_lstm: int, n_heads: int=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_lstm = d_lstm
        self.queries = nn.ParameterList([nn.Parameter(torch.randn(d_lstm) * 0.01) for _ in range(n_heads)])
        self.d_context = n_heads * d_lstm

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = lstm_out.shape
        contexts: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for k in range(self.n_heads):
            q = self.queries[k]
            scores = torch.einsum('bld,d->bl', lstm_out, q)
            alpha = F.softmax(scores, dim=1)
            c = torch.einsum('bl,bld->bd', alpha, lstm_out)
            contexts.append(c)
            weights.append(alpha)
        context = torch.cat(contexts, dim=-1)
        attn_weights = torch.stack(weights, dim=1)
        return (context, attn_weights)

class MeanPooling(nn.Module):

    def __init__(self, d_lstm: int, n_heads: int=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_lstm = d_lstm
        self.d_context = n_heads * d_lstm

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = lstm_out.shape
        c = lstm_out.mean(dim=1)
        ctx = c.unsqueeze(1).expand(-1, self.n_heads, -1).reshape(B, -1)
        uniform = torch.ones(B, self.n_heads, L, device=lstm_out.device) / L
        return (ctx, uniform)