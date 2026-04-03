from __future__ import annotations
import torch
import torch.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

class StackedBiLSTM(nn.Module):

    def __init__(self, d_input: int, d_hidden: int=cfg.D_HIDDEN, n_layers: int=cfg.N_LAYERS, dropout: float=cfg.DROPOUT):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.out_dropout = nn.Dropout(p=dropout)
        self.d_out = 2 * d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.out_dropout(out)
        return out