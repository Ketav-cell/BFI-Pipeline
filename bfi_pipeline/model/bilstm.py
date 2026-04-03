"""
model/bilstm.py — 2-layer stacked Bidirectional LSTM.

Input  : (batch, L, d_input)
Output : (batch, L, 2 * d_hidden)  — outputs for all time steps

Dropout is applied between LSTM layers (PyTorch's built-in inter-layer dropout)
and on the final output sequence before attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


class StackedBiLSTM(nn.Module):
    """
    2-layer stacked Bidirectional LSTM.

    Parameters
    ----------
    d_input   : input feature dimension (= 4 * d_proj from BranchEncoders)
    d_hidden  : hidden units per direction per layer
    n_layers  : number of LSTM layers
    dropout   : dropout between LSTM layers and on output
    """

    def __init__(
        self,
        d_input:  int,
        d_hidden: int   = cfg.D_HIDDEN,
        n_layers: int   = cfg.N_LAYERS,
        dropout:  float = cfg.DROPOUT,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(p=dropout)
        self.d_out = 2 * d_hidden  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, L, d_input)

        Returns
        -------
        (batch, L, 2 * d_hidden)
        """
        out, _ = self.lstm(x)           # (batch, L, 2*d_hidden)
        out    = self.out_dropout(out)
        return out
