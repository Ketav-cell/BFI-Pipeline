from __future__ import annotations
import numpy as np
from typing import Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_AMP_THRESH_UV = cfg.AMP_THRESHOLD_UV
_FLAT_THRESH_UV = 0.5
_FLAT_DUR_SAMP = int(cfg.FLATLINE_THRESH_S * cfg.TARGET_SFREQ)
_WIN_SAMP = int(cfg.WINDOW_S * cfg.TARGET_SFREQ)
_ZSCORE_WIN_SAMP = int(cfg.ZSCORE_WINDOW_S * cfg.TARGET_SFREQ)

def _amplitude_bad(window: np.ndarray) -> bool:
    ptp = window.max(axis=1) - window.min(axis=1)
    return bool(np.any(ptp > _AMP_THRESH_UV))

def _flatline_bad(window: np.ndarray) -> bool:
    n_ch, n_samp = window.shape
    for ch in range(n_ch):
        sig = window[ch]
        if n_samp < _FLAT_DUR_SAMP:
            if sig.std() < _FLAT_THRESH_UV:
                return True
            continue
        shape = (n_samp - _FLAT_DUR_SAMP + 1, _FLAT_DUR_SAMP)
        strides = (sig.strides[0], sig.strides[0])
        rolling = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)
        roll_std = rolling.std(axis=1)
        if np.any(roll_std < _FLAT_THRESH_UV):
            return True
    return False

def zscore_normalize(eeg: np.ndarray, sfreq: float=cfg.TARGET_SFREQ) -> np.ndarray:
    n_ch, n_samp = eeg.shape
    block_size = int(cfg.ZSCORE_WINDOW_S * sfreq)
    out = np.empty_like(eeg, dtype=np.float32)
    for start in range(0, n_samp, block_size):
        end = min(start + block_size, n_samp)
        block = eeg[:, start:end]
        mu = block.mean(axis=1, keepdims=True)
        sigma = block.std(axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-08, 1.0, sigma)
        out[:, start:end] = ((block - mu) / sigma).astype(np.float32)
    return out

def detect_bad_windows(eeg: np.ndarray, win_size: int=_WIN_SAMP, step: int=None) -> np.ndarray:
    if step is None:
        step = win_size
    n_samp = eeg.shape[1]
    starts = list(range(0, n_samp - win_size + 1, step))
    bad_mask = np.zeros(len(starts), dtype=bool)
    for idx, s in enumerate(starts):
        win = eeg[:, s:s + win_size]
        if _amplitude_bad(win) or _flatline_bad(win):
            bad_mask[idx] = True
    return bad_mask

def reject_and_normalize(eeg: np.ndarray, sfreq: float=cfg.TARGET_SFREQ) -> Tuple[np.ndarray, np.ndarray]:
    bad_mask = detect_bad_windows(eeg, win_size=_WIN_SAMP, step=_WIN_SAMP)
    eeg_norm = zscore_normalize(eeg, sfreq)
    return (eeg_norm, bad_mask)