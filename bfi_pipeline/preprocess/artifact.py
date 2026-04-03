"""
preprocess/artifact.py — Artifact detection and rejection.

Three artifact criteria, applied per 10-second window:
  1. Amplitude > 500 µV on any channel (peak-to-peak within window).
  2. Flat-line: std < 0.5 µV for > 5 continuous seconds on any channel.
  3. Channel-wise z-score normalization via sliding 60-second window.

Output
------
- bad_mask : bool array (n_windows,) — True = artifact-contaminated window.
- eeg_norm : channel-wise z-scored EEG (same shape as input).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Constants (derived from config) ─────────────────────────────────────────
_AMP_THRESH_UV   = cfg.AMP_THRESHOLD_UV          # 500 µV
_FLAT_THRESH_UV  = 0.5                            # µV std for "flat"
_FLAT_DUR_SAMP   = int(cfg.FLATLINE_THRESH_S * cfg.TARGET_SFREQ)  # 5 s × 256 Hz = 1280
_WIN_SAMP        = int(cfg.WINDOW_S  * cfg.TARGET_SFREQ)          # 10 s window
_ZSCORE_WIN_SAMP = int(cfg.ZSCORE_WINDOW_S * cfg.TARGET_SFREQ)    # 60 s z-score window


# ─────────────────────────────────────────────────────────────────────────────
# Amplitude artifact
# ─────────────────────────────────────────────────────────────────────────────

def _amplitude_bad(window: np.ndarray) -> bool:
    """True if peak-to-peak amplitude > threshold on any channel."""
    ptp = window.max(axis=1) - window.min(axis=1)   # (n_channels,)
    return bool(np.any(ptp > _AMP_THRESH_UV))


# ─────────────────────────────────────────────────────────────────────────────
# Flat-line artifact
# ─────────────────────────────────────────────────────────────────────────────

def _flatline_bad(window: np.ndarray) -> bool:
    """
    True if any channel has std < _FLAT_THRESH_UV for > _FLAT_DUR_SAMP
    consecutive samples within the window.
    """
    n_ch, n_samp = window.shape
    for ch in range(n_ch):
        sig = window[ch]
        # Sliding std with a step-1 approach using a short kernel
        # Use stride-tricks for efficiency on 10-second windows
        if n_samp < _FLAT_DUR_SAMP:
            if sig.std() < _FLAT_THRESH_UV:
                return True
            continue

        # Compute rolling std with np.lib.stride_tricks
        shape   = (n_samp - _FLAT_DUR_SAMP + 1, _FLAT_DUR_SAMP)
        strides = (sig.strides[0], sig.strides[0])
        rolling = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)
        roll_std = rolling.std(axis=1)
        if np.any(roll_std < _FLAT_THRESH_UV):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Z-score normalization
# ─────────────────────────────────────────────────────────────────────────────

def zscore_normalize(eeg: np.ndarray, sfreq: float = cfg.TARGET_SFREQ) -> np.ndarray:
    """
    Channel-wise z-score using a sliding 60-second window.

    For efficiency the normalization statistics (mean, std) are computed on
    non-overlapping 60-second blocks. Samples within the first block use the
    global statistics of that first block.

    Parameters
    ----------
    eeg   : (n_channels, n_samples)
    sfreq : sample rate (Hz)

    Returns
    -------
    (n_channels, n_samples) float32
    """
    n_ch, n_samp = eeg.shape
    block_size   = int(cfg.ZSCORE_WINDOW_S * sfreq)
    out          = np.empty_like(eeg, dtype=np.float32)

    for start in range(0, n_samp, block_size):
        end   = min(start + block_size, n_samp)
        block = eeg[:, start:end]
        mu    = block.mean(axis=1, keepdims=True)
        sigma = block.std(axis=1, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)  # avoid division by zero
        out[:, start:end] = ((block - mu) / sigma).astype(np.float32)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_bad_windows(
    eeg: np.ndarray,
    win_size: int = _WIN_SAMP,
    step: int     = None,
) -> np.ndarray:
    """
    Return a boolean mask over non-overlapping windows of *eeg*.

    Parameters
    ----------
    eeg      : (n_channels, n_samples) — already resampled/filtered
    win_size : samples per window (default 10 s × 256 Hz)
    step     : step between windows; defaults to win_size (non-overlapping)
               for artifact detection purposes

    Returns
    -------
    bad_mask : (n_windows,) bool — True = bad/artifact window
    """
    if step is None:
        step = win_size

    n_samp   = eeg.shape[1]
    starts   = list(range(0, n_samp - win_size + 1, step))
    bad_mask = np.zeros(len(starts), dtype=bool)

    for idx, s in enumerate(starts):
        win = eeg[:, s : s + win_size]
        if _amplitude_bad(win) or _flatline_bad(win):
            bad_mask[idx] = True

    return bad_mask


def reject_and_normalize(
    eeg: np.ndarray,
    sfreq: float = cfg.TARGET_SFREQ,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Compute artifact mask on non-overlapping 10-second windows.
    2. Apply channel-wise z-score normalization.

    Returns
    -------
    eeg_norm : (n_channels, n_samples) normalized EEG
    bad_mask : (n_windows,) bool mask for downstream segmentation
    """
    bad_mask = detect_bad_windows(eeg, win_size=_WIN_SAMP, step=_WIN_SAMP)
    eeg_norm = zscore_normalize(eeg, sfreq)
    return eeg_norm, bad_mask
