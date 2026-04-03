"""
preprocess/filter.py — Resample → bandpass → notch for each patient recording.

Steps applied in order:
  1. Resample to TARGET_SFREQ (256 Hz)
  2. Bandpass filter 0.5–45 Hz (4th-order Butterworth, zero-phase)
  3. Notch filter at 50 Hz and 60 Hz (IIR notch, Q=30)

Input / output: np.ndarray of shape (n_channels, n_samples), float32.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import (
    butter,
    sosfiltfilt,
    iirnotch,
    resample_poly,
)
from math import gcd
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── Butterworth bandpass (cache filter coefficients) ────────────────────────

def _butter_bandpass_sos(
    lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    sos  = butter(order, [low, high], btype="band", output="sos")
    return sos


def _iir_notch_ba(freq: float, fs: float, Q: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    w0 = freq / (fs / 2.0)
    b, a = iirnotch(w0, Q)
    return b, a


# ─── Public functions ─────────────────────────────────────────────────────────

def resample_signal(
    eeg: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float = cfg.TARGET_SFREQ,
) -> np.ndarray:
    """
    Polyphase resample from *orig_sfreq* → *target_sfreq*.

    Parameters
    ----------
    eeg          : (n_channels, n_samples)
    orig_sfreq   : source sample rate
    target_sfreq : destination sample rate (default from config)

    Returns
    -------
    (n_channels, n_samples_resampled) float32
    """
    if abs(orig_sfreq - target_sfreq) < 0.01:
        return eeg.astype(np.float32)

    up   = int(target_sfreq)
    down = int(orig_sfreq)
    g    = gcd(up, down)
    up  //= g
    down //= g

    resampled = resample_poly(eeg, up, down, axis=1)
    return resampled.astype(np.float32)


def bandpass_filter(
    eeg: np.ndarray,
    sfreq: float = cfg.TARGET_SFREQ,
    lowcut: float  = cfg.BANDPASS_LOW,
    highcut: float = cfg.BANDPASS_HIGH,
    order: int     = cfg.BANDPASS_ORDER,
) -> np.ndarray:
    """
    Apply zero-phase bandpass to each channel independently.

    Parameters
    ----------
    eeg    : (n_channels, n_samples)
    sfreq  : sample rate of *eeg*

    Returns
    -------
    (n_channels, n_samples) float32
    """
    sos = _butter_bandpass_sos(lowcut, highcut, sfreq, order)
    filtered = sosfiltfilt(sos, eeg, axis=1)
    return filtered.astype(np.float32)


def notch_filter(
    eeg: np.ndarray,
    sfreq: float = cfg.TARGET_SFREQ,
    freqs: list  = cfg.NOTCH_FREQS,
    Q: float     = 30.0,
) -> np.ndarray:
    """
    Apply IIR notch at each frequency in *freqs*.

    Parameters
    ----------
    eeg   : (n_channels, n_samples)
    sfreq : sample rate of *eeg*
    freqs : list of notch frequencies in Hz
    Q     : quality factor

    Returns
    -------
    (n_channels, n_samples) float32
    """
    out = eeg.copy()
    for freq in freqs:
        if freq >= sfreq / 2.0:
            continue   # Nyquist — skip
        b, a = _iir_notch_ba(freq, sfreq, Q)
        # filtfilt each channel
        from scipy.signal import filtfilt
        out = filtfilt(b, a, out, axis=1).astype(np.float32)
    return out


def preprocess_signal(
    eeg: np.ndarray,
    orig_sfreq: float,
) -> np.ndarray:
    """
    Full filter pipeline: resample → bandpass → notch.

    Parameters
    ----------
    eeg        : (n_channels, n_samples) raw EEG
    orig_sfreq : original sample rate in Hz

    Returns
    -------
    (n_channels, n_samples_at_256Hz) float32
    """
    eeg = resample_signal(eeg, orig_sfreq)
    eeg = bandpass_filter(eeg)
    eeg = notch_filter(eeg)
    return eeg
