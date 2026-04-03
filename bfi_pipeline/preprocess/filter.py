from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, resample_poly
from math import gcd
from typing import Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def _butter_bandpass_sos(lowcut: float, highcut: float, fs: float, order: int=4) -> np.ndarray:
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def _iir_notch_ba(freq: float, fs: float, Q: float=30.0) -> Tuple[np.ndarray, np.ndarray]:
    w0 = freq / (fs / 2.0)
    b, a = iirnotch(w0, Q)
    return (b, a)

def resample_signal(eeg: np.ndarray, orig_sfreq: float, target_sfreq: float=cfg.TARGET_SFREQ) -> np.ndarray:
    if abs(orig_sfreq - target_sfreq) < 0.01:
        return eeg.astype(np.float32)
    up = int(target_sfreq)
    down = int(orig_sfreq)
    g = gcd(up, down)
    up //= g
    down //= g
    resampled = resample_poly(eeg, up, down, axis=1)
    return resampled.astype(np.float32)

def bandpass_filter(eeg: np.ndarray, sfreq: float=cfg.TARGET_SFREQ, lowcut: float=cfg.BANDPASS_LOW, highcut: float=cfg.BANDPASS_HIGH, order: int=cfg.BANDPASS_ORDER) -> np.ndarray:
    sos = _butter_bandpass_sos(lowcut, highcut, sfreq, order)
    filtered = sosfiltfilt(sos, eeg, axis=1)
    return filtered.astype(np.float32)

def notch_filter(eeg: np.ndarray, sfreq: float=cfg.TARGET_SFREQ, freqs: list=cfg.NOTCH_FREQS, Q: float=30.0) -> np.ndarray:
    out = eeg.copy()
    for freq in freqs:
        if freq >= sfreq / 2.0:
            continue
        b, a = _iir_notch_ba(freq, sfreq, Q)
        from scipy.signal import filtfilt
        out = filtfilt(b, a, out, axis=1).astype(np.float32)
    return out

def preprocess_signal(eeg: np.ndarray, orig_sfreq: float) -> np.ndarray:
    eeg = resample_signal(eeg, orig_sfreq)
    eeg = bandpass_filter(eeg)
    eeg = notch_filter(eeg)
    return eeg