from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import welch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_SFREQ = cfg.TARGET_SFREQ
_WIN_SAMP = int(cfg.WINDOW_S * _SFREQ)
_NPERSEG = min(256, _WIN_SAMP)
_NOVERLAP = _NPERSEG // 2

def _welch_psd(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    freqs, psd = welch(signal, fs=_SFREQ, nperseg=_NPERSEG, noverlap=_NOVERLAP, window='hann', detrend='constant')
    return (freqs, psd)

def _band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if idx.sum() < 2:
        return 0.0
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(_trapz(psd[idx], freqs[idx]))

def _spectral_slope(freqs: np.ndarray, psd: np.ndarray, fmin: float=1.0, fmax: float=45.0) -> float:
    idx = (freqs >= fmin) & (freqs <= fmax) & (psd > 0) & (freqs > 0)
    if idx.sum() < 3:
        return 0.0
    lf = np.log10(freqs[idx])
    lp = np.log10(psd[idx] + 1e-30)
    coeffs = np.polyfit(lf, lp, 1)
    return float(coeffs[0])

def extract_spectral(window: np.ndarray) -> Dict[str, float]:
    n_ch = window.shape[0]
    ch_band_powers: Dict[str, List[float]] = {b: [] for b in cfg.BANDS}
    ch_rel_powers: Dict[str, List[float]] = {b: [] for b in cfg.BANDS}
    ch_slopes: List[float] = []
    ch_theta_alpha: List[float] = []
    ch_delta_alpha: List[float] = []
    for ch in range(n_ch):
        freqs, psd = _welch_psd(window[ch])
        bp: Dict[str, float] = {}
        for band, (fmin, fmax) in cfg.BANDS.items():
            bp[band] = _band_power(freqs, psd, fmin, fmax)
            ch_band_powers[band].append(bp[band])
        total = sum(bp.values()) + 1e-30
        for band in cfg.BANDS:
            ch_rel_powers[band].append(bp[band] / total)
        alpha = bp['alpha'] + 1e-30
        ch_theta_alpha.append(bp['theta'] / alpha)
        ch_delta_alpha.append(bp['delta'] / alpha)
        ch_slopes.append(_spectral_slope(freqs, psd))
    feats: Dict[str, float] = {}
    for band in cfg.BANDS:
        feats[f'bp_{band}'] = float(np.mean(ch_band_powers[band]))
        feats[f'rel_{band}'] = float(np.mean(ch_rel_powers[band]))
    feats['ratio_theta_alpha'] = float(np.mean(ch_theta_alpha))
    feats['ratio_delta_alpha'] = float(np.mean(ch_delta_alpha))
    feats['spectral_slope'] = float(np.mean(ch_slopes))
    feats['broadband_power'] = float(np.mean([sum((ch_band_powers[b][ch] for b in cfg.BANDS)) for ch in range(n_ch)]))
    return feats

def spectral_feature_vector(window: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    d = extract_spectral(window)
    names = sorted(d.keys())
    vec = np.array([d[k] for k in names], dtype=np.float32)
    return (vec, names)