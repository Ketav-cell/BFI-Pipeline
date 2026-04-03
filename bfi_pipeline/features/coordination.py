from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
from scipy.signal import welch, coherence, hilbert, butter, sosfiltfilt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_SFREQ = cfg.TARGET_SFREQ
_NPERSEG = min(256, int(cfg.WINDOW_S * _SFREQ))
_NOVERLAP = _NPERSEG // 2
_CH_IDX: Dict[str, int] = {ch: i for i, ch in enumerate(cfg.STANDARD_CHANNELS)}
_LEFT_IDX = [_CH_IDX[ch] for ch in cfg.LEFT_CHANNELS if ch in _CH_IDX]
_RIGHT_IDX = [_CH_IDX[ch] for ch in cfg.RIGHT_CHANNELS if ch in _CH_IDX]
_COH_PAIRS: List[Tuple[int, int]] = []
for a, b in cfg.COHERENCE_PAIRS:
    if a in _CH_IDX and b in _CH_IDX:
        _COH_PAIRS.append((_CH_IDX[a], _CH_IDX[b]))

def _bandpass(signal: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    nyq = _SFREQ / 2.0
    sos = butter(4, [fmin / nyq, fmax / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, signal).astype(np.float32)

def _alpha_coherence(window: np.ndarray) -> Dict[str, float]:
    fmin, fmax = cfg.BANDS['alpha']
    feats: Dict[str, float] = {}
    pair_names = [f'{a}-{b}' for a, b in cfg.COHERENCE_PAIRS if a in _CH_IDX and b in _CH_IDX]
    coh_vals: List[float] = []
    for (i, j), name in zip(_COH_PAIRS, pair_names):
        try:
            f, Cxy = coherence(window[i], window[j], fs=_SFREQ, nperseg=_NPERSEG, noverlap=_NOVERLAP)
            idx = (f >= fmin) & (f <= fmax)
            coh = float(Cxy[idx].mean()) if idx.any() else 0.0
        except Exception:
            coh = 0.0
        feats[f"coh_alpha_{name.replace('-', '_')}"] = coh
        coh_vals.append(coh)
    feats['coh_alpha_mean'] = float(np.mean(coh_vals)) if coh_vals else 0.0
    return feats

def _wpli_pair(x: np.ndarray, y: np.ndarray, fmin: float, fmax: float) -> float:
    try:
        n = len(x)
        seg_len = _NPERSEG
        step = seg_len // 2
        imag_parts: List[float] = []
        for start in range(0, n - seg_len + 1, step):
            seg_x = x[start:start + seg_len] * np.hanning(seg_len)
            seg_y = y[start:start + seg_len] * np.hanning(seg_len)
            Fx = np.fft.rfft(seg_x)
            Fy = np.fft.rfft(seg_y)
            Cxy = Fx * np.conj(Fy)
            imag_parts.append(np.imag(Cxy))
        if not imag_parts:
            return 0.0
        freqs = np.fft.rfftfreq(seg_len, 1.0 / _SFREQ)
        idx = (freqs >= fmin) & (freqs <= fmax)
        if not idx.any():
            return 0.0
        Im = np.array(imag_parts)[:, idx]
        num = np.abs(np.mean(np.abs(Im) * np.sign(Im), axis=0)).mean()
        denom = np.mean(np.abs(Im)) + 1e-30
        return float(num / denom)
    except Exception:
        return 0.0

def _alpha_wpli(window: np.ndarray) -> Dict[str, float]:
    fmin, fmax = cfg.BANDS['alpha']
    wpli_vals: List[float] = []
    feats: Dict[str, float] = {}
    for i, j in _COH_PAIRS:
        w = _wpli_pair(window[i], window[j], fmin, fmax)
        wpli_vals.append(w)
    feats['wpli_alpha_mean'] = float(np.mean(wpli_vals)) if wpli_vals else 0.0
    n_ch = window.shape[0]
    wpli_mat = np.zeros((n_ch, n_ch), dtype=np.float32)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            w = _wpli_pair(window[i], window[j], fmin, fmax)
            wpli_mat[i, j] = w
            wpli_mat[j, i] = w
    feats['_wpli_matrix'] = wpli_mat
    return feats

def _modulation_index(phase_sig: np.ndarray, amp_sig: np.ndarray, n_bins: int=18) -> float:
    try:
        phase = np.angle(hilbert(phase_sig))
        amp = np.abs(hilbert(amp_sig))
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        A = np.zeros(n_bins)
        for k in range(n_bins):
            mask = (phase >= bins[k]) & (phase < bins[k + 1])
            A[k] = amp[mask].mean() if mask.any() else 0.0
        A_sum = A.sum()
        if A_sum <= 0:
            return 0.0
        P = A / A_sum
        P = np.clip(P, 1e-30, None)
        kl = np.sum(P * (np.log(P) - np.log(1.0 / n_bins)))
        return float(kl / np.log(n_bins))
    except Exception:
        return 0.0

def _pac_features(window: np.ndarray) -> Dict[str, float]:
    theta_fmin, theta_fmax = cfg.BANDS['theta']
    gamma_fmin, gamma_fmax = cfg.BANDS['gamma']
    mi_vals: List[float] = []
    for ch in range(window.shape[0]):
        phase_sig = _bandpass(window[ch], theta_fmin, theta_fmax)
        amp_sig = _bandpass(window[ch], gamma_fmin, gamma_fmax)
        mi_vals.append(_modulation_index(phase_sig, amp_sig))
    return {'pac_theta_gamma_mi': float(np.mean(mi_vals))}

def _asymmetry_features(window: np.ndarray) -> Dict[str, float]:
    from features.spectral import _band_power, _welch_psd
    feats: Dict[str, float] = {}
    for band, (fmin, fmax) in cfg.BANDS.items():
        left_bp = []
        right_bp = []
        for ch in _LEFT_IDX:
            f, psd = _welch_psd(window[ch])
            left_bp.append(_band_power(f, psd, fmin, fmax))
        for ch in _RIGHT_IDX:
            f, psd = _welch_psd(window[ch])
            right_bp.append(_band_power(f, psd, fmin, fmax))
        L = np.mean(left_bp) if left_bp else 0.0
        R = np.mean(right_bp) if right_bp else 0.0
        denom = R + L + 1e-30
        feats[f'asymmetry_{band}'] = float((R - L) / denom)
    feats['asymmetry_mean'] = float(np.mean(list(feats.values())))
    return feats

def extract_coordination(window: np.ndarray) -> Dict[str, object]:
    feats: Dict[str, object] = {}
    feats.update(_alpha_coherence(window))
    feats.update(_alpha_wpli(window))
    feats.update(_pac_features(window))
    feats.update(_asymmetry_features(window))
    return feats

def coordination_feature_vector(window: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
    d = extract_coordination(window)
    wpli_mat = d.pop('_wpli_matrix', np.zeros((cfg.N_CHANNELS, cfg.N_CHANNELS)))
    names = sorted(d.keys())
    vec = np.array([float(d[k]) for k in names], dtype=np.float32)
    return (vec, names, wpli_mat)