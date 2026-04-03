from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_SFREQ = cfg.TARGET_SFREQ
_DFA_MIN = cfg.DFA_MIN_WIN
_DFA_MAX = cfg.DFA_MAX_WIN
try:
    import antropy as ant
    _HAS_ANTROPY = True
except ImportError:
    _HAS_ANTROPY = False
    import warnings
    warnings.warn('antropy not installed — DFA will be 0.0. pip install antropy', ImportWarning)

def _rolling_variance(signal: np.ndarray, win_samp: int) -> float:
    n = len(signal)
    if n < win_samp:
        return float(np.var(signal))
    variances = [float(np.var(signal[i:i + win_samp])) for i in range(0, n - win_samp + 1, win_samp // 2)]
    return float(np.mean(variances))

def _coeff_variation(signal: np.ndarray) -> float:
    mu = np.abs(np.mean(signal)) + 1e-30
    return float(np.std(signal) / mu)

def _lag1_autocorr(signal: np.ndarray) -> float:
    n = len(signal)
    if n < 3:
        return 0.0
    x = signal - signal.mean()
    c0 = np.dot(x, x)
    if c0 <= 0:
        return 0.0
    c1 = np.dot(x[:-1], x[1:])
    return float(c1 / c0)

def _dfa_exponent(signal: np.ndarray) -> float:
    if not _HAS_ANTROPY:
        return 0.0
    n = len(signal)
    lo = max(_DFA_MIN, 4)
    hi = min(_DFA_MAX, n // 4)
    if hi <= lo:
        return 0.0
    try:
        return float(ant.detrended_fluctuation(signal))
    except Exception:
        return 0.0

def _trend_magnitude(history: Optional[List[float]]) -> float:
    if history is None or len(history) < 3:
        return float('nan')
    x = np.arange(len(history), dtype=float)
    y = np.array(history, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 3:
        return float('nan')
    slope = float(np.polyfit(x[valid], y[valid], 1)[0])
    return slope

def extract_instability(window: np.ndarray, history_amplitude: Optional[List[float]]=None, history_entropy: Optional[List[float]]=None) -> Dict[str, float]:
    envelope = np.abs(window).mean(axis=0).astype(np.float64)
    win_samp_60s = int(60.0 * _SFREQ)
    feats: Dict[str, float] = {'rolling_variance': _rolling_variance(envelope, win_samp_60s), 'coeff_variation': _coeff_variation(envelope), 'lag1_autocorr': _lag1_autocorr(envelope), 'dfa_exponent': _dfa_exponent(envelope), 'trend_amplitude': _trend_magnitude(history_amplitude), 'trend_entropy': _trend_magnitude(history_entropy)}
    components = [feats['rolling_variance'], feats['coeff_variation'], abs(feats['lag1_autocorr'] - 1.0)]
    feats['overall_instability'] = float(np.nanmean(components))
    return feats

def instability_feature_vector(window: np.ndarray, history_amplitude: Optional[List[float]]=None, history_entropy: Optional[List[float]]=None) -> Tuple[np.ndarray, List[str]]:
    d = extract_instability(window, history_amplitude, history_entropy)
    names = sorted(d.keys())
    vec = np.array([d[k] for k in names], dtype=np.float32)
    return (vec, names)