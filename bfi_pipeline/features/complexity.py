from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import welch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_SFREQ = cfg.TARGET_SFREQ
_NPERSEG = min(256, int(cfg.WINDOW_S * _SFREQ))
_NOVERLAP = _NPERSEG // 2
try:
    import antropy as ant
    _HAS_ANTROPY = True
except ImportError:
    _HAS_ANTROPY = False
    import warnings
    warnings.warn('antropy not installed — permutation/sample entropy will be 0.0. Install with: pip install antropy', ImportWarning)

def _spectral_entropy_ch(signal: np.ndarray) -> float:
    _, psd = welch(signal, fs=_SFREQ, nperseg=_NPERSEG, noverlap=_NOVERLAP, window='hann', detrend='constant')
    psd = psd / (psd.sum() + 1e-30)
    psd = np.clip(psd, 1e-30, None)
    H = -np.sum(psd * np.log2(psd))
    H_max = np.log2(len(psd))
    return float(H / H_max) if H_max > 0 else 0.0

def _perm_entropy_ch(signal: np.ndarray) -> float:
    if not _HAS_ANTROPY:
        return 0.0
    try:
        return float(ant.perm_entropy(signal, order=cfg.PE_ORDER, delay=cfg.PE_DELAY, normalize=True))
    except Exception:
        return 0.0

def _sample_entropy_ch(signal: np.ndarray) -> float:
    if not _HAS_ANTROPY:
        return 0.0
    try:
        r = cfg.SE_R * float(np.std(signal))
        if r <= 0:
            return 0.0
        return float(ant.sample_entropy(signal, order=cfg.SE_M, metric='chebyshev'))
    except Exception:
        return 0.0

def extract_complexity(window: np.ndarray, history_pe: Optional[List[float]]=None) -> Dict[str, float]:
    n_ch = window.shape[0]
    sp_ent = []
    pe_vals = []
    se_vals = []
    for ch in range(n_ch):
        sig = window[ch].astype(np.float64)
        sp_ent.append(_spectral_entropy_ch(sig))
        pe_vals.append(_perm_entropy_ch(sig))
        se_vals.append(_sample_entropy_ch(sig))
    feats: Dict[str, float] = {'spectral_entropy': float(np.mean(sp_ent)), 'perm_entropy': float(np.mean(pe_vals)), 'sample_entropy': float(np.mean(se_vals))}
    if history_pe is not None and len(history_pe) >= 3:
        x = np.arange(len(history_pe), dtype=float)
        y = np.array(history_pe, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = float('nan')
    feats['entropy_slope'] = slope
    return feats

def complexity_feature_vector(window: np.ndarray, history_pe: Optional[List[float]]=None) -> Tuple[np.ndarray, List[str]]:
    d = extract_complexity(window, history_pe)
    names = sorted(d.keys())
    vec = np.array([d[k] for k in names], dtype=np.float32)
    return (vec, names)