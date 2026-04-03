from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from features.spectral import extract_spectral
from features.complexity import extract_complexity
from features.coordination import coordination_feature_vector
from features.instability import extract_instability
from features.network import extract_network, network_feature_vector
_P1_KEYS = ['coh_alpha_F3_P3', 'coh_alpha_F4_P4', 'coh_alpha_F3_F4', 'coh_alpha_P3_P4', 'coh_alpha_Fz_Pz', 'wpli_alpha_mean', 'global_efficiency', 'modularity', 'posterior_clustering', 'frontal_clustering']
_P2_KEYS = ['coh_alpha_mean', 'asymmetry_alpha', 'asymmetry_beta', 'asymmetry_theta', 'asymmetry_mean', 'hemispheric_density']
_P3_KEYS = ['coh_alpha_F3_P3', 'coh_alpha_Fz_Pz', 'coh_alpha_F3_F4', 'ratio_theta_alpha', 'bc_frontal', 'bc_temporal', 'network_variance']
_GL_KEYS = ['spectral_slope', 'broadband_power', 'bp_delta', 'bp_theta', 'bp_alpha', 'bp_beta', 'bp_gamma', 'spectral_entropy', 'perm_entropy', 'sample_entropy', 'rolling_variance', 'overall_instability', 'lag1_autocorr', 'pac_theta_gamma_mi']
_FEATURE_NAMES_CACHE: Optional[List[str]] = None

def _safe_get(d: Dict[str, Any], key: str, default: float=0.0) -> float:
    val = d.get(key, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def _extract_window_features(window: np.ndarray, history_pe: Optional[List[float]]=None, history_amp: Optional[List[float]]=None) -> Tuple[Dict[str, float], np.ndarray]:
    spectral = extract_spectral(window)
    complexity = extract_complexity(window, history_pe=history_pe)
    coord_vec, coord_names, wpli_mat = coordination_feature_vector(window)
    coord_dict = {n: float(coord_vec[i]) for i, n in enumerate(coord_names)}
    instab = extract_instability(window, history_amplitude=history_amp, history_entropy=history_pe)
    network = extract_network(wpli_mat)
    all_feats: Dict[str, float] = {}
    all_feats.update(spectral)
    all_feats.update(complexity)
    all_feats.update(coord_dict)
    all_feats.update(instab)
    all_feats.update(network)
    return (all_feats, wpli_mat)

def _feats_to_pattern_vectors(feats: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1 = np.array([_safe_get(feats, k) for k in _P1_KEYS], dtype=np.float32)
    p2 = np.array([_safe_get(feats, k) for k in _P2_KEYS], dtype=np.float32)
    p3 = np.array([_safe_get(feats, k) for k in _P3_KEYS], dtype=np.float32)
    gl = np.array([_safe_get(feats, k) for k in _GL_KEYS], dtype=np.float32)
    return (p1, p2, p3, gl)

def _all_feature_names(sample_feats: Dict[str, float]) -> List[str]:
    return sorted(sample_feats.keys())

def extract_sequence_features(sequence: np.ndarray, events: Optional[List[Dict]]=None) -> Dict[str, np.ndarray]:
    global _FEATURE_NAMES_CACHE
    L = sequence.shape[0]
    p1_list: List[np.ndarray] = []
    p2_list: List[np.ndarray] = []
    p3_list: List[np.ndarray] = []
    gl_list: List[np.ndarray] = []
    raw_list: List[np.ndarray] = []
    history_pe: List[float] = []
    history_amp: List[float] = []
    for t in range(L):
        window = sequence[t]
        hist_pe = history_pe[-int(cfg.ENTROPY_HISTORY_S / cfg.WINDOW_S):] if history_pe else None
        hist_amp = history_amp[-int(cfg.INSTAB_HISTORY_S / cfg.WINDOW_S):] if history_amp else None
        feats, _ = _extract_window_features(window, hist_pe, hist_amp)
        history_pe.append(_safe_get(feats, 'perm_entropy'))
        history_amp.append(_safe_get(feats, 'broadband_power'))
        p1, p2, p3, gl = _feats_to_pattern_vectors(feats)
        p1_list.append(p1)
        p2_list.append(p2)
        p3_list.append(p3)
        gl_list.append(gl)
        if _FEATURE_NAMES_CACHE is None:
            _FEATURE_NAMES_CACHE = _all_feature_names(feats)
        names = _FEATURE_NAMES_CACHE or sorted(feats.keys())
        raw = np.array([_safe_get(feats, k) for k in names], dtype=np.float32)
        raw_list.append(raw)
    return {'p1': np.stack(p1_list, axis=0), 'p2': np.stack(p2_list, axis=0), 'p3': np.stack(p3_list, axis=0), 'gl': np.stack(gl_list, axis=0), 'raw': np.stack(raw_list, axis=0)}

def extract_all_records_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in records:
        seqs = rec.get('sequences')
        if seqs is None or len(seqs) == 0:
            rec['feature_seqs'] = []
            continue
        if rec.get('eeg') is None:
            rec['feature_seqs'] = []
            continue
        feature_seqs = []
        for s_idx in range(len(seqs)):
            seq = seqs[s_idx]
            try:
                fseq = extract_sequence_features(seq)
            except Exception as exc:
                warnings.warn(f"[{rec['patient_id']}] seq {s_idx}: feature extraction failed: {exc}")
                fseq = None
            feature_seqs.append(fseq)
        rec['feature_seqs'] = feature_seqs
        print(f"  [{rec['patient_id'][:40]:40s}] feature_seqs={sum((1 for f in feature_seqs if f is not None)):5d}")
    return records

def get_feature_dims() -> Dict[str, int]:
    return {'D1': len(_P1_KEYS), 'D2': len(_P2_KEYS), 'D3': len(_P3_KEYS), 'D4': len(_GL_KEYS), 'D_total': len(_P1_KEYS) + len(_P2_KEYS) + len(_P3_KEYS) + len(_GL_KEYS)}