from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.stats import wilcoxon
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def benjamini_hochberg(p_values: np.ndarray, alpha: float=cfg.FDR_ALPHA) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.arange(1, n + 1)
    p_adj = np.empty(n)
    sorted_p = p_values[order]
    adj = sorted_p * n / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    p_adj[order] = adj
    return p_adj

def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    try:
        stat, _ = wilcoxon(x, y, zero_method='wilcox')
        max_w = n * (n + 1) / 2.0
        r = 1.0 - 2.0 * stat / max_w
        return float(r)
    except Exception:
        return 0.0

def _paired_test(pre_event: np.ndarray, control: np.ndarray) -> Tuple[float, float, float, float]:
    d = pre_event - control
    d = d[~np.isnan(d)]
    if len(d) < 5:
        return (float('nan'), 1.0, 0.0, float('nan'))
    try:
        stat, pval = wilcoxon(pre_event[~np.isnan(d)], control[~np.isnan(d)], zero_method='wilcox')
    except Exception:
        stat, pval = (float('nan'), 1.0)
    r = rank_biserial(pre_event[~np.isnan(d)], control[~np.isnan(d)])
    diff = float(np.nanmedian(pre_event) - np.nanmedian(control))
    return (float(stat), float(pval), float(r), float(diff))

def analyze_feature_family(feature_matrix_pre: np.ndarray, feature_matrix_ctrl: np.ndarray, feature_names: List[str], fdr_alpha: float=cfg.FDR_ALPHA) -> Dict[str, Any]:
    D = len(feature_names)
    stats = np.zeros(D)
    pvals = np.ones(D)
    effects = np.zeros(D)
    med_diff = np.zeros(D)
    for j in range(D):
        s, p, r, d = _paired_test(feature_matrix_pre[:, j], feature_matrix_ctrl[:, j])
        stats[j] = s
        pvals[j] = p
        effects[j] = r
        med_diff[j] = d
    p_adj = benjamini_hochberg(pvals, fdr_alpha)
    per_feature = {}
    for j, name in enumerate(feature_names):
        per_feature[name] = {'statistic': float(stats[j]), 'p_raw': float(pvals[j]), 'p_corrected': float(p_adj[j]), 'effect_size': float(effects[j]), 'median_diff': float(med_diff[j]), 'significant': bool(p_adj[j] < fdr_alpha)}
    best_j = int(np.argmax(np.abs(effects)))
    return {'per_feature': per_feature, 'median_diff': float(med_diff[best_j]), 'effect_size': float(effects[best_j]), 'p_corrected': float(p_adj[best_j]), 'n_significant': int((p_adj < fdr_alpha).sum())}

def run_feature_discrimination(pre_event_features: Dict[str, np.ndarray], control_features: Dict[str, np.ndarray], family_names: Dict[str, List[str]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for family in pre_event_features:
        pre = pre_event_features[family]
        ctrl = control_features.get(family, pre)
        names = family_names.get(family, [f'f{i}' for i in range(pre.shape[1])])
        print(f'  Testing family: {family}  (D={pre.shape[1]}, N={pre.shape[0]})')
        results[family] = analyze_feature_family(pre, ctrl, names)
    return results