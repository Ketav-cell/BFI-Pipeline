"""
evaluate/stats.py — Statistical hypothesis tests for feature discrimination.

Tests:
  • Wilcoxon signed-rank test (paired pre-event vs matched control)
  • Benjamini-Hochberg FDR correction (FDR α = 0.01)
  • Rank-biserial effect size  r = 1 − 2W / (n*(n+1)/2)
  • Median difference

Feature families:
  spectral / complexity / coordination / instability / network

Per-family analysis groups all features belonging to that family and
reports per-feature statistics, then corrects p-values across all tests
within the family.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple

import numpy as np
from scipy.stats import wilcoxon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


# ─── BH correction ────────────────────────────────────────────────────────────

def benjamini_hochberg(p_values: np.ndarray, alpha: float = cfg.FDR_ALPHA) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Returns
    -------
    p_adjusted : corrected p-values (same order as input)
    """
    n  = len(p_values)
    order  = np.argsort(p_values)
    ranks  = np.arange(1, n + 1)
    p_adj  = np.empty(n)

    sorted_p = p_values[order]
    adj      = sorted_p * n / ranks
    # Enforce monotonicity from right to left
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    p_adj[order] = adj
    return p_adj


# ─── Rank-biserial effect size ────────────────────────────────────────────────

def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial correlation for Wilcoxon signed-rank test.

    r = 1 − 2·W / (n·(n+1)/2)
    where W = Wilcoxon statistic and n = number of non-zero differences.
    """
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    try:
        stat, _ = wilcoxon(x, y, zero_method="wilcox")
        max_w   = n * (n + 1) / 2.0
        r       = 1.0 - 2.0 * stat / max_w
        return float(r)
    except Exception:
        return 0.0


# ─── Per-feature discrimination test ─────────────────────────────────────────

def _paired_test(
    pre_event: np.ndarray,   # (N_pairs,)
    control:   np.ndarray,   # (N_pairs,)
) -> Tuple[float, float, float, float]:
    """
    Returns (statistic, p_value, effect_size, median_diff).
    """
    d    = pre_event - control
    d    = d[~np.isnan(d)]
    if len(d) < 5:
        return float("nan"), 1.0, 0.0, float("nan")

    try:
        stat, pval = wilcoxon(pre_event[~np.isnan(d)], control[~np.isnan(d)],
                              zero_method="wilcox")
    except Exception:
        stat, pval = float("nan"), 1.0

    r    = rank_biserial(pre_event[~np.isnan(d)], control[~np.isnan(d)])
    diff = float(np.nanmedian(pre_event) - np.nanmedian(control))

    return float(stat), float(pval), float(r), float(diff)


# ─── Family-level analysis ────────────────────────────────────────────────────

def analyze_feature_family(
    feature_matrix_pre:  np.ndarray,   # (N_pairs, D_family)
    feature_matrix_ctrl: np.ndarray,   # (N_pairs, D_family)
    feature_names:       List[str],
    fdr_alpha:           float = cfg.FDR_ALPHA,
) -> Dict[str, Any]:
    """
    Test all features in a family for pre-event vs control differences.

    Parameters
    ----------
    feature_matrix_pre  : feature values for pre-event windows
    feature_matrix_ctrl : matched control windows (same shape)
    feature_names       : list of feature names (length D_family)

    Returns
    -------
    dict with per-feature stats and family-level summary.
    """
    D = len(feature_names)
    stats    = np.zeros(D)
    pvals    = np.ones(D)
    effects  = np.zeros(D)
    med_diff = np.zeros(D)

    for j in range(D):
        s, p, r, d = _paired_test(
            feature_matrix_pre[:, j],
            feature_matrix_ctrl[:, j],
        )
        stats[j]    = s
        pvals[j]    = p
        effects[j]  = r
        med_diff[j] = d

    # BH correction
    p_adj = benjamini_hochberg(pvals, fdr_alpha)

    per_feature = {}
    for j, name in enumerate(feature_names):
        per_feature[name] = {
            "statistic":    float(stats[j]),
            "p_raw":        float(pvals[j]),
            "p_corrected":  float(p_adj[j]),
            "effect_size":  float(effects[j]),
            "median_diff":  float(med_diff[j]),
            "significant":  bool(p_adj[j] < fdr_alpha),
        }

    # Family summary: use feature with largest |effect size|
    best_j = int(np.argmax(np.abs(effects)))
    return {
        "per_feature":  per_feature,
        "median_diff":  float(med_diff[best_j]),
        "effect_size":  float(effects[best_j]),
        "p_corrected":  float(p_adj[best_j]),
        "n_significant": int((p_adj < fdr_alpha).sum()),
    }


# ─── Top-level: all families ──────────────────────────────────────────────────

def run_feature_discrimination(
    pre_event_features:  Dict[str, np.ndarray],  # family → (N, D)
    control_features:    Dict[str, np.ndarray],  # family → (N, D)
    family_names:        Dict[str, List[str]],   # family → [feature names]
) -> Dict[str, Any]:
    """
    Run feature discrimination analysis for all families.

    Parameters
    ----------
    pre_event_features : {family: (N, D)} arrays of pre-event feature windows
    control_features   : {family: (N, D)} matched controls
    family_names       : {family: [name, ...]} feature names per family

    Returns
    -------
    {family: {median_diff, effect_size, p_corrected, n_significant}}
    """
    results: Dict[str, Any] = {}
    for family in pre_event_features:
        pre  = pre_event_features[family]
        ctrl = control_features.get(family, pre)  # fallback if ctrl missing
        names = family_names.get(family, [f"f{i}" for i in range(pre.shape[1])])

        print(f"  Testing family: {family}  (D={pre.shape[1]}, N={pre.shape[0]})")
        results[family] = analyze_feature_family(pre, ctrl, names)

    return results
