from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int=cfg.ECE_BINS) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)

def compute_metrics(class_probs: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    results: Dict[str, float] = {}
    n_classes = class_probs.shape[1]
    unique_classes = np.unique(true_labels)
    try:
        if len(unique_classes) > 1:
            auroc = roc_auc_score(true_labels, class_probs, multi_class='ovr', average='macro')
        else:
            auroc = 0.5
    except Exception:
        auroc = 0.5
    results['auroc'] = float(auroc)
    try:
        pr_aucs = []
        for c in range(n_classes):
            y_bin = (true_labels == c).astype(int)
            if y_bin.sum() == 0:
                continue
            ap = average_precision_score(y_bin, class_probs[:, c])
            pr_aucs.append(ap)
        results['pr_auc'] = float(np.mean(pr_aucs)) if pr_aucs else 0.0
    except Exception:
        results['pr_auc'] = 0.0
    try:
        briers = []
        for c in range(n_classes):
            y_bin = (true_labels == c).astype(float)
            briers.append(brier_score_loss(y_bin, class_probs[:, c]))
        results['brier'] = float(np.mean(briers))
    except Exception:
        results['brier'] = 1.0
    results['ece'] = expected_calibration_error(class_probs, true_labels)
    return results

def bootstrap_metrics(class_probs: np.ndarray, true_labels: np.ndarray, patient_ids: List[str], n_iter: int=cfg.BOOTSTRAP_ITERS, seed: int=cfg.RANDOM_SEED) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_pids = list(dict.fromkeys(patient_ids))
    pid_to_idx: Dict[str, List[int]] = {p: [] for p in unique_pids}
    for i, pid in enumerate(patient_ids):
        pid_to_idx[pid].append(i)
    boot_aurocs: List[float] = []
    boot_praucs: List[float] = []
    for _ in range(n_iter):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idx = []
        for pid in sampled_pids:
            idx.extend(pid_to_idx[pid])
        idx = np.array(idx)
        bp = class_probs[idx]
        bl = true_labels[idx]
        m = compute_metrics(bp, bl)
        boot_aurocs.append(m['auroc'])
        boot_praucs.append(m['pr_auc'])
    point = compute_metrics(class_probs, true_labels)
    lo, hi = np.percentile(boot_aurocs, [2.5, 97.5])
    plo, phi = np.percentile(boot_praucs, [2.5, 97.5])
    return {'auroc': point['auroc'], 'auroc_ci': [float(lo), float(hi)], 'pr_auc': point['pr_auc'], 'pr_auc_ci': [float(plo), float(phi)], 'brier': point['brier'], 'ece': point['ece']}

def aggregate_fold_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_probs = np.concatenate([f['class_probs'] for f in fold_results], axis=0)
    all_labels = np.concatenate([f['true_labels'] for f in fold_results], axis=0)
    all_pids = []
    for f in fold_results:
        all_pids.extend(f['patient_ids'])
    return bootstrap_metrics(all_probs, all_labels, all_pids)