"""
evaluate/sensitivity.py — Sensitivity analyses.

1. First-event-only analysis:
   For each patient, keep only the first event sequence (earliest pre_collapse
   or imminent window per patient). Recompute AUROC, lead time, FP/h.

2. Sham-label permutation test:
   Permute event onset timestamps 10 times → retrain → report mean AUROC ± std.
"""

from __future__ import annotations

import copy
import random
import warnings
from typing import Dict, List, Any, Optional

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

from data.splits     import make_loso_folds, filter_records_by_ids
from model.bfi_model import build_model
from train.trainer   import Trainer, records_to_samples
from evaluate.metrics import bootstrap_metrics, aggregate_fold_results
from features.extract import get_feature_dims


# ─── First-event-only filter ──────────────────────────────────────────────────

def first_event_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From each patient, keep only the first event sequence (label > 0)
    plus all stable (label = 0) sequences from that patient.

    This removes later events to avoid counting the same patient's repeated
    deteriorations as independent observations.
    """
    from collections import defaultdict

    # Group by patient
    by_patient: Dict[str, List] = defaultdict(list)
    for s in samples:
        by_patient[s["patient_id"]].append(s)

    kept: List[Dict[str, Any]] = []
    for pid, pat_samples in by_patient.items():
        # Find first event index
        event_idx: Optional[int] = None
        for i, s in enumerate(pat_samples):
            if s["label"] > 0:
                event_idx = i
                break

        if event_idx is None:
            # No events — keep all stable samples
            kept.extend(pat_samples)
        else:
            # Keep stable samples + the first event sequence only
            for i, s in enumerate(pat_samples):
                if s["label"] == 0 or i == event_idx:
                    kept.append(s)

    return kept


# ─── Sham permutation ─────────────────────────────────────────────────────────

def permute_event_onsets(
    records: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Return a deep copy of records with event onset_sample times randomly
    permuted within each recording (sham labels).
    """
    rng     = random.Random(seed)
    records_copy = copy.deepcopy(records)

    for rec in records_copy:
        events = rec.get("events", [])
        n_samp = rec.get("eeg").shape[1] if rec.get("eeg") is not None else 0
        if n_samp == 0 or not events:
            continue
        # Randomly shuffle onset times within the recording duration
        random_onsets = [rng.randint(0, max(0, n_samp - 1)) for _ in events]
        for ev, new_onset in zip(events, random_onsets):
            ev["onset_sample"]  = new_onset
            ev["offset_sample"] = new_onset

    return records_copy


# ─── Sham analysis ────────────────────────────────────────────────────────────

def run_sham_analysis(
    records:        List[Dict[str, Any]],
    model_kwargs:   Dict,
    trainer_kwargs: Dict,
    n_permutations: int = cfg.SHAM_REPEATS,
    max_folds:      int = 2,
) -> Dict[str, Any]:
    """
    Run permutation test: permute onset times, re-extract features,
    retrain, and measure AUROC.

    Note: For efficiency, this reuses pre-extracted features and permutes
    the *sequence labels* rather than re-extracting from scratch.

    Parameters
    ----------
    records          : original records (with feature_seqs)
    model_kwargs     : model build kwargs
    trainer_kwargs   : Trainer kwargs
    n_permutations   : number of sham repeats
    max_folds        : number of LOSO folds per permutation

    Returns
    -------
    {"auroc_mean": float, "auroc_std": float, "all_aurocs": list}
    """
    dims   = get_feature_dims()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds  = make_loso_folds(records)[:max_folds]

    sham_aurocs: List[float] = []

    for perm_idx in range(n_permutations):
        seed = cfg.RANDOM_SEED + perm_idx + 1000
        print(f"  Sham permutation {perm_idx + 1}/{n_permutations} (seed={seed})")

        # Permute labels in all samples
        rng = np.random.default_rng(seed)
        fold_results = []

        for fold in folds:
            train_recs = filter_records_by_ids(records, fold["train_ids"])
            val_recs   = filter_records_by_ids(records, fold["val_ids"])
            test_recs  = filter_records_by_ids(records, fold["test_ids"])

            train_s = records_to_samples(train_recs)
            val_s   = records_to_samples(val_recs)
            test_s  = records_to_samples(test_recs)

            if not train_s or not test_s:
                continue

            # Permute labels
            all_labels = np.array([s["label"] for s in train_s + val_s + test_s])
            perm_labels = rng.permutation(all_labels)

            n_tr = len(train_s)
            n_va = len(val_s)
            for i, s in enumerate(train_s):
                s = dict(s); s["label"] = int(perm_labels[i]); train_s[i] = s
            for i, s in enumerate(val_s):
                s = dict(s); s["label"] = int(perm_labels[n_tr + i]); val_s[i] = s
            for i, s in enumerate(test_s):
                s = dict(s); s["label"] = int(perm_labels[n_tr + n_va + i]); test_s[i] = s

            model = build_model(
                d_p1=dims["D1"], d_p2=dims["D2"],
                d_p3=dims["D3"], d_global=dims["D4"],
                **model_kwargs,
            )

            trainer = Trainer(model, device, max_epochs=20, patience=5,
                              **{k: v for k, v in trainer_kwargs.items()
                                 if k not in ("max_epochs", "patience")})
            trainer.fit(train_s, val_s if val_s else train_s[:max(1, len(train_s)//10)])

            probs, _ = trainer.predict(test_s)
            labels   = np.array([s["label"] for s in test_s])
            pids     = [s["patient_id"] for s in test_s]

            fold_results.append({
                "class_probs": probs,
                "true_labels": labels,
                "patient_ids": pids,
                "fold":        fold["fold"],
                "test_site":   fold["test_site"],
            })

        if fold_results:
            agg = aggregate_fold_results(fold_results)
            sham_aurocs.append(agg["auroc"])
            print(f"    → AUROC={agg['auroc']:.4f}")

    return {
        "auroc_mean": float(np.mean(sham_aurocs)) if sham_aurocs else float("nan"),
        "auroc_std":  float(np.std(sham_aurocs))  if sham_aurocs else float("nan"),
        "all_aurocs": [float(a) for a in sham_aurocs],
    }


# ─── First-event-only evaluation ─────────────────────────────────────────────

def run_first_event_analysis(
    fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Re-evaluate fold predictions using first-event-only samples.

    Parameters
    ----------
    fold_results : output of run_loso_cv

    Returns
    -------
    Metrics dict (same structure as bootstrap_metrics output).
    """
    all_probs:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_pids:   List[str]        = []

    for fold in fold_results:
        probs  = fold["class_probs"]    # (N, 3)
        labels = fold["true_labels"]    # (N,)
        pids   = fold["patient_ids"]    # list[str]

        # Build fake sample dicts to reuse first_event_samples
        samples = [
            {"label": int(labels[i]), "patient_id": pids[i], "_idx": i}
            for i in range(len(labels))
        ]
        kept = first_event_samples(samples)
        kept_idx = [s["_idx"] for s in kept]

        all_probs.append(probs[kept_idx])
        all_labels.append(labels[kept_idx])
        all_pids.extend([pids[i] for i in kept_idx])

    if not all_probs:
        return {}

    merged_probs  = np.concatenate(all_probs,  axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)

    return bootstrap_metrics(merged_probs, merged_labels, all_pids)
