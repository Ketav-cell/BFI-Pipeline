"""
data/splits.py — Leave-one-site-out CV + patient-level train/val splits.

Leave-one-site-out (LOSO) cross-validation:
  - 8 folds, each fold holds out one site entirely as the test set.
  - Within the training folds, 15% of patients (stratified by condition)
    are held out as a validation set.
  - All events from a single patient always stay in the same split (no
    within-patient leakage).

Output per fold:
  {
    "fold": int,
    "test_site": str,
    "train_ids": List[str],  # patient_id values
    "val_ids":   List[str],
    "test_ids":  List[str],
  }
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg


PatientRecord = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────

def _patient_level_records(records: List[PatientRecord]) -> Dict[str, List[PatientRecord]]:
    """Group recording-level records by patient_id prefix (before last '_')."""
    groups: Dict[str, List[PatientRecord]] = defaultdict(list)
    for rec in records:
        # patient_id may include a file suffix; use the base patient key
        pid = rec["patient_id"]
        groups[pid].append(rec)
    return dict(groups)


def _stratified_split(
    patient_ids: List[str],
    id_to_condition: Dict[str, str],
    val_frac: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    """
    Split patient_ids into train/val, stratified by condition.
    Returns (train_ids, val_ids).
    """
    rng = random.Random(seed)

    # Group by condition
    by_cond: Dict[str, List[str]] = defaultdict(list)
    for pid in patient_ids:
        by_cond[id_to_condition.get(pid, "unknown")].append(pid)

    train_ids: List[str] = []
    val_ids:   List[str] = []

    for cond, pids in by_cond.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        n_val = max(1, int(len(pids_shuffled) * val_frac))
        val_ids.extend(pids_shuffled[:n_val])
        train_ids.extend(pids_shuffled[n_val:])

    return train_ids, val_ids


def make_loso_folds(
    records: List[PatientRecord],
    val_frac: float = cfg.VALIDATION_FRAC,
    seed: int = cfg.RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """
    Build leave-one-site-out folds.

    Parameters
    ----------
    records : flat list of all PatientRecord dicts (all sites).
    val_frac : fraction of training patients held out for validation.
    seed     : random seed for reproducibility.

    Returns
    -------
    List of fold dicts:
      {fold, test_site, train_ids, val_ids, test_ids}
    """
    # Deduplicate to patient level (use first record for metadata)
    pid_to_record: Dict[str, PatientRecord] = {}
    for rec in records:
        pid = rec["patient_id"]
        if pid not in pid_to_record:
            pid_to_record[pid] = rec

    all_pids   = list(pid_to_record.keys())
    id_to_site = {pid: rec["site"] for pid, rec in pid_to_record.items()}
    id_to_cond = {pid: rec["condition"] for pid, rec in pid_to_record.items()}

    # Unique sites
    sites = sorted(set(id_to_site.values()))

    folds: List[Dict[str, Any]] = []
    for fold_idx, test_site in enumerate(sites):
        test_ids  = [pid for pid in all_pids if id_to_site[pid] == test_site]
        train_val = [pid for pid in all_pids if id_to_site[pid] != test_site]

        train_ids, val_ids = _stratified_split(
            train_val, id_to_cond, val_frac, seed=seed + fold_idx
        )

        folds.append({
            "fold":       fold_idx,
            "test_site":  test_site,
            "train_ids":  train_ids,
            "val_ids":    val_ids,
            "test_ids":   test_ids,
        })

        n_tr = len(train_ids)
        n_v  = len(val_ids)
        n_te = len(test_ids)
        print(
            f"  Fold {fold_idx} | test_site={test_site:<12} | "
            f"train={n_tr:4d}  val={n_v:4d}  test={n_te:4d}"
        )

    return folds


def filter_records_by_ids(
    records: List[PatientRecord],
    patient_ids: List[str],
) -> List[PatientRecord]:
    """Return only those records whose patient_id is in patient_ids."""
    id_set = set(patient_ids)
    return [r for r in records if r["patient_id"] in id_set]
