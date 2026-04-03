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

def _patient_level_records(records: List[PatientRecord]) -> Dict[str, List[PatientRecord]]:
    groups: Dict[str, List[PatientRecord]] = defaultdict(list)
    for rec in records:
        pid = rec['patient_id']
        groups[pid].append(rec)
    return dict(groups)

def _stratified_split(patient_ids: List[str], id_to_condition: Dict[str, str], val_frac: float, seed: int) -> tuple[List[str], List[str]]:
    rng = random.Random(seed)
    by_cond: Dict[str, List[str]] = defaultdict(list)
    for pid in patient_ids:
        by_cond[id_to_condition.get(pid, 'unknown')].append(pid)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for cond, pids in by_cond.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        n_val = max(1, int(len(pids_shuffled) * val_frac))
        val_ids.extend(pids_shuffled[:n_val])
        train_ids.extend(pids_shuffled[n_val:])
    return (train_ids, val_ids)

def make_loso_folds(records: List[PatientRecord], val_frac: float=cfg.VALIDATION_FRAC, seed: int=cfg.RANDOM_SEED) -> List[Dict[str, Any]]:
    pid_to_record: Dict[str, PatientRecord] = {}
    for rec in records:
        pid = rec['patient_id']
        if pid not in pid_to_record:
            pid_to_record[pid] = rec
    all_pids = list(pid_to_record.keys())
    id_to_site = {pid: rec['site'] for pid, rec in pid_to_record.items()}
    id_to_cond = {pid: rec['condition'] for pid, rec in pid_to_record.items()}
    sites = sorted(set(id_to_site.values()))
    folds: List[Dict[str, Any]] = []
    for fold_idx, test_site in enumerate(sites):
        test_ids = [pid for pid in all_pids if id_to_site[pid] == test_site]
        train_val = [pid for pid in all_pids if id_to_site[pid] != test_site]
        train_ids, val_ids = _stratified_split(train_val, id_to_cond, val_frac, seed=seed + fold_idx)
        folds.append({'fold': fold_idx, 'test_site': test_site, 'train_ids': train_ids, 'val_ids': val_ids, 'test_ids': test_ids})
        n_tr = len(train_ids)
        n_v = len(val_ids)
        n_te = len(test_ids)
        print(f'  Fold {fold_idx} | test_site={test_site:<12} | train={n_tr:4d}  val={n_v:4d}  test={n_te:4d}')
    return folds

def filter_records_by_ids(records: List[PatientRecord], patient_ids: List[str]) -> List[PatientRecord]:
    id_set = set(patient_ids)
    return [r for r in records if r['patient_id'] in id_set]