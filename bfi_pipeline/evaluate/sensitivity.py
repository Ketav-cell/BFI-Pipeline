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
from data.splits import make_loso_folds, filter_records_by_ids
from model.bfi_model import build_model
from train.trainer import Trainer, records_to_samples
from evaluate.metrics import bootstrap_metrics, aggregate_fold_results
from features.extract import get_feature_dims

def first_event_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    by_patient: Dict[str, List] = defaultdict(list)
    for s in samples:
        by_patient[s['patient_id']].append(s)
    kept: List[Dict[str, Any]] = []
    for pid, pat_samples in by_patient.items():
        event_idx: Optional[int] = None
        for i, s in enumerate(pat_samples):
            if s['label'] > 0:
                event_idx = i
                break
        if event_idx is None:
            kept.extend(pat_samples)
        else:
            for i, s in enumerate(pat_samples):
                if s['label'] == 0 or i == event_idx:
                    kept.append(s)
    return kept

def permute_event_onsets(records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    records_copy = copy.deepcopy(records)
    for rec in records_copy:
        events = rec.get('events', [])
        n_samp = rec.get('eeg').shape[1] if rec.get('eeg') is not None else 0
        if n_samp == 0 or not events:
            continue
        random_onsets = [rng.randint(0, max(0, n_samp - 1)) for _ in events]
        for ev, new_onset in zip(events, random_onsets):
            ev['onset_sample'] = new_onset
            ev['offset_sample'] = new_onset
    return records_copy

def run_sham_analysis(records: List[Dict[str, Any]], model_kwargs: Dict, trainer_kwargs: Dict, n_permutations: int=cfg.SHAM_REPEATS, max_folds: int=2) -> Dict[str, Any]:
    dims = get_feature_dims()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folds = make_loso_folds(records)[:max_folds]
    sham_aurocs: List[float] = []
    for perm_idx in range(n_permutations):
        seed = cfg.RANDOM_SEED + perm_idx + 1000
        print(f'  Sham permutation {perm_idx + 1}/{n_permutations} (seed={seed})')
        rng = np.random.default_rng(seed)
        fold_results = []
        for fold in folds:
            train_recs = filter_records_by_ids(records, fold['train_ids'])
            val_recs = filter_records_by_ids(records, fold['val_ids'])
            test_recs = filter_records_by_ids(records, fold['test_ids'])
            train_s = records_to_samples(train_recs)
            val_s = records_to_samples(val_recs)
            test_s = records_to_samples(test_recs)
            if not train_s or not test_s:
                continue
            all_labels = np.array([s['label'] for s in train_s + val_s + test_s])
            perm_labels = rng.permutation(all_labels)
            n_tr = len(train_s)
            n_va = len(val_s)
            for i, s in enumerate(train_s):
                s = dict(s)
                s['label'] = int(perm_labels[i])
                train_s[i] = s
            for i, s in enumerate(val_s):
                s = dict(s)
                s['label'] = int(perm_labels[n_tr + i])
                val_s[i] = s
            for i, s in enumerate(test_s):
                s = dict(s)
                s['label'] = int(perm_labels[n_tr + n_va + i])
                test_s[i] = s
            model = build_model(d_p1=dims['D1'], d_p2=dims['D2'], d_p3=dims['D3'], d_global=dims['D4'], **model_kwargs)
            trainer = Trainer(model, device, max_epochs=20, patience=5, **{k: v for k, v in trainer_kwargs.items() if k not in ('max_epochs', 'patience')})
            trainer.fit(train_s, val_s if val_s else train_s[:max(1, len(train_s) // 10)])
            probs, _ = trainer.predict(test_s)
            labels = np.array([s['label'] for s in test_s])
            pids = [s['patient_id'] for s in test_s]
            fold_results.append({'class_probs': probs, 'true_labels': labels, 'patient_ids': pids, 'fold': fold['fold'], 'test_site': fold['test_site']})
        if fold_results:
            agg = aggregate_fold_results(fold_results)
            sham_aurocs.append(agg['auroc'])
            print(f"    → AUROC={agg['auroc']:.4f}")
    return {'auroc_mean': float(np.mean(sham_aurocs)) if sham_aurocs else float('nan'), 'auroc_std': float(np.std(sham_aurocs)) if sham_aurocs else float('nan'), 'all_aurocs': [float(a) for a in sham_aurocs]}

def run_first_event_analysis(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_pids: List[str] = []
    for fold in fold_results:
        probs = fold['class_probs']
        labels = fold['true_labels']
        pids = fold['patient_ids']
        samples = [{'label': int(labels[i]), 'patient_id': pids[i], '_idx': i} for i in range(len(labels))]
        kept = first_event_samples(samples)
        kept_idx = [s['_idx'] for s in kept]
        all_probs.append(probs[kept_idx])
        all_labels.append(labels[kept_idx])
        all_pids.extend([pids[i] for i in kept_idx])
    if not all_probs:
        return {}
    merged_probs = np.concatenate(all_probs, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    return bootstrap_metrics(merged_probs, merged_labels, all_pids)