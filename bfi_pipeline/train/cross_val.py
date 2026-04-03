from __future__ import annotations
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
from features.extract import get_feature_dims

def _device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_loso_cv(records: List[Dict[str, Any]], model_kwargs: Optional[Dict]=None, trainer_kwargs: Optional[Dict]=None) -> List[Dict[str, Any]]:
    model_kwargs = model_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    device = _device()
    print(f'\nDevice: {device}')
    dims = get_feature_dims()
    d_p1 = dims['D1']
    d_p2 = dims['D2']
    d_p3 = dims['D3']
    d_global = dims['D4']
    print(f"Feature dims: P1={d_p1}  P2={d_p2}  P3={d_p3}  Global={d_global}  Total={dims['D_total']}")
    print('\nBuilding LOSO folds:')
    folds = make_loso_folds(records)
    fold_results: List[Dict[str, Any]] = []
    for fold in folds:
        fold_idx = fold['fold']
        test_site = fold['test_site']
        print(f"\n{'=' * 60}")
        print(f'FOLD {fold_idx}  |  Test site: {test_site}')
        print(f"{'=' * 60}")
        train_recs = filter_records_by_ids(records, fold['train_ids'])
        val_recs = filter_records_by_ids(records, fold['val_ids'])
        test_recs = filter_records_by_ids(records, fold['test_ids'])
        train_samples = records_to_samples(train_recs)
        val_samples = records_to_samples(val_recs)
        test_samples = records_to_samples(test_recs)
        if not train_samples:
            warnings.warn(f'Fold {fold_idx}: no training samples — skipping.')
            continue
        if not val_samples:
            warnings.warn(f'Fold {fold_idx}: no validation samples — using 10% of train.')
            n_val = max(1, len(train_samples) // 10)
            val_samples = train_samples[:n_val]
            train_samples = train_samples[n_val:]
        if not test_samples:
            warnings.warn(f'Fold {fold_idx}: no test samples — skipping.')
            continue
        print(f'  Samples: train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}')
        model = build_model(d_p1, d_p2, d_p3, d_global, **model_kwargs)
        n_params = sum((p.numel() for p in model.parameters()))
        print(f'  Model parameters: {n_params:,}')
        trainer = Trainer(model, device, **trainer_kwargs)
        train_result = trainer.fit(train_samples, val_samples)
        class_probs, bfi_scores = trainer.predict(test_samples)
        true_labels = np.array([s['label'] for s in test_samples], dtype=np.int64)
        patient_ids = [s['patient_id'] for s in test_samples]
        fold_results.append({'fold': fold_idx, 'test_site': test_site, 'class_probs': class_probs, 'bfi_scores': bfi_scores, 'true_labels': true_labels, 'patient_ids': patient_ids, 'train_result': train_result, 'model': model})
        best = train_result['best_val_auroc']
        ep = train_result['best_epoch']
        print(f'  → Best val AUROC={best:.4f} at epoch {ep}')
    return fold_results