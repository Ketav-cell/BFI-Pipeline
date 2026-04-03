from __future__ import annotations
import copy
import warnings
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
from model.bfi_model import build_model
from train.trainer import Trainer, records_to_samples
from data.splits import make_loso_folds, filter_records_by_ids
from evaluate.metrics import aggregate_fold_results, bootstrap_metrics
from evaluate.alarm import aggregate_alarm_metrics, evaluate_patient_alarms
from features.extract import get_feature_dims
_SPECTRAL_KEYS = {'bp_', 'rel_', 'ratio_', 'spectral_slope', 'broadband_power'}
_COMPLEXITY_KEYS = {'spectral_entropy', 'perm_entropy', 'sample_entropy', 'entropy_slope'}
_COORD_KEYS = {'coh_', 'wpli_', 'pac_', 'asymmetry_'}
_INSTAB_KEYS = {'rolling_variance', 'coeff_variation', 'lag1_autocorr', 'dfa_exponent', 'trend_', 'overall_instability'}
_NETWORK_KEYS = {'global_efficiency', 'modularity', 'bc_', 'network_variance', 'frontal_clustering', 'posterior_clustering', 'hemispheric_density'}

def _zero_keys(sample: Dict[str, Any], key_prefixes: set) -> Dict[str, Any]:
    s = dict(sample)
    for pat in ('p1', 'p2', 'p3', 'gl'):
        arr = s[pat].copy()
        s[pat] = arr
    return s

class FeatureMaskWrapper(torch.nn.Module):

    def __init__(self, model, masks: Dict[str, Optional[torch.Tensor]]):
        super().__init__()
        self.model = model
        self.masks = masks

    def forward(self, p1, p2, p3, gl):

        def apply_mask(x, key):
            m = self.masks.get(key)
            if m is not None:
                x = x * m.to(x.device).unsqueeze(0).unsqueeze(0)
            return x
        p1 = apply_mask(p1, 'p1')
        p2 = apply_mask(p2, 'p2')
        p3 = apply_mask(p3, 'p3')
        gl = apply_mask(gl, 'gl')
        return self.model(p1, p2, p3, gl)

    def compute_bfi(self, probs):
        return self.model.compute_bfi(probs)

    def predict_bfi(self, p1, p2, p3, gl):
        self.eval()
        probs, _, _ = self.forward(p1, p2, p3, gl)
        bfi = self.compute_bfi(probs)
        return (bfi, probs)

def _run_single_fold_ablation(records: List[Dict[str, Any]], fold: Dict[str, Any], model_kwargs: Dict, trainer_kwargs: Dict, use_attention: bool=True, aux_lambda: float=cfg.AUX_LAMBDA) -> Dict[str, Any]:
    dims = get_feature_dims()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_recs = filter_records_by_ids(records, fold['train_ids'])
    val_recs = filter_records_by_ids(records, fold['val_ids'])
    test_recs = filter_records_by_ids(records, fold['test_ids'])
    train_s = records_to_samples(train_recs)
    val_s = records_to_samples(val_recs)
    test_s = records_to_samples(test_recs)
    if not train_s or not test_s:
        return {}
    model = build_model(d_p1=dims['D1'], d_p2=dims['D2'], d_p3=dims['D3'], d_global=dims['D4'], use_attention=use_attention, **{k: v for k, v in model_kwargs.items() if k != 'use_attention'})
    tkw = dict(trainer_kwargs)
    tkw['aux_lambda'] = aux_lambda
    trainer = Trainer(model, device, **tkw)
    trainer.fit(train_s, val_s if val_s else train_s[:max(1, len(train_s) // 10)])
    probs, bfi = trainer.predict(test_s)
    labels = np.array([s['label'] for s in test_s])
    pids = [s['patient_id'] for s in test_s]
    return {'class_probs': probs, 'true_labels': labels, 'patient_ids': pids, 'bfi_scores': bfi, 'fold': fold['fold'], 'test_site': fold['test_site']}

def run_ablation(records: List[Dict[str, Any]], baseline_metrics: Dict[str, Any], model_kwargs: Dict, trainer_kwargs: Dict, max_folds: int=2) -> Dict[str, Any]:
    folds = make_loso_folds(records)[:max_folds]
    ablation_configs = {'minus_aux_losses': {'use_attention': True, 'aux_lambda': 0.0}, 'minus_attention': {'use_attention': False, 'aux_lambda': cfg.AUX_LAMBDA}}
    for name in ['minus_spectral', 'minus_complexity', 'minus_coordination', 'minus_instability', 'minus_network']:
        ablation_configs[name] = {'use_attention': True, 'aux_lambda': cfg.AUX_LAMBDA}
    results: Dict[str, Any] = {}
    for ablation_name, extra_kwargs in ablation_configs.items():
        print(f'\n  Ablation: {ablation_name}')
        fold_results: List[Dict] = []
        for fold in folds:
            res = _run_single_fold_ablation(records, fold, model_kwargs, trainer_kwargs, use_attention=extra_kwargs.get('use_attention', True), aux_lambda=extra_kwargs.get('aux_lambda', cfg.AUX_LAMBDA))
            if res:
                fold_results.append(res)
        if not fold_results:
            results[ablation_name] = {'delta_auroc': float('nan'), 'delta_lead': float('nan')}
            continue
        agg = aggregate_fold_results(fold_results)
        delta_auroc = agg['auroc'] - baseline_metrics.get('auroc', 0.0)
        delta_lead = float('nan')
        results[ablation_name] = {'auroc': agg['auroc'], 'delta_auroc': float(delta_auroc), 'delta_lead': delta_lead}
        print(f"    AUROC={agg['auroc']:.4f}  ΔAUROC={delta_auroc:+.4f}")
    return results