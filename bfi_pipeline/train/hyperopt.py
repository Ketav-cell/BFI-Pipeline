"""
train/hyperopt.py — Optuna hyperparameter search for the BFI model.

Searches over:
  lr         : 1e-4 → 1e-2 (log)
  d_proj     : 32, 64, 128
  d_hidden   : 64, 128, 256
  dropout    : 0.1 → 0.5
  aux_lambda : 0.1 → 0.5
  batch_size : 16, 32, 64

Uses a single random fold (first fold by default) for speed.
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    warnings.warn("optuna not installed — hyperopt disabled. pip install optuna", ImportWarning)

from data.splits    import make_loso_folds, filter_records_by_ids
from model.bfi_model import build_model
from train.trainer   import Trainer, records_to_samples
from features.extract import get_feature_dims


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _objective(
    trial: "optuna.Trial",
    records: List[Dict[str, Any]],
    fold: Dict[str, Any],
    dims: Dict[str, int],
    device: torch.device,
) -> float:
    """Optuna objective: returns negative val AUROC (minimization)."""
    lr         = trial.suggest_float("lr",         1e-4, 1e-2, log=True)
    d_proj     = trial.suggest_categorical("d_proj",     [32, 64, 128])
    d_hidden   = trial.suggest_categorical("d_hidden",   [64, 128, 256])
    dropout    = trial.suggest_float("dropout",    0.1, 0.5)
    aux_lambda = trial.suggest_float("aux_lambda", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_recs = filter_records_by_ids(records, fold["train_ids"])
    val_recs   = filter_records_by_ids(records, fold["val_ids"])

    train_samples = records_to_samples(train_recs)
    val_samples   = records_to_samples(val_recs)

    if not train_samples or not val_samples:
        return 0.5   # report chance level

    model = build_model(
        d_p1=dims["D1"], d_p2=dims["D2"],
        d_p3=dims["D3"], d_global=dims["D4"],
        d_proj=d_proj, d_hidden=d_hidden, dropout=dropout,
    )

    trainer = Trainer(
        model, device,
        lr=lr,
        batch_size=batch_size,
        max_epochs=30,   # reduced for hyperopt
        patience=5,
        aux_lambda=aux_lambda,
    )

    result = trainer.fit(train_samples, val_samples)
    return -result["best_val_auroc"]   # minimize → maximize AUROC


def run_hyperopt(
    records: List[Dict[str, Any]],
    n_trials: int    = cfg.OPTUNA_TRIALS,
    timeout:  int    = cfg.OPTUNA_TIMEOUT,
    fold_idx: int    = 0,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search.

    Parameters
    ----------
    records  : all records with feature_seqs
    n_trials : number of Optuna trials
    timeout  : wall-clock timeout in seconds
    fold_idx : which LOSO fold to use for evaluation

    Returns
    -------
    dict with "best_params" and "best_val_auroc"
    """
    if not _HAS_OPTUNA:
        warnings.warn("optuna not available — returning config defaults.")
        return {
            "best_params": {
                "lr":         cfg.LEARNING_RATE,
                "d_proj":     cfg.D_PROJ,
                "d_hidden":   cfg.D_HIDDEN,
                "dropout":    cfg.DROPOUT,
                "aux_lambda": cfg.AUX_LAMBDA,
                "batch_size": cfg.BATCH_SIZE,
            },
            "best_val_auroc": float("nan"),
        }

    device = _device()
    dims   = get_feature_dims()
    folds  = make_loso_folds(records)

    if fold_idx >= len(folds):
        fold_idx = 0
    fold = folds[fold_idx]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
    )

    print(f"\nRunning Optuna ({n_trials} trials, timeout={timeout}s) "
          f"on fold {fold_idx} ({fold['test_site']}) ...")

    study.optimize(
        lambda trial: _objective(trial, records, fold, dims, device),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best = study.best_trial
    best_params = best.params
    best_auroc  = -best.value

    print(f"\nBest val AUROC: {best_auroc:.4f}")
    print(f"Best params:    {best_params}")

    return {
        "best_params":     best_params,
        "best_val_auroc":  best_auroc,
    }
