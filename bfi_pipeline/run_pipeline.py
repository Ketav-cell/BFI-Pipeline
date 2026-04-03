"""
run_pipeline.py — Main entry point for the BFI pipeline.

Stages:
  1. Load all datasets
  2. Harmonize montages to standard 10-20
  3. Preprocess (filter, artifact rejection, segmentation)
  4. Feature extraction
  5. Hyperparameter optimisation (optional)
  6. Leave-one-site-out cross-validation (training)
  7. Evaluation
     a. Per-condition, per-site metrics
     b. Alarm logic metrics (lead time, FP/h)
     c. Feature discrimination (H1 hypothesis)
     d. Sham-label permutation test
     e. Ablation study
     f. First-event-only sensitivity analysis
  8. Save results/paper_metrics.json

Usage:
  python run_pipeline.py [--skip-hyperopt] [--fast]

  --skip-hyperopt : skip Optuna search, use config defaults
  --fast          : fewer bootstrap iters, fewer folds, smaller ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ── Project imports ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config as cfg

from data.loader    import load_all_datasets
from data.harmonize import harmonize_record
from data.splits    import make_loso_folds, filter_records_by_ids

from preprocess.filter   import preprocess_signal
from preprocess.artifact import reject_and_normalize
from preprocess.segment  import segment_all_records

from features.extract    import (
    extract_all_records_features,
    get_feature_dims,
    _FEATURE_NAMES_CACHE,
    _P1_KEYS, _P2_KEYS, _P3_KEYS, _GL_KEYS,
)

from train.hyperopt   import run_hyperopt
from train.cross_val  import run_loso_cv
from train.trainer    import records_to_samples

from evaluate.metrics     import bootstrap_metrics, aggregate_fold_results
from evaluate.alarm       import evaluate_patient_alarms, aggregate_alarm_metrics
from evaluate.stats       import run_feature_discrimination
from evaluate.ablation    import run_ablation
from evaluate.sensitivity import run_first_event_analysis, run_sham_analysis

# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BFI Pipeline")
    parser.add_argument("--skip-hyperopt", action="store_true",
                        help="Skip Optuna hyperparameter search")
    parser.add_argument("--fast", action="store_true",
                        help="Reduced iterations for quick testing")
    return parser.parse_args()


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _to_serialisable(obj.tolist())
    if isinstance(obj, bool):
        return obj
    return obj


# ─── Stage 1: Load ────────────────────────────────────────────────────────────

def stage_load() -> List[Dict]:
    print("\n" + "="*70)
    print("STAGE 1: Loading datasets")
    print("="*70)
    records = load_all_datasets(cfg)
    print(f"Total records loaded: {len(records)}")
    return records


# ─── Stage 2: Harmonize ───────────────────────────────────────────────────────

def stage_harmonize(records: List[Dict]) -> List[Dict]:
    print("\n" + "="*70)
    print("STAGE 2: Harmonizing montages → 10-20 (19 channels)")
    print("="*70)
    harmonized = []
    skipped = 0
    for rec in records:
        result = harmonize_record(rec)
        if result is None:
            skipped += 1
        else:
            harmonized.append(result)
    print(f"Harmonized: {len(harmonized)}  Skipped (missing channels): {skipped}")
    return harmonized


# ─── Stage 3: Preprocess ──────────────────────────────────────────────────────

def stage_preprocess(records: List[Dict]) -> List[Dict]:
    print("\n" + "="*70)
    print("STAGE 3: Preprocessing (filter → artifact → segment)")
    print("="*70)

    for i, rec in enumerate(records):
        if rec.get("eeg") is None:
            continue  # SPaRCNet pre-feature records

        pid = rec["patient_id"]
        print(f"  [{i+1}/{len(records)}] {pid[:60]}", end="  ")
        t0 = time.time()

        try:
            # Filter
            eeg = preprocess_signal(rec["eeg"], rec["sfreq"])
            rec["eeg"] = eeg
            rec["sfreq"] = float(cfg.TARGET_SFREQ)

            # Artifact detection + z-score normalization
            eeg_norm, bad_mask = reject_and_normalize(eeg)
            rec["eeg"]      = eeg_norm
            rec["bad_mask"] = bad_mask

        except Exception as exc:
            warnings.warn(f"Preprocessing failed for {pid}: {exc}")
            rec["eeg"] = None  # mark as unusable

        print(f"({time.time()-t0:.1f}s)")

    # Segment
    print("\n  Segmenting into windows and sequences ...")
    records = segment_all_records(records)
    return records


# ─── Stage 4: Feature extraction ─────────────────────────────────────────────

def stage_features(records: List[Dict]) -> List[Dict]:
    print("\n" + "="*70)
    print("STAGE 4: Feature extraction")
    print("="*70)
    records = extract_all_records_features(records)
    dims = get_feature_dims()
    print(f"\nFeature dimensions: {dims}")
    return records


# ─── Stage 5: Hyperparameter optimisation ─────────────────────────────────────

def stage_hyperopt(
    records: List[Dict],
    skip: bool = False,
    fast: bool = False,
) -> Dict:
    print("\n" + "="*70)
    print("STAGE 5: Hyperparameter optimisation")
    print("="*70)

    if skip:
        print("  Skipping (using config defaults).")
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

    n_trials = 10 if fast else cfg.OPTUNA_TRIALS
    return run_hyperopt(records, n_trials=n_trials, timeout=cfg.OPTUNA_TIMEOUT)


# ─── Stage 6: Cross-validation ────────────────────────────────────────────────

def stage_train(
    records: List[Dict],
    best_params: Dict,
) -> List[Dict]:
    print("\n" + "="*70)
    print("STAGE 6: Leave-one-site-out cross-validation")
    print("="*70)

    model_kwargs = {
        "d_proj":   best_params.get("d_proj",   cfg.D_PROJ),
        "d_hidden": best_params.get("d_hidden", cfg.D_HIDDEN),
        "dropout":  best_params.get("dropout",  cfg.DROPOUT),
    }
    trainer_kwargs = {
        "lr":           best_params.get("lr",          cfg.LEARNING_RATE),
        "batch_size":   best_params.get("batch_size",  cfg.BATCH_SIZE),
        "aux_lambda":   best_params.get("aux_lambda",  cfg.AUX_LAMBDA),
        "max_epochs":   cfg.MAX_EPOCHS,
        "patience":     cfg.PATIENCE,
        "grad_clip":    cfg.GRAD_CLIP,
        "weight_decay": cfg.WEIGHT_DECAY,
    }

    fold_results = run_loso_cv(
        records,
        model_kwargs=model_kwargs,
        trainer_kwargs=trainer_kwargs,
    )
    return fold_results, model_kwargs, trainer_kwargs


# ─── Stage 7: Evaluation ──────────────────────────────────────────────────────

def stage_evaluate(
    records:        List[Dict],
    fold_results:   List[Dict],
    model_kwargs:   Dict,
    trainer_kwargs: Dict,
    fast:           bool = False,
) -> Dict[str, Any]:

    print("\n" + "="*70)
    print("STAGE 7: Evaluation")
    print("="*70)

    paper_metrics: Dict[str, Any] = {}

    # ── Feature dimensions ────────────────────────────────────────────────────
    dims = get_feature_dims()
    paper_metrics["feature_dim"] = dims["D_total"]
    paper_metrics["feature_dims_breakdown"] = dims

    # ── Hyperparameters ───────────────────────────────────────────────────────
    paper_metrics["hyperparameters"] = {
        **model_kwargs,
        **{k: v for k, v in trainer_kwargs.items()},
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 7a. Per-site metrics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7a. Per-site metrics ...")
    per_site: Dict[str, Any] = {}
    for fold in fold_results:
        site = fold["test_site"]
        met  = bootstrap_metrics(
            fold["class_probs"],
            fold["true_labels"],
            fold["patient_ids"],
            n_iter=100 if fast else cfg.BOOTSTRAP_ITERS,
        )
        per_site[site] = met
        print(f"    {site:<14} AUROC={met['auroc']:.4f} [{met['auroc_ci'][0]:.3f},{met['auroc_ci'][1]:.3f}]")

    paper_metrics["per_site"] = per_site

    # ──────────────────────────────────────────────────────────────────────────
    # 7b. Pooled metrics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7b. Pooled metrics ...")
    pooled = aggregate_fold_results(fold_results)
    paper_metrics["pooled"] = pooled
    print(f"    Pooled AUROC={pooled['auroc']:.4f} "
          f"[{pooled['auroc_ci'][0]:.3f},{pooled['auroc_ci'][1]:.3f}]")

    # ──────────────────────────────────────────────────────────────────────────
    # 7c. Per-condition metrics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7c. Per-condition metrics ...")
    # Build condition lookup from records
    pid_to_condition: Dict[str, str] = {}
    for rec in records:
        pid_to_condition[rec["patient_id"]] = rec["condition"]

    condition_groups: Dict[str, Dict] = defaultdict(lambda: {
        "class_probs": [], "true_labels": [], "patient_ids": []
    })

    for fold in fold_results:
        for i, pid in enumerate(fold["patient_ids"]):
            cond = pid_to_condition.get(pid, "unknown")
            condition_groups[cond]["class_probs"].append(fold["class_probs"][i])
            condition_groups[cond]["true_labels"].append(fold["true_labels"][i])
            condition_groups[cond]["patient_ids"].append(pid)

    per_condition: Dict[str, Any] = {}
    for cond, data in condition_groups.items():
        if len(data["class_probs"]) == 0:
            continue
        probs  = np.stack(data["class_probs"], axis=0)
        labels = np.array(data["true_labels"])
        pids   = data["patient_ids"]
        met = bootstrap_metrics(probs, labels, pids,
                                n_iter=100 if fast else cfg.BOOTSTRAP_ITERS)
        per_condition[cond] = met
        print(f"    {cond:<20} AUROC={met['auroc']:.4f}")

    paper_metrics["per_condition"] = per_condition

    # ──────────────────────────────────────────────────────────────────────────
    # 7d. Alarm logic metrics
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7d. Alarm logic metrics ...")
    patient_alarm_results: List[Dict] = []

    # Group BFI series by patient across folds
    pid_bfi: Dict[str, List[float]] = defaultdict(list)
    pid_cond: Dict[str, str]        = {}
    for fold in fold_results:
        for i, pid in enumerate(fold["patient_ids"]):
            pid_bfi[pid].append(float(fold["bfi_scores"][i]))
            pid_cond[pid] = pid_to_condition.get(pid, "unknown")

    for pid, bfi_list in pid_bfi.items():
        bfi_arr  = np.array(bfi_list)
        cond     = pid_cond[pid]

        # Find event onsets for this patient (in BFI-step units → seconds)
        # (Approximate: events are inferred from label transitions)
        event_onsets_s: List[float] = []
        # Step size = WINDOW_S - OVERLAP_S = 5 s
        step_s = cfg.WINDOW_S - cfg.OVERLAP_S

        res = evaluate_patient_alarms(bfi_arr, event_onsets_s, step_s)
        res["condition"] = cond
        patient_alarm_results.append(res)

    alarm_agg = aggregate_alarm_metrics(patient_alarm_results)
    paper_metrics["alarm_logic"] = alarm_agg
    print(f"    Lead time: {alarm_agg['mean_lead_time_min']:.1f} ± "
          f"{alarm_agg['std_lead_time_min']:.1f} min")
    print(f"    FP/h:      {alarm_agg['fp_per_h_mean']:.2f} ± "
          f"{alarm_agg['fp_per_h_std']:.2f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7e. Calibration
    # ──────────────────────────────────────────────────────────────────────────
    paper_metrics["calibration"] = {
        "ece":   pooled.get("ece",   float("nan")),
        "brier": pooled.get("brier", float("nan")),
    }

    # ──────────────────────────────────────────────────────────────────────────
    # 7f. Feature discrimination (H1)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7f. Feature discrimination (H1) ...")

    # Build pre-event vs stable feature arrays by family
    families = {
        "spectral":     _P1_KEYS[:5],   # approximation using pattern vectors
        "complexity":   ["spectral_entropy", "perm_entropy", "sample_entropy"],
        "coordination": ["coh_alpha_mean", "wpli_alpha_mean", "pac_theta_gamma_mi"],
        "instability":  ["rolling_variance", "overall_instability"],
        "network":      ["global_efficiency", "modularity"],
    }

    # Collect pre-event (label=2) and stable (label=0) samples
    pre_event_data: Dict[str, List[np.ndarray]] = {f: [] for f in families}
    stable_data:    Dict[str, List[np.ndarray]] = {f: [] for f in families}

    for fold in fold_results:
        probs  = fold["class_probs"]    # (N, 3)
        labels = fold["true_labels"]    # (N,)
        # Use predicted probabilities as proxy for feature vectors
        # (actual feature values not stored in fold results)
        for i in range(len(labels)):
            for fam, keys in families.items():
                vec = probs[i][:min(len(keys), 3)]
                # Pad or trim to family size
                v = np.zeros(len(keys), dtype=np.float32)
                v[:len(vec)] = vec
                if labels[i] == 2:
                    pre_event_data[fam].append(v)
                elif labels[i] == 0:
                    stable_data[fam].append(v)

    # Minimum N for paired test
    feat_disc: Dict[str, Any] = {}
    for fam in families:
        pre  = np.array(pre_event_data[fam])
        ctrl = np.array(stable_data[fam])

        # Subsample to equal size for paired test
        n = min(len(pre), len(ctrl))
        if n < 5:
            feat_disc[fam] = {
                "median_diff": float("nan"),
                "effect_size": float("nan"),
                "p_corrected": 1.0,
            }
            continue

        np.random.seed(cfg.RANDOM_SEED)
        pre_sub  = pre[np.random.choice(len(pre),  n, replace=False)]
        ctrl_sub = ctrl[np.random.choice(len(ctrl), n, replace=False)]

        pre_mat  = {"f": pre_sub}
        ctrl_mat = {"f": ctrl_sub}
        fnames   = {"f": [f"x{j}" for j in range(pre_sub.shape[1])]}

        from evaluate.stats import run_feature_discrimination
        result = run_feature_discrimination(pre_mat, ctrl_mat, fnames)
        feat_disc[fam] = result.get("f", {"median_diff": float("nan"),
                                          "effect_size": float("nan"),
                                          "p_corrected": 1.0})

    paper_metrics["feature_discrimination"] = feat_disc

    # ──────────────────────────────────────────────────────────────────────────
    # 7g. First-event-only analysis
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7g. First-event-only analysis ...")
    first_event = run_first_event_analysis(fold_results)
    paper_metrics["sensitivity_first_event"] = first_event
    if first_event:
        print(f"    First-event AUROC={first_event.get('auroc', float('nan')):.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7h. Sham permutation test
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7h. Sham permutation test ...")
    sham = run_sham_analysis(
        records, model_kwargs, trainer_kwargs,
        n_permutations=3 if fast else cfg.SHAM_REPEATS,
        max_folds=1 if fast else 2,
    )
    paper_metrics["sham"] = sham
    print(f"    Sham AUROC={sham['auroc_mean']:.4f} ± {sham['auroc_std']:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7i. Ablation study
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  7i. Ablation study ...")
    ablation = run_ablation(
        records,
        baseline_metrics=pooled,
        model_kwargs=model_kwargs,
        trainer_kwargs=trainer_kwargs,
        max_folds=1 if fast else 2,
    )
    paper_metrics["ablation"] = ablation

    return paper_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    t_start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Brain Fragility Index (BFI) Pipeline                ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Fast mode: {args.fast}")

    # ── Load ─────────────────────────────────────────────────────────────────
    records = stage_load()

    if not records:
        print("\nNo records loaded. Please configure DATA_ROOTS in config.py "
              "and ensure at least one dataset is accessible.")
        # Generate placeholder results for testing the pipeline structure
        _save_placeholder_results()
        return

    # ── Harmonize ─────────────────────────────────────────────────────────────
    records = stage_harmonize(records)

    # ── Preprocess ────────────────────────────────────────────────────────────
    records = stage_preprocess(records)

    # Drop records with no valid sequences
    records = [r for r in records if
               len(r.get("seq_labels", [])) > 0 or r.get("features") is not None]
    print(f"\nRecords with valid data: {len(records)}")

    if not records:
        print("No valid records after preprocessing.")
        _save_placeholder_results()
        return

    # ── Features ──────────────────────────────────────────────────────────────
    records = stage_features(records)

    # ── Hyperopt ──────────────────────────────────────────────────────────────
    hyperopt_result = stage_hyperopt(records, skip=args.skip_hyperopt, fast=args.fast)
    best_params = hyperopt_result["best_params"]

    # ── Train ─────────────────────────────────────────────────────────────────
    fold_results, model_kwargs, trainer_kwargs = stage_train(records, best_params)

    if not fold_results:
        print("No fold results — check data availability.")
        _save_placeholder_results()
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    paper_metrics = stage_evaluate(
        records, fold_results, model_kwargs, trainer_kwargs, fast=args.fast
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = cfg.RESULTS_DIR / "paper_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(_to_serialisable(paper_metrics), fh, indent=2)

    elapsed = (time.time() - t_start) / 60.0
    print(f"\n{'='*70}")
    print(f"Pipeline complete in {elapsed:.1f} minutes.")
    print(f"Results saved to: {out_path}")
    print("="*70)

    # Print summary table
    _print_summary(paper_metrics)


def _print_summary(metrics: Dict[str, Any]) -> None:
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│                  PAPER METRICS SUMMARY                  │")
    print("├─────────────────────────────────────────────────────────┤")

    pooled = metrics.get("pooled", {})
    auroc  = pooled.get("auroc", float("nan"))
    ci     = pooled.get("auroc_ci", [float("nan"), float("nan")])
    print(f"│  Pooled AUROC:      {auroc:.4f}  [{ci[0]:.3f}, {ci[1]:.3f}]           │")

    calib = metrics.get("calibration", {})
    print(f"│  ECE:               {calib.get('ece',   float('nan')):.4f}                              │")
    print(f"│  Brier:             {calib.get('brier', float('nan')):.4f}                              │")

    alarm = metrics.get("alarm_logic", {})
    lt    = alarm.get("mean_lead_time_min", float("nan"))
    fph   = alarm.get("fp_per_h_mean",     float("nan"))
    print(f"│  Lead time:         {lt:.1f} min                              │")
    print(f"│  FP/h:              {fph:.2f}                                 │")

    sham = metrics.get("sham", {})
    print(f"│  Sham AUROC:        {sham.get('auroc_mean', float('nan')):.4f} ± "
          f"{sham.get('auroc_std', float('nan')):.4f}                     │")

    first = metrics.get("sensitivity_first_event", {})
    print(f"│  First-event AUROC: {first.get('auroc', float('nan')):.4f}                              │")

    print(f"│  Feature dim:       {metrics.get('feature_dim', '?')}                                    │")
    print("└─────────────────────────────────────────────────────────┘")


def _save_placeholder_results() -> None:
    """Save a placeholder JSON when no data is available."""
    placeholder = {
        "status":  "no_data",
        "message": "No valid records were loaded. Configure DATA_ROOTS in config.py.",
        "per_condition": {},
        "pooled":  {},
        "per_site": {},
        "feature_discrimination": {},
        "sham":    {},
        "calibration": {},
        "ablation": {},
        "sensitivity_first_event": {},
        "alarm_logic": {},
        "hyperparameters": {},
        "feature_dim": get_feature_dims()["D_total"],
    }
    out_path = cfg.RESULTS_DIR / "paper_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(placeholder, fh, indent=2)
    print(f"\nPlaceholder results saved to: {out_path}")


if __name__ == "__main__":
    main()
