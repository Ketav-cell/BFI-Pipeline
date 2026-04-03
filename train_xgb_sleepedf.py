#!/usr/bin/env python3
"""
Sleep-EDF: Wake vs Sleep Classification
Distinguishes conscious (wake) from unconscious (sleep) states
"""

import os
import json
import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

SLEEP_PATH = Path("physionet.org/files/sleep-edfx/1.0.0/sleep-cassette")
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

def extract_features(signal, fs):
    """Extract spectral + statistical features."""
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(signal) < 1e-10:
        return None
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs*4)))
    except:
        return None
    
    delta = trapezoid(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)]) if any((freqs >= 0.5) & (freqs < 4)) else 0
    theta = trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)]) if any((freqs >= 4) & (freqs < 8)) else 0
    alpha = trapezoid(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)]) if any((freqs >= 8) & (freqs < 13)) else 0
    beta = trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)]) if any((freqs >= 13) & (freqs < 30)) else 0
    gamma = trapezoid(psd[(freqs >= 30) & (freqs < 45)], freqs[(freqs >= 30) & (freqs < 45)]) if any((freqs >= 30) & (freqs < 45)) else 0
    
    total = delta + theta + alpha + beta + gamma + 1e-10
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    # Hjorth parameters
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    
    return np.array([
        delta/total, theta/total, alpha/total, beta/total, gamma/total,
        theta/(alpha+1e-10), delta/(alpha+1e-10), (alpha+beta)/(delta+theta+1e-10),
        np.log10(total+1e-10), np.log10(delta+1e-10), np.log10(alpha+1e-10),
        np.var(signal), skew(signal), kurtosis(signal),
        np.mean(np.abs(np.diff(signal))),
        np.sum(np.diff(np.sign(signal)) != 0)/len(signal),
        spectral_entropy, activity, mobility, complexity
    ])

def load_recording(psg_file, hypno_file):
    """Load PSG, extract Wake and Sleep epochs."""
    try:
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
        annot = mne.read_annotations(str(hypno_file))
        raw.set_annotations(annot)
        fs = raw.info['sfreq']
        
        # Get EEG channel
        eeg_ch = [ch for ch in raw.ch_names if 'EEG' in ch or 'Fpz' in ch]
        if not eeg_ch:
            eeg_ch = raw.ch_names[:1]
        raw.pick_channels(eeg_ch[:1])
        
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Map stages
        wake_ids = [eid for name, eid in event_id.items() if 'W' in name]
        sleep_ids = [eid for name, eid in event_id.items() if any(s in name for s in ['1','2','3','4','R'])]
        
        if not wake_ids or not sleep_ids:
            return None, None
        
        epoch_len = int(30 * fs)
        X_wake, X_sleep = [], []
        
        for i in range(len(events)):
            start = events[i][0]
            if start + epoch_len > raw.n_times:
                continue
            
            data = raw.get_data(start=start, stop=start+epoch_len)[0]
            feats = extract_features(data, fs)
            
            if feats is None:
                continue
            
            if events[i][2] in wake_ids:
                X_wake.append(feats)
            elif events[i][2] in sleep_ids:
                X_sleep.append(feats)
        
        return X_wake, X_sleep
    except Exception as e:
        return None, None

def main():
    print("=" * 60)
    print("Sleep-EDF: Wake vs Sleep Classification")
    print("=" * 60)
    
    psg_files = sorted(SLEEP_PATH.glob("*-PSG.edf"))[:50]
    print(f"Processing {len(psg_files)} recordings...")
    
    all_wake, all_sleep = [], []
    loaded = 0
    
    for psg in psg_files:
        base = psg.name[:6]
        hypno = list(SLEEP_PATH.glob(f"{base}*Hypnogram.edf"))
        if not hypno:
            continue
        
        wake, sleep = load_recording(psg, hypno[0])
        if wake and sleep and len(wake) > 3 and len(sleep) > 3:
            all_wake.extend(wake)
            all_sleep.extend(sleep)
            loaded += 1
            print(f"  {psg.name}: {len(wake)} wake, {len(sleep)} sleep")
    
    print(f"\nTotal: {len(all_wake)} wake, {len(all_sleep)} sleep from {loaded} subjects")
    
    # Balance classes
    n_samples = min(len(all_wake), len(all_sleep), 500)
    np.random.seed(42)
    
    wake_idx = np.random.choice(len(all_wake), n_samples, replace=False)
    sleep_idx = np.random.choice(len(all_sleep), n_samples, replace=False)
    
    X_wake = np.array([all_wake[i] for i in wake_idx])
    X_sleep = np.array([all_sleep[i] for i in sleep_idx])
    
    X = np.vstack([X_wake, X_sleep])
    y = np.array([0]*n_samples + [1]*n_samples)  # 0=Wake, 1=Sleep
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X = np.nan_to_num(X)
    
    print(f"Balanced dataset: {len(X)} epochs ({n_samples} wake, {n_samples} sleep)")
    
    # Scale
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs, accs = [], []
    
    print("\nTraining XGBoost...")
    for fold, (tr, te) in enumerate(cv.split(X_sc, y)):
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
        model.fit(X_sc[tr], y[tr])
        prob = model.predict_proba(X_sc[te])[:,1]
        pred = model.predict(X_sc[te])
        aurocs.append(roc_auc_score(y[te], prob))
        accs.append(accuracy_score(y[te], pred))
        print(f"  Fold {fold+1}: AUROC={aurocs[-1]:.3f}, Acc={accs[-1]:.3f}")
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Subjects: {loaded}")
    print(f"Epochs: {len(X)} ({n_samples} per class)")
    print(f"AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    results = {
        "dataset": "Sleep-EDF",
        "task": "wake_vs_sleep_classification",
        "n_subjects": loaded,
        "n_epochs": len(X),
        "n_per_class": n_samples,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs))
    }
    
    with open(RESULTS_PATH / "sleepedf_xgb_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/sleepedf_xgb_results.json")

if __name__ == "__main__":
    main()
