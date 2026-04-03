#!/usr/bin/env python3
"""
I-CARE Coma: Segment-level outcome prediction
Detects EEG signatures associated with poor neurological outcome
"""

import json
import numpy as np
from scipy.io import loadmat
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

ICARE_PATH = Path("physionet.org/files/i-care/2.1/training")
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)
FS = 250

def extract_features(signal, fs=FS):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < fs or np.std(signal) < 1e-10:
        return None
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs*2))
    except:
        return None
    
    delta = trapezoid(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)]) if any((freqs >= 0.5) & (freqs < 4)) else 0
    theta = trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)]) if any((freqs >= 4) & (freqs < 8)) else 0
    alpha = trapezoid(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)]) if any((freqs >= 8) & (freqs < 13)) else 0
    beta = trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)]) if any((freqs >= 13) & (freqs < 30)) else 0
    gamma = trapezoid(psd[(freqs >= 30) & (freqs < 45)], freqs[(freqs >= 30) & (freqs < 45)]) if any((freqs >= 30) & (freqs < 45)) else 0
    
    total = delta + theta + alpha + beta + gamma + 1e-10
    
    # Suppression ratio
    threshold = 0.1 * np.std(signal)
    suppression_ratio = np.mean(np.abs(signal) < threshold)
    
    # Hjorth
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    return np.array([
        delta/total, theta/total, alpha/total, beta/total, gamma/total,
        theta/(alpha+1e-10), delta/(alpha+1e-10), (alpha+beta)/(delta+theta+1e-10),
        np.log10(total+1e-10), suppression_ratio,
        activity, mobility, complexity, spectral_entropy,
        skew(signal), kurtosis(signal),
        np.mean(np.abs(np.diff(signal))),
        np.sum(np.diff(np.sign(signal)) != 0)/len(signal),
        np.max(np.abs(signal)), np.percentile(np.abs(signal), 95)
    ])

def main():
    print("=" * 60)
    print("I-CARE Coma: Segment-Level Outcome Prediction")
    print("=" * 60)
    
    patients = sorted([p.name for p in ICARE_PATH.iterdir() if p.is_dir() and list(p.glob("*_EEG.mat"))])
    
    X_good, X_poor = [], []
    
    for pid in patients:
        patient_path = ICARE_PATH / pid
        outcome_file = patient_path / f"{pid}.txt"
        
        if not outcome_file.exists():
            continue
        
        with open(outcome_file) as f:
            content = f.read()
        
        outcome = None
        for line in content.split('\n'):
            if line.startswith('Outcome:'):
                outcome = line.split(':')[1].strip()
                break
        
        if outcome is None:
            continue
        
        is_poor = (outcome == 'Poor')
        eeg_files = sorted(patient_path.glob("*_EEG.mat"))
        patient_segments = 0
        
        for eeg_file in eeg_files:
            try:
                mat = loadmat(str(eeg_file))
                eeg_data = mat.get('val', None)
                if eeg_data is None:
                    for key in mat.keys():
                        if not key.startswith('__'):
                            eeg_data = mat[key]
                            break
                
                if eeg_data is None or eeg_data.size == 0:
                    continue
                
                if eeg_data.ndim == 1:
                    eeg_data = eeg_data.reshape(1, -1)
                
                # Extract 10-second segments
                window_samples = 10 * FS
                for start in range(0, eeg_data.shape[1] - window_samples, window_samples):
                    segment = np.mean(eeg_data[:, start:start+window_samples], axis=0)
                    feats = extract_features(segment)
                    
                    if feats is not None:
                        if is_poor:
                            X_poor.append(feats)
                        else:
                            X_good.append(feats)
                        patient_segments += 1
                        
            except:
                continue
        
        print(f"  {pid}: {patient_segments} segments, {outcome}")
    
    print(f"\nTotal: {len(X_good)} Good, {len(X_poor)} Poor segments")
    
    if len(X_good) < 50 or len(X_poor) < 50:
        print("Not enough segments")
        return
    
    # Balance classes
    n_samples = min(len(X_good), len(X_poor), 500)
    np.random.seed(42)
    
    idx_good = np.random.choice(len(X_good), n_samples, replace=False)
    idx_poor = np.random.choice(len(X_poor), n_samples, replace=False)
    
    X = np.vstack([[X_good[i] for i in idx_good], [X_poor[i] for i in idx_poor]])
    y = np.array([0]*n_samples + [1]*n_samples)  # 0=Good, 1=Poor
    
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    X = np.nan_to_num(X)
    
    print(f"Balanced: {len(X)} segments ({n_samples} per class)")
    
    # 5-fold CV
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    
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
    print(f"Task: Detect EEG signatures of poor coma outcome")
    print(f"Segments: {len(X)} ({n_samples} per class)")
    print(f"AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    
    results = {
        "dataset": "I-CARE",
        "task": "coma_outcome_segment_level",
        "n_segments": len(X),
        "n_per_class": n_samples,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs))
    }
    
    with open(RESULTS_PATH / "coma_segment_xgb_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/coma_segment_xgb_results.json")

if __name__ == "__main__":
    main()
