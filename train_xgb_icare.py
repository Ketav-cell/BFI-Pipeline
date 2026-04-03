#!/usr/bin/env python3
"""
I-CARE Coma Outcome Prediction using XGBoost
"""

import os
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ICARE_PATH = Path("physionet.org/files/i-care/2.1/training")
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)
FS = 250
N_FEATURES = 150  # Fixed feature size (10 channels x 15 features)

def extract_features(eeg_data, fs=FS, n_channels=10):
    """Extract features, pad/truncate to fixed channel count."""
    all_features = []
    
    # Limit to n_channels
    n_ch = min(eeg_data.shape[0], n_channels)
    
    for ch in range(n_ch):
        signal = np.nan_to_num(eeg_data[ch, :], nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(signal) < 1e-10:
            all_features.extend([0] * 15)
            continue
        try:
            freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs*2))
        except:
            all_features.extend([0] * 15)
            continue
        
        delta = trapezoid(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)]) if any((freqs >= 0.5) & (freqs < 4)) else 0
        theta = trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)]) if any((freqs >= 4) & (freqs < 8)) else 0
        alpha = trapezoid(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)]) if any((freqs >= 8) & (freqs < 13)) else 0
        beta = trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)]) if any((freqs >= 13) & (freqs < 30)) else 0
        
        total_power = delta + theta + alpha + beta + 1e-10
        rel_delta, rel_theta, rel_alpha, rel_beta = delta/total_power, theta/total_power, alpha/total_power, beta/total_power
        theta_alpha, delta_alpha = theta/(alpha+1e-10), delta/(alpha+1e-10)
        variance, sig_skew, sig_kurt = np.var(signal), skew(signal), kurtosis(signal)
        line_length = np.mean(np.abs(np.diff(signal)))
        zero_cross = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        
        all_features.extend([rel_delta, rel_theta, rel_alpha, rel_beta, theta_alpha, delta_alpha, 
                            np.log10(total_power+1e-10), variance, sig_skew, sig_kurt, 
                            line_length, zero_cross, delta, theta, alpha])
    
    # Pad if fewer channels
    while len(all_features) < N_FEATURES:
        all_features.append(0)
    
    return np.array(all_features[:N_FEATURES])

def load_patient_data(patient_id):
    patient_path = ICARE_PATH / patient_id
    outcome_file = patient_path / f"{patient_id}.txt"
    if not outcome_file.exists():
        return None, None
    
    with open(outcome_file) as f:
        content = f.read()
    
    outcome = None
    for line in content.split('\n'):
        if line.startswith('Outcome:'):
            outcome = line.split(':')[1].strip()
            break
    
    if outcome is None:
        return None, None
    
    label = 0 if outcome == 'Good' else 1
    eeg_files = sorted(patient_path.glob("*_EEG.mat"))
    if not eeg_files:
        return None, None
    
    all_features = []
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
            
            feats = extract_features(eeg_data)
            if len(feats) == N_FEATURES and not np.all(feats == 0):
                all_features.append(feats)
        except Exception as e:
            continue
    
    if not all_features:
        return None, None
    
    return np.mean(all_features, axis=0), label

def main():
    print("=" * 60)
    print("I-CARE Coma Outcome Prediction")
    print("=" * 60)
    
    patients = sorted([p.name for p in ICARE_PATH.iterdir() if p.is_dir() and list(p.glob("*_EEG.mat"))])
    print(f"Found {len(patients)} patients with EEG data")
    
    X, y, patient_ids = [], [], []
    for pid in patients:
        print(f"Loading {pid}...", end=" ")
        features, label = load_patient_data(pid)
        if features is not None and len(features) == N_FEATURES:
            X.append(features)
            y.append(label)
            patient_ids.append(pid)
            print(f"{'Good' if label == 0 else 'Poor'}")
        else:
            print("skipped")
    
    X = np.array(X)
    y = np.array(y)
    X = np.nan_to_num(X)
    
    print(f"\nLoaded {len(X)} patients: {np.sum(y==0)} Good, {np.sum(y==1)} Poor")
    
    if len(X) < 4:
        print("Not enough patients for analysis")
        return
    
    loo = LeaveOneOut()
    y_true, y_pred, y_prob = [], [], []
    
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X[train_idx])
        X_test_sc = scaler.transform(X[test_idx])
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                                   random_state=42, use_label_encoder=False, 
                                   eval_metric='logloss', verbosity=0)
        model.fit(X_train_sc, y[train_idx])
        
        y_true.append(y[test_idx][0])
        y_pred.append(model.predict(X_test_sc)[0])
        y_prob.append(model.predict_proba(X_test_sc)[0, 1])
    
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"Patients: {len(y_true)} ({np.sum(y_true==0)} Good, {np.sum(y_true==1)} Poor)")
    print(f"AUROC: {auroc:.3f}")
    print(f"Accuracy: {acc:.3f}")
    
    print("\nPer-patient predictions:")
    for i, pid in enumerate(patient_ids):
        true_str = "Good" if y_true[i] == 0 else "Poor"
        pred_str = "Good" if y_pred[i] == 0 else "Poor"
        mark = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"  {pid}: True={true_str}, Pred={pred_str}, Prob={y_prob[i]:.3f} {mark}")
    
    results = {"dataset": "I-CARE", "task": "coma_outcome", "auroc": float(auroc), 
               "accuracy": float(acc), "n_patients": len(y_true),
               "n_good": int(np.sum(y_true==0)), "n_poor": int(np.sum(y_true==1))}
    
    with open(RESULTS_PATH / "icare_xgb_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/icare_xgb_results.json")

if __name__ == "__main__":
    main()
