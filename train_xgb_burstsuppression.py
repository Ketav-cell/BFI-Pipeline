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
ICARE_PATH = Path('physionet.org/files/i-care/2.1/training')
RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(exist_ok=True)
FS = 250

def extract_features(signal, fs=FS):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < fs * 2:
        return None
    if np.std(signal) < 1e-10:
        return None
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
    except:
        return None
    delta = trapezoid(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)]) if any((freqs >= 0.5) & (freqs < 4)) else 0
    theta = trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)]) if any((freqs >= 4) & (freqs < 8)) else 0
    alpha = trapezoid(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)]) if any((freqs >= 8) & (freqs < 13)) else 0
    beta = trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)]) if any((freqs >= 13) & (freqs < 30)) else 0
    total = delta + theta + alpha + beta + 1e-10
    threshold = 0.1 * np.std(signal)
    suppression_ratio = np.mean(np.abs(signal) < threshold)
    window_size = int(fs * 0.5)
    n_windows = len(signal) // window_size
    if n_windows > 1:
        window_amps = [np.std(signal[i * window_size:(i + 1) * window_size]) for i in range(n_windows)]
        amp_variance = np.var(window_amps)
        amp_cv = np.std(window_amps) / (np.mean(window_amps) + 1e-10)
    else:
        amp_variance = 0
        amp_cv = 0
    burst_threshold = 2 * np.std(signal)
    bursts = np.abs(signal) > burst_threshold
    burst_ratio = np.mean(bursts)
    zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    line_length = np.mean(np.abs(np.diff(signal)))
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return np.array([delta / total, theta / total, alpha / total, beta / total, np.log10(total + 1e-10), suppression_ratio, amp_variance, amp_cv, burst_ratio, zcr, line_length, activity, mobility, complexity, spectral_entropy, skew(signal), kurtosis(signal), np.max(np.abs(signal)), np.percentile(np.abs(signal), 95)])

def detect_bs_segments(eeg_data, fs=FS, window_sec=10):
    n_channels, n_samples = eeg_data.shape
    window_samples = int(window_sec * fs)
    segments = []
    labels = []
    for start in range(0, n_samples - window_samples, window_samples):
        end = start + window_samples
        segment = np.mean(eeg_data[:, start:end], axis=0)
        threshold = 0.15 * np.std(segment)
        suppression_ratio = np.mean(np.abs(segment) < threshold)
        if suppression_ratio > 0.3:
            labels.append(1)
            segments.append(segment)
        elif suppression_ratio < 0.1:
            labels.append(0)
            segments.append(segment)
    return (segments, labels)

def main():
    print('=' * 60)
    print('Burst Suppression Detection (I-CARE)')
    print('=' * 60)
    patients = sorted([p.name for p in ICARE_PATH.iterdir() if p.is_dir() and list(p.glob('*_EEG.mat'))])
    print(f'Found {len(patients)} patients with EEG data')
    all_X, all_y = ([], [])
    for pid in patients:
        patient_path = ICARE_PATH / pid
        eeg_files = sorted(patient_path.glob('*_EEG.mat'))
        patient_segments = 0
        patient_bs = 0
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
                segments, labels = detect_bs_segments(eeg_data)
                for seg, lab in zip(segments, labels):
                    feats = extract_features(seg)
                    if feats is not None:
                        all_X.append(feats)
                        all_y.append(lab)
                        patient_segments += 1
                        if lab == 1:
                            patient_bs += 1
            except Exception as e:
                continue
        if patient_segments > 0:
            print(f'  {pid}: {patient_segments} segments, {patient_bs} BS')
    if not all_X:
        print('No data extracted!')
        return
    X = np.array(all_X)
    y = np.array(all_y)
    X = np.nan_to_num(X)
    print(f'\nTotal: {len(X)} segments')
    print(f'Normal: {np.sum(y == 0)}, BS: {np.sum(y == 1)}')
    if np.sum(y == 0) < 20 or np.sum(y == 1) < 20:
        print('Not enough samples in each class')
        return
    n_min = min(np.sum(y == 0), np.sum(y == 1))
    np.random.seed(42)
    idx0 = np.random.choice(np.where(y == 0)[0], n_min, replace=False)
    idx1 = np.random.choice(np.where(y == 1)[0], n_min, replace=False)
    idx = np.concatenate([idx0, idx1])
    np.random.shuffle(idx)
    X_bal, y_bal = (X[idx], y[idx])
    print(f'Balanced: {len(X_bal)} segments ({n_min} each class)')
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_bal)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs, accs = ([], [])
    print('\nTraining XGBoost...')
    for fold, (tr, te) in enumerate(cv.split(X_sc, y_bal)):
        model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        model.fit(X_sc[tr], y_bal[tr])
        prob = model.predict_proba(X_sc[te])[:, 1]
        pred = model.predict(X_sc[te])
        aurocs.append(roc_auc_score(y_bal[te], prob))
        accs.append(accuracy_score(y_bal[te], pred))
        print(f'  Fold {fold + 1}: AUROC={aurocs[-1]:.3f}, Acc={accs[-1]:.3f}')
    print(f"\n{'=' * 60}\nRESULTS\n{'=' * 60}")
    print(f'Segments: {len(X_bal)} ({n_min} per class)')
    print(f'AUROC: {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f}')
    print(f'Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    results = {'dataset': 'I-CARE', 'task': 'burst_suppression_detection', 'n_segments': len(X_bal), 'n_per_class': int(n_min), 'auroc_mean': float(np.mean(aurocs)), 'auroc_std': float(np.std(aurocs)), 'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs))}
    with open(RESULTS_PATH / 'burstsuppression_xgb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to results/burstsuppression_xgb_results.json')
if __name__ == '__main__':
    main()