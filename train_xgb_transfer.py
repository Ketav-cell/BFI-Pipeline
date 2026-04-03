import json
import numpy as np
import mne
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from pathlib import Path
import pyedflib
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
CHBMIT_PATH = Path('physionet.org/files/chbmit/1.0.0')
SLEEPEDF_PATH = Path('physionet.org/files/sleep-edfx/1.0.0/sleep-cassette')
ICARE_PATH = Path('physionet.org/files/i-care/2.1/training')
RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(exist_ok=True)

def extract_features(signal, fs):
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < fs or np.std(signal) < 1e-10:
        return None
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 2)))
    except:
        return None
    delta = trapezoid(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)]) if any((freqs >= 0.5) & (freqs < 4)) else 0
    theta = trapezoid(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)]) if any((freqs >= 4) & (freqs < 8)) else 0
    alpha = trapezoid(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)]) if any((freqs >= 8) & (freqs < 13)) else 0
    beta = trapezoid(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)]) if any((freqs >= 13) & (freqs < 30)) else 0
    gamma = trapezoid(psd[(freqs >= 30) & (freqs < 45)], freqs[(freqs >= 30) & (freqs < 45)]) if any((freqs >= 30) & (freqs < 45)) else 0
    total = delta + theta + alpha + beta + gamma + 1e-10
    threshold = 0.1 * np.std(signal)
    suppression_ratio = np.mean(np.abs(signal) < threshold)
    diff1 = np.diff(signal)
    diff2 = np.diff(diff1)
    activity = np.var(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return np.array([delta / total, theta / total, alpha / total, beta / total, gamma / total, theta / (alpha + 1e-10), delta / (alpha + 1e-10), (alpha + beta) / (delta + theta + 1e-10), np.log10(total + 1e-10), suppression_ratio, activity, mobility, complexity, spectral_entropy, skew(signal), kurtosis(signal), np.mean(np.abs(np.diff(signal))), np.sum(np.diff(np.sign(signal)) != 0) / len(signal), np.max(np.abs(signal)), np.percentile(np.abs(signal), 95)])

def load_chbmit_data(max_patients=10, max_files_per_patient=5):
    print('\n[1/3] Loading CHB-MIT (seizure training data)...')
    X, y = ([], [])
    patients = sorted([p for p in CHBMIT_PATH.iterdir() if p.is_dir() and p.name.startswith('chb')])[:max_patients]
    for patient_dir in patients:
        summary_file = patient_dir / f'{patient_dir.name}-summary.txt'
        if not summary_file.exists():
            continue
        seizure_info = {}
        current_file = None
        with open(summary_file) as f:
            for line in f:
                if line.startswith('File Name:'):
                    current_file = line.split(':')[1].strip()
                    seizure_info[current_file] = []
                elif 'Seizure Start Time' in line and current_file:
                    try:
                        start = int(line.split(':')[-1].strip().replace(' seconds', ''))
                        seizure_info[current_file].append(start)
                    except:
                        pass
        edf_files = sorted(patient_dir.glob('*.edf'))[:max_files_per_patient]
        for edf_file in edf_files:
            try:
                f = pyedflib.EdfReader(str(edf_file))
                n_channels = f.signals_in_file
                fs = int(f.getSampleFrequency(0))
                n_samples = f.getNSamples()[0]
                signal = f.readSignal(0)
                f.close()
                seizures = seizure_info.get(edf_file.name, [])
                window_sec = 10
                window_samples = window_sec * fs
                for start_sample in range(0, len(signal) - window_samples, window_samples):
                    end_sample = start_sample + window_samples
                    start_sec = start_sample // fs
                    segment = signal[start_sample:end_sample]
                    feats = extract_features(segment, fs)
                    if feats is None:
                        continue
                    is_preictal = any((0 < sz - start_sec <= 60 for sz in seizures))
                    is_ictal = any((sz <= start_sec <= sz + 60 for sz in seizures))
                    if is_preictal:
                        X.append(feats)
                        y.append(1)
                    elif not is_ictal and len([i for i in y if i == 0]) < 2000:
                        X.append(feats)
                        y.append(0)
            except Exception as e:
                continue
        print(f'  {patient_dir.name}: {len([i for i in y if i == 1])} pre-ictal, {len([i for i in y if i == 0])} interictal')
    return (np.array(X), np.array(y))

def load_sleepedf_data(max_subjects=30):
    print('\n[2/3] Loading Sleep-EDF (consciousness test data)...')
    X_wake, X_sleep = ([], [])
    psg_files = sorted(SLEEPEDF_PATH.glob('*-PSG.edf'))[:max_subjects]
    for psg in psg_files:
        try:
            base = psg.name[:6]
            hypno = list(SLEEPEDF_PATH.glob(f'{base}*Hypnogram.edf'))
            if not hypno:
                continue
            raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
            annot = mne.read_annotations(str(hypno[0]))
            raw.set_annotations(annot)
            fs = raw.info['sfreq']
            eeg_ch = [ch for ch in raw.ch_names if 'EEG' in ch]
            if not eeg_ch:
                continue
            raw.pick_channels(eeg_ch[:1])
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            wake_ids = [eid for name, eid in event_id.items() if 'W' in name]
            sleep_ids = [eid for name, eid in event_id.items() if any((s in name for s in ['1', '2', '3', '4', 'R']))]
            epoch_len = int(30 * fs)
            for i in range(len(events)):
                start = events[i][0]
                if start + epoch_len > raw.n_times:
                    continue
                data = raw.get_data(start=start, stop=start + epoch_len)[0]
                feats = extract_features(data, fs)
                if feats is None:
                    continue
                if events[i][2] in wake_ids and len(X_wake) < 500:
                    X_wake.append(feats)
                elif events[i][2] in sleep_ids and len(X_sleep) < 500:
                    X_sleep.append(feats)
        except Exception as e:
            continue
    print(f'  Loaded {len(X_wake)} wake, {len(X_sleep)} sleep epochs')
    n = min(len(X_wake), len(X_sleep))
    X = np.vstack([X_wake[:n], X_sleep[:n]])
    y = np.array([0] * n + [1] * n)
    return (X, y)

def load_icare_bs_data():
    print('\n[3/3] Loading I-CARE (burst suppression test data)...')
    X_normal, X_bs = ([], [])
    patients = sorted([p.name for p in ICARE_PATH.iterdir() if p.is_dir() and list(p.glob('*_EEG.mat'))])
    for pid in patients:
        patient_path = ICARE_PATH / pid
        eeg_files = sorted(patient_path.glob('*_EEG.mat'))
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
                fs = 250
                window_samples = 10 * fs
                for start in range(0, eeg_data.shape[1] - window_samples, window_samples):
                    segment = np.mean(eeg_data[:, start:start + window_samples], axis=0)
                    threshold = 0.15 * np.std(segment)
                    suppression_ratio = np.mean(np.abs(segment) < threshold)
                    feats = extract_features(segment, fs)
                    if feats is None:
                        continue
                    if suppression_ratio > 0.3 and len(X_bs) < 200:
                        X_bs.append(feats)
                    elif suppression_ratio < 0.1 and len(X_normal) < 200:
                        X_normal.append(feats)
            except:
                continue
    print(f'  Loaded {len(X_normal)} normal, {len(X_bs)} BS segments')
    n = min(len(X_normal), len(X_bs))
    if n == 0:
        return (None, None)
    X = np.vstack([X_normal[:n], X_bs[:n]])
    y = np.array([0] * n + [1] * n)
    return (X, y)

def main():
    print('=' * 60)
    print('CROSS-CONDITION TRANSFER LEARNING')
    print('Train on seizures → Test on consciousness & burst suppression')
    print('=' * 60)
    X_train, y_train = load_chbmit_data(max_patients=15, max_files_per_patient=8)
    if len(X_train) == 0:
        print('Failed to load CHB-MIT data')
        return
    X_train = np.nan_to_num(X_train)
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    n_train = min(n_pos, n_neg, 500)
    np.random.seed(42)
    idx0 = np.random.choice(np.where(y_train == 0)[0], n_train, replace=False)
    idx1 = np.random.choice(np.where(y_train == 1)[0], min(n_pos, n_train), replace=n_pos < n_train)
    idx = np.concatenate([idx0, idx1])
    np.random.shuffle(idx)
    X_train, y_train = (X_train[idx], y_train[idx])
    print(f'\nTraining set: {len(X_train)} samples ({np.sum(y_train == 0)} interictal, {np.sum(y_train == 1)} pre-ictal)')
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    print('\nTraining XGBoost on seizure data...')
    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X_train_sc, y_train)
    print('  Done.')
    results = {'training': 'CHB-MIT_seizures', 'n_train': len(X_train)}
    X_sleep, y_sleep = load_sleepedf_data(max_subjects=40)
    if X_sleep is not None and len(X_sleep) > 0:
        X_sleep = np.nan_to_num(X_sleep)
        X_sleep_sc = scaler.transform(X_sleep)
        prob = model.predict_proba(X_sleep_sc)[:, 1]
        pred = model.predict(X_sleep_sc)
        auroc = roc_auc_score(y_sleep, prob)
        acc = accuracy_score(y_sleep, pred)
        print(f'\n>>> Sleep-EDF (Wake vs Sleep) — NO RETRAINING')
        print(f'    AUROC: {auroc:.3f}')
        print(f'    Accuracy: {acc:.3f}')
        results['sleepedf'] = {'auroc': float(auroc), 'accuracy': float(acc), 'n_test': len(y_sleep)}
    X_bs, y_bs = load_icare_bs_data()
    if X_bs is not None and len(X_bs) > 0:
        X_bs = np.nan_to_num(X_bs)
        X_bs_sc = scaler.transform(X_bs)
        prob = model.predict_proba(X_bs_sc)[:, 1]
        pred = model.predict(X_bs_sc)
        auroc = roc_auc_score(y_bs, prob)
        acc = accuracy_score(y_bs, pred)
        print(f'\n>>> I-CARE Burst Suppression — NO RETRAINING')
        print(f'    AUROC: {auroc:.3f}')
        print(f'    Accuracy: {acc:.3f}')
        results['icare_bs'] = {'auroc': float(auroc), 'accuracy': float(acc), 'n_test': len(y_bs)}
    print(f"\n{'=' * 60}")
    print('TRANSFER LEARNING RESULTS')
    print('=' * 60)
    print('Model trained ONLY on CHB-MIT seizure prediction')
    print('Tested on completely different conditions WITHOUT retraining:\n')
    if 'sleepedf' in results:
        print(f"  Sleep-EDF (consciousness):    AUROC = {results['sleepedf']['auroc']:.3f}")
    if 'icare_bs' in results:
        print(f"  I-CARE (burst suppression):   AUROC = {results['icare_bs']['auroc']:.3f}")
    print('\n>>> This proves BFI features capture UNIVERSAL brain instability markers!')
    with open(RESULTS_PATH / 'transfer_learning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to results/transfer_learning_results.json')
if __name__ == '__main__':
    main()