import numpy as np
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
import mne
import re
import gc
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

def extract_features(data, sfreq):
    feats = {}
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    psd_all = []
    for ch in range(min(data.shape[0], 18)):
        f, psd = signal.welch(data[ch], sfreq, nperseg=min(256, data.shape[1]))
        psd_all.append(psd)
    psd_mean = np.mean(psd_all, axis=0)
    total = np.sum(psd_mean)
    for band, (lo, hi) in bands.items():
        idx = (f >= lo) & (f < hi)
        feats[f'rel_{band}'] = np.sum(psd_mean[idx]) / total if total > 0 else 0
    feats['theta_alpha'] = feats['rel_theta'] / (feats['rel_alpha'] + 1e-10)
    feats['delta_alpha'] = feats['rel_delta'] / (feats['rel_alpha'] + 1e-10)
    feats['theta_beta'] = feats['rel_theta'] / (feats['rel_beta'] + 1e-10)
    feats['spectral_entropy'] = entropy(psd_mean / np.sum(psd_mean) + 1e-10)
    feats['spectral_centroid'] = np.sum(f * psd_mean) / (total + 1e-10)
    idx = (f >= 1) & (f <= 30)
    if np.sum(idx) > 2:
        feats['spectral_slope'], _ = np.polyfit(np.log(f[idx] + 1e-10), np.log(psd_mean[idx] + 1e-10), 1)
    else:
        feats['spectral_slope'] = 0
    x = data.flatten()[:10000]
    feats['std'] = np.std(x)
    feats['line_length'] = np.mean(np.abs(np.diff(x)))
    feats['kurtosis'] = kurtosis(x)
    feats['skewness'] = skew(x)
    feats['zero_crossings'] = np.sum(np.diff(np.sign(x)) != 0) / len(x)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x, var_dx, var_ddx = (np.var(x), np.var(dx), np.var(ddx))
    feats['hjorth_mobility'] = np.sqrt(var_dx / (var_x + 1e-10))
    feats['hjorth_complexity'] = np.sqrt(var_ddx / (var_dx + 1e-10)) / (feats['hjorth_mobility'] + 1e-10)
    n, order, delay = (min(len(x), 1000), 3, 1)
    nn = n - (order - 1) * delay
    if nn > 0:
        patterns = {}
        for i in range(nn):
            p = tuple(np.argsort([x[i + j * delay] for j in range(order)]))
            patterns[p] = patterns.get(p, 0) + 1
        probs = np.array(list(patterns.values())) / nn
        feats['perm_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        feats['perm_entropy'] = 0
    feats['sample_entropy'] = feats['perm_entropy'] * feats['hjorth_complexity']
    ch_vars = np.var(data, axis=1)
    feats['channel_var_mean'] = np.mean(ch_vars)
    feats['channel_var_std'] = np.std(ch_vars)
    return feats

def get_patient_features(pdir):
    patient = pdir.name
    summary = pdir / f'{patient}-summary.txt'
    seizures = {}
    if summary.exists():
        txt = summary.read_text()
        for m in re.finditer('File Name:\\s*(\\S+).*?Seizure.*?Start.*?:\\s*(\\d+).*?End.*?:\\s*(\\d+)', txt, re.DOTALL):
            fn, start, end = m.groups()
            seizures[fn] = (int(start), int(end))
    pre_feats, ctrl_feats = ([], [])
    for edf in sorted(pdir.glob('*.edf')):
        try:
            raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            win = int(30 * sfreq)
            if edf.name in seizures:
                s, e = seizures[edf.name]
                evt_s = int(s * sfreq)
                start = max(0, evt_s - int(60 * 60 * sfreq))
                for i in range(start, evt_s - win, win):
                    if i + win <= data.shape[1]:
                        f = extract_features(data[:, i:i + win], sfreq)
                        if not any((np.isnan(v) or np.isinf(v) for v in f.values())):
                            pre_feats.append(list(f.values()))
            else:
                for i in range(0, min(data.shape[1] - win, int(10 * 60 * sfreq)), win):
                    f = extract_features(data[:, i:i + win], sfreq)
                    if not any((np.isnan(v) or np.isinf(v) for v in f.values())):
                        ctrl_feats.append(list(f.values()))
            del raw, data
            gc.collect()
        except:
            pass
    return (patient, pre_feats, ctrl_feats)
PATH = Path('physionet.org/files/chbmit/1.0.0')
patient_dirs = sorted([p for p in PATH.glob('chb*') if p.is_dir()])
print(f'Processing {len(patient_dirs)} patients...')
all_data = {}
for pdir in patient_dirs:
    patient, pre, ctrl = get_patient_features(pdir)
    if pre:
        all_data[patient] = {'pre': pre, 'ctrl': ctrl}
        print(f'  {patient}: {len(pre)} pre, {len(ctrl)} ctrl')
    gc.collect()
patients_with_sz = list(all_data.keys())
print(f'\n{len(patients_with_sz)} patients with seizures')
all_ctrl = []
for p, d in all_data.items():
    all_ctrl.extend(d['ctrl'])
X_ctrl_pool = np.array(all_ctrl)
print(f'Control pool: {len(X_ctrl_pool)} windows')
print(f'\nPatient-specific fine-tuning CV...')
results_base = []
results_tuned = []
for tp in patients_with_sz:
    pre_data = np.array(all_data[tp]['pre'])
    n_finetune = max(10, int(len(pre_data) * 0.3))
    n_test = len(pre_data) - n_finetune
    if n_test < 20:
        continue
    X_pre_finetune = pre_data[:n_finetune]
    X_pre_test = pre_data[n_finetune:]
    ctrl_idx = np.random.choice(len(X_ctrl_pool), min(n_test, len(X_ctrl_pool)), replace=False)
    X_ctrl_test = X_ctrl_pool[ctrl_idx]
    X_test = np.vstack([X_pre_test, X_ctrl_test])
    y_test = np.array([1] * len(X_pre_test) + [0] * len(X_ctrl_test))
    train_pre, train_ctrl = ([], [])
    for p, d in all_data.items():
        if p != tp:
            train_pre.extend(d['pre'])
            train_ctrl.extend(d['ctrl'])
    X_pre_train = np.array(train_pre)
    X_ctrl_train = np.array(train_ctrl[:len(train_pre) * 2])
    X_train_base = np.vstack([X_pre_train, X_ctrl_train])
    y_train_base = np.array([1] * len(X_pre_train) + [0] * len(X_ctrl_train))
    sc = StandardScaler()
    X_train_base_scaled = sc.fit_transform(X_train_base)
    X_finetune_scaled = sc.transform(X_pre_finetune)
    X_test_scaled = sc.transform(X_test)
    base_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc', verbosity=0)
    base_model.fit(X_train_base_scaled, y_train_base)
    probs_base = base_model.predict_proba(X_test_scaled)[:, 1]
    auc_base = roc_auc_score(y_test, probs_base)
    results_base.append(auc_base)
    ctrl_finetune_idx = np.random.choice(len(X_ctrl_pool), min(n_finetune * 2, len(X_ctrl_pool)), replace=False)
    X_ctrl_finetune = X_ctrl_pool[ctrl_finetune_idx]
    X_finetune_full = np.vstack([X_pre_finetune, X_ctrl_finetune])
    y_finetune = np.array([1] * len(X_pre_finetune) + [0] * len(X_ctrl_finetune))
    X_finetune_full_scaled = sc.transform(X_finetune_full)
    X_combined = np.vstack([X_train_base_scaled, np.tile(X_finetune_full_scaled, (3, 1))])
    y_combined = np.concatenate([y_train_base, np.tile(y_finetune, 3)])
    tuned_model = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='auc', verbosity=0)
    tuned_model.fit(X_combined, y_combined)
    probs_tuned = tuned_model.predict_proba(X_test_scaled)[:, 1]
    auc_tuned = roc_auc_score(y_test, probs_tuned)
    results_tuned.append(auc_tuned)
    print(f"  {tp}: base={auc_base:.3f} → tuned={auc_tuned:.3f} ({('+' if auc_tuned > auc_base else '')}{auc_tuned - auc_base:.3f})")
print(f"\n{'=' * 50}")
print(f'Base model:  AUROC {np.mean(results_base):.3f} +/- {np.std(results_base):.3f}')
print(f'Fine-tuned:  AUROC {np.mean(results_tuned):.3f} +/- {np.std(results_tuned):.3f}')
print(f'Improvement: +{np.mean(results_tuned) - np.mean(results_base):.3f}')
print(f'Range: {np.min(results_tuned):.3f} - {np.max(results_tuned):.3f}')
Path('results').mkdir(exist_ok=True)
with open('results/xgb_finetuned.json', 'w') as f:
    json.dump({'base_auroc': float(np.mean(results_base)), 'tuned_auroc': float(np.mean(results_tuned)), 'improvement': float(np.mean(results_tuned) - np.mean(results_base)), 'std': float(np.std(results_tuned)), 'n': len(results_tuned)}, f)
print('Saved to results/xgb_finetuned.json')