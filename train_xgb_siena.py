#!/usr/bin/env python3
"""XGBoost seizure prediction on Siena Scalp EEG dataset"""

import os, re, json, warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

import mne
from scipy import stats
from scipy.signal import welch
from scipy.integrate import trapezoid
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

mne.set_log_level('ERROR')

SIENA_PATH = Path("physionet.org/files/siena-scalp-eeg/1.0.0")
WINDOW_SEC = 30
FS_TARGET = 256

def compute_features(data, fs):
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    freqs, psd = welch(data, fs, nperseg=min(fs*2, len(data)))
    total = trapezoid(psd, freqs)
    if total == 0:
        bp = {f'rel_{b}': 0 for b in bands}
    else:
        bp = {f'rel_{b}': trapezoid(psd[(freqs >= lo) & (freqs <= hi)], freqs[(freqs >= lo) & (freqs <= hi)]) / total 
              for b, (lo, hi) in bands.items()}
    
    psd_norm = (psd + 1e-10) / (psd.sum() + 1e-10)
    idx = (freqs >= 1) & (freqs <= 40)
    slope = stats.linregress(np.log(freqs[idx] + 1e-10), np.log(psd[idx] + 1e-10))[0] if idx.sum() > 2 else 0
    
    diff1 = np.diff(data)
    var0, var1 = np.var(data), np.var(diff1)
    mobility = np.sqrt(var1 / (var0 + 1e-10))
    
    return {**bp, 'spectral_entropy': float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10))),
            'spectral_slope': float(slope), 'std': float(np.std(data)),
            'line_length': float(np.sum(np.abs(diff1))), 'hjorth_mobility': float(mobility)}

def extract_features(window, fs):
    all_ch = [compute_features(window[ch], fs) for ch in range(window.shape[0])]
    avg = {k: np.mean([f[k] for f in all_ch]) for k in all_ch[0]}
    avg['theta_alpha_ratio'] = avg['rel_theta'] / (avg['rel_alpha'] + 1e-10)
    return avg

def parse_seizures(sf):
    with open(sf) as f:
        content = f.read()
    seizures = []
    for block in re.split(r'Seizure n\s*\d+', content)[1:]:
        fm = re.search(r'File name:\s*(\S+)', block)
        rm = re.search(r'Registration start time:\s*(\d+)[.\:](\d+)[.\:](\d+)', block)
        sm = re.search(r'Seizure start time:\s*(\d+)[.\:](\d+)[.\:](\d+)', block)
        if fm and rm and sm:
            fn = fm.group(1) + ('' if fm.group(1).endswith('.edf') else '.edf')
            reg = int(rm.group(1))*3600 + int(rm.group(2))*60 + int(rm.group(3))
            sez = int(sm.group(1))*3600 + int(sm.group(2))*60 + int(sm.group(3))
            offset = sez - reg
            if offset < 0: offset += 86400
            seizures.append({'file': fn.lower().replace('pno', 'pn'), 'start': offset})
    return seizures

def main():
    print("="*50 + "\nSiena - XGBoost Seizure Prediction\n" + "="*50)
    all_data = {}
    ws = WINDOW_SEC * FS_TARGET
    
    for pdir in sorted(SIENA_PATH.iterdir()):
        if not pdir.is_dir() or not pdir.name.startswith('PN'):
            continue
        pid = pdir.name
        sf = pdir / f"Seizures-list-{pid}.txt"
        if not sf.exists():
            continue
        seizures = parse_seizures(sf)
        if not seizures:
            print(f"  {pid}: no seizures")
            continue
        recs = {}
        for edf in pdir.glob("*.edf"):
            try:
                raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)
                if raw.info['sfreq'] != FS_TARGET:
                    raw.resample(FS_TARGET)
                picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                if len(picks) >= 4:
                    recs[edf.name.lower()] = raw.get_data(picks=picks)
            except:
                pass
        if not recs:
            continue
        pre_win, ctrl_win = [], []
        for fn, data in recs.items():
            matching = [s for s in seizures if s['file'] == fn]
            if matching:
                for s in matching:
                    ps = max(0, int((s['start'] - 3600) * FS_TARGET))
                    pe = max(0, int((s['start'] - 300) * FS_TARGET))
                    for i in range(ps, pe - ws, ws):
                        if i + ws <= data.shape[1]:
                            pre_win.append(data[:, i:i+ws])
                    cs = int((s['start'] + 600) * FS_TARGET)
                    for i in range(min(5, (data.shape[1] - cs) // ws)):
                        if cs + (i+1)*ws <= data.shape[1]:
                            ctrl_win.append(data[:, cs + i*ws : cs + (i+1)*ws])
            else:
                for i in range(min(15, data.shape[1] // ws)):
                    ctrl_win.append(data[:, i*ws:(i+1)*ws])
        if len(pre_win) < 5 or len(ctrl_win) < 5:
            print(f"  {pid}: {len(pre_win)} pre, {len(ctrl_win)} ctrl - skip")
            continue
        fnames = list(extract_features(pre_win[0], FS_TARGET).keys())
        X_pre = np.array([[extract_features(w, FS_TARGET)[k] for k in fnames] for w in pre_win])
        X_ctrl = np.array([[extract_features(w, FS_TARGET)[k] for k in fnames] for w in ctrl_win])
        all_data[pid] = {'X_pre': X_pre, 'X_ctrl': X_ctrl}
        print(f"  {pid}: {len(pre_win)} pre, {len(ctrl_win)} ctrl OK")
    
    print(f"\n{len(all_data)} patients ready")
    if len(all_data) < 3:
        print("Not enough patients")
        return
    print("\nLeave-one-out CV...")
    base_aucs, tuned_aucs = [], []
    for pid in all_data:
        X_tr = np.vstack([np.vstack([all_data[p]['X_pre'], all_data[p]['X_ctrl']]) for p in all_data if p != pid])
        y_tr = np.concatenate([[1]*len(all_data[p]['X_pre']) + [0]*len(all_data[p]['X_ctrl']) for p in all_data if p != pid])
        X_te = np.vstack([all_data[pid]['X_pre'], all_data[pid]['X_ctrl']])
        y_te = np.array([1]*len(all_data[pid]['X_pre']) + [0]*len(all_data[pid]['X_ctrl']))
        m = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
        m.fit(X_tr, y_tr)
        base_auc = roc_auc_score(y_te, m.predict_proba(X_te)[:,1])
        n_pre, n_ctrl = len(all_data[pid]['X_pre']), len(all_data[pid]['X_ctrl'])
        nt_pre, nt_ctrl = max(2, n_pre//3), max(2, n_ctrl//3)
        X_tune = np.vstack([all_data[pid]['X_pre'][:nt_pre], all_data[pid]['X_ctrl'][:nt_ctrl]])
        y_tune = np.array([1]*nt_pre + [0]*nt_ctrl)
        X_eval = np.vstack([all_data[pid]['X_pre'][nt_pre:], all_data[pid]['X_ctrl'][nt_ctrl:]])
        y_eval = np.array([1]*(n_pre-nt_pre) + [0]*(n_ctrl-nt_ctrl))
        if len(y_eval) < 5:
            continue
        m2 = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
        m2.fit(np.vstack([X_tr, X_tune, X_tune, X_tune]), np.concatenate([y_tr, y_tune, y_tune, y_tune]))
        tuned_auc = roc_auc_score(y_eval, m2.predict_proba(X_eval)[:,1])
        base_aucs.append(base_auc)
        tuned_aucs.append(tuned_auc)
        print(f"  {pid}: {base_auc:.3f} -> {tuned_auc:.3f}")
    print(f"\n{'='*50}")
    print(f"Base:  {np.mean(base_aucs):.3f} +/- {np.std(base_aucs):.3f}")
    print(f"Tuned: {np.mean(tuned_aucs):.3f} +/- {np.std(tuned_aucs):.3f}")
    os.makedirs('results', exist_ok=True)
    json.dump({'dataset': 'Siena', 'n': len(all_data), 
               'base_mean': round(np.mean(base_aucs),3), 'tuned_mean': round(np.mean(tuned_aucs),3)},
              open('results/siena_xgb_results.json', 'w'), indent=2)
    print("Saved to results/siena_xgb_results.json")

if __name__ == "__main__":
    main()
