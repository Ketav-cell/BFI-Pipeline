import sys
import json
import warnings
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
CHB_MIT_PATH = Path('physionet.org/files/chbmit/1.0.0')
sys.path.insert(0, str(Path(__file__).parent / 'bfi_pipeline'))
warnings.filterwarnings('ignore')
import config as cfg
from data.loader import load_chbmit
from preprocess.filter import preprocess_signal
from preprocess.artifact import reject_and_normalize
from features.spectral import extract_spectral
from features.complexity import extract_complexity
from features.instability import extract_instability

def extract_features_simple(window):
    feats = {}
    try:
        feats.update(extract_spectral(window))
    except Exception:
        pass
    try:
        feats.update(extract_complexity(window))
    except Exception:
        pass
    try:
        feats.update(extract_instability(window))
    except Exception:
        pass
    feats = {k: v for k, v in feats.items() if isinstance(v, (int, float)) and (not np.isnan(v))}
    return feats

def extract_windows_for_patient(record, window_s=10, sfreq=256):
    eeg = record['eeg']
    events = record['events']
    n_win = int(window_s * sfreq)
    pre_event_features = []
    control_features = []
    for evt in events:
        onset = evt['onset_sample']
        pre_start = max(0, onset - 60 * 60 * sfreq)
        pre_end = onset
        if pre_end - pre_start < n_win:
            continue
        for win_start in range(pre_start, pre_end - n_win, n_win * 3):
            win = eeg[:, win_start:win_start + n_win]
            if win.shape[1] < n_win:
                continue
            feats = extract_features_simple(win)
            if len(feats) > 10:
                pre_event_features.append(feats)
    ctrl_end = min(30 * 60 * sfreq, eeg.shape[1] // 2)
    for win_start in range(0, ctrl_end - n_win, n_win * 3):
        skip = False
        for evt in events:
            if abs(win_start - evt['onset_sample']) < 60 * 60 * sfreq:
                skip = True
                break
        if skip:
            continue
        win = eeg[:, win_start:win_start + n_win]
        if win.shape[1] < n_win:
            continue
        feats = extract_features_simple(win)
        if len(feats) > 10:
            control_features.append(feats)
    return (pre_event_features, control_features)

def main():
    print('=' * 60)
    print('BFI Quickstart — CHB-MIT Feature Extraction')
    print('=' * 60)
    if not CHB_MIT_PATH.exists():
        print(f'\nERROR: CHB-MIT data not found at {CHB_MIT_PATH}')
        print('Download from: https://physionet.org/content/chbmit/1.0.0/')
        print('Then edit CHB_MIT_PATH at the top of this script.')
        sys.exit(1)
    cfg.DATA_ROOTS['CHB_MIT'] = CHB_MIT_PATH
    print(f'\nLoading CHB-MIT from {CHB_MIT_PATH}...')
    records = load_chbmit(CHB_MIT_PATH)
    records = [r for r in records if r is not None and r.get('eeg') is not None]
    n_events = sum((len(r['events']) for r in records))
    n_with_events = sum((1 for r in records if len(r['events']) > 0))
    print(f'Loaded {len(records)} recordings, {n_with_events} with seizures, {n_events} total events')
    print('\nSkipping montage harmonization (using bipolar channels directly)...')
    print('\nPreprocessing (resample, filter, artifact rejection)...')
    preprocessed = []
    for r in tqdm(records, desc='Preprocessing'):
        try:
            r['eeg'] = preprocess_signal(r['eeg'], r['sfreq'])
            eeg_norm, bad_mask = reject_and_normalize(r['eeg'], cfg.TARGET_SFREQ)
            r['eeg'] = eeg_norm
            r['sfreq'] = cfg.TARGET_SFREQ
            preprocessed.append(r)
        except Exception as e:
            pass
    print(f'  {len(preprocessed)} records preprocessed')
    with_events = [r for r in preprocessed if len(r['events']) > 0]
    without_events = [r for r in preprocessed if len(r['events']) == 0]
    print(f'  {len(with_events)} with seizures, {len(without_events)} without')
    print('\nExtracting features...')
    all_pre = []
    all_ctrl = []
    for r in tqdm(with_events, desc='Seizure recordings'):
        try:
            pre, ctrl = extract_windows_for_patient(r)
            all_pre.extend(pre)
            all_ctrl.extend(ctrl)
        except Exception as e:
            pass
    for r in tqdm(without_events[:20], desc='Control recordings'):
        try:
            eeg = r['eeg']
            n_win = int(10 * 256)
            for win_start in range(0, min(30 * 60 * 256, eeg.shape[1]) - n_win, n_win * 3):
                win = eeg[:, win_start:win_start + n_win]
                if win.shape[1] < n_win:
                    continue
                feats = extract_features_simple(win)
                if len(feats) > 10:
                    all_ctrl.append(feats)
        except Exception:
            pass
    print(f'\nExtracted {len(all_pre)} pre-event windows, {len(all_ctrl)} control windows')
    if len(all_pre) == 0 or len(all_ctrl) == 0:
        print('ERROR: No features extracted.')
        sys.exit(1)
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    common_keys = set(all_pre[0].keys())
    for d in all_pre[1:]:
        common_keys &= set(d.keys())
    for d in all_ctrl:
        common_keys &= set(d.keys())
    common_keys = sorted(common_keys)
    print(f"\n{'=' * 60}")
    print(f'FEATURE DISCRIMINATION (H1) — {len(common_keys)} features')
    print(f"{'=' * 60}")
    n_compare = min(len(all_pre), len(all_ctrl))
    pre_sample = all_pre[:n_compare]
    ctrl_sample = all_ctrl[:n_compare]
    results = {}
    p_values = []
    feature_names_tested = []
    for key in common_keys:
        pre_vals = np.array([d[key] for d in pre_sample])
        ctrl_vals = np.array([d[key] for d in ctrl_sample])
        mask = ~(np.isnan(pre_vals) | np.isnan(ctrl_vals))
        if mask.sum() < 10:
            continue
        try:
            stat, p = wilcoxon(pre_vals[mask], ctrl_vals[mask])
            n = mask.sum()
            r_rb = 1 - 2 * stat / (n * (n + 1) / 2)
            median_diff = float(np.median(pre_vals[mask]) - np.median(ctrl_vals[mask]))
            results[key] = {'median_diff': median_diff, 'effect_size_r_rb': r_rb, 'p_value_raw': p}
            p_values.append(p)
            feature_names_tested.append(key)
        except Exception:
            continue
    if p_values:
        reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        for i, key in enumerate(feature_names_tested):
            results[key]['p_corrected'] = float(p_corrected[i])
            results[key]['significant'] = bool(reject[i])
    sig_count = sum((1 for r in results.values() if r.get('significant', False)))
    print(f'\nSignificant features (FDR < 0.05): {sig_count}/{len(results)}')
    print(f"\n{'Feature':<30} {'Med.Diff':>10} {'Effect(r)':>10} {'p(corr)':>10} {'Sig':>5}")
    print('-' * 70)
    for key in sorted(results.keys(), key=lambda k: results[k].get('p_corrected', 1)):
        r = results[key]
        sig = '***' if r.get('significant', False) else ''
        print(f"{key:<30} {r['median_diff']:>10.4f} {r['effect_size_r_rb']:>10.4f} {r.get('p_corrected', 999):>10.4f} {sig:>5}")
    output = {'dataset': 'CHB-MIT', 'n_recordings': len(preprocessed), 'n_with_seizures': len(with_events), 'n_seizure_events': sum((len(r['events']) for r in with_events)), 'n_pre_event_windows': len(all_pre), 'n_control_windows': len(all_ctrl), 'n_features': len(common_keys), 'feature_discrimination': results, 'significant_features_count': sig_count, 'total_features_tested': len(results)}
    out_path = Path('results/chbmit_metrics.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\nResults saved to {out_path}')
    print('Done!')
if __name__ == '__main__':
    main()