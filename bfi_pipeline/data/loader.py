from __future__ import annotations
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
PatientRecord = Dict[str, Any]

def _read_edf(path: Path) -> 'mne.io.Raw':
    import mne
    mne.set_log_level('WARNING')
    return mne.io.read_raw_edf(str(path), preload=True, verbose=False)

def _to_samples(time_s: float, sfreq: float) -> int:
    return int(round(time_s * sfreq))

def _warn_skip(dataset: str, reason: str) -> None:
    warnings.warn(f'[{dataset}] Skipping — {reason}', RuntimeWarning, stacklevel=3)

def _parse_chbmit_summary(summary_path: Path) -> Dict[str, List[Dict]]:
    events: Dict[str, List[Dict]] = {}
    current_file: Optional[str] = None
    seizure_starts: List[float] = []
    seizure_ends: List[float] = []
    n_seizures = 0
    with open(summary_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match('File Name:\\s+(\\S+)', line)
            if m:
                if current_file and seizure_starts:
                    events[current_file] = [{'onset_s': s, 'offset_s': e} for s, e in zip(seizure_starts, seizure_ends)]
                current_file = m.group(1)
                seizure_starts = []
                seizure_ends = []
                n_seizures = 0
                continue
            m = re.match('Number of Seizures in File:\\s+(\\d+)', line)
            if m:
                n_seizures = int(m.group(1))
                continue
            m = re.match('Seizure(?:\\s+\\d+)?\\s+Start\\s+Time\\s*:\\s+(\\d+)\\s+seconds', line)
            if m:
                seizure_starts.append(float(m.group(1)))
                continue
            m = re.match('Seizure(?:\\s+\\d+)?\\s+End\\s+Time\\s*:\\s+(\\d+)\\s+seconds', line)
            if m:
                seizure_ends.append(float(m.group(1)))
                continue
    if current_file and seizure_starts:
        events[current_file] = [{'onset_s': s, 'offset_s': e} for s, e in zip(seizure_starts, seizure_ends)]
    return events

def load_chbmit(root: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip('CHB-MIT', f'path not found: {root}')
        return records
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        summary_files = list(patient_dir.glob('*-summary.txt'))
        if not summary_files:
            continue
        summary_events = _parse_chbmit_summary(summary_files[0])
        for edf_path in sorted(patient_dir.glob('*.edf')):
            fname = edf_path.name
            file_events_raw = summary_events.get(fname, [])
            try:
                raw = _read_edf(edf_path)
            except Exception as exc:
                warnings.warn(f'[CHB-MIT] Cannot read {edf_path}: {exc}')
                continue
            sfreq = raw.info['sfreq']
            data, _ = raw[:]
            ch_names = raw.ch_names
            events = [{'onset_sample': _to_samples(ev['onset_s'], sfreq), 'offset_sample': _to_samples(ev['offset_s'], sfreq), 'type': 'seizure'} for ev in file_events_raw]
            records.append({'patient_id': f'CHBMIT_{patient_id}_{fname}', 'site': 'CHB_MIT', 'condition': 'seizure', 'eeg': data.astype(np.float32), 'sfreq': float(sfreq), 'channel_names': ch_names, 'events': events, 'age': float('nan'), 'sex': 'unknown'})
    print(f'[CHB-MIT] Loaded {len(records)} recordings.')
    return records

def _parse_tse(tse_path: Path, sfreq: float, min_prob: float=0.5) -> List[Dict]:
    events = []
    with open(tse_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('version'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                start, end, label, prob = (parts[0], parts[1], parts[2], float(parts[3]))
            except ValueError:
                continue
            if label == 'seiz' and prob >= min_prob:
                events.append({'onset_sample': _to_samples(float(start), sfreq), 'offset_sample': _to_samples(float(end), sfreq), 'type': 'seizure'})
    return events

def load_tuh(root: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip('TUH', f'path not found: {root}')
        return records
    for edf_path in sorted(root.rglob('*.edf')):
        tse_path = edf_path.with_suffix('.tse')
        if not tse_path.exists():
            tse_path = edf_path.with_suffix('.tse_bi')
        if not tse_path.exists():
            continue
        try:
            raw = _read_edf(edf_path)
        except Exception as exc:
            warnings.warn(f'[TUH] Cannot read {edf_path}: {exc}')
            continue
        sfreq = raw.info['sfreq']
        data, _ = raw[:]
        events = _parse_tse(tse_path, sfreq)
        parts = edf_path.parts
        patient_id = '_'.join(parts[max(0, len(parts) - 3):])
        records.append({'patient_id': f'TUH_{patient_id}', 'site': 'TUH', 'condition': 'seizure', 'eeg': data.astype(np.float32), 'sfreq': float(sfreq), 'channel_names': raw.ch_names, 'events': events, 'age': float('nan'), 'sex': 'unknown'})
    print(f'[TUH] Loaded {len(records)} recordings.')
    return records

def _parse_siena_summary(summary_path: Path, sfreq: float) -> List[Dict]:
    events = []
    with open(summary_path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        ms = re.match('.*[Ss]eizure.*[Ss]tart.*?(\\d+(?:\\.\\d+)?)\\s*s', line)
        if ms:
            onset_s = float(ms.group(1))
            for j in range(i + 1, min(i + 5, len(lines))):
                me = re.match('.*[Ss]eizure.*[Ee]nd.*?(\\d+(?:\\.\\d+)?)\\s*s', lines[j].strip())
                if me:
                    events.append({'onset_sample': _to_samples(onset_s, sfreq), 'offset_sample': _to_samples(float(me.group(1)), sfreq), 'type': 'seizure'})
                    break
        i += 1
    return events

def load_siena(root: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip('Siena', f'path not found: {root}')
        return records
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        summaries = list(patient_dir.glob('*.txt')) + list(patient_dir.glob('*summary*'))
        for edf_path in sorted(patient_dir.glob('*.edf')):
            try:
                raw = _read_edf(edf_path)
            except Exception as exc:
                warnings.warn(f'[Siena] Cannot read {edf_path}: {exc}')
                continue
            sfreq = raw.info['sfreq']
            data, _ = raw[:]
            events: List[Dict] = []
            for s in summaries:
                events.extend(_parse_siena_summary(s, sfreq))
            records.append({'patient_id': f'SIENA_{patient_id}_{edf_path.stem}', 'site': 'SIENA', 'condition': 'seizure', 'eeg': data.astype(np.float32), 'sfreq': float(sfreq), 'channel_names': raw.ch_names, 'events': events, 'age': float('nan'), 'sex': 'unknown'})
    print(f'[Siena] Loaded {len(records)} recordings.')
    return records

def load_sparcnet(csv_path: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not csv_path.exists():
        _warn_skip('SPaRCNet', f'CSV not found: {csv_path}')
        return records
    df = pd.read_csv(csv_path)
    required_cols = {'patient_id', 'label'}
    if not required_cols.issubset(df.columns):
        _warn_skip('SPaRCNet', f'CSV missing required columns: {required_cols - set(df.columns)}')
        return records
    feature_cols = [c for c in df.columns if c not in ('patient_id', 'label', 'age', 'sex')]
    for pid, group in df.groupby('patient_id'):
        features = group[feature_cols].values.astype(np.float32)
        labels = group['label'].values
        records.append({'patient_id': f'SPARCNET_{pid}', 'site': 'SPARCNET', 'condition': 'seizure', 'eeg': None, 'features': features, 'labels': labels, 'sfreq': None, 'channel_names': [], 'events': [], 'age': float(group['age'].iloc[0]) if 'age' in group else float('nan'), 'sex': str(group['sex'].iloc[0]) if 'sex' in group else 'unknown'})
    print(f'[SPaRCNet] Loaded {len(records)} patient feature sets.')
    return records

def load_icare(root: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip('I-CARE', f'path not found: {root}')
        return records
    try:
        import wfdb
    except ImportError:
        _warn_skip('I-CARE', 'wfdb package not installed (pip install wfdb)')
        return records
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name
        header_files = list(patient_dir.glob('*.hea'))
        for hea in sorted(header_files):
            record_name = str(hea.with_suffix(''))
            try:
                record = wfdb.rdrecord(record_name)
            except Exception as exc:
                warnings.warn(f'[I-CARE] Cannot read {hea}: {exc}')
                continue
            data = record.p_signal.T.astype(np.float32)
            sfreq = float(record.fs)
            ch_names = record.sig_name
            events: List[Dict] = []
            records.append({'patient_id': f'ICARE_{patient_id}_{hea.stem}', 'site': 'ICARE', 'condition': 'coma', 'eeg': data, 'sfreq': sfreq, 'channel_names': list(ch_names), 'events': events, 'age': float('nan'), 'sex': 'unknown'})
    print(f'[I-CARE] Loaded {len(records)} recordings.')
    return records

def load_ecams(root: Path, labels_csv: Path) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip('E-CAM-S', f'path not found: {root}')
        return records
    if not labels_csv.exists():
        _warn_skip('E-CAM-S', f'labels CSV not found: {labels_csv}')
        return records
    labels_df = pd.read_csv(labels_csv)
    delirium_onsets: Dict[str, float] = {}
    for pid, grp in labels_df.groupby('patient_id'):
        grp_sorted = grp.sort_values('assessment_time_h')
        first_del = grp_sorted[grp_sorted['cam_s_score'] >= 4]
        if not first_del.empty:
            delirium_onsets[str(pid)] = float(first_del.iloc[0]['assessment_time_h'])
    try:
        from scipy.io import loadmat
    except ImportError:
        _warn_skip('E-CAM-S', 'scipy not installed')
        return records
    for mat_path in sorted(root.rglob('*.mat')):
        patient_id = mat_path.stem
        try:
            mat = loadmat(str(mat_path))
        except Exception as exc:
            warnings.warn(f'[E-CAM-S] Cannot read {mat_path}: {exc}')
            continue
        if 'data' not in mat:
            continue
        data = mat['data'].astype(np.float32)
        sfreq = float(mat.get('sfreq', [[256]])[0][0])
        raw_ch = mat.get('ch_names', None)
        if raw_ch is not None:
            try:
                ch_names = [str(c).strip() for c in raw_ch.flatten()]
            except Exception:
                ch_names = [f'EEG{i}' for i in range(data.shape[0])]
        else:
            ch_names = [f'EEG{i}' for i in range(data.shape[0])]
        events: List[Dict] = []
        if patient_id in delirium_onsets:
            onset_h = delirium_onsets[patient_id]
            onset_sample = _to_samples(onset_h * 3600.0, sfreq)
            events.append({'onset_sample': onset_sample, 'offset_sample': onset_sample, 'type': 'delirium_onset'})
        condition = 'delirium'
        pid_rows = labels_df[labels_df['patient_id'].astype(str) == patient_id]
        if not pid_rows.empty and pid_rows['cam_s_score'].max() <= 1:
            condition = 'delirium_control'
        records.append({'patient_id': f'ECAMS_{patient_id}', 'site': 'ECAMS', 'condition': condition, 'eeg': data, 'sfreq': sfreq, 'channel_names': ch_names, 'events': events, 'age': float('nan'), 'sex': 'unknown'})
    print(f'[E-CAM-S] Loaded {len(records)} recordings.')
    return records

def _load_sah_generic(root: Path, master_xlsx: Path, site_tag: str) -> List[PatientRecord]:
    records: List[PatientRecord] = []
    if not root.exists():
        _warn_skip(site_tag, f'path not found: {root}')
        return records
    if not master_xlsx.exists():
        _warn_skip(site_tag, f'master XLSX not found: {master_xlsx}')
        return records
    master = pd.read_excel(str(master_xlsx))
    master['patient_id'] = master['patient_id'].astype(str)
    dci_lookup: Dict[str, Dict] = {}
    for _, row in master.iterrows():
        pid = str(row['patient_id'])
        dci_lookup[pid] = {'dci': int(row.get('dci', 0)), 'age': float(row.get('age', float('nan'))), 'sex': str(row.get('sex', 'unknown')), 'dci_h': float(row.get('dci_onset_h', float('nan')))}
    try:
        import mne
        mne.set_log_level('WARNING')
    except ImportError:
        _warn_skip(site_tag, 'mne not installed')
        return records
    for edf_path in sorted(root.rglob('*.edf')):
        pid = edf_path.stem.split('_')[0]
        meta = dci_lookup.get(pid, {'dci': 0, 'age': float('nan'), 'sex': 'unknown', 'dci_h': float('nan')})
        try:
            raw = _read_edf(edf_path)
        except Exception as exc:
            warnings.warn(f'[{site_tag}] Cannot read {edf_path}: {exc}')
            continue
        sfreq = raw.info['sfreq']
        data, _ = raw[:]
        events: List[Dict] = []
        if meta['dci'] == 1 and (not np.isnan(meta['dci_h'])):
            onset_sample = _to_samples(meta['dci_h'] * 3600.0, sfreq)
            events.append({'onset_sample': onset_sample, 'offset_sample': onset_sample, 'type': 'dci_onset'})
        records.append({'patient_id': f'{site_tag}_{pid}_{edf_path.stem}', 'site': site_tag, 'condition': 'stroke_dci', 'eeg': data.astype(np.float32), 'sfreq': float(sfreq), 'channel_names': raw.ch_names, 'events': events, 'age': meta['age'], 'sex': meta['sex']})
    print(f'[{site_tag}] Loaded {len(records)} recordings.')
    return records

def load_sah_prospective(root: Path, master_xlsx: Path) -> List[PatientRecord]:
    return _load_sah_generic(root, master_xlsx, 'SAH_PROS')

def load_sah_automated(root: Path, master_xlsx: Path) -> List[PatientRecord]:
    return _load_sah_generic(root, master_xlsx, 'SAH_AUTO')

def load_all_datasets(cfg) -> List[PatientRecord]:
    all_records: List[PatientRecord] = []
    loaders = [('CHB_MIT', lambda: load_chbmit(cfg.DATA_ROOTS['CHB_MIT'])), ('TUH', lambda: load_tuh(cfg.DATA_ROOTS['TUH'])), ('SIENA', lambda: load_siena(cfg.DATA_ROOTS['SIENA'])), ('SPARCNET', lambda: load_sparcnet(cfg.SPARCNET_FEATURES_CSV)), ('ICARE', lambda: load_icare(cfg.DATA_ROOTS['ICARE'])), ('ECAMS', lambda: load_ecams(cfg.DATA_ROOTS['ECAMS'], cfg.ECAMS_LABELS_CSV)), ('SAH_PROS', lambda: load_sah_prospective(cfg.DATA_ROOTS['SAH_PROS'], cfg.SAH_MASTER_XLSX)), ('SAH_AUTO', lambda: load_sah_automated(cfg.DATA_ROOTS['SAH_AUTO'], cfg.SAH_MASTER_XLSX))]
    for name, loader_fn in loaders:
        print(f'\n--- Loading {name} ---')
        try:
            records = loader_fn()
            all_records.extend(records)
        except Exception as exc:
            warnings.warn(f'[{name}] Loader failed: {exc}', RuntimeWarning)
    print(f'\n=== Total recordings loaded: {len(all_records)} ===\n')
    return all_records