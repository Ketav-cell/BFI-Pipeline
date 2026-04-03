from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_ALIASES: dict[str, str] = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6', 'FP1': 'Fp1', 'FP2': 'Fp2', 'CZ': 'Cz', 'FZ': 'Fz', 'PZ': 'Pz', 'FP1-REF': 'Fp1', 'FP2-REF': 'Fp2', 'F3-REF': 'F3', 'F4-REF': 'F4', 'F7-REF': 'F7', 'F8-REF': 'F8', 'FZ-REF': 'Fz', 'C3-REF': 'C3', 'C4-REF': 'C4', 'CZ-REF': 'Cz', 'T3-REF': 'T3', 'T4-REF': 'T4', 'T5-REF': 'T5', 'T6-REF': 'T6', 'T7-REF': 'T3', 'T8-REF': 'T4', 'P7-REF': 'T5', 'P8-REF': 'T6', 'P3-REF': 'P3', 'P4-REF': 'P4', 'PZ-REF': 'Pz', 'O1-REF': 'O1', 'O2-REF': 'O2', 'FP1-LE': 'Fp1', 'FP2-LE': 'Fp2', 'F3-LE': 'F3', 'F4-LE': 'F4', 'F7-LE': 'F7', 'F8-LE': 'F8', 'FZ-LE': 'Fz', 'C3-LE': 'C3', 'C4-LE': 'C4', 'CZ-LE': 'Cz', 'T3-LE': 'T3', 'T4-LE': 'T4', 'T5-LE': 'T5', 'T6-LE': 'T6', 'T7-LE': 'T3', 'T8-LE': 'T4', 'P7-LE': 'T5', 'P8-LE': 'T6', 'P3-LE': 'P3', 'P4-LE': 'P4', 'PZ-LE': 'Pz', 'O1-LE': 'O1', 'O2-LE': 'O2'}
_STD_UPPER = {ch.upper(): ch for ch in cfg.STANDARD_CHANNELS}
_ALIAS_UPPER = {k.upper(): v for k, v in _ALIASES.items()}

def _normalize(name: str) -> str:
    name = name.strip().upper()
    for prefix in ('EEG ', 'EEG-'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

def _resolve_channel(name: str) -> Optional[str]:
    key = _normalize(name)
    if key in _STD_UPPER:
        return _STD_UPPER[key]
    if key in _ALIAS_UPPER:
        alias_std = _ALIAS_UPPER[key]
        return alias_std
    return None

def harmonize(eeg: np.ndarray, channel_names: List[str]) -> Optional[Tuple[np.ndarray, List[str]]]:
    available: dict[str, int] = {}
    for idx, name in enumerate(channel_names):
        std_name = _resolve_channel(name)
        if std_name is not None and std_name not in available:
            available[std_name] = idx
    missing = [ch for ch in cfg.STANDARD_CHANNELS if ch not in available]
    if missing:
        return None
    n_samples = eeg.shape[1]
    eeg_std = np.zeros((cfg.N_CHANNELS, n_samples), dtype=np.float32)
    for out_idx, ch in enumerate(cfg.STANDARD_CHANNELS):
        eeg_std[out_idx] = eeg[available[ch]]
    return (eeg_std, cfg.STANDARD_CHANNELS[:])

def harmonize_record(record: dict) -> Optional[dict]:
    if record.get('eeg') is None:
        return record
    result = harmonize(record['eeg'], record['channel_names'])
    if result is None:
        return None
    eeg_std, ch_std = result
    record['eeg'] = eeg_std
    record['channel_names'] = ch_std
    return record