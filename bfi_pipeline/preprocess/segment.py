from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_SFREQ = cfg.TARGET_SFREQ
_WIN_SAMP = int(cfg.WINDOW_S * _SFREQ)
_STEP_SAMP = int((cfg.WINDOW_S - cfg.OVERLAP_S) * _SFREQ)
_SEQ_LEN = cfg.SEQ_LEN
_IMM_HOR_S = 5 * 60
_PRE_HOR_S = 60 * 60
_IMM_SAMP = int(_IMM_HOR_S * _SFREQ)
_PRE_SAMP = int(_PRE_HOR_S * _SFREQ)
CLASS_STABLE = 0
CLASS_PRE_COLLAPSE = 1
CLASS_IMMINENT = 2
CLASS_NAMES = ['stable', 'pre_collapse', 'imminent']

def _window_labels(n_samples: int, events: List[Dict[str, Any]]) -> np.ndarray:
    starts = np.arange(0, n_samples - _WIN_SAMP + 1, _STEP_SAMP)
    n_wins = len(starts)
    labels = np.zeros(n_wins, dtype=np.int8)
    onsets = np.array([ev['onset_sample'] for ev in events], dtype=np.int64)
    if len(onsets) == 0:
        return labels
    for w_idx, ws in enumerate(starts):
        we = ws + _WIN_SAMP
        mid = (ws + we) // 2
        dists = onsets - mid
        future = dists[dists > 0]
        if len(future) == 0:
            continue
        min_future = future.min()
        if min_future <= _IMM_SAMP:
            labels[w_idx] = CLASS_IMMINENT
        elif min_future <= _PRE_SAMP:
            labels[w_idx] = CLASS_PRE_COLLAPSE
    return labels

def segment_recording(eeg: np.ndarray, events: List[Dict[str, Any]], bad_mask: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    n_ch, n_samp = eeg.shape
    starts = np.arange(0, n_samp - _WIN_SAMP + 1, _STEP_SAMP)
    n_wins = len(starts)
    if n_wins == 0:
        return (np.empty((0, _SEQ_LEN, n_ch, _WIN_SAMP), dtype=np.float32), np.empty((0,), dtype=np.int64))
    if bad_mask is not None:
        win_bad = np.zeros(n_wins, dtype=bool)
        for w_idx, ws in enumerate(starts):
            block_idx = ws // _WIN_SAMP
            if block_idx < len(bad_mask) and bad_mask[block_idx]:
                win_bad[w_idx] = True
    else:
        win_bad = np.zeros(n_wins, dtype=bool)
    win_labels = _window_labels(n_samp, events)
    sequences: List[np.ndarray] = []
    seq_labels: List[int] = []
    for seq_start in range(0, n_wins - _SEQ_LEN + 1, _SEQ_LEN):
        seq_indices = list(range(seq_start, seq_start + _SEQ_LEN))
        n_bad = win_bad[seq_indices].sum()
        if n_bad > _SEQ_LEN * 0.25:
            continue
        seq = np.stack([eeg[:, starts[i]:starts[i] + _WIN_SAMP] for i in seq_indices], axis=0)
        label = int(win_labels[seq_indices[-1]])
        sequences.append(seq)
        seq_labels.append(label)
    if len(sequences) == 0:
        return (np.empty((0, _SEQ_LEN, n_ch, _WIN_SAMP), dtype=np.float32), np.empty((0,), dtype=np.int64))
    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    labels_arr = np.array(seq_labels, dtype=np.int64)
    return (sequences_arr, labels_arr)

def segment_all_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in records:
        if rec.get('eeg') is None:
            rec.setdefault('sequences', np.empty((0,), dtype=np.float32))
            rec.setdefault('seq_labels', np.empty((0,), dtype=np.int64))
            continue
        bad_mask = rec.get('bad_mask', None)
        seqs, labels = segment_recording(rec['eeg'], rec.get('events', []), bad_mask)
        rec['sequences'] = seqs
        rec['seq_labels'] = labels
        n_event = (labels > 0).sum() if len(labels) else 0
        print(f"  [{rec['patient_id'][:40]:40s}] seqs={len(seqs):5d}  event_seqs={n_event:5d}")
    return records