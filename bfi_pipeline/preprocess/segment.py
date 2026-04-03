"""
preprocess/segment.py — Sliding-window segmentation into sequences.

Segmentation parameters (from config):
  WINDOW_S  = 10 s  (window length)
  OVERLAP_S =  5 s  (overlap, so step = 5 s)
  SEQ_LEN   = 36    (windows per sequence = 6 minutes)

Each sequence is a 3-D array: (SEQ_LEN, n_channels, window_samples).

Labels for each sequence:
  - "stable"       : no event within look-ahead horizon
  - "pre_collapse" : event within 5–60 min (pre-ictal / pre-deterioration)
  - "imminent"     : event within 0–5 min

The label of a sequence is determined by the label of its *last* window.
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

# ─── Constants ────────────────────────────────────────────────────────────────
_SFREQ      = cfg.TARGET_SFREQ
_WIN_SAMP   = int(cfg.WINDOW_S  * _SFREQ)   # 10 × 256 = 2560
_STEP_SAMP  = int((cfg.WINDOW_S - cfg.OVERLAP_S) * _SFREQ)  # 5 × 256 = 1280
_SEQ_LEN    = cfg.SEQ_LEN                    # 36

# Label horizon (in samples)
_IMM_HOR_S  = 5  * 60   # 5 min → imminent
_PRE_HOR_S  = 60 * 60   # 60 min → pre-collapse
_IMM_SAMP   = int(_IMM_HOR_S * _SFREQ)
_PRE_SAMP   = int(_PRE_HOR_S * _SFREQ)

# Integer class codes
CLASS_STABLE       = 0
CLASS_PRE_COLLAPSE = 1
CLASS_IMMINENT     = 2
CLASS_NAMES        = ["stable", "pre_collapse", "imminent"]


# ─────────────────────────────────────────────────────────────────────────────
# Window label assignment
# ─────────────────────────────────────────────────────────────────────────────

def _window_labels(
    n_samples: int,
    events: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Assign a class label (0/1/2) to every window (indexed by its start sample).

    Parameters
    ----------
    n_samples : total samples in the recording
    events    : list of {"onset_sample": int, ...}

    Returns
    -------
    labels : (n_windows,) int8 array
    """
    starts  = np.arange(0, n_samples - _WIN_SAMP + 1, _STEP_SAMP)
    n_wins  = len(starts)
    labels  = np.zeros(n_wins, dtype=np.int8)  # default: stable

    # Collect all onset samples
    onsets = np.array([ev["onset_sample"] for ev in events], dtype=np.int64)

    if len(onsets) == 0:
        return labels

    for w_idx, ws in enumerate(starts):
        we = ws + _WIN_SAMP  # window end sample
        mid = (ws + we) // 2  # window midpoint

        # Distance from window midpoint to each event onset
        dists = onsets - mid  # positive = onset is in the future

        future = dists[dists > 0]
        if len(future) == 0:
            # All events are in the past → stable
            continue

        min_future = future.min()
        if min_future <= _IMM_SAMP:
            labels[w_idx] = CLASS_IMMINENT
        elif min_future <= _PRE_SAMP:
            labels[w_idx] = CLASS_PRE_COLLAPSE
        # else: stable (0)

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_recording(
    eeg: np.ndarray,
    events: List[Dict[str, Any]],
    bad_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a recording into overlapping windows and group into sequences.

    Parameters
    ----------
    eeg      : (n_channels, n_samples) normalized EEG at TARGET_SFREQ
    events   : list of event dicts with "onset_sample"
    bad_mask : (n_nonoverlap_windows,) bool — bad windows flagged by artifact.py
               (non-overlapping, step = WIN_SAMP).  If None, all assumed good.

    Returns
    -------
    sequences : (N_seq, SEQ_LEN, n_channels, WIN_SAMP) float32
    seq_labels: (N_seq,) int64  — label of last window in each sequence
    """
    n_ch, n_samp = eeg.shape

    # ── Sliding window indices (step = STEP_SAMP = 5 s = 1280 samples) ──────
    starts = np.arange(0, n_samp - _WIN_SAMP + 1, _STEP_SAMP)
    n_wins = len(starts)

    if n_wins == 0:
        return (
            np.empty((0, _SEQ_LEN, n_ch, _WIN_SAMP), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # Map non-overlapping bad_mask to overlapping windows (mark win as bad if
    # its non-overlapping block was bad)
    if bad_mask is not None:
        win_bad = np.zeros(n_wins, dtype=bool)
        for w_idx, ws in enumerate(starts):
            block_idx = ws // _WIN_SAMP   # nearest non-overlap block
            if block_idx < len(bad_mask) and bad_mask[block_idx]:
                win_bad[w_idx] = True
    else:
        win_bad = np.zeros(n_wins, dtype=bool)

    # ── Per-window labels ──────────────────────────────────────────────────
    win_labels = _window_labels(n_samp, events)

    # ── Assemble into sequences of SEQ_LEN consecutive windows ───────────
    sequences:  List[np.ndarray] = []
    seq_labels: List[int]        = []

    for seq_start in range(0, n_wins - _SEQ_LEN + 1, _SEQ_LEN):
        seq_indices = list(range(seq_start, seq_start + _SEQ_LEN))

        # Skip sequence if > 25% windows are bad
        n_bad = win_bad[seq_indices].sum()
        if n_bad > _SEQ_LEN * 0.25:
            continue

        seq = np.stack(
            [eeg[:, starts[i]: starts[i] + _WIN_SAMP] for i in seq_indices],
            axis=0,
        )  # (SEQ_LEN, n_channels, WIN_SAMP)

        label = int(win_labels[seq_indices[-1]])

        sequences.append(seq)
        seq_labels.append(label)

    if len(sequences) == 0:
        return (
            np.empty((0, _SEQ_LEN, n_ch, _WIN_SAMP), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)
    labels_arr    = np.array(seq_labels, dtype=np.int64)

    return sequences_arr, labels_arr


def segment_all_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply segmentation to every record in-place, adding 'sequences' and
    'seq_labels' keys.  Records with no valid sequences are kept (empty arrays).

    Returns the list of records (mutated).
    """
    for rec in records:
        if rec.get("eeg") is None:
            # Pre-feature records (SPaRCNet) — skip raw segmentation
            rec.setdefault("sequences", np.empty((0,), dtype=np.float32))
            rec.setdefault("seq_labels", np.empty((0,), dtype=np.int64))
            continue

        bad_mask = rec.get("bad_mask", None)
        seqs, labels = segment_recording(rec["eeg"], rec.get("events", []), bad_mask)
        rec["sequences"]  = seqs
        rec["seq_labels"] = labels

        n_event = (labels > 0).sum() if len(labels) else 0
        print(
            f"  [{rec['patient_id'][:40]:40s}] "
            f"seqs={len(seqs):5d}  event_seqs={n_event:5d}"
        )

    return records
