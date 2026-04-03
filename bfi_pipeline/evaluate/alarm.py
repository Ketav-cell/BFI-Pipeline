"""
evaluate/alarm.py — Sustained-threshold alarm logic and alarm-quality metrics.

Alarm rule:
  BFI ≥ 70 sustained for ≥ 120 seconds → alarm fires.

Metrics:
  • Lead time (min): time from first sustained alarm to actual event onset.
  • FP/h: false alarms per patient-hour of non-event recording.
  • % time below threshold (for event-free controls).
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

_THRESHOLD    = cfg.BFI_ALARM_THRESHOLD        # 70
_SUSTAINED_S  = cfg.ALARM_SUSTAINED_S          # 120 s
_WIN_STEP_S   = cfg.WINDOW_S - cfg.OVERLAP_S  # 5 s  (BFI output rate)


# ─── Core alarm logic ─────────────────────────────────────────────────────────

def detect_alarms(
    bfi_series: np.ndarray,
    step_s:     float = _WIN_STEP_S,
    threshold:  float = _THRESHOLD,
    sustained_s: float = _SUSTAINED_S,
) -> List[float]:
    """
    Detect sustained alarm onset times from a BFI time series.

    Parameters
    ----------
    bfi_series   : (T,) BFI scores (0–100), one per step
    step_s       : seconds between consecutive BFI values
    threshold    : BFI alarm threshold
    sustained_s  : required sustained duration in seconds

    Returns
    -------
    alarm_times_s : list of alarm onset times (seconds from recording start)
                    — one entry per alarm episode
    """
    T              = len(bfi_series)
    min_steps      = int(np.ceil(sustained_s / step_s))
    above          = bfi_series >= threshold   # (T,) bool

    alarm_times: List[float] = []
    in_alarm     = False
    run_start    = 0

    for t in range(T):
        if above[t]:
            if not in_alarm:
                run_start = t
                in_alarm  = True
            else:
                run_len = t - run_start + 1
                if run_len >= min_steps and (
                    not alarm_times or
                    run_start * step_s > alarm_times[-1]
                ):
                    alarm_times.append(run_start * step_s)
        else:
            in_alarm = False

    return alarm_times


# ─── Lead time ────────────────────────────────────────────────────────────────

def compute_lead_time(
    alarm_times_s: List[float],
    event_onset_s: float,
) -> Optional[float]:
    """
    First alarm before the event onset.

    Returns lead time in minutes, or None if no alarm before event.
    """
    before = [t for t in alarm_times_s if t <= event_onset_s]
    if not before:
        return None
    first_alarm = min(before)
    lead_s = event_onset_s - first_alarm
    return float(lead_s / 60.0)  # convert to minutes


# ─── False positive rate ───────────────────────────────────────────────────────

def compute_fp_per_hour(
    alarm_times_s: List[float],
    non_event_duration_s: float,
) -> float:
    """
    False alarms per patient-hour of non-event recording.

    Parameters
    ----------
    alarm_times_s         : alarm onset times (seconds)
    non_event_duration_s  : total non-event recording duration (seconds)
    """
    if non_event_duration_s <= 0:
        return 0.0
    fp_per_hour = len(alarm_times_s) / (non_event_duration_s / 3600.0)
    return float(fp_per_hour)


# ─── Per-patient alarm evaluation ─────────────────────────────────────────────

def evaluate_patient_alarms(
    bfi_series:    np.ndarray,
    event_onsets_s: List[float],
    step_s:        float = _WIN_STEP_S,
) -> Dict[str, Any]:
    """
    Evaluate alarm performance for a single patient.

    Parameters
    ----------
    bfi_series      : (T,) BFI series
    event_onsets_s  : list of ground-truth event onset times (seconds)

    Returns
    -------
    dict with lead_times (list[float|None]), fp_count, fp_per_h,
    pct_above_threshold, total_duration_s
    """
    T            = len(bfi_series)
    duration_s   = T * step_s
    alarm_times  = detect_alarms(bfi_series, step_s)

    # Classify each alarm as TP (within 60 min before event) or FP
    _TP_HORIZON_S = 60.0 * 60.0  # 60 min

    matched_alarms: set = set()
    lead_times: List[Optional[float]] = []

    for onset_s in event_onsets_s:
        lt = compute_lead_time(alarm_times, onset_s)
        lead_times.append(lt)
        # Mark alarms within TP horizon as matched
        for i, at in enumerate(alarm_times):
            if 0 <= (onset_s - at) <= _TP_HORIZON_S:
                matched_alarms.add(i)

    fp_alarms = [at for i, at in enumerate(alarm_times) if i not in matched_alarms]

    # Non-event duration: total duration minus event windows
    event_dur_s = len(event_onsets_s) * _TP_HORIZON_S
    non_event_s = max(0.0, duration_s - event_dur_s)

    fp_per_h = compute_fp_per_hour(fp_alarms, non_event_s)
    pct_above = float((bfi_series >= _THRESHOLD).mean() * 100.0)

    return {
        "lead_times":          lead_times,
        "fp_count":            len(fp_alarms),
        "fp_per_h":            fp_per_h,
        "pct_above_threshold": pct_above,
        "total_duration_s":    duration_s,
        "n_alarms":            len(alarm_times),
    }


# ─── Aggregate across patients ────────────────────────────────────────────────

def aggregate_alarm_metrics(
    patient_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate alarm metrics across patients.

    Returns
    -------
    {
      "mean_lead_time_min": float, "std_lead_time_min": float,
      "fp_per_h_mean": float, "fp_per_h_std": float,
      "pct_below_threshold": float,
    }
    """
    all_lead: List[float] = []
    all_fp:   List[float] = []
    all_pct:  List[float] = []

    for pr in patient_results:
        for lt in pr["lead_times"]:
            if lt is not None:
                all_lead.append(lt)
        all_fp.append(pr["fp_per_h"])
        all_pct.append(100.0 - pr["pct_above_threshold"])

    return {
        "mean_lead_time_min": float(np.mean(all_lead))   if all_lead else float("nan"),
        "std_lead_time_min":  float(np.std(all_lead))    if all_lead else float("nan"),
        "fp_per_h_mean":      float(np.mean(all_fp))     if all_fp   else float("nan"),
        "fp_per_h_std":       float(np.std(all_fp))      if all_fp   else float("nan"),
        "control_pct_below_threshold": float(np.mean(all_pct)) if all_pct else float("nan"),
    }
