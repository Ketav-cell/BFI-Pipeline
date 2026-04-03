from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg
_THRESHOLD = cfg.BFI_ALARM_THRESHOLD
_SUSTAINED_S = cfg.ALARM_SUSTAINED_S
_WIN_STEP_S = cfg.WINDOW_S - cfg.OVERLAP_S

def detect_alarms(bfi_series: np.ndarray, step_s: float=_WIN_STEP_S, threshold: float=_THRESHOLD, sustained_s: float=_SUSTAINED_S) -> List[float]:
    T = len(bfi_series)
    min_steps = int(np.ceil(sustained_s / step_s))
    above = bfi_series >= threshold
    alarm_times: List[float] = []
    in_alarm = False
    run_start = 0
    for t in range(T):
        if above[t]:
            if not in_alarm:
                run_start = t
                in_alarm = True
            else:
                run_len = t - run_start + 1
                if run_len >= min_steps and (not alarm_times or run_start * step_s > alarm_times[-1]):
                    alarm_times.append(run_start * step_s)
        else:
            in_alarm = False
    return alarm_times

def compute_lead_time(alarm_times_s: List[float], event_onset_s: float) -> Optional[float]:
    before = [t for t in alarm_times_s if t <= event_onset_s]
    if not before:
        return None
    first_alarm = min(before)
    lead_s = event_onset_s - first_alarm
    return float(lead_s / 60.0)

def compute_fp_per_hour(alarm_times_s: List[float], non_event_duration_s: float) -> float:
    if non_event_duration_s <= 0:
        return 0.0
    fp_per_hour = len(alarm_times_s) / (non_event_duration_s / 3600.0)
    return float(fp_per_hour)

def evaluate_patient_alarms(bfi_series: np.ndarray, event_onsets_s: List[float], step_s: float=_WIN_STEP_S) -> Dict[str, Any]:
    T = len(bfi_series)
    duration_s = T * step_s
    alarm_times = detect_alarms(bfi_series, step_s)
    _TP_HORIZON_S = 60.0 * 60.0
    matched_alarms: set = set()
    lead_times: List[Optional[float]] = []
    for onset_s in event_onsets_s:
        lt = compute_lead_time(alarm_times, onset_s)
        lead_times.append(lt)
        for i, at in enumerate(alarm_times):
            if 0 <= onset_s - at <= _TP_HORIZON_S:
                matched_alarms.add(i)
    fp_alarms = [at for i, at in enumerate(alarm_times) if i not in matched_alarms]
    event_dur_s = len(event_onsets_s) * _TP_HORIZON_S
    non_event_s = max(0.0, duration_s - event_dur_s)
    fp_per_h = compute_fp_per_hour(fp_alarms, non_event_s)
    pct_above = float((bfi_series >= _THRESHOLD).mean() * 100.0)
    return {'lead_times': lead_times, 'fp_count': len(fp_alarms), 'fp_per_h': fp_per_h, 'pct_above_threshold': pct_above, 'total_duration_s': duration_s, 'n_alarms': len(alarm_times)}

def aggregate_alarm_metrics(patient_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_lead: List[float] = []
    all_fp: List[float] = []
    all_pct: List[float] = []
    for pr in patient_results:
        for lt in pr['lead_times']:
            if lt is not None:
                all_lead.append(lt)
        all_fp.append(pr['fp_per_h'])
        all_pct.append(100.0 - pr['pct_above_threshold'])
    return {'mean_lead_time_min': float(np.mean(all_lead)) if all_lead else float('nan'), 'std_lead_time_min': float(np.std(all_lead)) if all_lead else float('nan'), 'fp_per_h_mean': float(np.mean(all_fp)) if all_fp else float('nan'), 'fp_per_h_std': float(np.std(all_fp)) if all_fp else float('nan'), 'control_pct_below_threshold': float(np.mean(all_pct)) if all_pct else float('nan')}