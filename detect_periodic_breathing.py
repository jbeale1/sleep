#!/usr/bin/env python3
"""
detect_periodic_breathing.py - J.Beale 2026-02-22

Detects episodes of periodic breathing (Cheyne-Stokes-like oscillations)
from overnight sleep study data.

A window is flagged when ALL THREE conditions hold simultaneously:
  1. SPECTRAL PROMINENCE: the breath amplitude envelope has a clear spectral
     peak in the 30-120 s period band, with power significantly above
     the broadband background (10-300 s band).
  2. HR COHERENCE: the ECG heart rate is phase-coherent with the breath
     envelope at the dominant periodic-breathing frequency.  Random
     variability in either signal produces low coherence; genuine autonomic
     entrainment produces high coherence.
  3. OSCILLATION DEPTH: the amplitude of the periodic modulation is large
     relative to mean breathing amplitude (filters out faint narrowband
     noise that might pass the spectral test).

Input files (same directory as generate_sleep_dashboard.py):
  MOT_*_breath.csv     - tilt-based breath amplitude envelope (required)
  ECG_*_beats.csv      - ECG beat timestamps (required for coherence test)

Output:
  Printed report to stdout
  periodic_breathing_report.csv written to the input directory

Usage:
  python detect_periodic_breathing.py <input_directory>
"""

import os
import sys
import csv
import glob
import bisect
import calendar
import io
from datetime import datetime, date

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import numpy as np
    from scipy.signal import welch, coherence, butter, filtfilt
except ImportError:
    print("Error: numpy and scipy are required.  pip install numpy scipy")
    sys.exit(1)


# =============================================================================
# DETECTION THRESHOLDS  (all tunable)
# =============================================================================

# Analysis grid
GRID_DT        = 2.0    # seconds — interpolation / analysis sample interval

# Sliding window
WIN_SEC        = 600    # 10-minute analysis window
STEP_SEC       = 120    # 2-minute step between windows

# Periodic-breathing frequency band
PB_LOW_PERIOD  = 30     # s  (0.033 Hz)  — upper frequency edge of PB band
PB_HIGH_PERIOD = 120    # s  (0.0083 Hz) — lower frequency edge of PB band

# Broadband reference used to normalise the PB peak (wider than PB band)
BB_LOW_PERIOD  = 10     # s
BB_HIGH_PERIOD = 300    # s

# Welch PSD parameters
NPERSEG        = 100    # samples → 200 s sub-segments; ~5 averages in WIN_SEC

# ── detection thresholds ──────────────────────────────────────────────────────
# 1. Spectral prominence: PB peak / median of broadband PSD
#    Flat spectrum → ~1; strong narrowband oscillation → 5-20+
PROM_THRESH    = 15.0

# 2. Magnitude-squared coherence at the dominant PB frequency.
#    With ~5 Welch averages, the 95 % significance level under H0 is ~0.53.
#    Using 0.55 as threshold gives a small margin above chance.
COH_THRESH     = 0.55

# 3. Oscillation depth: RMS of band-passed envelope / mean of raw envelope.
#    Captures genuine amplitude modulation; rejects spectrally peaked but
#    physically tiny oscillations.
DEPTH_THRESH   = 0.20

# 4. Crest factor upper limit: peak / RMS of the band-passed envelope.
#    A pure sine wave gives sqrt(2) ~ 1.41.  Genuine CS with noise runs
#    ~2-3.  Sharp transient spikes that alias into the PB band give 5-15+.
#    Setting an upper bound rejects spike-dominated windows that pass the
#    spectral and coherence tests but are not sustained sinusoidal cycling.
CREST_MAX      = 4.0

# Episode assembly
MIN_EP_MINUTES = 20.0   # discard episodes shorter than this
GAP_MERGE_MIN  = 4.0    # merge flagged windows separated by less than this


# =============================================================================
# File discovery  (mirrors generate_sleep_dashboard.py)
# =============================================================================

def find_input_files(input_dir):
    """Return dict of relevant file paths, or raise FileNotFoundError."""
    files = {}

    # Tilt-based breath envelope (required)
    breath_matches = sorted(glob.glob(os.path.join(input_dir, "MOT_*_breath.csv")))
    if not breath_matches:
        raise FileNotFoundError("No MOT_*_breath.csv file found — tilt breath data is required.")
    files['tilt_breath'] = breath_matches[-1]
    if len(breath_matches) > 1:
        print(f"  Note: {len(breath_matches)} breath files found, using {os.path.basename(breath_matches[-1])}")

    # ECG beats (required for coherence test; detector still runs without it
    # but coherence column will be absent)
    ecg_matches = sorted(glob.glob(os.path.join(input_dir, "ECG_*_beats.csv")))
    if ecg_matches:
        files['ecg_beats'] = ecg_matches[-1]
        if len(ecg_matches) > 1:
            print(f"  Note: {len(ecg_matches)} ECG beat files found, using {os.path.basename(ecg_matches[-1])}")
    else:
        print("  Note: no ECG_*_beats.csv found — coherence test will be skipped.")

    # Apnea events (optional — used only to annotate report)
    apnea_path = os.path.join(input_dir, "breathing_analysis_report_apnea_events.csv")
    if os.path.isfile(apnea_path):
        files['apnea'] = apnea_path

    return files


# =============================================================================
# Time helpers
# =============================================================================

def parse_sleepu_time(ts_str):
    return datetime.strptime(ts_str.strip(), "%H:%M:%S %d/%m/%Y")


def minutes_to_clock(m):
    h = int(m / 60) % 24
    mn = int(m % 60)
    return f"{h:02d}:{mn:02d}"


# =============================================================================
# Data readers  (adapted from generate_sleep_dashboard.py)
# =============================================================================

def read_tilt_breath(filepath, ref_date):
    """Return (t_seconds_array, env_deg_array) as 1-D numpy arrays.
    t is wall-clock seconds since midnight of ref_date."""
    with open(filepath, 'r') as f:
        first = f.readline().strip()
    if not first.startswith('# start '):
        raise ValueError(f"Cannot find start time in {filepath}")
    start_time_str = first[8:].split('sync_millis')[0].strip()
    t0_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    ref_midnight = datetime.combine(ref_date, datetime.min.time())
    t0_sec = (t0_dt - ref_midnight).total_seconds()

    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith('seconds,')), None)
    if header_idx is None:
        raise ValueError(f"No header row in {filepath}")

    reader = csv.DictReader(io.StringIO(''.join(lines[header_idx:])))
    t_list, env_list = [], []
    for row in reader:
        try:
            sec = float(row['seconds'])
            env = float(row['envelope_deg'])
        except (ValueError, KeyError):
            continue
        t_list.append(t0_sec + sec)
        env_list.append(env)

    return np.array(t_list), np.array(env_list)


def read_ecg_hr_1hz(filepath, ref_date, pleth_delay_s=0.25):
    """Return (t_seconds_array, hr_bpm_array) on a 1-Hz grid, as numpy arrays.
    t is wall-clock seconds since midnight of ref_date (PST = UTC-8)."""
    utc_midnight_epoch = calendar.timegm(ref_date.timetuple())
    local_midnight_epoch = utc_midnight_epoch + 8 * 3600

    beats = []
    try:
        with open(filepath, newline='') as f:
            lines = [l for l in f if not l.startswith('#')]
        for row in csv.DictReader(lines):
            try:
                epoch_s = float(row['epoch_s']) + pleth_delay_s
                rr_ms   = float(row['rr_ms'])
            except (ValueError, KeyError):
                continue
            if 100 < rr_ms < 3000:
                beats.append((epoch_s, rr_ms))
    except OSError:
        return np.array([]), np.array([])

    if len(beats) < 4:
        return np.array([]), np.array([])

    beat_times = [b[0] for b in beats]
    t_start, t_end = beat_times[0], beat_times[-1]

    # 1-Hz instantaneous HR from last R-R interval
    grid = []
    t = t_start + 1.0
    while t <= t_end:
        idx = bisect.bisect_right(beat_times, t) - 1
        if idx >= 1:
            iv = (beat_times[idx] - beat_times[idx - 1]) * 1000.0
            grid.append(60000.0 / iv if 300 < iv < 3000 else None)
        else:
            grid.append(None)
        t += 1.0

    # Causal median-3 spike filter
    med3 = list(grid)
    for i in range(2, len(med3)):
        window = [v for v in [grid[i-2], grid[i-1], grid[i]] if v is not None]
        if len(window) >= 2:
            med3[i] = sorted(window)[len(window) // 2]

    # Fill remaining None by linear interpolation across gaps
    arr = np.array([v if v is not None else np.nan for v in med3], dtype=float)
    nans = np.isnan(arr)
    if nans.any() and (~nans).any():
        arr[nans] = np.interp(np.flatnonzero(nans),
                              np.flatnonzero(~nans),
                              arr[~nans])

    # Wall-clock seconds since local midnight
    t_arr = np.arange(len(arr), dtype=float) + (t_start + 1.0 - local_midnight_epoch)
    return t_arr, arr


def read_apnea_events(filepath, ref_date):
    """Return list of (t_minutes, duration_sec) tuples."""
    from datetime import datetime as _dt
    events = []
    ref_midnight = datetime.combine(ref_date, datetime.min.time())
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            try:
                dt = _dt.strptime(row['wall_clock_time'].strip(), "%Y-%m-%d %H:%M:%S")
            except (ValueError, KeyError):
                continue
            t_min = (dt - ref_midnight).total_seconds() / 60.0
            events.append((t_min, float(row['duration_sec'].strip())))
    return events


# =============================================================================
# Signal utilities
# =============================================================================

def resample_to_grid(t_src, v_src, t_grid):
    """Linear interpolation onto a uniform grid; NaN outside source range."""
    out = np.interp(t_grid, t_src, v_src,
                    left=np.nan, right=np.nan)
    return out


def bandpass_filter(x, fs, low_period, high_period, order=2):
    """Zero-phase Butterworth bandpass between low_period and high_period (seconds)."""
    nyq = 0.5 * fs
    lo  = 1.0 / high_period / nyq   # lower cutoff frequency (normalised)
    hi  = 1.0 / low_period  / nyq   # upper cutoff frequency (normalised)
    lo  = np.clip(lo, 1e-4, 0.999)
    hi  = np.clip(hi, lo + 1e-4, 0.999)
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, x)


def spectral_prominence(x, fs, pb_lo, pb_hi, bb_lo, bb_hi, nperseg):
    """
    Ratio of peak PSD in the PB band to the median PSD of the broadband
    reference.  Returns (prominence, dominant_period_s).
    period arguments are in seconds.
    """
    freqs, psd = welch(x, fs=fs, nperseg=min(nperseg, len(x)//2))
    if len(freqs) < 3:
        return 0.0, np.nan

    pb_mask = (freqs >= 1.0/pb_hi) & (freqs <= 1.0/pb_lo)
    bb_mask = (freqs >= 1.0/bb_hi) & (freqs <= 1.0/bb_lo)

    if pb_mask.sum() == 0 or bb_mask.sum() == 0:
        return 0.0, np.nan

    pb_peak  = psd[pb_mask].max()
    bb_med   = np.median(psd[bb_mask])
    if bb_med <= 0:
        return 0.0, np.nan

    dom_f = freqs[pb_mask][np.argmax(psd[pb_mask])]
    dom_period = 1.0 / dom_f if dom_f > 0 else np.nan

    return pb_peak / bb_med, dom_period


def hr_coherence_at_pb(env, hr, fs, dom_period, nperseg):
    """
    Magnitude-squared coherence between env and hr, evaluated at the
    frequency bin nearest to dom_period.
    Returns coherence value (0-1), or NaN if unavailable.
    """
    if hr is None or len(hr) != len(env):
        return np.nan
    if np.isnan(hr).any() or np.isnan(env).any():
        return np.nan
    f, coh = coherence(env, hr, fs=fs, nperseg=min(nperseg, len(env)//2))
    if len(f) < 2 or np.isnan(dom_period):
        return np.nan
    target_f = 1.0 / dom_period
    idx = np.argmin(np.abs(f - target_f))
    return float(coh[idx])


def oscillation_depth(env_raw, env_bp):
    """RMS of bandpass-filtered envelope / mean of raw envelope."""
    mean_env = np.mean(env_raw)
    if mean_env <= 0:
        return 0.0
    return float(np.sqrt(np.mean(env_bp**2)) / mean_env)


def crest_factor(env_bp):
    """Peak / RMS of the band-passed envelope.  Pure sine = sqrt(2) ~ 1.41.
    High values indicate spiky transients rather than sustained oscillation."""
    rms = float(np.sqrt(np.mean(env_bp**2)))
    if rms <= 0:
        return 0.0
    return float(np.max(np.abs(env_bp)) / rms)


# =============================================================================
# Core window-by-window analysis
# =============================================================================

def analyse_windows(env_t, env_v, hr_t, hr_v, has_hr):
    """
    Slide a window across the recording and compute per-window metrics.
    Returns list of dicts, one per window.
    """
    fs        = 1.0 / GRID_DT
    win_samp  = int(WIN_SEC  / GRID_DT)
    step_samp = int(STEP_SEC / GRID_DT)

    # Build common uniform time grid clipped to the overlap of both signals.
    # When HR data is present, only analyse the period where both are available
    # so that every window can be tested against all three criteria.
    if has_hr and len(hr_t) > 0:
        overlap_start = max(env_t[0], hr_t[0])
        overlap_end   = min(env_t[-1], hr_t[-1])
    else:
        overlap_start = env_t[0]
        overlap_end   = env_t[-1]

    if overlap_end - overlap_start < WIN_SEC:
        return []

    grid_t = np.arange(overlap_start, overlap_end, GRID_DT)

    env_grid = resample_to_grid(env_t, env_v, grid_t)
    hr_grid  = resample_to_grid(hr_t,  hr_v,  grid_t) if has_hr else None

    n_total = len(grid_t)
    results = []

    for i in range(0, n_total - win_samp + 1, step_samp):
        seg_env = env_grid[i : i + win_samp]
        seg_hr  = hr_grid [i : i + win_samp] if has_hr else None

        # Skip windows with too many NaNs (>20 %)
        env_valid = ~np.isnan(seg_env)
        if env_valid.sum() < win_samp * 0.8:
            continue

        # Fill any remaining NaNs by interpolation within the segment
        if not env_valid.all():
            idx = np.arange(win_samp)
            seg_env = np.interp(idx, idx[env_valid], seg_env[env_valid])

        if has_hr and seg_hr is not None:
            hr_valid = ~np.isnan(seg_hr)
            if hr_valid.sum() < win_samp * 0.8:
                seg_hr = None
            elif not hr_valid.all():
                idx = np.arange(win_samp)
                seg_hr = np.interp(idx, idx[hr_valid], seg_hr[hr_valid])

        # 1. Spectral prominence of breath envelope
        prom, dom_period = spectral_prominence(
            seg_env, fs,
            PB_LOW_PERIOD, PB_HIGH_PERIOD,
            BB_LOW_PERIOD, BB_HIGH_PERIOD,
            NPERSEG
        )

        # 2. HR coherence at dominant PB frequency
        coh = hr_coherence_at_pb(seg_env, seg_hr, fs, dom_period, NPERSEG) \
              if (has_hr and seg_hr is not None) else np.nan

        # 3. Oscillation depth and crest factor
        try:
            seg_bp = bandpass_filter(seg_env, fs, PB_LOW_PERIOD, PB_HIGH_PERIOD)
            depth  = oscillation_depth(seg_env, seg_bp)
            crest  = crest_factor(seg_bp)
        except Exception:
            depth = 0.0
            crest = 0.0

        # Determine if this window is flagged
        prom_ok  = prom  >= PROM_THRESH
        coh_ok   = (not has_hr) or np.isnan(coh) or (coh >= COH_THRESH)
        depth_ok = depth >= DEPTH_THRESH
        crest_ok = crest <= CREST_MAX
        flagged  = prom_ok and coh_ok and depth_ok and crest_ok

        t_centre_min = grid_t[i + win_samp // 2] / 60.0

        results.append({
            't_centre_min' : t_centre_min,
            't_start_min'  : grid_t[i]              / 60.0,
            't_end_min'    : grid_t[i + win_samp - 1] / 60.0,
            'prominence'   : round(prom,  1),
            'coherence'    : round(float(coh), 3) if not np.isnan(coh) else None,
            'depth'        : round(depth, 3),
            'crest'        : round(crest, 2),
            'dom_period_s' : round(dom_period, 1) if not np.isnan(dom_period) else None,
            'flagged'      : flagged,
        })

    return results


# =============================================================================
# Episode assembly
# =============================================================================

def assemble_episodes(windows):
    """
    Merge consecutive flagged windows (allowing small gaps) into episodes.
    Returns list of episode dicts.
    """
    flagged = [w for w in windows if w['flagged']]
    if not flagged:
        return []

    episodes = []
    ep_start = flagged[0]['t_start_min']
    ep_end   = flagged[0]['t_end_min']
    ep_wins  = [flagged[0]]

    for w in flagged[1:]:
        if w['t_start_min'] - ep_end <= GAP_MERGE_MIN:
            ep_end = max(ep_end, w['t_end_min'])
            ep_wins.append(w)
        else:
            episodes.append((ep_start, ep_end, ep_wins))
            ep_start = w['t_start_min']
            ep_end   = w['t_end_min']
            ep_wins  = [w]
    episodes.append((ep_start, ep_end, ep_wins))

    # Filter by minimum duration and compute summary stats
    result = []
    for ep_start, ep_end, ep_wins in episodes:
        dur = ep_end - ep_start
        if dur < MIN_EP_MINUTES:
            continue

        proms  = [w['prominence']   for w in ep_wins]
        depths = [w['depth']        for w in ep_wins]
        crests = [w['crest']        for w in ep_wins]
        cohs   = [w['coherence']    for w in ep_wins if w['coherence'] is not None]
        periods= [w['dom_period_s'] for w in ep_wins if w['dom_period_s'] is not None]

        result.append({
            't_start_min'   : round(ep_start, 2),
            't_end_min'     : round(ep_end,   2),
            'duration_min'  : round(dur, 1),
            'mean_prominence': round(float(np.mean(proms)),  1),
            'max_prominence' : round(float(np.max(proms)),   1),
            'mean_depth'    : round(float(np.mean(depths)),  3),
            'mean_crest'    : round(float(np.mean(crests)),  2),
            'mean_coherence': round(float(np.mean(cohs)),    3) if cohs   else None,
            'mean_period_s' : round(float(np.mean(periods)), 1) if periods else None,
        })

    return result


# =============================================================================
# Apnea overlap annotation
# =============================================================================

def apnea_overlap_fraction(ep, apnea_events):
    """Fraction of episode minutes covered by apnea events."""
    if not apnea_events:
        return 0.0
    ep_s = ep['t_start_min'] * 60
    ep_e = ep['t_end_min']   * 60
    ep_dur = ep_e - ep_s
    overlap = 0.0
    for (t_min, dur_s) in apnea_events:
        ev_s = t_min * 60
        ev_e = ev_s + dur_s
        overlap += max(0.0, min(ev_e, ep_e) - max(ev_s, ep_s))
    return min(1.0, overlap / ep_dur) if ep_dur > 0 else 0.0


# =============================================================================
# Reporting
# =============================================================================

SEPARATOR = "─" * 84

def print_report(episodes, apnea_events, has_hr, total_minutes, input_dir, ref_date,
                 analysis_start_min=None):
    print()
    print(SEPARATOR)
    print(f"  Periodic Breathing Detection Report")
    print(f"  Date: {ref_date}    Directory: {os.path.basename(input_dir)}")
    span_start = analysis_start_min if analysis_start_min is not None else 0
    print(f"  Analysis span: {minutes_to_clock(span_start)} – {minutes_to_clock(span_start + total_minutes)}  "
          f"({total_minutes:.0f} min)")
    print(f"  Analysis window: {WIN_SEC//60} min, step: {STEP_SEC//60} min, "
          f"grid: {GRID_DT:.0f} s")
    print(f"  PB band: {PB_LOW_PERIOD}–{PB_HIGH_PERIOD} s period   "
          f"Thresholds: prominence≥{PROM_THRESH}, "
          + (f"coherence≥{COH_THRESH}, " if has_hr else "coherence: N/A (no ECG), ")
          + f"depth≥{DEPTH_THRESH}")
    print(SEPARATOR)

    if not episodes:
        print("  No periodic breathing episodes detected.")
        print(SEPARATOR)
        return

    total_pb_min = sum(ep['duration_min'] for ep in episodes)
    frac = total_pb_min / total_minutes * 100 if total_minutes > 0 else 0

    print(f"  Episodes found: {len(episodes)}   "
          f"Total PB time: {total_pb_min:.1f} min  ({frac:.1f} % of recording)")
    print()

    hdr = (f"  {'#':>2}  {'Start':>5}  {'End':>5}  {'Dur':>5}  "
           f"{'Period':>7}  {'Prom':>5}  {'Depth':>6}  {'Crest':>5}"
           + (f"  {'Coh':>5}" if has_hr else "")
           + f"  {'Apnea%':>7}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for n, ep in enumerate(episodes, 1):
        ap_frac = apnea_overlap_fraction(ep, apnea_events)
        coh_str = f"{ep['mean_coherence']:.2f}" if ep['mean_coherence'] is not None else "  N/A"
        per_str = f"{ep['mean_period_s']:.0f}s" if ep['mean_period_s'] is not None else "  N/A"
        line = (f"  {n:>2}  "
                f"{minutes_to_clock(ep['t_start_min']):>5}  "
                f"{minutes_to_clock(ep['t_end_min']):>5}  "
                f"{ep['duration_min']:>4.1f}m  "
                f"{per_str:>7}  "
                f"{ep['mean_prominence']:>5.1f}  "
                f"{ep['mean_depth']:>6.3f}  "
                f"{ep['mean_crest']:>5.2f}"
                + (f"  {coh_str:>5}" if has_hr else "")
                + f"  {ap_frac*100:>6.1f}%")
        print(line)

    print()
    print("  Column notes:")
    print("    Period  — dominant oscillation period within PB band (30–120 s)")
    print("    Prom    — mean spectral prominence (PB peak / broadband median PSD)")
    print("    Depth   — RMS of band-passed breath envelope / mean breath amplitude")
    print(f"    Crest   — peak / RMS of band-passed envelope  (pure sine=1.41; threshold <{CREST_MAX})")
    if has_hr:
        print("    Coh     — magnitude-squared coherence between envelope and ECG HR")
    print("    Apnea%  — fraction of episode overlapping apnea events")
    print(SEPARATOR)
    print()


def write_csv(episodes, apnea_events, has_hr, output_path):
    fields = ['episode', 'start_clock', 'end_clock', 't_start_min', 't_end_min',
              'duration_min', 'mean_period_s', 'mean_prominence', 'max_prominence',
              'mean_depth', 'mean_crest']
    if has_hr:
        fields.append('mean_coherence')
    fields.append('apnea_overlap_frac')

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, ep in enumerate(episodes, 1):
            row = {
                'episode'            : n,
                'start_clock'        : minutes_to_clock(ep['t_start_min']),
                'end_clock'          : minutes_to_clock(ep['t_end_min']),
                't_start_min'        : ep['t_start_min'],
                't_end_min'          : ep['t_end_min'],
                'duration_min'       : ep['duration_min'],
                'mean_period_s'      : ep['mean_period_s'],
                'mean_prominence'    : ep['mean_prominence'],
                'max_prominence'     : ep['max_prominence'],
                'mean_depth'         : ep['mean_depth'],
                'mean_crest'         : ep['mean_crest'],
                'apnea_overlap_frac' : round(apnea_overlap_fraction(ep, apnea_events), 3),
            }
            if has_hr:
                row['mean_coherence'] = ep['mean_coherence']
            w.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_periodic_breathing.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    print(f"\nPeriodic Breathing Detector")
    print(f"Input directory: {input_dir}\n")

    # ── locate files ──────────────────────────────────────────────────────────
    try:
        files = find_input_files(input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    for key, path in files.items():
        print(f"  {key:12s}: {os.path.basename(path)}")

    # ── determine reference date from tilt breath file header ─────────────────
    with open(files['tilt_breath'], 'r') as f:
        first_line = f.readline().strip()
    if not first_line.startswith('# start '):
        print("Error: cannot read start time from breath file header.")
        sys.exit(1)
    start_str = first_line[8:].split('sync_millis')[0].strip()
    ref_dt    = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    ref_date  = ref_dt.date()
    print(f"\n  Reference date: {ref_date}")

    # ── read breath envelope ──────────────────────────────────────────────────
    print("  Reading tilt breath envelope …")
    env_t, env_v = read_tilt_breath(files['tilt_breath'], ref_date)
    if len(env_t) < WIN_SEC / GRID_DT:
        print("Error: breath envelope too short for analysis.")
        sys.exit(1)

    total_minutes = (env_t[-1] - env_t[0]) / 60.0
    print(f"  Breath envelope: {len(env_t)} samples, "
          f"{total_minutes:.1f} min, "
          f"native dt ≈ {np.median(np.diff(env_t)):.2f} s")

    # ── read ECG HR ───────────────────────────────────────────────────────────
    has_hr = 'ecg_beats' in files
    hr_t = hr_v = None
    if has_hr:
        print("  Reading ECG beats (1 Hz HR) …")
        hr_t, hr_v = read_ecg_hr_1hz(files['ecg_beats'], ref_date)
        if len(hr_t) < WIN_SEC:
            print("  Warning: ECG HR too short — coherence test disabled.")
            has_hr = False
        else:
            print(f"  ECG HR: {len(hr_t)} samples at 1 Hz, "
                  f"{len(hr_t)/60:.1f} min")
    if not has_hr:
        hr_t = np.array([])
        hr_v = np.array([])

    # ── read apnea events (optional) ──────────────────────────────────────────
    apnea_events = []
    if 'apnea' in files:
        apnea_events = read_apnea_events(files['apnea'], ref_date)
        print(f"  Apnea events: {len(apnea_events)}")

    # ── sliding window analysis ───────────────────────────────────────────────
    # Compute the analysis span (overlap of both signals when HR is available)
    if has_hr and len(hr_t) > 0:
        analysis_start_s = max(env_t[0], hr_t[0])
        analysis_end_s   = min(env_t[-1], hr_t[-1])
    else:
        analysis_start_s = env_t[0]
        analysis_end_s   = env_t[-1]
    analysis_minutes = (analysis_end_s - analysis_start_s) / 60.0

    print(f"\n  Analysing {analysis_minutes:.0f} min of data "
          f"({int(analysis_minutes//(STEP_SEC/60))} windows) …")

    windows = analyse_windows(env_t, env_v, hr_t, hr_v, has_hr)
    n_flagged = sum(1 for w in windows if w['flagged'])
    print(f"  Windows analysed: {len(windows)},  flagged: {n_flagged}")

    # ── assemble episodes ─────────────────────────────────────────────────────
    episodes = assemble_episodes(windows)

    # ── print report ──────────────────────────────────────────────────────────
    analysis_start_min = analysis_start_s / 60.0
    print_report(episodes, apnea_events, has_hr, analysis_minutes, input_dir, ref_date,
                 analysis_start_min=analysis_start_min)

    # ── write CSV ─────────────────────────────────────────────────────────────
    if episodes:
        csv_path = os.path.join(input_dir, "periodic_breathing_report.csv")
        write_csv(episodes, apnea_events, has_hr, csv_path)
        print(f"  CSV report written to: {csv_path}\n")
    else:
        print("  (No CSV written — no episodes detected.)\n")


if __name__ == "__main__":
    main()
