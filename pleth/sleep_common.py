"""
sleep_common.py — Shared utilities for ECG + Pleth sleep analysis tools.

File discovery, data loading, signal conditioning.
"""

import sys
import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from datetime import datetime, timezone, timedelta

# ── Constants ───────────────────────────────────────────────────────────
PST = timezone(timedelta(hours=-8))
ECG_FS = 250              # ECG sample rate (Hz)
PLETH_FS = 24             # Pleth sample rate (Hz)
PLETH_UP_FS = 240         # Upsampled pleth rate for trough detection (Hz)
PLETH_UP_RATIO = PLETH_UP_FS // PLETH_FS  # 10x

# ── File discovery ──────────────────────────────────────────────────────
def find_files(directory):
    """Auto-discover pleth, ECG raw, and ECG sync files in a directory.
    
    Expected patterns:
        *_pleth.csv          → pleth file
        ECG_*_sync.csv       → ECG sync file
        ECG_*.csv (not sync) → ECG raw file
    """
    directory = os.path.normpath(directory)
    if not os.path.isdir(directory):
        print(f"ERROR: '{directory}' is not a directory")
        sys.exit(1)
    
    # Pleth: *_pleth.csv
    pleth_files = glob.glob(os.path.join(directory, '*_pleth.csv'))
    if len(pleth_files) != 1:
        print(f"ERROR: Expected 1 *_pleth.csv file, found {len(pleth_files)}")
        sys.exit(1)
    
    # ECG sync: ECG_*_sync.csv
    sync_files = glob.glob(os.path.join(directory, 'ECG_*_sync.csv'))
    if len(sync_files) != 1:
        print(f"ERROR: Expected 1 ECG_*_sync.csv file, found {len(sync_files)}")
        sys.exit(1)
    
    # ECG raw: exactly ECG_YYYYMMDD_HHMMSS.csv (no suffix after the timestamp)
    import re
    ecg_pattern = re.compile(r'^ECG_\d{8}_\d{6}\.csv$')
    ecg_files = [f for f in glob.glob(os.path.join(directory, 'ECG_*.csv'))
                 if ecg_pattern.match(os.path.basename(f))]
    if len(ecg_files) != 1:
        print(f"ERROR: Expected 1 ECG_*.csv (non-sync) file, found {len(ecg_files)}")
        sys.exit(1)
    
    pleth_path = pleth_files[0]
    ecg_raw_path = ecg_files[0]
    ecg_sync_path = sync_files[0]
    
    print(f"Directory: {directory}")
    print(f"  Pleth:    {os.path.basename(pleth_path)}")
    print(f"  ECG raw:  {os.path.basename(ecg_raw_path)}")
    print(f"  ECG sync: {os.path.basename(ecg_sync_path)}")
    
    return pleth_path, ecg_raw_path, ecg_sync_path

# ── Data loading ────────────────────────────────────────────────────────
def load_pleth(path):
    """Load pleth CSV. De-jitter timestamps via causal linear fit.
    
    BT buffering introduces ~25ms std jitter where sample pairs arrive
    bunched.  Since sample rate is crystal-stable at ~24 sps with no
    dropped samples, a linear fit gives clean, evenly-spaced times.
    The intercept is shifted to the lower envelope (1st percentile)
    because buffering can only delay, never advance delivery.
    """
    print(f"Loading pleth from {os.path.basename(path)}...")
    df = pd.read_csv(path)
    
    # Parse timestamp to UTC epoch (timestamps are PST wall-clock)
    ts = pd.to_datetime(df['timestamp'])
    ts_epoch_pst = ts.values.astype('datetime64[ns]').astype(np.float64) / 1e9
    ts_epoch_utc = ts_epoch_pst + 8 * 3600  # PST -> UTC
    
    n_samples = len(df)
    sample_idx = np.arange(n_samples)
    
    # Linear fit for slope, causal shift for intercept
    slope, intercept = np.polyfit(sample_idx, ts_epoch_utc, 1)
    residuals = ts_epoch_utc - (intercept + slope * sample_idx)
    causal_shift = np.percentile(residuals, 1)
    intercept += causal_shift
    fitted_rate = 1.0 / slope
    
    epoch_utc = intercept + slope * sample_idx
    
    residuals_ms = (ts_epoch_utc - epoch_utc) * 1000
    n_neg = (residuals_ms < 0).sum()
    print(f"  {n_samples} samples, fitted rate: {fitted_rate:.4f} sps")
    print(f"  Causal shift: {causal_shift*1000:.1f} ms "
          f"({n_neg}/{n_samples} samples before fit line)")
    t0_pst = datetime.fromtimestamp(epoch_utc[0], PST)
    t1_pst = datetime.fromtimestamp(epoch_utc[-1], PST)
    print(f"  {t0_pst:%Y-%m-%d %H:%M:%S} – {t1_pst:%H:%M:%S} PST")
    
    values = df['pleth'].values.astype(np.float64)
    return epoch_utc, values

def load_ecg(raw_path, sync_path):
    """Load ECG raw + sync files. Interpolate timestamps for every sample."""
    print(f"Loading ECG sync from {os.path.basename(sync_path)}...")
    sync = pd.read_csv(sync_path)
    sample_idx = sync['sample_idx'].values
    unix_time = sync['unix_time'].values
    
    print(f"Loading ECG raw from {os.path.basename(raw_path)}...")
    raw = pd.read_csv(raw_path)
    ecg_uv = raw['ecg_raw_uV'].values.astype(np.float64)
    n = len(ecg_uv)
    
    print(f"  {n} samples, interpolating timestamps...")
    all_idx = np.arange(n)
    epoch_utc = np.interp(all_idx, sample_idx, unix_time)
    
    t0_pst = datetime.fromtimestamp(epoch_utc[0], PST)
    t1_pst = datetime.fromtimestamp(epoch_utc[-1], PST)
    print(f"  {t0_pst:%Y-%m-%d %H:%M:%S} – {t1_pst:%H:%M:%S} PST")
    return epoch_utc, ecg_uv

# ── Signal processing ───────────────────────────────────────────────────
def highpass_filter(signal, fs, cutoff_hz):
    """Apply zero-phase Butterworth high-pass filter."""
    sos = butter(2, cutoff_hz, btype='high', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)

def upsample_pleth(pleth_t, pleth_v, ratio=PLETH_UP_RATIO):
    """Upsample pleth via cubic spline for sub-sample trough detection.
    
    At 24 sps each sample is ~42ms apart, which creates ±42ms jitter in
    trough detection.  Upsampling 10x to 240 sps reduces this to ~4ms.
    """
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(pleth_t, pleth_v)
    n_up = (len(pleth_t) - 1) * ratio + 1
    t_up = np.linspace(pleth_t[0], pleth_t[-1], n_up)
    v_up = cs(t_up)
    return t_up, v_up

def detect_pulse_feet(t_up, v_up, fs=PLETH_UP_FS):
    """Detect pulse feet using the intersecting tangent method.
    
    The absolute minimum of the pleth waveform can occur before a
    secondary reflected wave bounce, especially at low heart rates
    where the arterial system's damped oscillation has time to manifest.
    The true pulse arrival (foot) is better defined as the point where
    the next systolic upstroke begins, found by the intersecting tangent
    method:
    
    1. Find approximate troughs (absolute minima between beats)
    2. For each trough, find the steepest point on the subsequent upstroke
    3. Draw a tangent line at the steepest point
    4. The foot is where this tangent intersects the trough's baseline value
    
    Returns arrays of foot_times and foot_vals.
    """
    from scipy.signal import find_peaks
    
    # Step 1: find approximate troughs (absolute minima)
    inv = v_up.max() - v_up
    amp = v_up.max() - v_up.min()
    if amp < 10:
        return np.array([]), np.array([])
    
    trough_idx, _ = find_peaks(inv, distance=int(fs * 0.3),
                                prominence=amp * 0.15)
    if len(trough_idx) == 0:
        return np.array([]), np.array([])
    
    # Step 2-4: refine each trough to pulse foot via intersecting tangent
    foot_times = []
    foot_vals = []
    
    # Search window for upstroke: up to 300ms after the trough
    upstroke_window = int(fs * 0.3)
    # Derivative smoothing kernel (small moving average to avoid noise spikes)
    kern_n = max(3, int(fs * 0.01))  # ~10ms smoothing
    kern = np.ones(kern_n) / kern_n
    
    for ti in trough_idx:
        # Region from trough to trough + upstroke window
        end = min(ti + upstroke_window, len(v_up) - 1)
        if end - ti < kern_n + 2:
            continue
        
        seg_t = t_up[ti:end]
        seg_v = v_up[ti:end]
        
        # First derivative (smoothed)
        dv = np.diff(seg_v) / np.diff(seg_t)
        if len(dv) > kern_n:
            dv_smooth = np.convolve(dv, kern, mode='same')
        else:
            dv_smooth = dv
        
        # Find maximum slope point
        max_slope_idx = np.argmax(dv_smooth)
        if dv_smooth[max_slope_idx] <= 0:
            continue  # no upstroke found
        
        # Tangent at steepest point
        slope = dv_smooth[max_slope_idx]
        t_steep = seg_t[max_slope_idx]
        v_steep = seg_v[max_slope_idx]
        
        # Baseline: value at the trough
        v_base = v_up[ti]
        
        # Intersect: where tangent line = baseline value
        # tangent: v = slope * (t - t_steep) + v_steep = v_base
        # t_foot = t_steep - (v_steep - v_base) / slope
        t_foot = t_steep - (v_steep - v_base) / slope
        
        # Sanity check: foot should be between trough and steepest point
        if t_foot < t_up[ti] - 0.05:  # allow 50ms before trough
            t_foot = t_up[ti]
        if t_foot > t_steep:
            t_foot = t_up[ti]
        
        foot_times.append(t_foot)
        foot_vals.append(v_base)  # display at baseline level
    
    return np.array(foot_times), np.array(foot_vals)

def despike_ecg(ecg):
    """Remove single-sample glitches (ADC reads of exactly 0.0)."""
    ecg = ecg.copy()
    d = np.diff(ecg)
    absd = np.abs(d)
    thresh = np.percentile(absd, 99.9) * 2
    count = 0
    for i in range(1, len(d)):
        if absd[i-1] > thresh and absd[i] > thresh:
            lo = max(0, i - 3)
            hi = min(len(ecg), i + 4)
            neighbors = np.concatenate([ecg[lo:i], ecg[i+1:hi]])
            ecg[i] = np.median(neighbors)
            count += 1
    if count:
        print(f"  Despiked {count} single-sample glitches")
    return ecg

def load_and_process(directory):
    """Full pipeline: find files, load, despike, filter, find overlap.
    
    Returns dict with all needed arrays and metadata.
    """
    pleth_path, ecg_raw_path, ecg_sync_path = find_files(directory)
    
    pleth_t, pleth_v = load_pleth(pleth_path)
    ecg_t, ecg_raw = load_ecg(ecg_raw_path, ecg_sync_path)
    
    # Overlap
    t_start = max(pleth_t[0], ecg_t[0])
    t_end = min(pleth_t[-1], ecg_t[-1])
    overlap_s = t_end - t_start
    if overlap_s <= 0:
        print(f"\nERROR: No time overlap!")
        print(f"  Pleth: {datetime.fromtimestamp(pleth_t[0], PST):%H:%M:%S} – "
              f"{datetime.fromtimestamp(pleth_t[-1], PST):%H:%M:%S} PST")
        print(f"  ECG:   {datetime.fromtimestamp(ecg_t[0], PST):%H:%M:%S} – "
              f"{datetime.fromtimestamp(ecg_t[-1], PST):%H:%M:%S} PST")
        sys.exit(1)
    print(f"\nOverlap: {overlap_s:.1f}s ({overlap_s/60:.1f} min)")
    
    print("Despiking ECG...")
    ecg_clean = despike_ecg(ecg_raw)
    print("Applying high-pass filter...")
    ecg_filt = highpass_filter(ecg_clean, ECG_FS, 0.5)
    
    return {
        'pleth_t': pleth_t, 'pleth_v': pleth_v,
        'ecg_t': ecg_t, 'ecg_raw': ecg_raw,
        'ecg_clean': ecg_clean, 'ecg_filt': ecg_filt,
        't_start': t_start, 't_end': t_end, 'overlap_s': overlap_s,
    }
