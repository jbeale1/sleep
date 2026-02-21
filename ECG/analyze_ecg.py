#!/usr/bin/env python3

"""
ECG single-lead analysis — comprehensive beat-by-beat metrics vs time.

Reads a single-column CSV (µV at known sample rate) and produces a
multi-panel dashboard of every standard and some non-standard ECG metrics.

Metrics per beat:
  - R-R interval, instantaneous HR
  - HRV: SDNN, RMSSD (rolling windows)
  - QRS amplitude (R-peak value)
  - QRS duration
  - P-wave: amplitude, duration, notch depth, notch presence
  - T-wave: amplitude, polarity
  - QT interval, QTc (Bazett)
  - ST-segment level
  - Respiratory sinus arrhythmia envelope (from R-R modulation)

Usage:
  python analyze_ecg.py <csv_file> [sample_rate] [--prefiltered] [--no-plot] [--plot hr] [--csv-out]
  python analyze_ecg.py ECG_20260213.csv 250
  python analyze_ecg.py ECG_filtered.csv 250 --plot hr --prefiltered
  python analyze_ecg.py ECG_filtered.csv 250 --no-plot --csv-out

Wall-clock time: auto-detects _sync.csv file for NTP-locked timestamps,
falls back to YYYYMMDD_HHMMSS from filename, or elapsed time.

J. Beale  2026-02
"""

import numpy as np
import matplotlib
from scipy import signal
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import os
import re
import argparse

# =============================================================
# PARSE ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='ECG single-lead analysis dashboard')
parser.add_argument('csv_file', help='CSV file with ECG data (single column, µV)')
parser.add_argument('sample_rate', nargs='?', type=int, default=250,
                    help='Sample rate in sps (default: 250)')
parser.add_argument('--prefiltered', action='store_true',
                    help='Data is already filtered (skip filtering)')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNGs but do not display plots')
parser.add_argument('--plot', choices=['all', 'hr'], default='all',
                    help='Which plot(s) to produce (default: all)')
parser.add_argument('--csv-out', action='store_true', dest='csv_out',
                    help='Export per-beat metrics to CSV (same dir as input)')
parser.add_argument('--save-summary', action='store_true', dest='save_summary',
                    help='Save summary statistics to text file')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

CSV_FILE = args.csv_file
FS = args.sample_rate

# =============================================================
# RESOLVE INPUT FILE (support both file and directory)
# =============================================================
def find_ecg_csv(directory):
    """
    Search directory for the first ECG_<YYYYMMDD_HHMMSS>.csv file.
    Returns the full path to the matching file, or None if not found.
    """
    ecg_pattern = re.compile(r'^ECG_\d{8}_\d{6}\.csv$')
    if not Path(directory).is_dir():
        return None
    
    files = sorted(Path(directory).iterdir())
    for filepath in files:
        if filepath.is_file() and ecg_pattern.match(filepath.name):
            return str(filepath)
    return None

# Check if input is a directory; if so, find the ECG CSV file
if Path(CSV_FILE).is_dir():
    found_file = find_ecg_csv(CSV_FILE)
    if found_file is None:
        print(f"Error: No ECG_*.csv file found in {CSV_FILE}")
        sys.exit(1)
    CSV_FILE = found_file
    print(f"Found ECG file: {CSV_FILE}")

# Filtering (applied with filtfilt for zero-phase)
HP_FREQ = 0.5    # highpass (Hz)
NOTCH_FREQ = 60   # powerline notch (Hz)
LP_FREQ = 40      # lowpass (Hz)

# Detection
REFRACT_SEC = 0.25        # refractory period after R-peak (sec) — supports up to ~220 bpm
QRS_SEARCH_MS = 80        # half-width for QRS onset/offset search (ms)
P_WINDOW_MS = (200, 40)   # P-wave search window: 200–40 ms before R
T_WINDOW_MS = (160, 550)  # T-wave search: 160–550 ms after R
ST_MEASURE_MS = 80        # ST level measured this far after R (J+80)

# Rolling HRV window
HRV_WINDOW_BEATS = 20

# =============================================================
# LOAD & FILTER
# =============================================================
data_raw = np.loadtxt(CSV_FILE, delimiter=",", skiprows=1).flatten()
N = len(data_raw)
t = np.arange(N) / FS

print(f"Loaded {N} samples ({N/FS:.1f}s) at {FS} sps from {CSV_FILE}")

# Zero-phase filtering
sos_hp = signal.butter(2, HP_FREQ, 'highpass', fs=FS, output='sos')
sos_lp = signal.butter(4, LP_FREQ, 'lowpass', fs=FS, output='sos')
b_n, a_n = signal.iirnotch(NOTCH_FREQ, Q=30, fs=FS)

if args.prefiltered:
    ecg = data_raw.copy()
    print("Data marked as pre-filtered, skipping filters")
else:
    ecg = signal.sosfiltfilt(sos_hp, data_raw)
    ecg = signal.filtfilt(b_n, a_n, ecg)
    ecg = signal.sosfiltfilt(sos_lp, ecg)
    print("Filtering applied (HP 0.5 Hz, notch 60 Hz, LP 40 Hz)")

# =============================================================
# WALL-CLOCK TIME (from sync file or filename timestamp)
# =============================================================
import matplotlib.dates as mdates

def parse_filename_timestamp(filepath):
    """Extract YYYYMMDD_HHMMSS from filename, return epoch float or None."""
    m = re.search(r'(\d{8})_(\d{6})', Path(filepath).stem)
    if m:
        ts = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        # Assume local time; use as-is (naive datetime → treat as UTC for plotting)
        return ts.timestamp()
    return None

def load_sync_file(csv_path):
    """Look for matching _sync.csv; return (sample_idx, unix_time) arrays or None."""
    sync_path = Path(csv_path).with_name(Path(csv_path).stem + '_sync.csv')
    if not sync_path.exists():
        return None
    try:
        sync_data = np.loadtxt(sync_path, delimiter=',', skiprows=1)
        return sync_data[:, 0], sync_data[:, 1]
    except Exception as e:
        print(f"Warning: could not read sync file: {e}")
        return None

def sample_to_epoch(sample_indices):
    """Map sample indices to unix epoch timestamps."""
    return np.interp(sample_indices, sync_idx, sync_epoch)

# Try sync file first, then filename
sync_result = load_sync_file(CSV_FILE)
if sync_result is not None:
    sync_idx, sync_epoch = sync_result
    t0_epoch = sample_to_epoch(np.array([0]))[0]
    t0_str = datetime.fromtimestamp(t0_epoch).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Sync file loaded ({len(sync_idx)} entries), start: {t0_str}")
    time_source = 'sync'
else:
    t0_epoch = parse_filename_timestamp(CSV_FILE)
    if t0_epoch is not None:
        # Create simple linear mapping: sample 0 → t0_epoch
        sync_idx = np.array([0, N - 1], dtype=float)
        sync_epoch = np.array([t0_epoch, t0_epoch + (N - 1) / FS])
        t0_str = datetime.fromtimestamp(t0_epoch).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Start time from filename: {t0_str} (estimated)")
        time_source = 'filename'
    else:
        # No time info: use elapsed seconds from 00:00:00
        sync_idx = np.array([0, N - 1], dtype=float)
        sync_epoch = np.array([0.0, (N - 1) / FS])
        print("No timestamp found; using elapsed time")
        time_source = 'elapsed'

def epoch_to_mpl(epoch_arr):
    """Convert unix epoch array to matplotlib date numbers."""
    ref_epoch = sync_epoch[0]
    if time_source == 'elapsed':
        # No real timestamp; use epoch 0 = 1970-01-01 00:00:00
        ref_mpl = mdates.date2num(datetime(1970, 1, 1))
    else:
        ref_mpl = mdates.date2num(datetime.fromtimestamp(ref_epoch))
    return ref_mpl + (np.asarray(epoch_arr) - ref_epoch) / 86400.0

def format_time_axis(ax):
    """Apply HH:MM formatting to an x-axis with reasonable tick density."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    loc = mdates.AutoDateLocator(minticks=6, maxticks=15)
    ax.xaxis.set_major_locator(loc)
    ax.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    ax.set_xlabel('Time')

# Pre-compute wall-clock arrays for sample times and beat times
t_epoch = sample_to_epoch(np.arange(N, dtype=float))
t_wall = epoch_to_mpl(t_epoch)

# =============================================================
# R-PEAK DETECTION (Pan-Tompkins inspired)
# =============================================================
# Bandpass 5-20 Hz for detection only
sos_det = signal.butter(2, [5, 20], 'bandpass', fs=FS, output='sos')
ecg_det = signal.sosfiltfilt(sos_det, ecg)
ecg_det_sq = ecg_det ** 2

# Moving average
ma_len = int(0.12 * FS)
ecg_ma = uniform_filter1d(ecg_det_sq, ma_len)

# Determine R-peak polarity from whole file to avoid argmax(abs) picking S-wave
r_polarity = 1 if np.max(ecg) >= np.abs(np.min(ecg)) else -1

# Adaptive threshold
refract = int(REFRACT_SEC * FS)
threshold = 0.3 * np.max(ecg_ma[:FS*2])  # init from first 2 sec
min_threshold = 0.05 * np.max(ecg_ma[:FS*2])  # floor to prevent P-wave triggers
peaks = []
i = int(0.5 * FS)  # skip first 0.5s

while i < N - int(0.5 * FS):
    if ecg_ma[i] > threshold:
        # Asymmetric window: look further back than forward.
        # If threshold is inflated the MA crossing may occur on the S-wave
        # downslope, leaving the R-peak behind i rather than ahead of it.
        search_start = max(0, i - int(0.25 * FS))
        search_end = min(N, i + int(0.10 * FS))
        seg = ecg[search_start:search_end]
        r_idx = search_start + (np.argmax(seg) if r_polarity > 0 else np.argmin(seg))
        peaks.append(r_idx)

        # Set threshold from last detected peak only — no EMA history.
        # EMA carried forward high-amplitude exercise beats and prevented
        # detection of subsequent lower-amplitude beats.
        ma_peak = np.max(ecg_ma[search_start:search_end])  # use MA peak, not rising-edge crossing value
        threshold = 0.4 * ma_peak
        min_threshold = max(min_threshold, 0.05 * ma_peak)
        threshold = max(threshold, min_threshold)

        i = max(i + 1, r_idx + refract)
    else:
        # Decay both threshold and floor
        threshold *= 0.9995
        min_threshold *= 0.9995
        threshold = max(threshold, min_threshold)
        i += 1

peaks = np.array(peaks)
print(f"Detected {len(peaks)} R-peaks (initial pass)")

# =============================================================
# MISSED-BEAT RECOVERY (second pass)
# =============================================================
# Scan for R-R gaps that are ~2x the local median, then re-search
# those gaps for the strongest energy peak that stands out above
# the noise floor within the gap itself.
RECOVERY_RR_RATIO = 1.6   # gap must be > this × local median R-R
RECOVERY_WINDOW = 10       # beats for local median R-R estimate
RECOVERY_SNR = 3.0         # gap peak must be this × gap median energy

recovered = []
if len(peaks) > RECOVERY_WINDOW + 2:
    rr_all = np.diff(peaks)
    for i in range(len(rr_all)):
        # Local median R-R from surrounding beats
        lo = max(0, i - RECOVERY_WINDOW // 2)
        hi = min(len(rr_all), i + RECOVERY_WINDOW // 2 + 1)
        local_rr = np.median(rr_all[lo:hi])

        if rr_all[i] > RECOVERY_RR_RATIO * local_rr:
            # Search the gap for energy peaks
            gap_start = peaks[i] + refract
            gap_end = peaks[i + 1] - refract
            if gap_end <= gap_start:
                continue

            gap_ma = ecg_ma[gap_start:gap_end]
            gap_noise = np.percentile(gap_ma, 25)
            gap_peak_val = np.max(gap_ma)

            # Accept if the best candidate stands out above the gap noise
            if gap_peak_val > RECOVERY_SNR * max(gap_noise, 1.0):
                # Refine position using squared bandpass signal (no MA delay)
                gap_sq = ecg_det_sq[gap_start:gap_end]
                qrs_pos = gap_start + np.argmax(gap_sq)

                # Find the actual R-peak (max |ecg|) within ±30ms; abs() handles inverted leads
                hw_refine = int(0.03 * FS)
                ss = max(0, qrs_pos - hw_refine)
                se = min(N, qrs_pos + hw_refine + 1)
                seg = ecg[ss:se]
                r_idx = ss + (np.argmax(seg) if r_polarity > 0 else np.argmin(seg))
                recovered.append(r_idx)

n_recovered = len(recovered)
if n_recovered:
    peaks = np.sort(np.concatenate([peaks, recovered]))
    # Remove any duplicates (peaks within refractory distance)
    keep = [0]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[keep[-1]] >= refract:
            keep.append(i)
    peaks = peaks[keep]
    print(f"Recovered {n_recovered} missed beats → {len(peaks)} total")
else:
    print(f"No missed beats found → {len(peaks)} total")

if len(peaks) < 5:
    print("Too few beats detected. Check data/filter settings.")
    sys.exit(1)

# =============================================================
# BEAT-BY-BEAT MEASUREMENTS
# =============================================================
def sample(ms):
    """Convert ms to samples."""
    return int(ms * FS / 1000)

n_beats = len(peaks)

# Pre-allocate arrays (NaN = unmeasured)
rr_ms       = np.full(n_beats, np.nan)
hr_bpm      = np.full(n_beats, np.nan)
qrs_amp     = np.full(n_beats, np.nan)
qrs_dur_ms  = np.full(n_beats, np.nan)

p_amp       = np.full(n_beats, np.nan)
p_dur_ms    = np.full(n_beats, np.nan)
p_notch_uv  = np.full(n_beats, np.nan)  # notch depth in µV (0 = no notch)
p_notched   = np.full(n_beats, False)    # boolean: notch detected

t_amp       = np.full(n_beats, np.nan)
qt_ms       = np.full(n_beats, np.nan)
qtc_ms      = np.full(n_beats, np.nan)
st_level    = np.full(n_beats, np.nan)

is_pvc       = np.full(n_beats, False)
pvc_score    = np.full(n_beats, np.nan)   # 0 = normal, higher = more abnormal
qrs_corr     = np.full(n_beats, np.nan)   # correlation with template

beat_time   = t[peaks]  # time of each R-peak
beat_epoch  = sample_to_epoch(peaks.astype(float))
beat_wall   = epoch_to_mpl(beat_epoch)

for i, r in enumerate(peaks):
    # --- QRS amplitude ---
    qrs_amp[i] = ecg[r]

    # --- R-R interval ---
    if i > 0:
        rr_samp = r - peaks[i-1]
        rr_ms[i] = rr_samp / FS * 1000
        if 300 < rr_ms[i] < 2000:
            hr_bpm[i] = 60000 / rr_ms[i]
        else:
            rr_ms[i] = np.nan

    # --- QRS duration (derivative-based) ---
    qrs_hw = sample(QRS_SEARCH_MS)
    if r - qrs_hw >= 0 and r + qrs_hw < N:
        seg = ecg[r - qrs_hw : r + qrs_hw + 1]
        deriv = np.abs(np.diff(seg))
        # Smooth derivative (5-sample / 20ms) to bridge momentary dips
        deriv_smooth = uniform_filter1d(deriv, min(5, len(deriv)))
        max_deriv = np.max(deriv_smooth)

        if max_deriv > 0:
            d_thresh = 0.10 * max_deriv  # 10% of peak slope
            above = np.where(deriv_smooth > d_thresh)[0]
            if len(above) >= 2:
                onset = above[0]
                offset = above[-1]
                dur = (offset - onset) / FS * 1000
                if 20 < dur < 200:
                    qrs_dur_ms[i] = dur

    # --- P-wave analysis ---
    p_start = r - sample(P_WINDOW_MS[0])
    p_end   = r - sample(P_WINDOW_MS[1])
    if p_start >= 0 and p_end < N and p_start < p_end:
        p_seg = ecg[p_start:p_end]
        p_baseline = np.median(ecg[max(0, p_start - sample(30)):p_start])

        p_seg_rel = p_seg - p_baseline

        if len(p_seg_rel) > 5:
            p_peak_idx = np.argmax(p_seg_rel)
            p_amp[i] = p_seg_rel[p_peak_idx]

            # P-wave duration: where it rises above 20% of peak
            if p_amp[i] > 15:  # only if P-wave is detectable
                p_thresh = 0.2 * p_amp[i]
                above = np.where(p_seg_rel > p_thresh)[0]
                if len(above) >= 2:
                    p_dur_ms[i] = (above[-1] - above[0]) / FS * 1000

                # --- P-wave notch detection ---
                # Find the two highest local maxima and the minimum between them
                # Smooth slightly to avoid noise-induced false notches
                if len(p_seg_rel) > 7:
                    p_smooth = uniform_filter1d(p_seg_rel.astype(float), 3)
                    # Find all local maxima
                    local_max_idx, _ = signal.find_peaks(p_smooth,
                                                         height=0.2 * p_amp[i],
                                                         distance=max(2, sample(15)))
                    if len(local_max_idx) >= 2:
                        # Take the two tallest
                        heights = p_smooth[local_max_idx]
                        top2 = np.argsort(heights)[-2:]
                        idx1, idx2 = sorted(local_max_idx[top2])

                        # Notch = minimum between the two peaks
                        if idx2 > idx1 + 1:
                            valley = np.min(p_smooth[idx1:idx2+1])
                            peak_avg = (p_smooth[idx1] + p_smooth[idx2]) / 2
                            notch_depth = peak_avg - valley

                            # Require notch to be meaningful
                            if notch_depth > 5 and notch_depth > 0.1 * p_amp[i]:
                                p_notch_uv[i] = notch_depth
                                p_notched[i] = True
                            else:
                                p_notch_uv[i] = 0
                        else:
                            p_notch_uv[i] = 0
                    else:
                        p_notch_uv[i] = 0

    # --- T-wave and QT ---
    t_start_samp = r + sample(T_WINDOW_MS[0])
    t_end_samp   = r + sample(T_WINDOW_MS[1])
    if t_start_samp >= 0 and t_end_samp < N:
        t_seg = ecg[t_start_samp:t_end_samp]
        t_baseline = np.median(ecg[max(0, r - sample(60)):r - sample(40)])

        t_peak_rel = np.argmax(t_seg)
        t_amp[i] = t_seg[t_peak_rel] - t_baseline

        # QT: Q-onset to T-end (tangent method)
        # Q-onset
        q_onset_offset = 0
        if r - sample(60) >= 0:
            pre_r = ecg[r - sample(60):r]
            q_baseline = np.median(ecg[max(0, r-sample(200)):r-sample(100)])
            below = np.where(pre_r <= q_baseline)[0]
            if len(below) > 0:
                q_onset_offset = sample(60) - below[-1]

        # T-end via tangent
        t_end_offset = t_peak_rel  # fallback
        if t_peak_rel + 5 < len(t_seg):
            post_tpeak = t_seg[t_peak_rel:]
            slopes = np.diff(post_tpeak)
            if len(slopes) > 2:
                steepest = np.argmin(slopes)
                slope_val = slopes[steepest]
                if slope_val < -0.5:
                    y_at = post_tpeak[steepest]
                    dx = (t_baseline - y_at) / slope_val
                    t_end_offset = t_peak_rel + steepest + max(0, dx)

        qt_samples = q_onset_offset + sample(T_WINDOW_MS[0]) + t_end_offset
        qt_ms[i] = qt_samples / FS * 1000

        # Bazett QTc
        if not np.isnan(rr_ms[i]) and rr_ms[i] > 300:
            rr_sec = rr_ms[i] / 1000
            qtc_ms[i] = qt_ms[i] / np.sqrt(rr_sec)

    # --- ST level (J-point + 80ms) ---
    st_idx = r + sample(ST_MEASURE_MS)
    if st_idx < N:
        st_baseline = np.median(ecg[max(0, r - sample(200)):r - sample(100)])
        st_level[i] = ecg[st_idx] - st_baseline

# =============================================================
# PVC DETECTION
# =============================================================
# Extract QRS windows (±100ms around R-peak) for morphology comparison
qrs_half = sample(100)
qrs_win_len = 2 * qrs_half + 1

def get_qrs_window(beat_idx):
    """Extract QRS region around R-peak, NaN-padded if at edges."""
    r = peaks[beat_idx]
    s = r - qrs_half
    e = r + qrs_half + 1
    if s < 0 or e > N:
        return None
    return ecg[s:e].copy()

# Build template from first 30 beats that have valid R-R and normal-looking amplitude
template_beats = []
median_amp = np.nanmedian(qrs_amp)
for i in range(min(n_beats, 200)):
    if np.isnan(qrs_amp[i]):
        continue
    # Skip obvious outliers for template building
    if abs(qrs_amp[i] - median_amp) > 0.5 * median_amp:
        continue
    win = get_qrs_window(i)
    if win is not None:
        template_beats.append(win)
    if len(template_beats) >= 30:
        break

if len(template_beats) >= 5:
    qrs_template = np.mean(template_beats, axis=0)

    for i in range(n_beats):
        win = get_qrs_window(i)
        if win is None:
            continue

        # Correlation with template
        cc = np.corrcoef(win, qrs_template)[0, 1]
        qrs_corr[i] = cc

        # PVC criteria (any of these flag it):
        # 1. Low morphology correlation (different shape)
        morph_abnormal = cc < 0.85

        # 2. QRS significantly wider than normal
        width_abnormal = (not np.isnan(qrs_dur_ms[i]) and
                          not np.isnan(np.nanmedian(qrs_dur_ms)) and
                          qrs_dur_ms[i] > np.nanmedian(qrs_dur_ms) * 1.5)

        # 3. Premature: R-R < 80% of recent median R-R
        premature = False
        if i > 2 and not np.isnan(rr_ms[i]):
            recent_rr = rr_ms[max(1, i-10):i]
            recent_rr = recent_rr[~np.isnan(recent_rr)]
            if len(recent_rr) >= 3:
                premature = rr_ms[i] < 0.80 * np.median(recent_rr)

        # 4. Inverted polarity (big negative QRS where normally positive)
        inverted = (not np.isnan(qrs_amp[i]) and
                    median_amp > 200 and
                    qrs_amp[i] < median_amp * 0.2)

        # Score: sum of weighted criteria
        score = 0
        if morph_abnormal:  score += 2
        if width_abnormal:  score += 1
        if premature:       score += 1
        if inverted:        score += 2
        pvc_score[i] = score

        # Flag as PVC if morphology is clearly abnormal, or multiple criteria
        is_pvc[i] = morph_abnormal or score >= 3

    n_pvc = np.sum(is_pvc)
    pvc_burden = 100 * n_pvc / n_beats if n_beats > 0 else 0
    print(f"PVCs detected: {n_pvc}/{n_beats} ({pvc_burden:.1f}%)")
else:
    print("Warning: insufficient beats for PVC template")

# =============================================================
# ARTIFACT / OUTLIER REJECTION
# =============================================================
# Flag beats where key metrics deviate wildly from local rolling median.
# Uses MAD (median absolute deviation) which is robust to the outliers
# themselves. Flagged beats are kept in arrays but excluded from
# summary stats and plotted as gray.

OUTLIER_WINDOW = 51   # beats for rolling median (odd number)
OUTLIER_THRESH = 5.0  # multiples of MAD

is_artifact = np.full(n_beats, False)

def rolling_median_mad(arr, window):
    """Compute rolling median and MAD for an array with NaNs."""
    half = window // 2
    med = np.full_like(arr, np.nan)
    mad = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        s = max(0, i - half)
        e = min(len(arr), i + half + 1)
        chunk = arr[s:e]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= 5:
            m = np.median(valid)
            med[i] = m
            mad[i] = np.median(np.abs(valid - m))
    return med, mad

# Check each key metric for outliers
metrics_to_check = [
    ("QRS amp",  qrs_amp),
    ("R-R",      rr_ms),
    ("P-wave",   p_amp),
    ("T-wave",   t_amp),
    ("QTc",      qtc_ms),
    ("ST level", st_level),
]

for name, arr in metrics_to_check:
    med, mad = rolling_median_mad(arr, OUTLIER_WINDOW)
    for i in range(n_beats):
        if np.isnan(arr[i]) or np.isnan(med[i]) or np.isnan(mad[i]):
            continue
        # MAD=0 happens if signal is very stable; use a small floor
        effective_mad = max(mad[i], 1.0)
        if abs(arr[i] - med[i]) > OUTLIER_THRESH * effective_mad:
            is_artifact[i] = True

# Don't flag PVCs as artifacts — they're real beats, just abnormal
is_artifact = is_artifact & ~is_pvc

n_artifact = np.sum(is_artifact)
print(f"Artifact beats: {n_artifact}/{n_beats} ({100*n_artifact/max(1,n_beats):.1f}%)")

# Create a "clean" mask: not artifact, not PVC
is_clean = ~is_artifact & ~is_pvc

# =============================================================
# HRV METRICS (rolling window)
# =============================================================
sdnn  = np.full(n_beats, np.nan)
rmssd = np.full(n_beats, np.nan)
W = HRV_WINDOW_BEATS

for i in range(W, n_beats):
    rr_win = rr_ms[i-W+1:i+1]
    clean_win = is_clean[i-W+1:i+1]
    valid = rr_win[~np.isnan(rr_win) & clean_win]
    if len(valid) >= W // 2:
        sdnn[i] = np.std(valid)
        diffs = np.diff(valid)
        rmssd[i] = np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else np.nan

# =============================================================
# RESPIRATORY SINUS ARRHYTHMIA (RSA) — from R-R modulation
# =============================================================
# Interpolate R-R to uniform time grid, then bandpass 0.1-0.5 Hz
valid_rr = ~np.isnan(rr_ms) & is_clean
if np.sum(valid_rr) > 10:
    rr_interp_t = np.arange(beat_time[1], beat_time[-1], 1.0/4)  # 4 Hz grid
    rr_interp = np.interp(rr_interp_t, beat_time[valid_rr], rr_ms[valid_rr])

    # Clip outliers before bandpassing to prevent ringing artifacts
    rr_med = np.median(rr_interp)
    rr_mad = np.median(np.abs(rr_interp - rr_med))
    clip_limit = 5 * max(rr_mad, 1.0)
    rr_interp = np.clip(rr_interp, rr_med - clip_limit, rr_med + clip_limit)

    # Median filter to remove isolated spikes that survived clipping
    from scipy.signal import medfilt
    rr_interp = medfilt(rr_interp, kernel_size=7)

    sos_rsa = signal.butter(2, [0.1, 0.5], 'bandpass', fs=4, output='sos')
    rsa_signal = signal.sosfiltfilt(sos_rsa, rr_interp)
    rsa_envelope = np.abs(signal.hilbert(rsa_signal))
else:
    rr_interp_t = np.array([])
    rsa_signal = np.array([])
    rsa_envelope = np.array([])

# =============================================================
# ROLLING HEART RATE AVERAGE (Gaussian-weighted, σ=5s)
# =============================================================
HR_AVG_SIGMA_SEC = 2.0     # Gaussian σ (effective ~6s window at ±1.5σ)
HR_AVG_CUTOFF = 3.0        # evaluate out to ±3σ
hr_avg = np.full(n_beats, np.nan)
clean_hr = ~np.isnan(hr_bpm) & ~is_pvc  # include artifact beats (valid timing)

clean_idx = np.where(clean_hr)[0]
if len(clean_idx) >= 3:
    clean_times = beat_time[clean_idx]
    clean_vals = hr_bpm[clean_idx]
    half_win = HR_AVG_SIGMA_SEC * HR_AVG_CUTOFF
    lo = 0
    for ci in range(len(clean_idx)):
        t_center = clean_times[ci]
        while lo < ci and clean_times[lo] < t_center - half_win:
            lo += 1
        hi = ci
        while hi < len(clean_idx) - 1 and clean_times[hi + 1] <= t_center + half_win:
            hi += 1
        if hi - lo + 1 >= 3:
            dt = clean_times[lo:hi + 1] - t_center
            weights = np.exp(-0.5 * (dt / HR_AVG_SIGMA_SEC) ** 2)
            hr_avg[clean_idx[ci]] = np.average(clean_vals[lo:hi + 1], weights=weights)

# =============================================================
# PLOT DASHBOARD
# =============================================================
stem = str(Path(CSV_FILE).parent / Path(CSV_FILE).stem)
title_base = f"{Path(CSV_FILE).name}  ({FS} sps, {N/FS:.0f}s)"

def plot_metric(ax, x, y, ylabel, title, color='steelblue', scatter=False):
    valid = ~np.isnan(y)
    clean = valid & is_clean
    dirty = valid & is_artifact
    if scatter:
        if np.any(dirty):
            ax.scatter(x[dirty], y[dirty], s=3, c='silver', alpha=0.7, zorder=1)
        ax.scatter(x[clean], y[clean], s=4, c=color, alpha=0.6, zorder=2)
    else:
        ax.plot(x[valid], y[valid], color=color, linewidth=0.8, alpha=0.8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

if args.plot == 'hr':
    # ----- HR-only plot -----
    fig_hr, ax_hr = plt.subplots(1, 1, figsize=(16, 4))
    fig_hr.suptitle(f"Heart Rate — {title_base}", fontsize=12, fontweight='bold')

    valid_hr = ~np.isnan(hr_bpm)
    clean_hr_mask = valid_hr & is_clean
    dirty_hr_mask = valid_hr & is_artifact

    if np.any(dirty_hr_mask):
        ax_hr.scatter(beat_wall[dirty_hr_mask], hr_bpm[dirty_hr_mask], s=3,
                      c='silver', alpha=0.7, zorder=1, label='Artifact')
    ax_hr.scatter(beat_wall[clean_hr_mask], hr_bpm[clean_hr_mask], s=4,
                  c='crimson', alpha=0.6, zorder=2, label='Heart Rate')

    # Rolling average
    valid_avg = ~np.isnan(hr_avg)
    if np.any(valid_avg):
        ax_hr.plot(beat_wall[valid_avg], hr_avg[valid_avg], color='navy',
                   linewidth=1.5, alpha=0.7, zorder=3, label=f'avg (σ={HR_AVG_SIGMA_SEC:.0f}s)')

    ax_hr.set_ylabel('bpm', fontsize=9)
    ax_hr.set_title('Heart Rate', fontsize=10, loc='left')
    ax_hr.legend(fontsize=8, loc='upper right')
    ax_hr.grid(True, alpha=0.3)
    format_time_axis(ax_hr)
    fig_hr.tight_layout()
    fig_hr.savefig(f"{stem}_hr.png", dpi=150, bbox_inches='tight')


else:
    # ----- Figure 1: ECG trace + Heart Rate + R-R + HRV -----
    fig1, axes1 = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig1.suptitle(f"Rhythm & Rate — {title_base}", fontsize=12, fontweight='bold')

    ax = axes1[0]
    ax.plot(t_wall, ecg, linewidth=0.3, color='steelblue')
    ax.plot(t_wall[peaks], ecg[peaks], 'r.', markersize=2)
    ax.set_ylabel('µV')
    ax.set_title('Filtered ECG with detected R-peaks', fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

    plot_metric(axes1[1], beat_wall, hr_bpm, 'bpm', 'Heart Rate',
                color='crimson', scatter=True)
    valid_avg = ~np.isnan(hr_avg)
    if np.any(valid_avg):
        axes1[1].plot(beat_wall[valid_avg], hr_avg[valid_avg], color='navy',
                      linewidth=1.5, alpha=0.7, zorder=3)

    ax = axes1[2]
    normal_rr = is_clean & ~np.isnan(rr_ms)
    artifact_rr = is_artifact & ~np.isnan(rr_ms)
    pvc_rr = is_pvc & ~np.isnan(rr_ms)
    if np.any(artifact_rr):
        ax.scatter(beat_wall[artifact_rr], rr_ms[artifact_rr], s=3, c='lightgray',
                   alpha=0.4, zorder=1)
    ax.scatter(beat_wall[normal_rr], rr_ms[normal_rr], s=4, c='darkgreen', alpha=0.6,
               zorder=2)
    if np.any(pvc_rr):
        ax.scatter(beat_wall[pvc_rr], rr_ms[pvc_rr], s=20, c='red', marker='x',
                   linewidths=1.5, zorder=5, label='PVC')
        ax.legend(fontsize=8)
    ax.set_ylabel('ms', fontsize=9)
    ax.set_title('R-R Interval', fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

    ax = axes1[3]
    v = ~np.isnan(sdnn)
    ax.plot(beat_wall[v], sdnn[v], color='purple', linewidth=1, label='SDNN', alpha=0.8)
    v = ~np.isnan(rmssd)
    ax.plot(beat_wall[v], rmssd[v], color='orange', linewidth=1, label='RMSSD', alpha=0.8)
    ax.set_ylabel('ms')
    ax.set_title(f'HRV (rolling {W}-beat window)', fontsize=10, loc='left')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    format_time_axis(axes1[3])

    fig1.tight_layout()
    fig1.savefig(f"{stem}_1_rhythm.png", dpi=150, bbox_inches='tight')

    # ----- Figure 2: QRS amplitude + duration + correlation + PVC + ST level -----
    fig2, axes2 = plt.subplots(5, 1, figsize=(16, 13), sharex=True)
    fig2.suptitle(f"QRS & ST Morphology — {title_base}", fontsize=12, fontweight='bold')

    ax = axes2[0]
    normal = is_clean
    artifact = is_artifact & ~np.isnan(qrs_amp)
    if np.any(artifact):
        ax.scatter(beat_wall[artifact], qrs_amp[artifact], s=3, c='lightgray',
                   alpha=0.4, zorder=1)
    ax.scatter(beat_wall[normal], qrs_amp[normal], s=4, c='steelblue', alpha=0.6,
               zorder=2)
    if np.any(is_pvc):
        ax.scatter(beat_wall[is_pvc], qrs_amp[is_pvc], s=20, c='red', marker='x',
                   linewidths=1.5, zorder=5, label=f'PVC ({np.sum(is_pvc)})')
        ax.legend(fontsize=8)
    ax.set_ylabel('µV', fontsize=9)
    ax.set_title('QRS Amplitude (R-peak) — red × = PVC', fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

    plot_metric(axes2[1], beat_wall, qrs_dur_ms, 'ms', 'QRS Duration',
                color='teal', scatter=True)

    ax = axes2[2]
    valid_cc = ~np.isnan(qrs_corr)
    colors = np.where(is_pvc[valid_cc], 'red', 'steelblue')
    ax.scatter(beat_wall[valid_cc], qrs_corr[valid_cc], s=4, c=colors, alpha=0.6)
    ax.axhline(0.85, color='orange', linestyle='--', linewidth=0.8, alpha=0.5,
               label='PVC threshold')
    ax.set_ylabel('r', fontsize=9)
    ax.set_title('QRS Morphology Correlation with Template', fontsize=10, loc='left')
    ax.set_ylim(min(0.4, np.nanmin(qrs_corr) - 0.05) if np.any(valid_cc) else 0.4,
                1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[3]
    valid_sc = ~np.isnan(pvc_score)
    colors_sc = np.where(is_pvc[valid_sc], 'red', 'gray')
    ax.scatter(beat_wall[valid_sc], pvc_score[valid_sc], s=6, c=colors_sc, alpha=0.6)
    ax.set_ylabel('score', fontsize=9)
    ax.set_title('PVC Score (≥3 or morph abnormal → PVC)', fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

    plot_metric(axes2[4], beat_wall, st_level, 'µV', 'ST Level (J+80ms)',
                color='brown', scatter=True)
    axes2[4].axhline(0, color='gray', linestyle='-', linewidth=0.5)
    format_time_axis(axes2[4])

    fig2.tight_layout()
    fig2.savefig(f"{stem}_2_qrs_st.png", dpi=150, bbox_inches='tight')

    # ----- Figure 3: P-wave amplitude + notch + T-wave + QTc -----
    fig3, axes3 = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    fig3.suptitle(f"P-wave, T-wave & QTc — {title_base}", fontsize=12, fontweight='bold')

    plot_metric(axes3[0], beat_wall, p_amp, 'µV', 'P-wave Amplitude',
                color='darkorange', scatter=True)

    ax = axes3[1]
    valid_notch = ~np.isnan(p_notch_uv)
    ax.scatter(beat_wall[valid_notch], p_notch_uv[valid_notch], s=6,
               c=np.where(p_notched[valid_notch], 'red', 'gray'), alpha=0.6)
    ax.set_ylabel('µV')
    ax.set_title('P-wave Notch Depth (red = notch detected)', fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)

    plot_metric(axes3[2], beat_wall, t_amp, 'µV', 'T-wave Amplitude',
                color='green', scatter=True)

    plot_metric(axes3[3], beat_wall, qtc_ms, 'ms', 'QTc (Bazett)',
                color='darkred', scatter=True)
    axes3[3].axhline(350, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    axes3[3].axhline(450, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    format_time_axis(axes3[3])

    fig3.tight_layout()
    fig3.savefig(f"{stem}_3_pt_waves.png", dpi=150, bbox_inches='tight')

    # ----- Figure 4: Respiratory Sinus Arrhythmia -----
    fig4, ax4 = plt.subplots(1, 1, figsize=(16, 4))
    fig4.suptitle(f"Respiratory Sinus Arrhythmia — {title_base}",
                  fontsize=12, fontweight='bold')

    if len(rr_interp_t) > 0:
        rsa_epoch = sample_to_epoch(rr_interp_t * FS)  # rr_interp_t is in seconds
        rsa_wall = epoch_to_mpl(rsa_epoch)
        ax4.plot(rsa_wall, rsa_signal, color='teal', linewidth=0.6, alpha=0.6,
                 label='RSA')
        ax4.plot(rsa_wall, rsa_envelope, color='red', linewidth=1, alpha=0.8,
                 label='Envelope')
        ax4.legend(fontsize=8)
    ax4.set_ylabel('ms')
    ax4.set_title('R-R modulation bandpassed 0.1–0.5 Hz', fontsize=10, loc='left')
    ax4.grid(True, alpha=0.3)
    format_time_axis(ax4)

    fig4.tight_layout()
    fig4.savefig(f"{stem}_4_rsa.png", dpi=150, bbox_inches='tight')



if not args.no_plot:
    plt.show()

# =============================================================
# CSV EXPORT (per-beat metrics)
# =============================================================
if args.csv_out:
    import csv as csvmod
    csv_out_path = f"{stem}_beats.csv"
    with open(csv_out_path, 'w', newline='') as f:
        writer = csvmod.writer(f)
        # Header comment with metadata
        f.write(f"# source: {Path(CSV_FILE).name}\n")
        f.write(f"# sample_rate: {FS}\n")
        f.write(f"# time_source: {time_source}\n")
        writer.writerow([
            'epoch_s', 'sample_idx', 'hr_bpm', 'hr_avg_bpm', 'rr_ms',
            'qrs_amp_uv', 'qrs_dur_ms', 'p_amp_uv', 'p_dur_ms',
            't_amp_uv', 'qt_ms', 'qtc_ms', 'st_level_uv',
            'is_artifact', 'is_pvc', 'qrs_corr', 'pvc_score'
        ])
        for i in range(n_beats):
            writer.writerow([
                f"{beat_epoch[i]:.6f}",
                int(peaks[i]),
                f"{hr_bpm[i]:.1f}" if not np.isnan(hr_bpm[i]) else '',
                f"{hr_avg[i]:.1f}" if not np.isnan(hr_avg[i]) else '',
                f"{rr_ms[i]:.1f}" if not np.isnan(rr_ms[i]) else '',
                f"{qrs_amp[i]:.1f}" if not np.isnan(qrs_amp[i]) else '',
                f"{qrs_dur_ms[i]:.1f}" if not np.isnan(qrs_dur_ms[i]) else '',
                f"{p_amp[i]:.1f}" if not np.isnan(p_amp[i]) else '',
                f"{p_dur_ms[i]:.1f}" if not np.isnan(p_dur_ms[i]) else '',
                f"{t_amp[i]:.1f}" if not np.isnan(t_amp[i]) else '',
                f"{qt_ms[i]:.1f}" if not np.isnan(qt_ms[i]) else '',
                f"{qtc_ms[i]:.1f}" if not np.isnan(qtc_ms[i]) else '',
                f"{st_level[i]:.1f}" if not np.isnan(st_level[i]) else '',
                int(is_artifact[i]),
                int(is_pvc[i]),
                f"{qrs_corr[i]:.3f}" if not np.isnan(qrs_corr[i]) else '',
                f"{pvc_score[i]:.0f}" if not np.isnan(pvc_score[i]) else '',
            ])


# =============================================================
# SUMMARY OUTPUT (optional save to file)
# =============================================================
_summary_file = None
_original_stdout = sys.stdout

if args.save_summary:
    summary_path = f"{stem}_summary.txt"
    _summary_file = open(summary_path, 'w')
    # Write header info to file only
    _summary_file.write(f"Source: {Path(CSV_FILE).name}\n")
    _summary_file.write(f"Samples: {N} ({N/FS:.1f}s) at {FS} sps\n")
    if time_source != 'elapsed':
        _summary_file.write(f"Start: {t0_str}\n")
    _summary_file.write(f"Beats: {n_beats} (artifacts: {n_artifact}, "
                        f"PVCs: {np.sum(is_pvc)})\n\n")

    class _Tee:
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
        def flush(self):
            self.stream.flush()
            self.file.flush()

    sys.stdout = _Tee(_summary_file, _original_stdout)

# =============================================================
# STUDY WINDOW MASK  (exclude data after 7:00 AM local time)
# =============================================================
# Beats recorded after 7 AM may include electrode disconnection artifacts.
# We use this mask for extreme-value stats (longest pause, brady/tachy)
# but not for general morphology stats which already have artifact rejection.

STUDY_END_HOUR = 7  # 7:00 AM local time

in_study = np.ones(n_beats, dtype=bool)
if time_source != 'elapsed':
    for i in range(n_beats):
        dt = datetime.fromtimestamp(beat_epoch[i])
        # If recording starts before midnight, hours < start_hour are next day
        if dt.hour >= STUDY_END_HOUR and dt.hour < 20:
            in_study[i] = False

n_in_study = np.sum(in_study)
n_excluded_study = n_beats - n_in_study
if n_excluded_study > 0:
    print(f"Study window: {n_in_study} beats before {STUDY_END_HOUR}:00 AM "
          f"({n_excluded_study} post-study beats masked)")

# =============================================================
# SUMMARY STATS
# =============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

def stat_line(name, arr, unit='', mask=None):
    if mask is None:
        mask = is_clean[:len(arr)] if len(arr) == n_beats else np.ones(len(arr), dtype=bool)
    v = arr[~np.isnan(arr) & mask[:len(arr)]]
    if len(v) > 0:
        print(f"  {name:25s}: {np.mean(v):7.1f} ± {np.std(v):5.1f} {unit}"
              f"  [{np.min(v):.1f} – {np.max(v):.1f}]")

stat_line("Heart Rate", hr_bpm, "bpm")
stat_line("R-R Interval", rr_ms, "ms")

# --- Median HR ---
hr_clean = hr_bpm[~np.isnan(hr_bpm) & is_clean]
if len(hr_clean) > 0:
    print(f"  {'HR Median':25s}: {np.median(hr_clean):7.1f} bpm")

# --- HR Distribution (5 bpm bins) ---
if len(hr_clean) > 0:
    bin_edges = [0, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 999]
    bin_labels = ['<45','45-','50-','55-','60-','65-','70-','75-',
                  '80-','85-','90-','95-','100+']
    counts, _ = np.histogram(hr_clean, bins=bin_edges)
    pcts = 100.0 * counts / len(hr_clean)
    # Compact vector display: only show bins that have >0%
    parts = []
    for lbl, pct in zip(bin_labels, pcts):
        if pct >= 0.05:  # show if ≥0.05%
            parts.append(f"{lbl}:{pct:.1f}")
    print(f"  {'HR Distribution (%)':25s}: [{', '.join(parts)}]")
    # Text-mode bar chart
    max_pct = max((p for p in pcts if p >= 0.05), default=1.0)
    bar_width = 40  # max bar length in characters
    print()
    for lbl, pct in zip(bin_labels, pcts):
        if pct >= 0.05:
            bar_len = max(1, round(pct / max_pct * bar_width))
            bar = '*' * bar_len
            print(f"    {lbl:>5}  {bar:<{bar_width}}  {pct:5.1f}%")
    print()

# --- Bradycardia / Tachycardia burden (within study window) ---
hr_study = hr_bpm[~np.isnan(hr_bpm) & is_clean & in_study]
if len(hr_study) > 0:
    n_brady = np.sum(hr_study < 50)
    n_tachy = np.sum(hr_study > 100)
    print(f"  {'Brady (<50) / Tachy (>100)':25s}: "
          f"{n_brady} ({100*n_brady/len(hr_study):.2f}%) / "
          f"{n_tachy} ({100*n_tachy/len(hr_study):.2f}%)")

# --- Min / Max 5-minute average HR ---
HR_AVG_WINDOW_SEC = 300  # 5 minutes
if len(hr_clean) > 0 and n_beats > 1:
    hr_5min = np.full(n_beats, np.nan)
    valid_hr_mask = ~np.isnan(hr_bpm) & is_clean & in_study
    valid_hr_idx = np.where(valid_hr_mask)[0]
    if len(valid_hr_idx) >= 10:
        vt = beat_time[valid_hr_idx]
        vh = hr_bpm[valid_hr_idx]
        lo = 0
        for ci in range(len(valid_hr_idx)):
            t_center = vt[ci]
            while lo < ci and vt[lo] < t_center - HR_AVG_WINDOW_SEC / 2:
                lo += 1
            hi = ci
            while hi < len(valid_hr_idx) - 1 and vt[hi+1] <= t_center + HR_AVG_WINDOW_SEC / 2:
                hi += 1
            span = vt[hi] - vt[lo]
            if span >= HR_AVG_WINDOW_SEC * 0.8 and hi - lo + 1 >= 10:
                hr_5min[valid_hr_idx[ci]] = np.mean(vh[lo:hi+1])
        hr_5min_valid = hr_5min[~np.isnan(hr_5min)]
        if len(hr_5min_valid) > 0:
            print(f"  {'HR 5-min avg (min/max)':25s}: "
                  f"{np.min(hr_5min_valid):5.1f} / {np.max(hr_5min_valid):.1f} bpm")

stat_line("QRS Amplitude", qrs_amp, "µV")
stat_line("QRS Duration", qrs_dur_ms, "ms")
stat_line("P-wave Amplitude", p_amp, "µV")
stat_line("P-wave Duration", p_dur_ms, "ms")
stat_line("P-wave Notch Depth", p_notch_uv[p_notched], "µV",
          mask=is_clean[p_notched])
stat_line("T-wave Amplitude", t_amp, "µV")
stat_line("QT Interval", qt_ms, "ms")
stat_line("QTc (Bazett)", qtc_ms, "ms")
qtc_clean = qtc_ms[~np.isnan(qtc_ms) & is_clean]
if len(qtc_clean) > 0:
    p5, p95 = np.percentile(qtc_clean, [5, 95])
    print(f"  {'QTc 5th–95th %ile':25s}: {p5:7.1f} – {p95:.1f} ms")
stat_line("ST Level (J+80)", st_level, "µV")
stat_line("SDNN", sdnn, "ms", mask=np.ones(n_beats, dtype=bool))
stat_line("RMSSD", rmssd, "ms", mask=np.ones(n_beats, dtype=bool))

# --- pNN50 ---
rr_clean_mask = ~np.isnan(rr_ms) & is_clean
rr_clean_vals = rr_ms[rr_clean_mask]
if len(rr_clean_vals) > 1:
    rr_diffs = np.abs(np.diff(rr_clean_vals))
    pnn50 = 100.0 * np.sum(rr_diffs > 50) / len(rr_diffs)
    print(f"  {'pNN50':25s}: {pnn50:7.1f} %")

# --- HRV Triangular Index ---
# Total clean R-R beats / mode-bin height, using 1/128s (~7.8ms) bins per standard
if len(rr_clean_vals) > 10:
    bin_width = 1000.0 / 128  # ~7.8125 ms per HRV standard
    rr_min = np.floor(np.min(rr_clean_vals) / bin_width) * bin_width
    rr_max = np.ceil(np.max(rr_clean_vals) / bin_width) * bin_width
    hist_bins = np.arange(rr_min, rr_max + bin_width, bin_width)
    hist_counts, _ = np.histogram(rr_clean_vals, bins=hist_bins)
    mode_count = np.max(hist_counts)
    if mode_count > 0:
        hrv_tri = len(rr_clean_vals) / mode_count
        print(f"  {'HRV Triangular Index':25s}: {hrv_tri:7.1f}")

# --- Longest R-R pause (within study window, clean beats only) ---
rr_study_mask = ~np.isnan(rr_ms) & in_study
rr_study_vals = rr_ms[rr_study_mask]
rr_study_idx = np.where(rr_study_mask)[0]
if len(rr_study_vals) > 0:
    longest_rr = np.max(rr_study_vals)
    longest_rr_beat = rr_study_idx[np.argmax(rr_study_vals)]
    pause_time_str = ""
    if time_source != 'elapsed':
        pause_dt = datetime.fromtimestamp(beat_epoch[longest_rr_beat])
        pause_time_str = f" at {pause_dt.strftime('%H:%M:%S')}"
    flag = " ⚠" if longest_rr > 2000 else ""
    print(f"  {'Longest R-R pause':25s}: {longest_rr:7.0f} ms "
          f"({longest_rr/1000:.2f}s){pause_time_str}{flag}")

n_notched = np.sum(p_notched & is_clean)
n_measured = np.sum(~np.isnan(p_notch_uv) & is_clean)
print(f"\n  P-wave notch: {n_notched}/{n_measured} beats "
      f"({100*n_notched/max(1,n_measured):.0f}%)")
print(f"  Artifact beats excluded: {n_artifact}/{n_beats} "
      f"({100*n_artifact/max(1,n_beats):.1f}%)")
print(f"  Missed beats recovered: {n_recovered}")
if n_excluded_study > 0:
    print(f"  Post-study (>{STUDY_END_HOUR}AM) excluded: {n_excluded_study} beats")

n_pvc_total = np.sum(is_pvc)
duration_min = (peaks[-1] - peaks[0]) / FS / 60 if n_beats > 1 else 0
print(f"  PVCs: {n_pvc_total}/{n_beats} beats "
      f"({100*n_pvc_total/max(1,n_beats):.1f}% burden, "
      f"{n_pvc_total/max(0.01,duration_min):.1f}/min)")
if n_pvc_total > 0:
    stat_line("PVC QRS Correlation", qrs_corr[is_pvc], "")
    # Check for couplets/triplets
    pvc_idx = np.where(is_pvc)[0]
    couplets = 0
    triplets = 0
    i_pvc = 0
    while i_pvc < len(pvc_idx):
        run = 1
        while (i_pvc + run < len(pvc_idx) and
               pvc_idx[i_pvc + run] == pvc_idx[i_pvc + run - 1] + 1):
            run += 1
        if run == 2:
            couplets += 1
        elif run >= 3:
            triplets += 1
        i_pvc += run
    if couplets > 0 or triplets > 0:
        print(f"  PVC runs: {couplets} couplets, {triplets} triplets/runs")

    # Longest PVC-free run (in beats and time)
    if len(pvc_idx) >= 1:
        # Gaps between consecutive PVCs (in beat indices)
        pvc_gaps = np.diff(pvc_idx) - 1  # number of non-PVC beats between PVCs
        # Also check gap before first PVC and after last PVC
        gaps = [pvc_idx[0]]  # beats before first PVC
        if len(pvc_gaps) > 0:
            gaps.extend(pvc_gaps.tolist())
        gaps.append(n_beats - 1 - pvc_idx[-1])  # beats after last PVC
        max_gap_beats = max(gaps)
        # Estimate time from beats (use mean R-R)
        mean_rr_sec = np.nanmean(rr_ms[is_clean]) / 1000 if np.any(is_clean) else 0.85
        max_gap_min = max_gap_beats * mean_rr_sec / 60
        print(f"  Longest PVC-free run   : {max_gap_beats} beats ({max_gap_min:.1f} min)")

# =============================================================
# CLOSE SUMMARY FILE
# =============================================================
if _summary_file:
    sys.stdout = _original_stdout
    _summary_file.close()
    print(f"Summary saved: {summary_path}")