#!/usr/bin/env python3
"""
Extract breathing amplitude envelope and rate from MPU-6050 motion CSV.
Outputs a sub-sampled CSV at 1-second intervals with:
  - seconds from start
  - breathing envelope amplitude (degrees)
  - breathing rate (breaths/min, 30s sliding window)

Usage: python3 breath_envelope_csv.py <input.csv> [output.csv]
If no output file given, uses input name with _breath suffix.
J.Beale 2026-02-11
"""

import sys
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, hilbert

# --- Configuration ---
BREATH_LO = 0.1    # Hz (6 breaths/min)
BREATH_HI = 0.5    # Hz (30 breaths/min)
ENV_SMOOTH = 0.05   # Hz lowpass for envelope smoothing
OUTPUT_INTERVAL = 1.0  # seconds between output samples
RATE_WINDOW = 60.0  # seconds, sliding window for breath rate

ABS_FLOOR = 0.10 # reject tilt peaks below this absolute level, to avoid noise peaks when breathing is very shallow


def process(csv_path, out_path=None):
    # parse header for start time
    start_time = None
    skip = 1
    with open(csv_path, 'r') as f:
        first = f.readline().strip()
        if first.startswith('# start ') and 'unknown' not in first:
            start_time = first[8:].split('sync_millis')[0].strip()
            skip = 2

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=skip)
    if data.ndim != 2 or data.shape[1] != 6:
        print(f"Error: expected 6 columns, got {data.shape}")
        sys.exit(1)

    if start_time:
        t = data[:, 0] / 1000.0
    else:
        t = (data[:, 0] - data[0, 0]) / 1000.0

    pitch = data[:, 1]
    dt_median = np.median(np.diff(t))
    fs = 1.0 / dt_median
    duration_min = (t[-1] - t[0]) / 60.0

    print(f"File: {csv_path}")
    print(f"Samples: {len(t)},  Duration: {duration_min:.1f} min,  Fs: {fs:.2f} Hz")
    if start_time:
        print(f"Start: {start_time}")

    # bandpass filter for breathing
    breath_sos = butter(2, [BREATH_LO, BREATH_HI], btype='bandpass', fs=fs, output='sos')
    breath_filt = sosfiltfilt(breath_sos, pitch)

    # Hilbert envelope + smoothing
    analytic = hilbert(breath_filt)
    env_amplitude = np.abs(analytic)
    env_lp_sos = butter(2, ENV_SMOOTH, btype='low', fs=fs, output='sos')
    env_amplitude = sosfiltfilt(env_lp_sos, env_amplitude)

    # find all breath peaks using adaptive local prominence
    min_peak_dist = max(1, int(1.5 * fs))
    # Use local envelope as adaptive prominence threshold:
    # a peak counts as a breath if it's at least 15% of the local envelope amplitude
    # First pass: find peaks with very low fixed prominence
    candidate_idx, candidate_props = find_peaks(breath_filt, distance=min_peak_dist,
                                                 prominence=0.001)
    # Filter: keep only peaks whose prominence >= 25% of local envelope
    if len(candidate_idx) > 0:
        local_env = env_amplitude[candidate_idx]
        adaptive_thresh = local_env * 0.25
        # Absolute floor: reject peaks below noise level
        adaptive_thresh = np.maximum(adaptive_thresh, ABS_FLOOR)
        keep = candidate_props['prominences'] >= adaptive_thresh
        peak_idx = candidate_idx[keep]
    else:
        peak_idx = candidate_idx
    peak_times = t[peak_idx]

    # sub-sample at 1-second intervals
    t_out = np.arange(t[0], t[-1], OUTPUT_INTERVAL)
    env_out = np.interp(t_out, t, env_amplitude)

    # breathing rate via sliding window
    half_win = RATE_WINDOW / 2.0
    rate_out = np.full_like(t_out, np.nan)
    for i, tc in enumerate(t_out):
        mask = (peak_times >= tc - half_win) & (peak_times <= tc + half_win)
        win_peaks = peak_times[mask]
        if len(win_peaks) >= 2:
            intervals = np.diff(win_peaks)
            rate_out[i] = 60.0 / np.mean(intervals)

    # write output
    if out_path is None:
        base = os.path.splitext(csv_path)[0]
        out_path = base + '_breath.csv'

    header_lines = []
    if start_time:
        header_lines.append(f'# start {start_time}')
    header_lines.append('seconds,envelope_deg,breaths_per_min')

    with open(out_path, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')
        for i in range(len(t_out)):
            sec = f'{t_out[i]:.1f}'
            env = f'{env_out[i]:.4f}'
            bpm = f'{rate_out[i]:.1f}' if not np.isnan(rate_out[i]) else ''
            f.write(f'{sec},{env},{bpm}\n')

    n_valid = np.sum(~np.isnan(rate_out))
    print(f"Output: {out_path}")
    print(f"Rows: {len(t_out)},  Rate coverage: {n_valid}/{len(t_out)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 breath_envelope_csv.py <input.csv> [output.csv]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    process(csv_path, out_path)
