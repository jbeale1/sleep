#!/usr/bin/env python3
"""
ECG + Pleth Beat-by-Beat PTT and HR Analysis
----------------------------------------------
Sweeps through the full overlap period, detecting R-peaks and pleth troughs,
and produces time series of:
  - Heart rate (from R-R intervals)
  - Pulse transit time (R-peak to pleth trough)
  - Relative BP proxy (1/PTT^2, proportional to BP via Moens-Korteweg)

Usage:
    python ecg_pleth_ptt.py <data_directory> [--offset SECONDS]
"""

import sys
import argparse
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from datetime import datetime

from sleep_common import (PST, ECG_FS, PLETH_FS, PLETH_UP_FS,
                          load_and_process, upsample_pleth, detect_pulse_feet)

# ── R-peak detection (full signal) ──────────────────────────────────────
def detect_r_peaks(ecg_filt, ecg_t):
    """Detect R-peaks across the full ECG signal.
    
    Uses a sliding window to adapt the height threshold to local amplitude
    variations (e.g. from movement artifacts or lead contact changes).
    """
    print("Detecting R-peaks...")
    
    # Global pass: generous threshold to catch most peaks
    # Use a rolling max approach: process in 30s chunks with adaptive threshold
    chunk_s = 30
    chunk_n = int(chunk_s * ECG_FS)
    all_peaks = []
    
    for start in range(0, len(ecg_filt), chunk_n):
        end = min(start + chunk_n, len(ecg_filt))
        seg = ecg_filt[start:end]
        if len(seg) < ECG_FS:  # need at least 1s
            continue
        
        # Adaptive threshold: 35% of the 95th percentile in this chunk
        # (using percentile rather than max to be robust to residual spikes)
        thresh = np.percentile(seg, 95) * 0.35
        if thresh < 50:  # minimum threshold to avoid noise peaks
            thresh = 50
        
        pks, _ = find_peaks(seg, height=thresh, distance=int(ECG_FS * 0.4))
        all_peaks.extend(pks + start)
    
    peaks = np.array(all_peaks)
    
    # Remove duplicate detections at chunk boundaries
    if len(peaks) > 1:
        keep = np.concatenate([[True], np.diff(peaks) > int(ECG_FS * 0.4)])
        peaks = peaks[keep]
    
    r_times = ecg_t[peaks]
    print(f"  Found {len(peaks)} R-peaks")
    
    # R-R intervals
    rr_ms = np.diff(r_times) * 1000
    valid = (rr_ms > 300) & (rr_ms < 3000)  # 20–200 bpm range
    print(f"  R-R intervals: {np.median(rr_ms[valid]):.0f} ms median "
          f"({60000/np.median(rr_ms[valid]):.0f} bpm)")
    
    return peaks, r_times

# ── Pleth trough detection (full signal) ────────────────────────────────
def detect_pleth_feet(pleth_v_up, pleth_t_up):
    """Detect pulse feet on upsampled (240 sps) signal using intersecting
    tangent method.  Processes in chunks to adapt to varying amplitude.
    """
    print("Detecting pulse feet (intersecting tangent method)...")
    
    chunk_s = 30
    chunk_n = int(chunk_s * PLETH_UP_FS)
    # Use overlapping chunks to avoid edge effects
    overlap_n = int(2 * PLETH_UP_FS)  # 2s overlap
    
    all_times = []
    all_vals = []
    
    start = 0
    while start < len(pleth_v_up):
        end = min(start + chunk_n + overlap_n, len(pleth_v_up))
        seg_t = pleth_t_up[start:end]
        seg_v = pleth_v_up[start:end]
        
        if len(seg_v) < PLETH_UP_FS:
            break
        
        ft, fv = detect_pulse_feet(seg_t, seg_v, PLETH_UP_FS)
        
        if len(ft) > 0:
            # Only keep feet within the non-overlap region (except last chunk)
            if end < len(pleth_v_up):
                cutoff_t = pleth_t_up[min(start + chunk_n, len(pleth_t_up) - 1)]
                mask = ft <= cutoff_t
                all_times.extend(ft[mask])
                all_vals.extend(fv[mask])
            else:
                all_times.extend(ft)
                all_vals.extend(fv)
        
        start += chunk_n
    
    foot_times = np.array(all_times)
    foot_vals = np.array(all_vals)
    
    # Remove any duplicates from chunk boundaries
    if len(foot_times) > 1:
        keep = np.concatenate([[True], np.diff(foot_times) > 0.25])
        foot_times = foot_times[keep]
        foot_vals = foot_vals[keep]
    
    print(f"  Found {len(foot_times)} pulse feet")
    return foot_times, foot_vals

# ── PTT computation ─────────────────────────────────────────────────────
def compute_ptt(r_times, trough_times, offset=0.0):
    """For each pleth trough, find the nearest preceding R-peak and compute PTT.
    
    Returns arrays of: beat_time (midpoint), ptt_ms, r_time, trough_time
    """
    trough_shifted = trough_times + offset
    
    ptt_list = []
    r_idx = 0
    
    for t_time in trough_shifted:
        # Advance r_idx to the last R-peak before this trough
        while r_idx < len(r_times) - 1 and r_times[r_idx + 1] < t_time:
            r_idx += 1
        
        if r_idx < len(r_times) and r_times[r_idx] < t_time:
            ptt_ms = (t_time - r_times[r_idx]) * 1000
            if 50 < ptt_ms < 800:
                ptt_list.append({
                    'r_time': r_times[r_idx],
                    'trough_time': t_time,
                    'beat_time': r_times[r_idx],  # timestamp at R-peak
                    'ptt_ms': ptt_ms,
                })
    
    if not ptt_list:
        print("WARNING: No valid PTT measurements found!")
        return None
    
    result = {k: np.array([d[k] for d in ptt_list]) for k in ptt_list[0]}
    print(f"  {len(ptt_list)} PTT measurements, "
          f"median={np.median(result['ptt_ms']):.0f} ms")
    return result

# ── HR from R-R intervals ──────────────────────────────────────────────
def compute_hr(r_times):
    """Compute instantaneous HR from R-R intervals.
    
    Returns arrays of: beat_time (midpoint of interval), hr_bpm, rr_ms
    """
    rr_s = np.diff(r_times)
    rr_ms = rr_s * 1000
    
    # Filter physiological range: 300–3000 ms (20–200 bpm)
    valid = (rr_ms > 300) & (rr_ms < 3000)
    
    # Midpoint time of each R-R interval
    mid_times = (r_times[:-1] + r_times[1:]) / 2
    
    hr_bpm = 60000.0 / rr_ms
    
    return {
        'beat_time': mid_times[valid],
        'hr_bpm': hr_bpm[valid],
        'rr_ms': rr_ms[valid],
    }

# ── Plotting ────────────────────────────────────────────────────────────
def plot_overnight(hr, ptt, t_start, t_end):
    """Plot overnight HR, PTT, and relative BP proxy."""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08, top=0.93, bottom=0.07)
    
    date_str = datetime.fromtimestamp(t_start, PST).strftime('%Y-%m-%d')
    fig.suptitle(f'Overnight HR / PTT / Relative BP  —  {date_str}', fontsize=12)
    
    # Time formatter
    def format_time(t, pos):
        try:
            return datetime.fromtimestamp(t, PST).strftime('%H:%M')
        except:
            return ''
    
    # ── Smoothing helper ────────────────────────────────────────────────
    def smooth(t, y, window_s=60):
        """Median-filter then moving average for display."""
        if len(y) < 5:
            return t, y
        # Median filter to reject outliers (width ~5 beats)
        y_med = median_filter(y, size=min(7, len(y) | 1))
        # Moving average over ~window_s seconds
        if len(t) > 1:
            dt_median = np.median(np.diff(t))
            n_avg = max(1, int(window_s / dt_median))
            if n_avg > 1 and len(y_med) > n_avg:
                kernel = np.ones(n_avg) / n_avg
                y_smooth = np.convolve(y_med, kernel, mode='same')
                # Trim edges where convolution is unreliable
                half = n_avg // 2
                return t[half:-half], y_smooth[half:-half]
        return t, y_med
    
    # ── Panel 1: Heart Rate ─────────────────────────────────────────────
    ax_hr = axes[0]
    ax_hr.scatter(hr['beat_time'], hr['hr_bpm'], s=0.3, alpha=0.2,
                  color='steelblue', rasterized=True)
    t_sm, hr_sm = smooth(hr['beat_time'], hr['hr_bpm'])
    ax_hr.plot(t_sm, hr_sm, 'b-', linewidth=1.2, label='60s smooth')
    ax_hr.set_ylabel('Heart Rate (bpm)')
    ax_hr.set_ylim(30, 120)
    ax_hr.legend(loc='upper right', fontsize=8)
    ax_hr.grid(True, alpha=0.3)
    
    # ── Panel 2: PTT ────────────────────────────────────────────────────
    ax_ptt = axes[1]
    ax_ptt.scatter(ptt['beat_time'], ptt['ptt_ms'], s=0.3, alpha=0.2,
                   color='green', rasterized=True)
    t_sm, ptt_sm = smooth(ptt['beat_time'], ptt['ptt_ms'])
    ax_ptt.plot(t_sm, ptt_sm, 'g-', linewidth=1.2, label='60s smooth')
    ax_ptt.set_ylabel('PTT (ms)')
    ax_ptt.legend(loc='upper right', fontsize=8)
    ax_ptt.grid(True, alpha=0.3)
    
    # ── Panel 3: Relative BP proxy (1/PTT) ─────────────────────────────
    ax_bp = axes[2]
    # 1/PTT is proportional to pulse wave velocity, which tracks BP
    # Normalize to arbitrary units centered on median
    bp_proxy = 1.0 / ptt['ptt_ms']
    bp_median = np.median(bp_proxy)
    bp_norm = bp_proxy / bp_median * 100  # percent of median
    
    ax_bp.scatter(ptt['beat_time'], bp_norm, s=0.3, alpha=0.2,
                  color='firebrick', rasterized=True)
    t_sm, bp_sm = smooth(ptt['beat_time'], bp_norm)
    ax_bp.plot(t_sm, bp_sm, 'r-', linewidth=1.2, label='60s smooth')
    ax_bp.set_ylabel('Relative BP\n1/PTT (% of median)')
    ax_bp.axhline(100, color='gray', linewidth=0.5, linestyle='--')
    ax_bp.legend(loc='upper right', fontsize=8)
    ax_bp.grid(True, alpha=0.3)
    ax_bp.set_xlabel('Time (PST)')
    
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    
    # Set x limits to overlap range
    ax_hr.set_xlim(t_start, t_end)
    
    # Print summary stats
    print(f"\n── Summary ──")
    print(f"  HR:  median={np.median(hr['hr_bpm']):.0f} bpm, "
          f"min={np.min(hr['hr_bpm']):.0f}, max={np.max(hr['hr_bpm']):.0f}")
    print(f"  PTT: median={np.median(ptt['ptt_ms']):.0f} ms, "
          f"min={np.min(ptt['ptt_ms']):.0f}, max={np.max(ptt['ptt_ms']):.0f}")
    print(f"  Relative BP range: {bp_sm.min():.0f}%–{bp_sm.max():.0f}% of median")
    
    plt.show()

# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Beat-by-beat PTT and HR analysis from ECG + pleth data')
    parser.add_argument('directory', help='Data directory containing CSV files')
    parser.add_argument('--offset', type=float, default=0.0,
                        help='Pleth time offset in seconds (default: 0)')
    args = parser.parse_args()
    
    d = load_and_process(args.directory)
    
    # Detect features
    r_peaks_idx, r_times = detect_r_peaks(d['ecg_filt'], d['ecg_t'])
    
    print("Upsampling pleth for pulse foot detection...")
    pleth_t_up, pleth_v_up = upsample_pleth(d['pleth_t'], d['pleth_v'])
    foot_times, foot_vals = detect_pleth_feet(pleth_v_up, pleth_t_up)
    
    # Compute beat-by-beat metrics
    hr = compute_hr(r_times)
    ptt = compute_ptt(r_times, foot_times, offset=args.offset)
    
    if ptt is None:
        print("No PTT data — cannot plot.")
        sys.exit(1)
    
    # Plot
    plot_overnight(hr, ptt, d['t_start'], d['t_end'])

if __name__ == '__main__':
    main()
