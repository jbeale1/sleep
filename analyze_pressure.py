#!/usr/bin/env python3
"""
analyze_pressure.py — Analyze pressure sensor recordings for cardiac and respiratory signals.

Reads CSV files with columns:
  - millis,pressure_hPa   (Pico SD-card logger format)
  - timestamp,pressure_hPa (USB real-time logger format)

Outputs:
  - Per-window statistics CSV  (*_stats.csv)
  - Summary plot PNG            (*_summary.png)
  - Detailed plot PNG           (*_detail.png)

Usage:
  python analyze_pressure.py <input.csv> [--window 30] [--step 15]
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
FS_NOMINAL = 31.5          # nominal sample rate (Hz)

# Cardiac detection
HR_MIN_BPM = 40
HR_MAX_BPM = 180
MIN_AUTOCORR_HR = 0.15     # minimum autocorrelation quality for HR
MIN_BEATS = 4
MAX_IBI_CV = 0.30

# Respiratory detection
RESP_MIN_BPM = 4           # 4 breaths/min (very slow)
RESP_MAX_BPM = 30          # 30 breaths/min
MIN_AUTOCORR_RESP = 0.10
MIN_BREATH_CYCLES = 2


def load_csv(filepath):
    """Load CSV, return (time_seconds, pressure_hPa, actual_fs)."""
    df = pd.read_csv(filepath)

    if 'millis' in df.columns:
        t_ms = df['millis'].values.astype(float)
        t_s = (t_ms - t_ms[0]) / 1000.0
    elif 'timestamp' in df.columns:
        # Use sequential index; timestamps are unreliable per-sample
        t_s = np.arange(len(df)) / FS_NOMINAL
    else:
        raise ValueError(f"Unrecognized CSV format. Columns: {list(df.columns)}")

    pressure = df['pressure_hPa'].values.astype(float)

    # Estimate actual sample rate from time vector
    if 'millis' in df.columns:
        dt = np.diff(t_s)
        dt_clean = dt[(dt > 0.01) & (dt < 0.1)]  # reject outliers
        if len(dt_clean) > 10:
            fs = 1.0 / np.median(dt_clean)
        else:
            fs = FS_NOMINAL
    else:
        fs = FS_NOMINAL

    return t_s, pressure, fs


def resample_uniform(t_s, pressure, fs):
    """Resample to uniform time grid (needed for filtering)."""
    t_uniform = np.arange(t_s[0], t_s[-1], 1.0 / fs)
    p_uniform = np.interp(t_uniform, t_s, pressure)
    return t_uniform, p_uniform


def compute_cardiac(data, fs):
    """Compute heart rate from a data window.
    Returns dict with hr_bpm, hr_quality, or None values if unreliable."""
    result = {'hr_bpm': None, 'hr_quality': 0.0}
    N = len(data)
    if N < int(3 * fs):
        return result

    try:
        sos_narrow = signal.butter(4, [0.7, 2.5], btype='bandpass', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos_narrow, data)
    except Exception:
        return result

    # Autocorrelation
    x = filtered - np.mean(filtered)
    if np.std(x) < 1e-9:
        return result
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] <= 0:
        return result
    corr = corr / corr[0]

    min_lag = int(60.0 / HR_MAX_BPM * fs)
    max_lag = min(int(60.0 / HR_MIN_BPM * fs), len(corr) - 1)
    if min_lag >= max_lag:
        return result

    peaks, _ = signal.find_peaks(corr[min_lag:max_lag], distance=int(0.15 * fs))
    peaks = peaks + min_lag
    if len(peaks) == 0:
        return result

    best = peaks[np.argmax(corr[peaks])]
    ac_quality = corr[best]
    if ac_quality < MIN_AUTOCORR_HR:
        return result

    est_period = best / fs

    # Derivative peak detection
    try:
        sos_lp = signal.butter(4, 8.0, btype='lowpass', fs=fs, output='sos')
        sos_hp = signal.butter(2, 0.5, btype='highpass', fs=fs, output='sos')
        smoothed = signal.sosfiltfilt(sos_lp, data)
        detrended = signal.sosfiltfilt(sos_hp, smoothed)
    except Exception:
        return result

    deriv = np.diff(detrended) * fs
    deriv_std = np.std(deriv)
    if deriv_std < 1e-9:
        return result

    min_dist = max(int(0.6 * est_period * fs), 3)
    d_peaks, _ = signal.find_peaks(deriv, height=deriv_std * 0.8,
                                   distance=min_dist, prominence=deriv_std * 0.5)
    if len(d_peaks) < MIN_BEATS:
        return result

    ibi = np.diff(d_peaks) / fs
    med_ibi = np.median(ibi)
    mad = np.median(np.abs(ibi - med_ibi))
    if mad > 0:
        ibi_good = ibi[np.abs(ibi - med_ibi) < 3 * mad]
    else:
        ibi_good = ibi
    if len(ibi_good) < 3:
        return result

    cv = np.std(ibi_good) / np.mean(ibi_good)
    if cv > MAX_IBI_CV:
        return result

    hr = 60.0 / np.median(ibi_good)
    if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
        return result

    result['hr_bpm'] = round(hr, 1)
    result['hr_quality'] = round(ac_quality * (1.0 - cv), 3)
    return result


def compute_respiratory(data, fs):
    """Compute respiratory stats from a data window.
    Returns dict with resp_rate, resp_quality, resp_amplitude, resp_max_interval, etc."""
    result = {
        'resp_rate_bpm': None,
        'resp_quality': 0.0,
        'resp_amplitude': None,
        'resp_max_interval': None,
        'resp_band_power': None,
    }
    N = len(data)
    # Need at least 1.5 cycles of the slowest expected breath rate
    min_samples = int(1.5 * (60.0 / RESP_MIN_BPM) * fs)
    if N < min_samples:
        return result

    try:
        # Respiratory bandpass
        f_lo = RESP_MIN_BPM / 60.0
        f_hi = RESP_MAX_BPM / 60.0
        sos_resp = signal.butter(3, [max(f_lo, 0.02), min(f_hi, fs/2 - 0.1)],
                                 btype='bandpass', fs=fs, output='sos')
        resp = signal.sosfiltfilt(sos_resp, data)
    except Exception:
        return result

    # Respiratory band power (signal strength)
    resp_power = np.mean(resp ** 2)
    result['resp_band_power'] = round(resp_power, 6)

    # Amplitude: peak-to-peak
    p2p = np.max(resp) - np.min(resp)
    result['resp_amplitude'] = round(p2p, 4)

    # Autocorrelation for breath rate
    x = resp - np.mean(resp)
    if np.std(x) < 1e-9:
        return result
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] <= 0:
        return result
    corr = corr / corr[0]

    min_lag = int(60.0 / RESP_MAX_BPM * fs)
    max_lag = min(int(60.0 / RESP_MIN_BPM * fs), len(corr) - 1, N // 2)
    if min_lag >= max_lag:
        return result

    peaks, _ = signal.find_peaks(corr[min_lag:max_lag], distance=int(1.0 * fs))
    peaks = peaks + min_lag
    if len(peaks) == 0:
        return result

    best = peaks[np.argmax(corr[peaks])]
    ac_quality = corr[best]
    if ac_quality < MIN_AUTOCORR_RESP:
        return result

    resp_period = best / fs
    resp_rate = 60.0 / resp_period
    result['resp_rate_bpm'] = round(resp_rate, 1)
    result['resp_quality'] = round(ac_quality, 3)

    # Detect individual breath peaks for inter-breath interval analysis
    min_breath_dist = int(0.5 * resp_period * fs)
    resp_peaks, _ = signal.find_peaks(resp, distance=min_breath_dist,
                                       prominence=p2p * 0.15)
    if len(resp_peaks) > 1:
        breath_intervals = np.diff(resp_peaks) / fs
        result['resp_max_interval'] = round(np.max(breath_intervals), 2)

    return result


def compute_window_stats(data, fs, t_start):
    """Compute all stats for one time window."""
    cardiac = compute_cardiac(data, fs)
    respiratory = compute_respiratory(data, fs)

    # Overall signal characteristics
    p2p_raw = np.max(data) - np.min(data)
    std_raw = np.std(data)

    stats = {
        't_start_s': round(t_start, 1),
        't_start_min': round(t_start / 60.0, 2),
        'hr_bpm': cardiac['hr_bpm'],
        'hr_quality': cardiac['hr_quality'],
        'resp_rate_bpm': respiratory['resp_rate_bpm'],
        'resp_quality': respiratory['resp_quality'],
        'resp_amplitude_hPa': respiratory['resp_amplitude'],
        'resp_max_interval_s': respiratory['resp_max_interval'],
        'resp_band_power': respiratory['resp_band_power'],
        'raw_p2p_hPa': round(p2p_raw, 4),
        'raw_std_hPa': round(std_raw, 4),
    }
    return stats


def analyze_file(filepath, window_sec=30, step_sec=15):
    """Main analysis: sliding window over entire file."""
    print(f"Loading {filepath}...")
    t_s, pressure, fs = load_csv(filepath)
    print(f"  Samples: {len(pressure)}, Duration: {t_s[-1] - t_s[0]:.1f}s, Fs: {fs:.1f} Hz")

    # Resample to uniform grid
    t_s, pressure = resample_uniform(t_s, pressure, fs)
    N = len(pressure)
    duration = t_s[-1] - t_s[0]
    print(f"  Resampled: {N} samples, {duration:.1f}s")

    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)

    all_stats = []

    if N < window_samples:
        # File shorter than one window — analyze the whole thing
        print(f"  Note: file shorter than {window_sec}s window, analyzing as single block")
        stats = compute_window_stats(pressure, fs, 0)
        all_stats.append(stats)
    else:
        n_windows = max(1, (N - window_samples) // step_samples + 1)
        print(f"  Analyzing {n_windows} windows ({window_sec}s window, {step_sec}s step)...")

        for i in range(0, N - window_samples + 1, step_samples):
            chunk = pressure[i:i + window_samples]
            t_start = t_s[i]
            stats = compute_window_stats(chunk, fs, t_start)
            all_stats.append(stats)

            pct = len(all_stats) / n_windows * 100
            if len(all_stats) % 20 == 0 or len(all_stats) == n_windows:
                print(f"    {len(all_stats)}/{n_windows} ({pct:.0f}%)")

    df_stats = pd.DataFrame(all_stats)
    return df_stats, t_s, pressure, fs


def plot_summary(df, t_s, pressure, fs, filepath, window_sec, step_sec):
    """Generate summary plot."""
    basename = Path(filepath).stem

    fig = plt.figure(figsize=(16, 18), constrained_layout=True)
    gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[2, 1, 1, 1, 1, 1])

    t_min = df['t_start_min'].values

    # 1. Raw pressure overview
    ax0 = fig.add_subplot(gs[0])
    t_plot = t_s / 60.0
    ax0.plot(t_plot, pressure, linewidth=0.2, color='steelblue', alpha=0.5)
    # Add smoothed trend
    if len(pressure) > int(5 * fs):
        try:
            sos_trend = signal.butter(2, 0.02, btype='lowpass', fs=fs, output='sos')
            trend = signal.sosfiltfilt(sos_trend, pressure)
            ax0.plot(t_plot, trend, linewidth=1.5, color='red', alpha=0.7, label='Trend')
            ax0.legend(fontsize=9)
        except Exception:
            pass
    ax0.set_ylabel('hPa')
    ax0.set_title(f'{basename} — Raw Pressure ({len(pressure)/fs/60:.1f} min)')

    # 2. Heart rate
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    hr = df['hr_bpm'].values.astype(float)
    hr_q = df['hr_quality'].values
    valid_hr = ~np.isnan(hr)
    if np.any(valid_hr):
        ax1.plot(t_min[valid_hr], hr[valid_hr], 'o-', markersize=3,
                 color='steelblue', linewidth=1)
        # Color by quality
        sc = ax1.scatter(t_min[valid_hr], hr[valid_hr], c=hr_q[valid_hr],
                        cmap='RdYlGn', s=15, vmin=0, vmax=0.8, zorder=5)
        plt.colorbar(sc, ax=ax1, label='Quality', shrink=0.6)
    ax1.set_ylabel('BPM')
    ax1.set_title('Heart Rate')
    ax1.set_ylim(HR_MIN_BPM, HR_MAX_BPM)

    # 3. Respiratory rate
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    rr = df['resp_rate_bpm'].values.astype(float)
    rr_q = df['resp_quality'].values
    valid_rr = ~np.isnan(rr)
    if np.any(valid_rr):
        ax2.plot(t_min[valid_rr], rr[valid_rr], 'o-', markersize=3,
                 color='steelblue', linewidth=1)
        sc2 = ax2.scatter(t_min[valid_rr], rr[valid_rr], c=rr_q[valid_rr],
                         cmap='RdYlGn', s=15, vmin=0, vmax=0.8, zorder=5)
        plt.colorbar(sc2, ax=ax2, label='Quality', shrink=0.6)
    ax2.set_ylabel('Breaths/min')
    ax2.set_title('Respiratory Rate')
    ax2.set_ylim(0, 30)

    # 4. Respiratory amplitude (signal strength indicator)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    amp = df['resp_amplitude_hPa'].values.astype(float)
    valid_amp = ~np.isnan(amp)
    if np.any(valid_amp):
        ax3.fill_between(t_min[valid_amp], 0, amp[valid_amp],
                         color='steelblue', alpha=0.4)
        ax3.plot(t_min[valid_amp], amp[valid_amp], linewidth=1, color='steelblue')
    ax3.set_ylabel('hPa (p-p)')
    ax3.set_title('Respiratory Amplitude (higher may indicate obstructed effort)')

    # 5. Longest inter-breath interval
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    mbi = df['resp_max_interval_s'].values.astype(float)
    valid_mbi = ~np.isnan(mbi)
    if np.any(valid_mbi):
        # Color relative to expected interval: red if >2x median resp period
        median_rr = df['resp_rate_bpm'].dropna().median()
        if not np.isnan(median_rr) and median_rr > 0:
            expected_interval = 60.0 / median_rr
        else:
            expected_interval = 6.0  # fallback ~10 breaths/min
        colors = ['red' if v > 2.0 * expected_interval else
                  'orange' if v > 1.5 * expected_interval else 'steelblue'
                  for v in mbi[valid_mbi]]
        ax4.bar(t_min[valid_mbi], mbi[valid_mbi], width=step_sec / 60 * 0.8,
                color=colors, alpha=0.7)
    ax4.axhline(10, color='red', linestyle='--', alpha=0.5, label='10s (clinical apnea)')
    # Also show expected inter-breath interval
    median_rr_val = df['resp_rate_bpm'].dropna().median()
    if not np.isnan(median_rr_val) and median_rr_val > 0:
        ax4.axhline(60.0 / median_rr_val, color='green', linestyle=':', alpha=0.5,
                     label=f'Expected ({60.0/median_rr_val:.1f}s at {median_rr_val:.0f}/min)')
    ax4.set_ylabel('Seconds')
    ax4.set_title('Longest Inter-Breath Interval per Window')
    ax4.legend(fontsize=9)

    # 6. Raw signal variability
    ax5 = fig.add_subplot(gs[5], sharex=ax0)
    ax5.plot(t_min, df['raw_std_hPa'], linewidth=1, color='steelblue', label='Std dev')
    ax5.set_ylabel('hPa')
    ax5.set_title('Raw Signal Variability (movement/activity indicator)')
    ax5.set_xlabel('Time (minutes)')
    ax5.legend(fontsize=9)

    outpath = Path(filepath).with_name(basename + '_summary.png')
    fig.savefig(outpath, dpi=150)
    print(f"  Summary plot: {outpath}")
    return str(outpath)


def print_report(df, filepath):
    """Print text summary to console."""
    basename = Path(filepath).stem
    print(f"\n{'='*60}")
    print(f"  Report: {basename}")
    print(f"{'='*60}")

    duration_min = df['t_start_min'].iloc[-1] - df['t_start_min'].iloc[0]
    print(f"  Duration: {duration_min:.1f} minutes")
    print(f"  Windows analyzed: {len(df)}")

    hr = df['hr_bpm'].dropna()
    if len(hr) > 0:
        print(f"\n  CARDIAC:")
        print(f"    Valid windows: {len(hr)}/{len(df)} ({100*len(hr)/len(df):.0f}%)")
        print(f"    HR range: {hr.min():.0f} – {hr.max():.0f} bpm")
        print(f"    HR mean:  {hr.mean():.1f} bpm")
        print(f"    HR median: {hr.median():.1f} bpm")
    else:
        print(f"\n  CARDIAC: no reliable measurements")

    rr = df['resp_rate_bpm'].dropna()
    amp = df['resp_amplitude_hPa'].dropna()
    mbi = df['resp_max_interval_s'].dropna()

    print(f"\n  RESPIRATORY:")
    if len(rr) > 0:
        print(f"    Valid windows: {len(rr)}/{len(df)} ({100*len(rr)/len(df):.0f}%)")
        print(f"    Rate range: {rr.min():.1f} – {rr.max():.1f} breaths/min")
        print(f"    Rate mean:  {rr.mean():.1f} breaths/min")
        print(f"    Rate median: {rr.median():.1f} breaths/min")
    else:
        print(f"    No reliable rate measurements")

    if len(amp) > 0:
        print(f"    Amplitude range: {amp.min():.3f} – {amp.max():.3f} hPa")
        print(f"    Amplitude mean:  {amp.mean():.3f} hPa")

    if len(mbi) > 0:
        print(f"    Max inter-breath interval overall: {mbi.max():.1f}s")
        apnea_windows = (mbi > 10).sum()
        if apnea_windows > 0:
            print(f"    *** Windows with >10s gap: {apnea_windows} ***")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze pressure sensor data for cardiac and respiratory signals.')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('--window', type=int, default=45, help='Analysis window size in seconds (default: 45)')
    parser.add_argument('--step', type=int, default=15, help='Step between windows in seconds (default: 15)')
    args = parser.parse_args()

    filepath = args.input
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    df_stats, t_s, pressure, fs = analyze_file(filepath, args.window, args.step)

    # Save outputs next to input file, or current dir if input dir is read-only
    basename = Path(filepath).stem
    out_dir = Path(filepath).parent
    try:
        test_file = out_dir / '.write_test'
        test_file.touch()
        test_file.unlink()
    except OSError:
        out_dir = Path('.')

    stats_path = out_dir / (basename + '_stats.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"  Stats CSV: {stats_path}")

    # Generate plots
    summary_path = plot_summary(df_stats, t_s, pressure, fs, str(out_dir / (basename + '.csv')),
                                args.window, args.step)

    # Print report
    print_report(df_stats, filepath)


if __name__ == '__main__':
    main()
