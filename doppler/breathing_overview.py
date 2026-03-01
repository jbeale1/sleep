#!/usr/bin/env python3
"""
breathing_overview.py — Overnight breathing analysis from 60 GHz CW radar IQ data.

Produces a multi-panel dashboard:
  1. Breathing spectrogram (displacement power in respiratory band)
  2. Breathing rate trend (breaths/min vs time)
  3. Breathing amplitude with apnea event markers
  4. Compact micro-Doppler strip for context

Usage:
    python breathing_overview.py <input.csv> [options]

Input CSV format (auto-detected):
    4-column:  seq,ms,I,Q       (integer counts, ms is modulo 1000)
    3-column:  elapsed_ms,I,Q   (float volts)
    2-column:  I,Q              (assumes uniform sample rate)

Options:
    --fs RATE           Override sample rate in Hz (default: auto)
    --win SEC           Analysis window in seconds (default: 60)
    --step SEC          Window step in seconds (default: 10)
    --breath-lo HZ      Breathing band low cutoff (default: 0.10)
    --breath-hi HZ      Breathing band high cutoff (default: 0.50)
    --apnea-thresh FRAC Apnea threshold as fraction of local baseline (default: 0.25)
    --apnea-min SEC     Minimum apnea duration in seconds (default: 10)
    --baseline-win SEC  Window for local amplitude baseline (default: 300)
    --output FILE       Save plot to file instead of displaying
    --dpi DPI           Output resolution (default: 150)
"""

import argparse
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal as sig
from scipy.ndimage import uniform_filter1d

# ── constants ────────────────────────────────────────────────────────────────
F_RADAR = 61e9
LAM = 3e8 / F_RADAR
DEFAULT_FS = 142.0

# ── file I/O (shared with micro_doppler.py) ──────────────────────────────────

def parse_start_time(filepath):
    basename = filepath.replace('\\', '/').split('/')[-1]
    m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', basename)
    if m:
        return datetime(*[int(g) for g in m.groups()])
    return None


def load_iq_csv(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    has_header = not first_line[0].isdigit() and not first_line[0] == '-'

    data = np.genfromtxt(filepath, delimiter=',',
                         skip_header=1 if has_header else 0)

    if data.ndim == 1:
        sys.exit("Error: CSV must have at least 2 columns")

    ncols = data.shape[1]

    if ncols >= 4:
        seq = data[:, 0].astype(int)
        ms_raw = data[:, 1].astype(int)
        I = data[:, 2].astype(np.float64)
        Q = data[:, 3].astype(np.float64)

        seq_diffs = np.diff(seq)
        gaps = np.where(seq_diffs != 1)[0]
        if len(gaps):
            total_dropped = int(np.sum(seq_diffs[gaps] - 1))
            print(f"WARNING: {total_dropped} dropped samples at {len(gaps)} gap(s)")

        elapsed_ms = np.zeros(len(ms_raw), dtype=np.float64)
        epoch = 0
        for i in range(1, len(ms_raw)):
            if ms_raw[i] < ms_raw[i-1] - 500:
                epoch += 1000
            elapsed_ms[i] = epoch + ms_raw[i] - ms_raw[0]

        duration_s = elapsed_ms[-1] / 1000.0
        fs = (len(I) - 1) / duration_s if duration_s > 0 else None
        print(f"Loaded {len(I)} samples, {duration_s:.1f}s ({duration_s/3600:.2f}h), "
              f"rate: {fs:.1f} Hz")
    elif ncols >= 3 and has_header:
        t_ms = data[:, 0]
        I = data[:, 1]
        Q = data[:, 2]
        duration_s = (t_ms[-1] - t_ms[0]) / 1000.0
        fs = (len(t_ms) - 1) / duration_s if duration_s > 0 else None
        print(f"Loaded {len(I)} samples, {duration_s:.1f}s, rate: {fs:.1f} Hz")
    elif ncols >= 2:
        I = data[:, 0]
        Q = data[:, 1]
        fs = None
        print(f"Loaded {len(I)} samples (no timestamps)")
    else:
        sys.exit("Error: CSV must have at least 2 columns")

    return I, Q, fs


# ── signal processing ────────────────────────────────────────────────────────

def iq_to_displacement(I_raw, Q_raw, fs, dc_window_s=10.0):
    """Convert raw IQ to displacement in mm via arctangent demodulation."""
    win = max(int(dc_window_s * fs), 3)
    I = I_raw - uniform_filter1d(I_raw, win)
    Q = Q_raw - uniform_filter1d(Q_raw, win)
    # IQ gain balance
    i_std, q_std = np.std(I), np.std(Q)
    if q_std > 0:
        Q = Q * (i_std / q_std)
    phase = np.unwrap(np.arctan2(Q, I))
    disp_mm = phase * LAM / (4 * np.pi) * 1000
    # High-pass to remove slow drift
    sos_hp = sig.butter(2, 0.03, btype='highpass', fs=fs, output='sos')
    disp_mm = sig.sosfiltfilt(sos_hp, disp_mm)
    return disp_mm


def extract_breathing(disp_mm, fs, win_s, step_s, breath_lo, breath_hi):
    """Extract breathing rate and amplitude in sliding windows.

    Returns:
        t_centres:  window centre times (s)
        br_rates:   breathing rate (breaths/min), NaN where uncertain
        br_spec_f:  frequency axis for breathing spectrogram
        br_spec:    breathing spectrogram (power, shape: n_freq × n_windows)
    """
    sos_br = sig.butter(4, [breath_lo, breath_hi], btype='bandpass',
                        fs=fs, output='sos')
    disp_br = sig.sosfiltfilt(sos_br, disp_mm)

    win_n = int(win_s * fs)
    step_n = int(step_s * fs)
    n = len(disp_mm)

    t_centres = []
    br_rates = []
    spec_list = []

    # Frequency axis for per-window PSD
    nperseg_psd = min(win_n, int(30 * fs))  # 30s segments within the window

    for start in range(0, n - win_n, step_n):
        end = start + win_n
        tc = (start + win_n / 2) / fs
        t_centres.append(tc)

        seg = disp_br[start:end]
        seg_raw = disp_mm[start:end]

        # Breathing spectrogram: PSD of raw displacement in respiratory band
        f_w, psd_w = sig.welch(seg_raw, fs=fs, nperseg=nperseg_psd,
                               noverlap=nperseg_psd // 2)
        spec_list.append(psd_w)

        # Rate estimation: autocorrelation of bandpassed signal
        seg_z = seg - np.mean(seg)
        ac = np.correlate(seg_z, seg_z, mode='full')
        ac = ac[len(ac) // 2:]
        if ac[0] > 0:
            ac = ac / ac[0]

        # Search for first peak in plausible breathing range
        min_lag = int(fs / breath_hi)   # fastest breath → shortest lag
        max_lag = int(fs / breath_lo)   # slowest breath → longest lag
        max_lag = min(max_lag, len(ac) - 1)

        rate = np.nan
        if max_lag > min_lag:
            pks, props = sig.find_peaks(ac[min_lag:max_lag],
                                        prominence=0.1, distance=int(0.5 * fs))
            if len(pks) > 0:
                best = pks[np.argmax(props['prominences'])]
                lag = min_lag + best
                rate = 60.0 * fs / lag
        br_rates.append(rate)

    t_centres = np.array(t_centres)
    br_rates = np.array(br_rates)

    # Build spectrogram matrix
    br_spec = np.array(spec_list).T  # shape: n_freq × n_windows

    return t_centres, br_rates, f_w, br_spec


def compute_breath_envelope(disp_mm, fs, breath_lo, breath_hi, smooth_s=5.0):
    """Compute high-resolution breathing amplitude envelope via Hilbert transform.

    Args:
        smooth_s: smoothing window in seconds (~1 breath, preserves minute-scale modulation)

    Returns:
        t_env: time axis (s) at full sample rate
        env_mm: smoothed amplitude envelope (mm, half pk-pk)
    """
    sos_br = sig.butter(4, [breath_lo, breath_hi], btype='bandpass',
                        fs=fs, output='sos')
    disp_br = sig.sosfiltfilt(sos_br, disp_mm)

    # Hilbert envelope
    analytic = sig.hilbert(disp_br)
    inst_amp = np.abs(analytic)

    # Smooth: moving average over ~1 breath period
    smooth_n = int(smooth_s * fs)
    if smooth_n % 2 == 0:
        smooth_n += 1
    env_mm = uniform_filter1d(inst_amp, smooth_n)

    t_env = np.arange(len(env_mm)) / fs
    return t_env, env_mm


def detect_periodic_breathing(t_env, env_mm, fs,
                               pb_lo_period=15, pb_hi_period=90,
                               pb_win_s=180, pb_step_s=10,
                               pb_thresh=0.3, pb_min_dur_s=120,
                               amp_clamp_mm=15.0):
    """Detect periodic breathing episodes via autocorrelation of the
    breathing amplitude envelope.

    Args:
        pb_lo_period, pb_hi_period: cycle period search range (seconds)
        pb_win_s: autocorrelation window (should span ~3+ cycles)
        pb_step_s: step between windows
        pb_thresh: minimum autocorrelation peak to consider periodic
        pb_min_dur_s: minimum contiguous duration to count as PB episode
        amp_clamp_mm: clamp envelope before analysis (reject body movements)

    Returns:
        t_pb: window centre times (s)
        periodicity: autocorrelation peak height per window (0–1)
        cycle_period: detected cycle period per window (s), NaN if none
        pb_episodes: list of (start_s, end_s, duration_s, mean_period_s)
    """
    # Downsample envelope to ~2 Hz
    ds_factor = max(1, int(fs / 2))
    env_ds = env_mm[::ds_factor]
    fs_ds = fs / ds_factor
    t_ds = np.arange(len(env_ds)) / fs_ds

    # Clamp to reject gross body movements
    env_ds = np.minimum(env_ds, amp_clamp_mm)

    # Remove slow trend
    trend_win = max(int(pb_hi_period * 3 * fs_ds), 3)
    if trend_win % 2 == 0:
        trend_win += 1
    env_detrend = env_ds - uniform_filter1d(env_ds, trend_win)

    win_n = int(pb_win_s * fs_ds)
    step_n = int(pb_step_s * fs_ds)

    min_lag = int(pb_lo_period * fs_ds)
    max_lag = int(pb_hi_period * fs_ds)

    t_pb = []
    periodicity = []
    cycle_period = []

    for start in range(0, len(env_detrend) - win_n, step_n):
        seg = env_detrend[start:start + win_n]
        tc = t_ds[start + win_n // 2]
        t_pb.append(tc)

        # Autocorrelation (normalized)
        seg_z = seg - np.mean(seg)
        ac = np.correlate(seg_z, seg_z, mode='full')
        ac = ac[len(ac) // 2:]
        if ac[0] > 0:
            ac = ac / ac[0]
        else:
            periodicity.append(0.0)
            cycle_period.append(np.nan)
            continue

        # Search for peaks in the PB period range
        search = ac[min_lag:min(max_lag + 1, len(ac))]
        if len(search) < 3:
            periodicity.append(0.0)
            cycle_period.append(np.nan)
            continue

        pks, props = sig.find_peaks(search, prominence=0.05)
        if len(pks) > 0:
            best = pks[np.argmax(props['prominences'])]
            pk_val = search[best]
            lag = min_lag + best
            periodicity.append(max(0, pk_val))
            cycle_period.append(lag / fs_ds)
        else:
            periodicity.append(0.0)
            cycle_period.append(np.nan)

    t_pb = np.array(t_pb)
    periodicity = np.array(periodicity)
    cycle_period = np.array(cycle_period)

    # Find contiguous PB episodes
    is_pb = periodicity >= pb_thresh
    raw_episodes = []
    in_ep = False
    ep_start = 0
    for i in range(len(is_pb)):
        if is_pb[i] and not in_ep:
            in_ep = True
            ep_start = i
        elif not is_pb[i] and in_ep:
            in_ep = False
            dur = t_pb[i - 1] - t_pb[ep_start]
            mean_per = np.nanmedian(cycle_period[ep_start:i])
            raw_episodes.append((t_pb[ep_start], t_pb[i - 1], dur, mean_per))
    if in_ep:
        dur = t_pb[-1] - t_pb[ep_start]
        mean_per = np.nanmedian(cycle_period[ep_start:])
        raw_episodes.append((t_pb[ep_start], t_pb[-1], dur, mean_per))

    # Bridge gaps shorter than 2.5× cycle period
    pb_episodes = []
    for ep in raw_episodes:
        if pb_episodes:
            prev_start, prev_end, prev_dur, prev_per = pb_episodes[-1]
            gap = ep[0] - prev_end
            bridge_limit = 2.5 * max(np.nanmedian([prev_per, ep[3]]),
                                    pb_lo_period) if not (np.isnan(prev_per) and np.isnan(ep[3])) \
                           else pb_hi_period
            if gap <= bridge_limit:
                # Merge with previous
                merged_dur = ep[1] - prev_start
                # Recompute median period over merged span
                mask = (t_pb >= prev_start) & (t_pb <= ep[1])
                merged_per = np.nanmedian(cycle_period[mask])
                pb_episodes[-1] = (prev_start, ep[1], merged_dur, merged_per)
                continue
        pb_episodes.append(ep)

    # Filter by minimum duration
    pb_episodes = [(s, e, d, p) for s, e, d, p in pb_episodes if d >= pb_min_dur_s]

    return t_pb, periodicity, cycle_period, pb_episodes


def detect_apneas(t_env, env_mm, apnea_thresh, apnea_min_s,
                  baseline_win_s, fs_env):
    """Detect apnea events where amplitude drops below threshold × local baseline.

    Works with the high-resolution Hilbert envelope.
    Returns list of (start_time, end_time, duration) tuples in seconds,
    and the baseline array.
    """
    # Downsample for efficiency (1 Hz is plenty for apnea detection)
    ds = max(1, int(fs_env))
    env_ds = env_mm[::ds]
    t_ds = t_env[::ds]

    # Local baseline: rolling median
    half_win = max(int(baseline_win_s / 2), 1)
    padded = np.pad(env_ds, half_win, mode='edge')
    baseline = np.array([np.median(padded[i:i + 2 * half_win + 1])
                         for i in range(len(env_ds))])

    # Flag points below threshold
    below = env_ds < (apnea_thresh * baseline)

    # Find contiguous regions
    events = []
    in_event = False
    ev_start = 0
    for i in range(len(below)):
        if below[i] and not in_event:
            in_event = True
            ev_start = i
        elif not below[i] and in_event:
            in_event = False
            duration = t_ds[i - 1] - t_ds[ev_start]
            if duration >= apnea_min_s:
                events.append((t_ds[ev_start], t_ds[i - 1], duration))
    if in_event:
        duration = t_ds[-1] - t_ds[ev_start]
        if duration >= apnea_min_s:
            events.append((t_ds[ev_start], t_ds[-1], duration))

    # Interpolate baseline back to full resolution
    baseline_full = np.interp(t_env, t_ds, baseline)

    return events, baseline_full


# ── plotting ─────────────────────────────────────────────────────────────────

def make_time_axis(t_seconds, start_time):
    """Convert array of seconds to datetime array, or return seconds if no start_time."""
    if start_time is not None:
        return np.array([start_time + timedelta(seconds=float(s)) for s in t_seconds])
    return t_seconds


def plot_overview(t_centres, br_rates, f_spec, br_spec,
                  t_env, env_mm, baseline,
                  apnea_events,
                  t_pb, periodicity, cycle_period, pb_episodes,
                  I_raw, Q_raw, fs,
                  start_time=None, breath_lo=0.10, breath_hi=0.50,
                  apnea_thresh=0.25, amp_clamp_pkpk=30.0, pb_thresh=0.3):
    """Create 5-panel breathing overview figure."""

    use_dt = start_time is not None
    t_ax = make_time_axis(t_centres, start_time)
    total_s = t_centres[-1] - t_centres[0]
    total_h = total_s / 3600
    med_rate = np.nanmedian(br_rates)
    n_apneas = len(apnea_events)
    total_apnea_s = sum(d for _, _, d in apnea_events)
    ahi_est = n_apneas / total_h if total_h > 0 else 0
    n_pb = len(pb_episodes)
    total_pb_min = sum(d for _, _, d, _ in pb_episodes) / 60

    fig, axes = plt.subplots(5, 1, figsize=(16, 18), layout='constrained',
                             sharex=True)

    # ── Panel 1: Breathing spectrogram ────────────────────────────────────────
    ax = axes[0]
    f_mask = (f_spec >= breath_lo * 0.5) & (f_spec <= breath_hi * 1.5)
    f_plot = f_spec[f_mask] * 60  # Hz → breaths/min
    spec_plot = 10 * np.log10(br_spec[f_mask, :] + 1e-20)
    vmin, vmax = np.percentile(spec_plot, [5, 97])

    if use_dt:
        im = ax.pcolormesh(t_ax, f_plot, spec_plot,
                           shading='gouraud', cmap='inferno',
                           vmin=vmin, vmax=vmax)
    else:
        t_hrs = t_centres / 3600
        im = ax.pcolormesh(t_hrs, f_plot, spec_plot,
                           shading='gouraud', cmap='inferno',
                           vmin=vmin, vmax=vmax)
    ax.set_ylabel('Rate (breaths/min)')
    ax.set_ylim(breath_lo * 60 * 0.5, breath_hi * 60 * 1.3)
    ax.set_title(f'Breathing Spectrogram  ({total_h:.1f} h)')
    plt.colorbar(im, ax=ax, label='dB', shrink=0.8, pad=0.01)

    # ── Panel 2: Breathing rate trend ─────────────────────────────────────────
    ax = axes[1]
    valid = ~np.isnan(br_rates)
    if use_dt:
        ax.scatter(t_ax[valid], br_rates[valid], s=2, c='steelblue', alpha=0.6)
    else:
        ax.scatter((t_centres/3600)[valid], br_rates[valid], s=2,
                   c='steelblue', alpha=0.6)
    if np.sum(valid) > 10:
        rates_filled = br_rates.copy()
        rates_filled[~valid] = np.interp(
            t_centres[~valid], t_centres[valid], br_rates[valid])
        smooth_n = max(3, int(120 / (t_centres[1] - t_centres[0])))
        if smooth_n % 2 == 0:
            smooth_n += 1
        rates_smooth = uniform_filter1d(rates_filled, smooth_n)
        if use_dt:
            ax.plot(t_ax, rates_smooth, 'r-', lw=1.5, alpha=0.8, label='2-min trend')
        else:
            ax.plot(t_hrs, rates_smooth, 'r-', lw=1.5, alpha=0.8, label='2-min trend')
        ax.legend(fontsize=8, loc='upper right')
    ax.set_ylabel('Breaths/min')
    ax.set_ylim(max(0, med_rate - 12), med_rate + 12)
    ax.set_title(f'Breathing Rate  (median {med_rate:.1f}/min)')
    ax.grid(True, alpha=0.3)

    # ── Panel 3: High-res breathing amplitude + apnea markers ─────────────────
    ax = axes[2]
    ds_plot = max(1, int(fs / 2))
    t_env_ds = t_env[::ds_plot]
    env_ds = env_mm[::ds_plot]
    baseline_ds = baseline[::ds_plot]
    amp_pkpk = np.minimum(env_ds * 2, amp_clamp_pkpk)  # clamp
    baseline_pkpk = baseline_ds * 2
    thresh_line = baseline_pkpk * apnea_thresh

    if use_dt:
        t_env_ax = make_time_axis(t_env_ds, start_time)
        ax.plot(t_env_ax, amp_pkpk, color='steelblue', lw=0.4, alpha=0.8,
                label='Breathing amplitude')
        ax.plot(t_env_ax, baseline_pkpk, 'k--', lw=0.8, alpha=0.5,
                label='Local baseline')
        ax.plot(t_env_ax, thresh_line, 'r:', lw=0.8, alpha=0.5,
                label=f'Apnea threshold ({apnea_thresh:.0%})')
    else:
        t_env_hrs = t_env_ds / 3600
        ax.plot(t_env_hrs, amp_pkpk, color='steelblue', lw=0.4, alpha=0.8,
                label='Breathing amplitude')
        ax.plot(t_env_hrs, baseline_pkpk, 'k--', lw=0.8, alpha=0.5,
                label='Local baseline')
        ax.plot(t_env_hrs, thresh_line, 'r:', lw=0.8, alpha=0.5,
                label=f'Apnea threshold ({apnea_thresh:.0%})')

    for ev_start, ev_end, dur in apnea_events:
        if use_dt:
            xs = start_time + timedelta(seconds=float(ev_start))
            xe = start_time + timedelta(seconds=float(ev_end))
        else:
            xs = ev_start / 3600
            xe = ev_end / 3600
        ax.axvspan(xs, xe, alpha=0.2, color='red')

    # Also shade PB episodes on amplitude panel
    for ev_start, ev_end, dur, per in pb_episodes:
        if use_dt:
            xs = start_time + timedelta(seconds=float(ev_start))
            xe = start_time + timedelta(seconds=float(ev_end))
        else:
            xs = ev_start / 3600
            xe = ev_end / 3600
        ax.axvspan(xs, xe, alpha=0.12, color='orange')

    ax.set_ylabel('Amplitude (mm pk-pk)')
    ax.set_title(f'Breathing Amplitude (Hilbert envelope, clamped {amp_clamp_pkpk:.0f} mm)  |  '
                 f'{n_apneas} apnea events  |  '
                 f'{n_pb} PB episodes (orange)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, amp_clamp_pkpk * 1.05)

    # ── Panel 4: Periodicity index + PB episodes ─────────────────────────────
    ax = axes[3]
    if use_dt:
        t_pb_ax = make_time_axis(t_pb, start_time)
        ax.plot(t_pb_ax, periodicity, color='steelblue', lw=0.8)
    else:
        t_pb_hrs = t_pb / 3600
        ax.plot(t_pb_hrs, periodicity, color='steelblue', lw=0.8)

    ax.axhline(pb_thresh, color='red', ls='--', lw=1, alpha=0.6,
               label=f'PB threshold ({pb_thresh})')

    # Shade PB episodes and annotate cycle period
    for ev_start, ev_end, dur, per in pb_episodes:
        if use_dt:
            xs = start_time + timedelta(seconds=float(ev_start))
            xe = start_time + timedelta(seconds=float(ev_end))
            xm = start_time + timedelta(seconds=float((ev_start + ev_end) / 2))
        else:
            xs = ev_start / 3600
            xe = ev_end / 3600
            xm = (ev_start + ev_end) / 2 / 3600
        ax.axvspan(xs, xe, alpha=0.15, color='orange')
        if not np.isnan(per):
            dur_min = dur / 60
            ax.text(xm, pb_thresh + 0.05,
                    f'{dur_min:.0f}m\nT={per:.0f}s',
                    ha='center', va='bottom', fontsize=7, color='darkorange',
                    fontweight='bold', linespacing=1.2)

    ax.set_ylabel('Periodicity index')
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f'Periodic Breathing Detection  |  '
                 f'{n_pb} episodes ≥2 min  ({total_pb_min:.0f} min total)'
                 + (f'  |  ~{total_pb_min/total_h*60:.0f} min/h' if total_h > 0 else ''))
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Compact micro-Doppler ────────────────────────────────────────
    ax = axes[4]
    dc_win = max(int(10.0 * fs), 3)
    I_dc = I_raw - uniform_filter1d(I_raw, dc_win)
    Q_dc = Q_raw - uniform_filter1d(Q_raw, dc_win)
    i_std, q_std = np.std(I_dc), np.std(Q_dc)
    if q_std > 0:
        Q_dc = Q_dc * (i_std / q_std)
    S = I_dc + 1j * Q_dc

    nfft_md = 256
    novl_md = nfft_md - nfft_md // 4
    f_md, t_md, Sxx_md = sig.spectrogram(S, fs=fs, nperseg=nfft_md,
                                          noverlap=novl_md,
                                          return_onesided=False, window='hann')
    f_md = np.fft.fftshift(f_md)
    Sxx_md = np.fft.fftshift(Sxx_md, axes=0)
    Sxx_db = 10 * np.log10(Sxx_md + 1e-20)
    fmax = 5.0
    fm = np.abs(f_md) <= fmax
    vmin_md, vmax_md = np.percentile(Sxx_db[fm, :], [5, 97])

    if use_dt:
        t_md_dt = np.array([start_time + timedelta(seconds=float(s)) for s in t_md])
        ax.pcolormesh(t_md_dt, f_md[fm], Sxx_db[fm, :],
                      shading='gouraud', cmap='magma',
                      vmin=vmin_md, vmax=vmax_md)
    else:
        ax.pcolormesh(t_md / 3600, f_md[fm], Sxx_db[fm, :],
                      shading='gouraud', cmap='magma',
                      vmin=vmin_md, vmax=vmax_md)

    ax.set_ylabel('Doppler (Hz)')
    ax.set_title('Micro-Doppler Overview (±5 Hz)')

    if use_dt:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.set_xlabel('Time')
        fig.autofmt_xdate(rotation=0, ha='center')
    else:
        ax.set_xlabel('Time (hours)')

    # Supertitle
    if start_time:
        date_str = start_time.strftime('%Y-%m-%d %H:%M')
    else:
        date_str = f'{total_s:.0f}s'
    fig.suptitle(f'Breathing Overview — {date_str} — {total_h:.1f} hours\n'
                 f'Median rate: {med_rate:.1f}/min  |  '
                 f'{n_apneas} amplitude drops ≥10s  |  '
                 f'{n_pb} PB episodes ({total_pb_min:.0f} min)',
                 fontsize=13, fontweight='bold')

    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Overnight breathing analysis from CW radar IQ data')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('--fs', type=float, default=None,
                        help='Sample rate override (Hz)')
    parser.add_argument('--win', type=float, default=60,
                        help='Analysis window (seconds)')
    parser.add_argument('--step', type=float, default=10,
                        help='Window step (seconds)')
    parser.add_argument('--breath-lo', type=float, default=0.10,
                        help='Breathing band low cutoff (Hz)')
    parser.add_argument('--breath-hi', type=float, default=0.50,
                        help='Breathing band high cutoff (Hz)')
    parser.add_argument('--apnea-thresh', type=float, default=0.25,
                        help='Apnea threshold fraction of local baseline')
    parser.add_argument('--apnea-min', type=float, default=10,
                        help='Minimum apnea duration (seconds)')
    parser.add_argument('--baseline-win', type=float, default=300,
                        help='Local amplitude baseline window (seconds)')
    parser.add_argument('--amp-clamp', type=float, default=30.0,
                        help='Clamp breathing amplitude display (mm pk-pk)')
    parser.add_argument('--pb-thresh', type=float, default=0.3,
                        help='Periodicity index threshold for PB detection')
    parser.add_argument('--pb-min-dur', type=float, default=120,
                        help='Minimum PB episode duration (seconds)')
    parser.add_argument('--pb-lo-period', type=float, default=15,
                        help='Shortest PB cycle period to detect (seconds)')
    parser.add_argument('--pb-hi-period', type=float, default=90,
                        help='Longest PB cycle period to detect (seconds)')
    parser.add_argument('--env-smooth', type=float, default=10.0,
                        help='Envelope smoothing window (seconds)')
    parser.add_argument('--output', default=None,
                        help='Save to file instead of displaying')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    args = parser.parse_args()

    I, Q, fs_measured = load_iq_csv(args.input)

    fs = args.fs or fs_measured or DEFAULT_FS
    if args.fs:
        print(f"Using override sample rate: {fs:.1f} Hz")
    elif fs_measured:
        print(f"Using measured sample rate: {fs:.1f} Hz")
    else:
        print(f"No timestamps, using default: {fs:.1f} Hz")

    start_time = parse_start_time(args.input)
    if start_time:
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print("Converting IQ to displacement...")
    disp_mm = iq_to_displacement(I, Q, fs)

    print(f"Extracting breathing ({args.win:.0f}s windows, {args.step:.0f}s step)...")
    t_centres, br_rates, f_spec, br_spec = extract_breathing(
        disp_mm, fs, args.win, args.step, args.breath_lo, args.breath_hi)

    print("Computing breathing envelope (Hilbert, {:.0f}s smooth)...".format(args.env_smooth))
    t_env, env_mm = compute_breath_envelope(
        disp_mm, fs, args.breath_lo, args.breath_hi, smooth_s=args.env_smooth)

    print("Detecting amplitude drops...")
    apnea_events, baseline = detect_apneas(
        t_env, env_mm, args.apnea_thresh, args.apnea_min,
        args.baseline_win, fs)

    print("Detecting periodic breathing...")
    t_pb, periodicity, cycle_period, pb_episodes = detect_periodic_breathing(
        t_env, env_mm, fs, pb_thresh=args.pb_thresh,
        pb_min_dur_s=args.pb_min_dur,
        pb_lo_period=args.pb_lo_period,
        pb_hi_period=args.pb_hi_period,
        amp_clamp_mm=args.amp_clamp / 2)  # pk-pk → half-amplitude

    valid_rate = br_rates[~np.isnan(br_rates)]
    print(f"  Breathing rate: {np.median(valid_rate):.1f}/min median "
          f"({np.percentile(valid_rate, 5):.1f}–{np.percentile(valid_rate, 95):.1f} "
          f"5th–95th pctl)")
    print(f"  Amplitude drops ≥{args.apnea_min:.0f}s: {len(apnea_events)}")
    for i, (es, ee, dur) in enumerate(apnea_events[:20]):
        if start_time:
            ts = (start_time + timedelta(seconds=float(es))).strftime('%H:%M:%S')
        else:
            ts = f'{es:.0f}s'
        print(f"    #{i+1}: {ts}  ({dur:.0f}s)")
    if len(apnea_events) > 20:
        print(f"    ... and {len(apnea_events) - 20} more")

    # Report periodic breathing
    print(f"  Periodic breathing episodes ≥{args.pb_min_dur:.0f}s: {len(pb_episodes)}")
    for i, (es, ee, dur, per) in enumerate(pb_episodes):
        if start_time:
            ts = (start_time + timedelta(seconds=float(es))).strftime('%H:%M:%S')
            te = (start_time + timedelta(seconds=float(ee))).strftime('%H:%M:%S')
        else:
            ts = f'{es:.0f}s'
            te = f'{ee:.0f}s'
        print(f"    #{i+1}: {ts}–{te}  ({dur/60:.1f} min, cycle {per:.0f}s)")

    print("Plotting...")
    fig = plot_overview(t_centres, br_rates, f_spec, br_spec,
                        t_env, env_mm, baseline,
                        apnea_events,
                        t_pb, periodicity, cycle_period, pb_episodes,
                        I, Q, fs,
                        start_time=start_time,
                        breath_lo=args.breath_lo, breath_hi=args.breath_hi,
                        apnea_thresh=args.apnea_thresh,
                        amp_clamp_pkpk=args.amp_clamp,
                        pb_thresh=args.pb_thresh)

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
