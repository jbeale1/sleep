#!/usr/bin/env python3

"""
ECG Beat Oscilloscope — R-peak triggered waveform viewer

Displays successive P-QRS-T complexes aligned on R-peak, like an
oscilloscope with stable triggering. Navigate beat-by-beat with
keyboard controls.

Overlays:
  - Current beat (bold)
  - Previous beat (ghost trace)
  - Rolling average of last N beats (dashed)

Controls:
  Right / D      → next beat
  Left  / A      → previous beat
  Shift+Right    → skip 10 beats forward
  Shift+Left     → skip 10 beats back
  Home           → first beat
  End            → last beat
  Up / Down      → adjust rolling average window (N)
  F              → toggle filter on/off (for raw files)
  G              → toggle ghost trace (previous beat)
  T              → toggle persistence (storage scope, last 20 beats)
  R              → toggle rolling average
  P              → toggle P-wave region highlight
  1 / 2 / 3     → show 1, 2, or 3 beats at once
  N              → toggle timeline: HR ↔ Noise
  S              → save current view as PNG
  Q / Esc        → quit

  Timeline strip: click to jump, click+drag to scrub through recording
                 scroll wheel to zoom, right-drag to pan, double-click or Home to reset

Usage:
  python ecg_scope.py <csv_file> [sample_rate] [--prefiltered]
  python ecg_scope.py ECG_20260212.csv 250
  python ecg_scope.py ECG_filtered.csv 250 --prefiltered

J. Beale  2026-02
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.rcParams['keymap.fullscreen'] = []  # free 'f' from fullscreen toggle
matplotlib.rcParams['keymap.save'] = []        # free 's' from save dialog
matplotlib.rcParams['keymap.pan'] = []         # free 'p' from pan mode
matplotlib.rcParams['keymap.grid'] = []        # free 'g' from grid toggle
matplotlib.rcParams['keymap.home'] = []        # free 'home' from reset view
matplotlib.rcParams['keymap.quit'] = []        # free 'q' from quit (we handle it)
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
from pathlib import Path
from datetime import datetime
import sys
import glob
import re

# =============================================================
# CONFIGURATION
# =============================================================
DEFAULT_FS = 250

# Display window around R-peak (ms)
PRE_R_MS  = 300    # show 300ms before R-peak
POST_R_MS = 500    # show 500ms after R-peak

# Filtering (for raw data)
HP_FREQ    = 0.5
NOTCH_FREQ = 60
NOTCH_Q    = 30
LP_FREQ    = 40

# Rolling average default window
AVG_WINDOW_DEFAULT = 10

# P-wave highlight region (ms before R)
P_REGION_MS = (200, 40)   # 200–40ms before R

# =============================================================
# PARSE ARGS
# =============================================================
import argparse

parser = argparse.ArgumentParser(description='ECG Beat Oscilloscope')
parser.add_argument('csv_file', help='CSV file with ECG data')
parser.add_argument('sample_rate', nargs='?', type=int, default=DEFAULT_FS,
                    help=f'Sample rate in sps (default: {DEFAULT_FS})')
parser.add_argument('--prefiltered', action='store_true',
                    help='Data is already filtered (skip filtering)')
args = parser.parse_args()

CSV_FILE = args.csv_file
FS = args.sample_rate

# =============================================================
# LOAD DATA
# =============================================================
data_raw = np.loadtxt(CSV_FILE, delimiter=",", skiprows=1).flatten()
N = len(data_raw)
print(f"Loaded {N} samples ({N/FS:.1f}s) at {FS} sps from {CSV_FILE}")

# =============================================================
# FILTER DESIGN
# =============================================================
sos_hp = signal.butter(2, HP_FREQ, 'highpass', fs=FS, output='sos')
b_n, a_n = signal.iirnotch(NOTCH_FREQ, Q=NOTCH_Q, fs=FS)
sos_lp = signal.butter(4, LP_FREQ, 'lowpass', fs=FS, output='sos')

def apply_filters(data):
    d = signal.sosfiltfilt(sos_hp, data)
    d = signal.filtfilt(b_n, a_n, d)
    d = signal.sosfiltfilt(sos_lp, d)
    return d

is_raw = not args.prefiltered
if is_raw:
    print(f"Data assumed raw, filtering enabled (use --prefiltered to skip)")
else:
    print(f"Data marked as pre-filtered, skipping filters")

ecg_filtered = apply_filters(data_raw)
ecg_display = ecg_filtered.copy()  # start with filtered view

# =============================================================
# R-PEAK DETECTION
# =============================================================
sos_det = signal.butter(2, [5, min(20, FS/2 - 1)], 'bandpass', fs=FS, output='sos')
ecg_det = signal.sosfiltfilt(sos_det, ecg_filtered)
ecg_det_sq = ecg_det ** 2
ma_len = max(1, int(0.12 * FS))
ecg_ma = uniform_filter1d(ecg_det_sq, ma_len)

refract = int(0.40 * FS)
threshold = 0.3 * np.max(ecg_ma[:FS * 2])
min_threshold = 0.05 * np.max(ecg_ma[:FS * 2])  # floor to prevent P-wave triggers
peaks = []
i = int(0.5 * FS)

while i < N - int(0.5 * FS):
    if ecg_ma[i] > threshold:
        # Search wider window for true R-peak in filtered ECG
        search_start = max(0, i - int(0.15 * FS))
        search_end = min(N, i + int(0.15 * FS))
        r_idx = search_start + np.argmax(ecg_filtered[search_start:search_end])
        peaks.append(r_idx)
        threshold = 0.3 * ecg_ma[i] + 0.7 * threshold
        threshold = max(threshold, min_threshold)
        i = r_idx + refract
    else:
        threshold *= 0.9995
        threshold = max(threshold, min_threshold)
        i += 1

peaks = np.array(peaks)
n_beats = len(peaks)
print(f"Detected {n_beats} R-peaks")

if n_beats < 3:
    print("Too few beats detected.")
    sys.exit(1)

# =============================================================
# WALL-CLOCK TIME
# =============================================================
def parse_timestamp_from_filename(filepath):
    """Extract datetime from YYYYMMDD_HHMMSS pattern in filename."""
    name = Path(filepath).stem
    parts = name.split('_')
    for i in range(len(parts) - 1):
        if len(parts[i]) == 8 and len(parts[i+1]) >= 6:
            try:
                dt = datetime.strptime(f"{parts[i]}_{parts[i+1][:6]}",
                                       "%Y%m%d_%H%M%S")
                return dt
            except ValueError:
                continue
    return None

def load_sync_file(csv_path):
    """Try to load *_sync.csv alongside the data file."""
    stem = Path(csv_path).stem
    parent = Path(csv_path).parent
    # Try exact match first
    sync_path = parent / f"{stem}_sync.csv"
    if sync_path.exists():
        idxs, epochs = [], []
        with open(sync_path) as f:
            for line in f:
                if line.startswith('#') or line.startswith('sample'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    idxs.append(int(parts[0]))
                    epochs.append(float(parts[1]))
        if len(idxs) >= 2:
            return np.array(idxs), np.array(epochs)
    return None, None

# Build epoch array for each beat
sync_idx, sync_epoch = load_sync_file(CSV_FILE)
time_source = 'unknown'
if sync_idx is not None:
    beat_epoch = np.interp(peaks.astype(float), sync_idx, sync_epoch)
    time_source = 'sync'
    print(f"Using sync file for wall-clock time")
else:
    dt = parse_timestamp_from_filename(CSV_FILE)
    if dt:
        t0_epoch = dt.timestamp()
        beat_epoch = t0_epoch + peaks / FS
        time_source = 'filename'
        print(f"Using filename timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        beat_epoch = peaks / FS
        time_source = 'elapsed'
        print(f"No timestamp found, using elapsed seconds")

# Convert to matplotlib date numbers
ref_epoch = beat_epoch[0]
ref_mpl = mdates.date2num(datetime.fromtimestamp(ref_epoch)) if time_source != 'elapsed' else 0
def epoch_to_mpl(ep):
    if time_source == 'elapsed':
        return ep / 60.0  # minutes
    return ref_mpl + (ep - ref_epoch) / 86400.0

beat_mpl = np.array([epoch_to_mpl(e) for e in beat_epoch])

# =============================================================
# PRE-COMPUTE HR TIMELINE
# =============================================================
# R-R based HR for each beat
hr_bpm = np.full(n_beats, np.nan)
for i in range(1, n_beats):
    rr_s = (peaks[i] - peaks[i - 1]) / FS
    if 0.3 < rr_s < 2.5:
        hr_bpm[i] = 60.0 / rr_s

# Gaussian rolling average (sigma=2s, same as analyze_ecg)
hr_avg = np.full(n_beats, np.nan)
SIGMA_S = 2.0
beat_time_s = peaks / FS
valid_hr = ~np.isnan(hr_bpm)
valid_idx = np.where(valid_hr)[0]
if len(valid_idx) > 2:
    for ii, ci in enumerate(valid_idx):
        t_c = beat_time_s[ci]
        lo = ii
        while lo > 0 and (t_c - beat_time_s[valid_idx[lo - 1]]) < 3 * SIGMA_S:
            lo -= 1
        hi = ii
        while hi < len(valid_idx) - 1 and (beat_time_s[valid_idx[hi + 1]] - t_c) < 3 * SIGMA_S:
            hi += 1
        win = valid_idx[lo:hi + 1]
        dt = beat_time_s[win] - t_c
        w = np.exp(-0.5 * (dt / SIGMA_S) ** 2)
        hr_avg[ci] = np.average(hr_bpm[win], weights=w)

print(f"HR timeline: {np.nanmin(hr_avg):.0f}-{np.nanmax(hr_avg):.0f} bpm")

# =============================================================
# EXTRACT BEAT WINDOWS
# =============================================================
pre_samp  = int(PRE_R_MS * FS / 1000)
post_samp = int(POST_R_MS * FS / 1000)
win_len   = pre_samp + post_samp + 1
t_win_ms  = np.arange(-pre_samp, post_samp + 1) / FS * 1000

# Highpass filter for noise metric (removes baseline wander from residual)
sos_noise_hp = signal.butter(2, 40, 'highpass', fs=FS, output='sos')

def get_beat(beat_idx, data):
    """Extract a single beat window, zero-padded if at edges."""
    r = peaks[beat_idx]
    start = r - pre_samp
    end = r + post_samp + 1

    if start < 0 or end > N:
        beat = np.full(win_len, np.nan)
        s = max(0, start)
        e = min(N, end)
        offset = s - start
        beat[offset:offset + (e - s)] = data[s:e]
        return beat
    return data[start:end].copy()

def compute_rolling_avg(center_idx, data, window):
    """Compute average of beats around center_idx."""
    start = max(0, center_idx - window + 1)
    end = center_idx + 1
    beats = []
    for j in range(start, end):
        b = get_beat(j, data)
        if not np.any(np.isnan(b)):
            beats.append(b)
    if len(beats) == 0:
        return np.full(win_len, np.nan)
    return np.mean(beats, axis=0)

# =============================================================
# INTERACTIVE PLOT
# =============================================================
PERSIST_DEPTH = 20  # number of historical beats to show in persistence mode

class BeatScope:
    def __init__(self):
        self.idx = 0
        self.avg_window = AVG_WINDOW_DEFAULT
        self.show_ghost = True
        self.show_persist = False  # storage scope persistence
        self.show_avg = True
        self.show_pwave = True
        self.use_filter = True
        self.num_beats = 1
        self.data = ecg_filtered
        self._dragging = False
        self.tl_mode = 'hr'          # 'hr' or 'noise'
        self._noise_timeline = None   # lazy-computed

        # --- Layout: beat scope on top, timeline strip on bottom ---
        self.fig = plt.figure(figsize=(14, 7))
        gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.25,
                      left=0.06, right=0.98, top=0.94, bottom=0.06)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_tl = self.fig.add_subplot(gs[1])
        self.fig.canvas.manager.set_window_title('ECG Beat Scope')

        # --- Timeline strip ---
        self._draw_timeline()

        # Position marker on timeline
        self.tl_marker = self.ax_tl.axvline(beat_mpl[0], color='red',
                                             linewidth=1.5, alpha=0.8)

        # --- Beat scope (main axes) ---
        # Persistence traces
        self.persist_lines = []
        for k in range(PERSIST_DEPTH):
            alpha = 0.04 + 0.16 * (k / max(1, PERSIST_DEPTH - 1))
            ln, = self.ax.plot([], [], color='green', linewidth=0.6,
                               alpha=alpha)
            self.persist_lines.append(ln)

        # Plot elements
        self.line_curr, = self.ax.plot([], [], 'b-', linewidth=1.5,
                                        label='Current beat')
        self.line_ghost, = self.ax.plot([], [], color='steelblue',
                                         linewidth=0.8, alpha=0.3,
                                         label='Previous beat')
        self.line_avg, = self.ax.plot([], [], 'r--', linewidth=1.2,
                                       alpha=0.6,
                                       label=f'Avg (last {self.avg_window})')

        # R-peak marker
        self.r_marker, = self.ax.plot([], [], 'rv', markersize=8)

        # P-wave region highlight
        p_start_ms = -P_REGION_MS[0]
        p_width_ms = P_REGION_MS[0] - P_REGION_MS[1]
        self.p_rect = Rectangle((p_start_ms, 0), p_width_ms, 1,
                                 alpha=0.08, color='orange',
                                 transform=self.ax.get_xaxis_transform())
        self.ax.add_patch(self.p_rect)

        # Vertical line at R=0
        self.ax.axvline(0, color='gray', linestyle=':', linewidth=0.5,
                         alpha=0.5)
        # Baseline
        self.ax.axhline(0, color='gray', linestyle='-', linewidth=0.3,
                         alpha=0.3)

        self.ax.set_xlim(-PRE_R_MS, POST_R_MS)
        self.ax.set_xlabel('Time relative to R-peak (ms)')
        self.ax.set_ylabel('µV')
        self.ax.grid(True, alpha=0.2)
        self.ax.legend(loc='upper right', fontsize=8)

        self.title = self.ax.set_title('', fontsize=11)
        self.status_text = self.ax.text(
            0.01, 0.01, '', transform=self.ax.transAxes,
            fontsize=8, verticalalignment='bottom',
            fontfamily='monospace', color='gray')

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Store full timeline range for zoom reset
        self._tl_full_xlim = self.ax_tl.get_xlim()
        self._panning = False
        self._pan_origin = None

        self.update()

    def _compute_noise_timeline(self):
        """Lazy-compute per-beat noise RMS (HP-filtered residual vs rolling avg)."""
        if self._noise_timeline is not None:
            return self._noise_timeline
        print("Computing noise timeline...", end=' ', flush=True)
        noise = np.full(n_beats, np.nan)
        for i in range(1, n_beats):
            b = get_beat(i, ecg_filtered)
            avg = compute_rolling_avg(i, ecg_filtered, self.avg_window)
            if np.any(np.isnan(b)) or np.any(np.isnan(avg)):
                continue
            residual = b - avg
            residual_hp = signal.sosfilt(sos_noise_hp, residual)
            noise[i] = np.sqrt(np.mean(residual_hp ** 2))
        self._noise_timeline = noise
        v = ~np.isnan(noise)
        print(f"done ({np.sum(v)} beats, "
              f"median {np.nanmedian(noise):.0f} µV)")
        return noise

    def _draw_timeline(self):
        """Clear and redraw the timeline strip for current mode."""
        xlim = self.ax_tl.get_xlim()
        has_marker = hasattr(self, 'tl_marker')
        self.ax_tl.clear()

        if self.tl_mode == 'noise':
            noise = self._compute_noise_timeline()
            v = ~np.isnan(noise)
            # Raw per-beat trace (dim)
            self.ax_tl.plot(beat_mpl[v], noise[v], color='purple',
                            linewidth=0.4, alpha=0.25)
            # Rolling median trendline
            if np.sum(v) > 30:
                noise_valid = noise[v]
                mpl_valid = beat_mpl[v]
                smoothed = median_filter(noise_valid, size=31)
                self.ax_tl.plot(mpl_valid, smoothed, color='purple',
                                linewidth=1.2, alpha=0.9)
            self.ax_tl.set_ylabel('µV', fontsize=8)
            if np.any(v):
                self.ax_tl.set_ylim(0, np.nanpercentile(noise[v], 99) * 1.2)
        else:
            v = ~np.isnan(hr_avg)
            self.ax_tl.plot(beat_mpl[v], hr_avg[v], color='navy',
                            linewidth=1.0, alpha=0.8)
            self.ax_tl.set_ylabel('bpm', fontsize=8)
            if np.any(v):
                ylo = max(30, np.nanmin(hr_avg[v]) - 5)
                yhi = np.nanmax(hr_avg[v]) + 5
                self.ax_tl.set_ylim(ylo, yhi)

        self.ax_tl.grid(True, alpha=0.2)
        self.ax_tl.tick_params(labelsize=7)

        # Time axis formatting
        if time_source != 'elapsed':
            self.ax_tl.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax_tl.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=6, maxticks=15))
            self.ax_tl.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
        else:
            self.ax_tl.set_xlabel('Elapsed (min)', fontsize=8)

        # Restore x limits if we had them
        if has_marker and xlim[0] != xlim[1]:
            self.ax_tl.set_xlim(xlim)

        # Re-add position marker
        self.tl_marker = self.ax_tl.axvline(beat_mpl[self.idx], color='red',
                                             linewidth=1.5, alpha=0.8)

    def _mpl_to_beat_idx(self, x_mpl):
        """Find nearest beat index to an x position on the timeline."""
        dists = np.abs(beat_mpl - x_mpl)
        return int(np.argmin(dists))

    def _reset_tl_zoom(self):
        """Reset timeline to full view."""
        self.ax_tl.set_xlim(self._tl_full_xlim)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax_tl:
            return
        # Double-click: reset zoom
        if event.dblclick:
            self._reset_tl_zoom()
            return
        # Left-click: scrub
        if event.button == 1:
            self._dragging = True
            idx = self._mpl_to_beat_idx(event.xdata)
            self.goto(idx)
        # Right-click: start pan
        elif event.button == 3:
            self._panning = True
            self._pan_origin = event.xdata

    def on_release(self, event):
        self._dragging = False
        self._panning = False
        self._pan_origin = None

    def on_motion(self, event):
        if event.inaxes != self.ax_tl:
            return
        # Left-drag: scrub
        if self._dragging:
            idx = self._mpl_to_beat_idx(event.xdata)
            self.goto(idx)
        # Right-drag: pan
        elif self._panning and self._pan_origin is not None:
            dx = self._pan_origin - event.xdata
            lo, hi = self.ax_tl.get_xlim()
            full_lo, full_hi = self._tl_full_xlim
            new_lo = lo + dx
            new_hi = hi + dx
            # Clamp to full range
            if new_lo < full_lo:
                new_lo, new_hi = full_lo, full_lo + (hi - lo)
            if new_hi > full_hi:
                new_hi, new_lo = full_hi, full_hi - (hi - lo)
            self.ax_tl.set_xlim(new_lo, new_hi)
            self._pan_origin = event.xdata
            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax_tl:
            return
        lo, hi = self.ax_tl.get_xlim()
        full_span = self._tl_full_xlim[1] - self._tl_full_xlim[0]
        span = hi - lo

        # Zoom factor per scroll step
        factor = 0.8 if event.button == 'up' else 1.25

        new_span = span * factor
        # Clamp: don't zoom out beyond full range, don't zoom in beyond 1/500th
        new_span = max(full_span / 500, min(full_span, new_span))

        # Zoom centered on cursor position
        cursor = event.xdata
        frac = (cursor - lo) / span if span > 0 else 0.5
        new_lo = cursor - frac * new_span
        new_hi = cursor + (1 - frac) * new_span

        # Clamp to full range
        full_lo, full_hi = self._tl_full_xlim
        if new_lo < full_lo:
            new_lo, new_hi = full_lo, full_lo + new_span
        if new_hi > full_hi:
            new_hi, new_lo = full_hi, full_hi - new_span

        self.ax_tl.set_xlim(new_lo, new_hi)
        self.fig.canvas.draw_idle()

    def update(self):
        # Determine how many beats we can actually show
        last_beat = min(self.idx + self.num_beats - 1, n_beats - 1)
        actual_beats = last_beat - self.idx + 1
        single = (actual_beats == 1)

        if single:
            beat = get_beat(self.idx, self.data)
            t_ms = t_win_ms
        else:
            start = peaks[self.idx] - pre_samp
            end = peaks[last_beat] + post_samp + 1
            win_n = end - start
            beat = np.full(win_n, np.nan)
            s = max(0, start)
            e = min(N, end)
            offset = s - start
            beat[offset:offset + (e - s)] = self.data[s:e]
            t_ms = np.arange(win_n) / FS * 1000 - PRE_R_MS

        self.line_curr.set_data(t_ms, beat)

        # Ghost
        if self.show_ghost and self.idx > 0 and single:
            ghost = get_beat(self.idx - 1, self.data)
            self.line_ghost.set_data(t_win_ms, ghost)
            self.line_ghost.set_visible(True)
        else:
            self.line_ghost.set_visible(False)

        # Persistence
        for k, ln in enumerate(self.persist_lines):
            hist_idx = self.idx - PERSIST_DEPTH + k
            if self.show_persist and single and 0 <= hist_idx < self.idx:
                b = get_beat(hist_idx, self.data)
                ln.set_data(t_win_ms, b)
                ln.set_visible(True)
            else:
                ln.set_visible(False)

        # Rolling average
        avg_beat = None
        if self.show_avg and self.idx >= 1 and single:
            avg_beat = compute_rolling_avg(self.idx, self.data, self.avg_window)
            self.line_avg.set_data(t_win_ms, avg_beat)
            self.line_avg.set_visible(True)
            self.line_avg.set_label(f'Avg (last {self.avg_window})')
        else:
            self.line_avg.set_visible(False)

        # R-peak markers
        r_times = []
        r_amps = []
        for bi in range(self.idx, last_beat + 1):
            t_r = (peaks[bi] - peaks[self.idx]) / FS * 1000
            if single:
                amp = beat[pre_samp] if not np.isnan(beat[pre_samp]) else 0
            else:
                si = peaks[bi] - (peaks[self.idx] - pre_samp)
                amp = beat[si] if 0 <= si < len(beat) and not np.isnan(beat[si]) else 0
            r_times.append(t_r)
            r_amps.append(amp)
        self.r_marker.set_data(r_times, r_amps)

        # P-wave highlight
        self.p_rect.set_visible(self.show_pwave and single)

        # Axis limits
        self.ax.set_xlim(t_ms[0], t_ms[-1])

        valid = beat[~np.isnan(beat)]
        if len(valid) > 0:
            ymin, ymax = np.min(valid), np.max(valid)
            margin = (ymax - ymin) * 0.15
            self.ax.set_ylim(ymin - margin, ymax + margin)

        # --- Timeline position marker ---
        self.tl_marker.set_xdata([beat_mpl[self.idx], beat_mpl[self.idx]])

        # --- Title with wall-clock time ---
        r_time = peaks[self.idx] / FS
        rr_ms = ''
        hr = ''
        if self.idx > 0:
            rr = (peaks[self.idx] - peaks[self.idx - 1]) / FS * 1000
            rr_ms = f'{rr:.0f}'
            hr = f'{60000/rr:.1f}'

        r_amp = r_amps[0] if r_amps else 0

        # Noise metric: RMS of highpass-filtered (current beat − rolling average)
        # Highpass removes slow baseline wander; keeps HF texture noise
        noise_str = ''
        if single and avg_beat is not None:
            residual = beat - avg_beat
            if not np.any(np.isnan(residual)):
                residual_hp = signal.sosfilt(sos_noise_hp, residual)
                noise_rms = np.sqrt(np.mean(residual_hp ** 2))
                noise_str = f'  Noise: {noise_rms:.0f}\u00b5V'

        # Wall-clock string
        if time_source != 'elapsed':
            wall_str = datetime.fromtimestamp(
                beat_epoch[self.idx]).strftime('%H:%M:%S')
        else:
            m, s = divmod(r_time, 60)
            wall_str = f'{int(m)}:{s:05.2f}'

        if single:
            beat_label = f'Beat {self.idx + 1}/{n_beats}'
        else:
            beat_label = f'Beats {self.idx + 1}\u2013{last_beat + 1}/{n_beats}'

        self.title.set_text(
            f'{beat_label}  '
            f'{wall_str}  '
            f'R-R: {rr_ms}ms  HR: {hr}bpm  '
            f'QRS: {r_amp:.0f}\u00b5V'
            f'{noise_str}'
        )

        flags = []
        if self.use_filter:   flags.append('FILT')
        if single and self.show_ghost:   flags.append('GHOST')
        if single and self.show_persist: flags.append(f'PERSIST:{PERSIST_DEPTH}')
        if single and self.show_avg:     flags.append(f'AVG:{self.avg_window}')
        if single and self.show_pwave:   flags.append('P-HL')
        if not single: flags.append(f'BEATS:{actual_beats}')
        flags.append(f'TL:{self.tl_mode.upper()}')
        self.status_text.set_text(
            f'[{"  ".join(flags)}]  '
            f'\u2190\u2192:step  Shift:\u00d710  '
            f'G:ghost  T:persist  R:avg  \u2191\u2193:N  F:filter  P:highlight  N:timeline  1-3:beats  S:save  Q:quit'
        )

        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def goto(self, idx):
        self.idx = max(0, min(n_beats - 1, idx))
        self.update()

    def on_key(self, event):
        if event.key in ('right', 'd'):
            self.goto(self.idx + 1)
        elif event.key in ('shift+right', 'D'):
            self.goto(self.idx + 10)
        elif event.key in ('left', 'a'):
            self.goto(self.idx - 1)
        elif event.key in ('shift+left', 'A'):
            self.goto(self.idx - 10)
        elif event.key == 'home':
            self._reset_tl_zoom()
            self.goto(0)
        elif event.key == 'end':
            self.goto(n_beats - 1)
        elif event.key == 'up':
            self.avg_window = min(100, self.avg_window + 5)
            self._noise_timeline = None
            if self.tl_mode == 'noise':
                self._draw_timeline()
            self.update()
        elif event.key == 'down':
            self.avg_window = max(2, self.avg_window - 5)
            self._noise_timeline = None
            if self.tl_mode == 'noise':
                self._draw_timeline()
            self.update()
        elif event.key == 'g':
            self.show_ghost = not self.show_ghost
            self.update()
        elif event.key == 't':
            self.show_persist = not self.show_persist
            self.update()
        elif event.key == 'r':
            self.show_avg = not self.show_avg
            self.update()
        elif event.key == 'p':
            self.show_pwave = not self.show_pwave
            self.update()
        elif event.key == 'f':
            self.use_filter = not self.use_filter
            self.data = ecg_filtered if self.use_filter else data_raw
            self.update()
        elif event.key in ('1', '2', '3'):
            self.num_beats = int(event.key)
            self.update()
        elif event.key == 'n':
            self.tl_mode = 'noise' if self.tl_mode == 'hr' else 'hr'
            self._draw_timeline()
            self.fig.canvas.draw_idle()
        elif event.key == 's':
            fname = str(Path(CSV_FILE).parent / f"beat_{self.idx+1:04d}.png")
            self.fig.savefig(fname, dpi=150, bbox_inches='tight')
            print(f"Saved {fname}")
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

# Launch
scope = BeatScope()
plt.show()
