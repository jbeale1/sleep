#!/usr/bin/env python3

"""
ECG beat overlay display — superimposed beats aligned at R-peak.

Extracts beat windows from the ECG signal using R-peak locations from
the beats CSV, then overlays them in panels segmented by sleep position
or by hour. Normal beats shown in blue, PVCs in red, with the median
template as a bold line.

Requires: ECG CSV + beats CSV (from analyze_ecg.py --csv-out)
Optional: positions CSV (from analyze_position.py)

Usage:
  python ecg_overlay.py <directory> [options]
  python ecg_overlay.py C:\\sleep\\20260215
  python ecg_overlay.py C:\\sleep\\20260215 --by hour
  python ecg_overlay.py C:\\sleep\\20260215 --by position --window -200 400

J. Beale  2026-02
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='ECG beat overlay display')
parser.add_argument('input_path', help='Directory or ECG CSV file')
parser.add_argument('sample_rate', nargs='?', type=int, default=250,
                    help='Sample rate in sps (default: 250)')
parser.add_argument('--prefiltered', action='store_true',
                    help='Data is already filtered')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNG but do not display')
parser.add_argument('--by', choices=['position', 'hour', 'all'], default='position',
                    help='How to segment beats (default: position)')
parser.add_argument('--window', nargs=2, type=int, default=[-200, 400],
                    metavar=('PRE_MS', 'POST_MS'),
                    help='Window around R-peak in ms (default: -200 400)')
parser.add_argument('--max-beats', type=int, default=2000, dest='max_beats',
                    help='Max beats per panel to avoid overplotting (default: 2000)')
parser.add_argument('--study-end', type=int, default=7, dest='study_end',
                    help='Hour to end study (default: 7 = 7AM)')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

FS = args.sample_rate
PRE_MS = abs(args.window[0])
POST_MS = args.window[1]
PRE_SAMP = int(PRE_MS * FS / 1000)
POST_SAMP = int(POST_MS * FS / 1000)
WIN_SAMP = PRE_SAMP + POST_SAMP
WIN_MS = np.arange(WIN_SAMP) * 1000 / FS - PRE_MS

# Filtering
HP_FREQ = 0.5
NOTCH_FREQ = 60
LP_FREQ = 40

# =============================================================
# RESOLVE FILES
# =============================================================
def find_file(directory, pattern):
    pat = re.compile(pattern, re.IGNORECASE)
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and pat.search(f.name):
            return str(f)
    return None

input_path = args.input_path
if Path(input_path).is_dir():
    directory = input_path
    ecg_file = find_file(directory, r'^ECG_\d{8}_\d{6}\.csv$')
    beats_file = find_file(directory, r'_beats\.csv$')
    pos_file = find_file(directory, r'_positions\.csv$')
    if ecg_file is None:
        print("Error: No ECG_*.csv found")
        sys.exit(1)
    if beats_file is None:
        print("Error: No *_beats.csv found (run analyze_ecg.py --csv-out first)")
        sys.exit(1)
else:
    ecg_file = input_path
    directory = str(Path(input_path).parent)
    beats_file = find_file(directory, r'_beats\.csv$')
    pos_file = find_file(directory, r'_positions\.csv$')
    if beats_file is None:
        print("Error: No *_beats.csv found")
        sys.exit(1)

print(f"ECG:   {Path(ecg_file).name}")
print(f"Beats: {Path(beats_file).name}")
if pos_file:
    print(f"Pos:   {Path(pos_file).name}")

# =============================================================
# LOAD & FILTER ECG
# =============================================================
data_raw = np.loadtxt(ecg_file, delimiter=",", skiprows=1).flatten()
N = len(data_raw)
print(f"Loaded {N} samples ({N/FS:.1f}s) at {FS} sps")

if args.prefiltered:
    ecg = data_raw.copy()
else:
    sos_hp = signal.butter(2, HP_FREQ, 'highpass', fs=FS, output='sos')
    sos_lp = signal.butter(4, LP_FREQ, 'lowpass', fs=FS, output='sos')
    b_n, a_n = signal.iirnotch(NOTCH_FREQ, Q=30, fs=FS)
    ecg = signal.sosfiltfilt(sos_hp, data_raw)
    ecg = signal.filtfilt(b_n, a_n, ecg)
    ecg = signal.sosfiltfilt(sos_lp, ecg)
    print("Filtering applied")

# =============================================================
# WALL-CLOCK TIME
# =============================================================
def parse_filename_timestamp(filepath):
    m = re.search(r'(\d{8})_(\d{6})', Path(filepath).stem)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
    return None

def load_sync_file(csv_path):
    sync_path = Path(csv_path).with_name(Path(csv_path).stem + '_sync.csv')
    if not sync_path.exists():
        return None
    try:
        sync_data = np.loadtxt(sync_path, delimiter=',', skiprows=1)
        return sync_data[:, 0], sync_data[:, 1]
    except Exception:
        return None

sync_result = load_sync_file(ecg_file)
if sync_result is not None:
    sync_idx, sync_epoch = sync_result
    t0_epoch = np.interp(0, sync_idx, sync_epoch)
    t0 = datetime.fromtimestamp(t0_epoch)
else:
    t0 = parse_filename_timestamp(ecg_file)
    if t0:
        t0_epoch = t0.timestamp()
        sync_idx = np.array([0, N - 1], dtype=float)
        sync_epoch = np.array([t0_epoch, t0_epoch + (N - 1) / FS])
    else:
        t0 = datetime(2000, 1, 1)
        t0_epoch = t0.timestamp()
        sync_idx = np.array([0, N - 1], dtype=float)
        sync_epoch = np.array([0.0, (N - 1) / FS])

def sample_to_epoch(sample_indices):
    return np.interp(sample_indices, sync_idx, sync_epoch)

# =============================================================
# LOAD BEATS CSV
# =============================================================
beat_samples = []
beat_epochs = []
beat_is_artifact = []
beat_is_pvc = []

with open(beats_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or line.startswith('epoch_s'):
            continue
        parts = line.split(',')
        if len(parts) < 17:
            continue
        try:
            epoch = float(parts[0])
            samp = int(parts[1])
            is_art = int(parts[13]) if parts[13] else 0
            is_pvc = int(parts[14]) if parts[14] else 0
            beat_samples.append(samp)
            beat_epochs.append(epoch)
            beat_is_artifact.append(is_art)
            beat_is_pvc.append(is_pvc)
        except (ValueError, IndexError):
            continue

beat_samples = np.array(beat_samples)
beat_epochs = np.array(beat_epochs)
beat_is_artifact = np.array(beat_is_artifact, dtype=bool)
beat_is_pvc = np.array(beat_is_pvc, dtype=bool)
n_beats = len(beat_samples)

print(f"Loaded {n_beats} beats ({np.sum(beat_is_pvc)} PVCs, "
      f"{np.sum(beat_is_artifact)} artifacts)")

# =============================================================
# LOAD POSITION SEGMENTS (optional)
# =============================================================
pos_segments = []

if pos_file and args.by == 'position':
    pos_t0 = None
    with open(pos_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# start:'):
                m = re.search(r'start:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if m:
                    pos_t0 = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
            elif line.startswith('#') or line.startswith('segment,'):
                continue
            else:
                parts = line.split(',')
                if len(parts) >= 9 and pos_t0:
                    seg_start_epoch = pos_t0.timestamp() + float(parts[1])
                    seg_end_epoch = pos_t0.timestamp() + float(parts[2])
                    pos_segments.append({
                        'epoch_start': seg_start_epoch,
                        'epoch_end': seg_end_epoch,
                        'position': parts[6],
                        'start_time': parts[3],
                        'duration_min': float(parts[5]) / 60,
                    })

    if pos_segments:
        print(f"Loaded {len(pos_segments)} position segments")

# =============================================================
# EXTRACT BEAT WINDOWS
# =============================================================
def extract_window(sample_idx):
    """Extract a window around an R-peak, baseline-corrected."""
    s = sample_idx - PRE_SAMP
    e = sample_idx + POST_SAMP
    if s < 0 or e > N:
        return None
    win = ecg[s:e].copy()
    # Baseline correction: subtract median of first 20ms (pre-QRS)
    baseline_samp = max(1, int(20 * FS / 1000))
    win -= np.median(win[:baseline_samp])
    return win

# =============================================================
# SEGMENT BEATS INTO GROUPS
# =============================================================
class BeatGroup:
    def __init__(self, label, description=""):
        self.label = label
        self.description = description
        self.normal_windows = []
        self.pvc_windows = []

groups = []

if args.by == 'position' and pos_segments:
    # Group by position label, combining segments of same position
    pos_labels = sorted(set(s['position'] for s in pos_segments))
    for pos in pos_labels:
        segs = [s for s in pos_segments if s['position'] == pos]
        total_min = sum(s['duration_min'] for s in segs)
        g = BeatGroup(f"Pos {pos}", f"{total_min:.0f} min total")

        for seg in segs:
            mask = ((beat_epochs >= seg['epoch_start']) &
                    (beat_epochs <= seg['epoch_end']))
            seg_indices = np.where(mask)[0]

            for bi in seg_indices:
                if beat_is_artifact[bi]:
                    continue
                win = extract_window(beat_samples[bi])
                if win is None:
                    continue
                if beat_is_pvc[bi]:
                    g.pvc_windows.append(win)
                else:
                    g.normal_windows.append(win)
        groups.append(g)

elif args.by == 'hour':
    # Group by hour of the day
    hours_seen = {}
    for bi in range(n_beats):
        if beat_is_artifact[bi]:
            continue
        bt = datetime.fromtimestamp(beat_epochs[bi])
        h = bt.hour
        # Skip post-study hours
        if h >= args.study_end and h < 20:
            continue
        if h not in hours_seen:
            hours_seen[h] = BeatGroup(f"{h:02d}:00", f"Hour {h}")
        win = extract_window(beat_samples[bi])
        if win is None:
            continue
        if beat_is_pvc[bi]:
            hours_seen[h].pvc_windows.append(win)
        else:
            hours_seen[h].normal_windows.append(win)

    # Sort by time order (handle overnight wrap)
    start_hour = datetime.fromtimestamp(beat_epochs[0]).hour
    def hour_sort_key(h):
        if h >= start_hour:
            return h
        return h + 24
    for h in sorted(hours_seen.keys(), key=hour_sort_key):
        groups.append(hours_seen[h])

else:
    # All beats in one group
    g = BeatGroup("All beats")
    for bi in range(n_beats):
        if beat_is_artifact[bi]:
            continue
        bt = datetime.fromtimestamp(beat_epochs[bi])
        if bt.hour >= args.study_end and bt.hour < 20:
            continue
        win = extract_window(beat_samples[bi])
        if win is None:
            continue
        if beat_is_pvc[bi]:
            g.pvc_windows.append(win)
        else:
            g.normal_windows.append(win)
    groups.append(g)

# Filter out empty groups
groups = [g for g in groups if len(g.normal_windows) > 0]
n_groups = len(groups)

if n_groups == 0:
    print("Error: no valid beat windows extracted")
    sys.exit(1)

for g in groups:
    print(f"  {g.label}: {len(g.normal_windows)} normal, "
          f"{len(g.pvc_windows)} PVCs")

# =============================================================
# PLOT
# =============================================================
# Layout: up to 4 columns
n_cols = min(n_groups, 4)
n_rows_plot = int(np.ceil(n_groups / n_cols))

fig_width = 4.5 * n_cols
fig_height = 4.0 * n_rows_plot + 0.8
fig, axes = plt.subplots(n_rows_plot, n_cols, figsize=(fig_width, fig_height),
                         squeeze=False, sharey=True)

fig.suptitle(f"Beat Overlay — {Path(ecg_file).name}  "
             f"(window: −{PRE_MS}–+{POST_MS} ms)",
             fontsize=12, fontweight='bold')

# Find global y-range from median templates
all_medians = []
for g in groups:
    if len(g.normal_windows) >= 5:
        med = np.median(g.normal_windows, axis=0)
        all_medians.append(med)
if all_medians:
    all_med = np.concatenate(all_medians)
    y_lo = np.percentile(all_med, 0.5) * 1.5
    y_hi = np.percentile(all_med, 99.5) * 1.5
    # Ensure some minimum range
    y_range = max(y_hi - y_lo, 500)
    y_lo = min(y_lo, -200)
    y_hi = max(y_hi, y_range + y_lo)
else:
    y_lo, y_hi = -500, 2000

for gi, g in enumerate(groups):
    row = gi // n_cols
    col = gi % n_cols
    ax = axes[row][col]

    normal = np.array(g.normal_windows)
    pvcs = np.array(g.pvc_windows) if g.pvc_windows else np.empty((0, WIN_SAMP))
    n_normal = len(normal)
    n_pvc = len(pvcs)

    # Subsample if too many beats
    if n_normal > args.max_beats:
        idx = np.random.default_rng(42).choice(n_normal, args.max_beats, replace=False)
        normal_plot = normal[idx]
    else:
        normal_plot = normal

    # Plot individual normal beats
    for win in normal_plot:
        ax.plot(WIN_MS, win, color='steelblue', linewidth=0.15,
                alpha=0.08, zorder=1)

    # Plot PVCs
    for win in pvcs:
        ax.plot(WIN_MS, win, color='red', linewidth=0.3,
                alpha=0.3, zorder=3)

    # Median template (bold)
    if n_normal >= 5:
        median_beat = np.median(normal, axis=0)
        ax.plot(WIN_MS, median_beat, color='navy', linewidth=1.5,
                alpha=0.9, zorder=4, label='Median')

        # 5th/95th percentile envelope
        p5 = np.percentile(normal, 5, axis=0)
        p95 = np.percentile(normal, 95, axis=0)
        ax.fill_between(WIN_MS, p5, p95, color='steelblue',
                        alpha=0.12, zorder=2, label='5–95th %ile')

    # PVC median if enough
    if n_pvc >= 3:
        pvc_median = np.median(pvcs, axis=0)
        ax.plot(WIN_MS, pvc_median, color='darkred', linewidth=1.5,
                alpha=0.8, zorder=4, linestyle='--', label='PVC median')

    # R-peak marker
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5, linestyle=':')

    # Labels
    ax.set_title(f"{g.label}  ({n_normal} beats"
                 f"{f', {n_pvc} PVCs' if n_pvc > 0 else ''})",
                 fontsize=10)
    if g.description:
        ax.text(0.02, 0.97, g.description, transform=ax.transAxes,
                fontsize=7, va='top', color='#666666')

    ax.set_xlim(WIN_MS[0], WIN_MS[-1])
    ax.set_ylim(y_lo, y_hi)

    if row == n_rows_plot - 1:
        ax.set_xlabel('ms', fontsize=9)
    if col == 0:
        ax.set_ylabel('µV', fontsize=9)

    ax.grid(True, alpha=0.2)

    # Legend on first panel only
    if gi == 0:
        ax.legend(fontsize=7, loc='upper right')

# Hide unused axes
for gi in range(n_groups, n_rows_plot * n_cols):
    row = gi // n_cols
    col = gi % n_cols
    axes[row][col].set_visible(False)

fig.tight_layout()

stem = str(Path(ecg_file).parent / Path(ecg_file).stem)
out_path = f"{stem}_overlay_{args.by}.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

if not args.no_plot:
    plt.show()
