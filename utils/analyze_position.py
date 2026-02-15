#!/usr/bin/env python3

"""
Sleep position segmentation from body roll angle (tilt sensor).

Reads MOT_*_breath.csv with columns: seconds, envelope_deg, breaths_per_min, roll_deg
Detects stable sleeping positions, clusters them, and outputs:
  - Annotated roll-angle plot with colored position segments
  - Segments CSV for downstream analysis (ECG, SpO2 correlation)
  - Printed summary table

Handles ±180° angle wrapping via unwrapping for smoothing and
circular statistics for clustering.

Usage:
  python analyze_position.py <csv_file_or_directory> [--no-plot] [--study-end 7]

J. Beale  2026-02
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='Sleep position analysis from roll angle')
parser.add_argument('input_path', help='CSV file or directory containing MOT_*_breath.csv')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNG but do not display')
parser.add_argument('--study-end', type=int, default=7, dest='study_end',
                    help='Hour to end study window (default: 7 = 7AM)')
parser.add_argument('--min-segment', type=float, default=3.0, dest='min_segment',
                    help='Minimum segment duration in minutes (default: 3)')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

STUDY_END_HOUR = args.study_end
MIN_SEGMENT_SEC = args.min_segment * 60

# Tuning parameters
SMOOTH_WINDOW_SEC = 30      # median filter window for smoothing
TRANSITION_THRESH_DEG = 15  # minimum angle change to count as transition
TRANSITION_WINDOW_SEC = 60  # window to detect transitions (rate of change)
CLUSTER_THRESH_DEG = 30     # max circular distance to merge into same position
TRANSITION_BUFFER_SEC = 30  # buffer to exclude around transitions

# =============================================================
# RESOLVE INPUT FILE
# =============================================================
def find_mot_csv(directory):
    pattern = re.compile(r'^MOT_.*_breath\.csv$', re.IGNORECASE)
    if not Path(directory).is_dir():
        return None
    for filepath in sorted(Path(directory).iterdir()):
        if filepath.is_file() and pattern.match(filepath.name):
            return str(filepath)
    return None

input_path = args.input_path
if Path(input_path).is_dir():
    found = find_mot_csv(input_path)
    if found is None:
        print(f"Error: No MOT_*_breath.csv found in {input_path}")
        sys.exit(1)
    input_path = found
    print(f"Found: {input_path}")

# =============================================================
# LOAD CSV
# =============================================================
t0 = None
with open(input_path, 'r') as f:
    first_line = f.readline().strip()
    m = re.search(r'start\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', first_line)
    if m:
        t0 = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')

data = np.genfromtxt(input_path, delimiter=',', names=True, skip_header=1)
seconds = data['seconds']
roll_raw = data['roll_deg']

N = len(seconds)
dt = np.median(np.diff(seconds[:100]))  # sample interval
fs = 1.0 / dt

print(f"Loaded {N} samples ({seconds[-1]-seconds[0]:.0f}s = "
      f"{(seconds[-1]-seconds[0])/3600:.1f}h) at {fs:.1f} sps from {Path(input_path).name}")
if t0:
    print(f"Start: {t0.strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================
# STUDY WINDOW MASK
# =============================================================
in_study = np.ones(N, dtype=bool)
if t0 is not None:
    for i in range(N):
        dt_i = t0 + timedelta(seconds=float(seconds[i]))
        h = dt_i.hour
        if h >= STUDY_END_HOUR and h < 20:
            in_study[i] = False

n_post_study = np.sum(~in_study)
if n_post_study > 0:
    # Trim to study window for analysis
    last_study = np.where(in_study)[0][-1]
    print(f"Study window ends at sample {last_study} "
          f"({n_post_study} post-study samples)")
else:
    last_study = N - 1

# =============================================================
# UNWRAP AND SMOOTH
# =============================================================
# Convert to radians, unwrap to eliminate ±180° discontinuities,
# then smooth to remove breathing oscillations.

roll_rad = np.deg2rad(roll_raw)
roll_unwrapped = np.rad2deg(np.unwrap(roll_rad))

# Median filter to remove spikes, then moving average for smoothness
from scipy.ndimage import median_filter, uniform_filter1d

med_win = max(3, int(SMOOTH_WINDOW_SEC * fs) | 1)  # ensure odd
roll_smooth = median_filter(roll_unwrapped, size=med_win)
avg_win = max(3, int(SMOOTH_WINDOW_SEC * fs))
roll_smooth = uniform_filter1d(roll_smooth, avg_win)

# =============================================================
# DETECT TRANSITIONS
# =============================================================
# A transition is where the smoothed angle changes by more than
# TRANSITION_THRESH_DEG within a TRANSITION_WINDOW_SEC window.

win_samples = int(TRANSITION_WINDOW_SEC * fs)
half_win = win_samples // 2

# Compute local range (max - min) in sliding window
rate = np.zeros(N)
for i in range(N):
    lo = max(0, i - half_win)
    hi = min(N, i + half_win + 1)
    rate[i] = np.max(roll_smooth[lo:hi]) - np.min(roll_smooth[lo:hi])

is_transition = rate > TRANSITION_THRESH_DEG

# Expand transition zones by buffer
buf_samples = int(TRANSITION_BUFFER_SEC * fs)
transition_idx = np.where(is_transition)[0]
for idx in transition_idx:
    lo = max(0, idx - buf_samples)
    hi = min(N, idx + buf_samples + 1)
    is_transition[lo:hi] = True

# =============================================================
# EXTRACT STABLE SEGMENTS
# =============================================================
# Contiguous runs of non-transition samples within study window

is_stable = ~is_transition & in_study

segments = []  # list of (start_idx, end_idx)
in_seg = False
seg_start = 0

for i in range(N):
    if is_stable[i] and not in_seg:
        seg_start = i
        in_seg = True
    elif not is_stable[i] and in_seg:
        seg_dur = seconds[i-1] - seconds[seg_start]
        if seg_dur >= MIN_SEGMENT_SEC:
            segments.append((seg_start, i - 1))
        in_seg = False

# Close final segment
if in_seg:
    seg_dur = seconds[N-1] - seconds[seg_start]
    if seg_dur >= MIN_SEGMENT_SEC:
        segments.append((seg_start, N - 1))

print(f"Found {len(segments)} stable segments (≥{MIN_SEGMENT_SEC/60:.0f} min)")

if len(segments) == 0:
    print("No stable segments found. Try adjusting parameters.")
    sys.exit(1)

# Compute circular mean angle for each segment
def circular_mean_deg(angles_deg):
    """Circular mean of angles in degrees."""
    rad = np.deg2rad(angles_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    return np.rad2deg(np.arctan2(mean_sin, mean_cos))

def circular_std_deg(angles_deg):
    """Circular standard deviation in degrees."""
    rad = np.deg2rad(angles_deg)
    R = np.sqrt(np.mean(np.sin(rad))**2 + np.mean(np.cos(rad))**2)
    R = min(R, 1.0)  # clamp for numerical safety
    return np.rad2deg(np.sqrt(-2 * np.log(R))) if R > 0 else 180.0

def circular_distance_deg(a, b):
    """Shortest angular distance between two angles in degrees."""
    d = (a - b + 180) % 360 - 180
    return abs(d)

seg_info = []
for s_start, s_end in segments:
    raw_angles = roll_raw[s_start:s_end+1]
    cmean = circular_mean_deg(raw_angles)
    cstd = circular_std_deg(raw_angles)
    dur = seconds[s_end] - seconds[s_start]
    seg_info.append({
        'start_idx': s_start,
        'end_idx': s_end,
        'start_sec': seconds[s_start],
        'end_sec': seconds[s_end],
        'duration': dur,
        'circ_mean': cmean,
        'circ_std': cstd,
    })

# =============================================================
# CLUSTER SEGMENTS INTO POSITIONS
# =============================================================
# Greedy clustering: assign each segment to nearest existing cluster,
# or start a new one if beyond threshold.

clusters = []  # list of {'mean': angle, 'members': [seg indices]}

for i, seg in enumerate(seg_info):
    best_cluster = None
    best_dist = CLUSTER_THRESH_DEG

    for ci, cl in enumerate(clusters):
        dist = circular_distance_deg(seg['circ_mean'], cl['mean'])
        if dist < best_dist:
            best_dist = dist
            best_cluster = ci

    if best_cluster is not None:
        clusters[best_cluster]['members'].append(i)
        # Update cluster mean (weighted by segment duration)
        members = clusters[best_cluster]['members']
        angles = [seg_info[m]['circ_mean'] for m in members]
        weights = [seg_info[m]['duration'] for m in members]
        # Weighted circular mean
        rad = np.deg2rad(angles)
        ws = np.array(weights) / np.sum(weights)
        mean_sin = np.sum(ws * np.sin(rad))
        mean_cos = np.sum(ws * np.cos(rad))
        clusters[best_cluster]['mean'] = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    else:
        clusters.append({'mean': seg['circ_mean'], 'members': [i]})

# Sort clusters by total duration (most time first)
for cl in clusters:
    cl['total_dur'] = sum(seg_info[m]['duration'] for m in cl['members'])
clusters.sort(key=lambda c: c['total_dur'], reverse=True)

# Assign position labels
pos_labels = []
for ci, cl in enumerate(clusters):
    label = chr(ord('A') + ci)
    cl['label'] = label
    cl['n_segments'] = len(cl['members'])

# Map segments to position label and color
colors = plt.cm.Set2(np.linspace(0, 1, max(len(clusters), 3)))
for ci, cl in enumerate(clusters):
    for mi in cl['members']:
        seg_info[mi]['position'] = cl['label']
        seg_info[mi]['color'] = colors[ci]
        seg_info[mi]['cluster_mean'] = cl['mean']

# =============================================================
# PRINT SUMMARY
# =============================================================
print("\n" + "="*70)
print("POSITION SUMMARY")
print("="*70)

print(f"\n  {'Pos':4s} {'Angle':>8s} {'Total':>8s} {'Segments':>8s}  Description")
print(f"  {'---':4s} {'-----':>8s} {'-----':>8s} {'--------':>8s}  -----------")
for cl in clusters:
    dur_min = cl['total_dur'] / 60
    dur_hr = cl['total_dur'] / 3600
    time_str = f"{dur_hr:.1f}h" if dur_hr >= 1 else f"{dur_min:.0f}m"
    print(f"  {cl['label']:4s} {cl['mean']:7.1f}° {time_str:>8s} {cl['n_segments']:8d}  "
          f"({100*cl['total_dur']/sum(c['total_dur'] for c in clusters):.0f}% of sleep)")

print(f"\n  {'Seg':4s} {'Start':>8s} {'End':>8s} {'Dur':>7s} {'Pos':>4s} "
      f"{'Angle':>8s} {'Stdev':>6s}")
print(f"  {'---':4s} {'-----':>8s} {'---':>8s} {'---':>7s} {'---':>4s} "
      f"{'-----':>8s} {'-----':>6s}")

for i, seg in enumerate(seg_info):
    if t0:
        t_start = (t0 + timedelta(seconds=float(seg['start_sec']))).strftime('%H:%M')
        t_end = (t0 + timedelta(seconds=float(seg['end_sec']))).strftime('%H:%M')
    else:
        t_start = f"{seg['start_sec']/3600:.2f}"
        t_end = f"{seg['end_sec']/3600:.2f}"
    dur_min = seg['duration'] / 60
    print(f"  {i+1:3d}  {t_start:>8s} {t_end:>8s} {dur_min:5.0f}m  "
          f"{seg['position']:>4s} {seg['circ_mean']:7.1f}° {seg['circ_std']:5.1f}°")

print("="*70)

# =============================================================
# SEGMENTS CSV OUTPUT
# =============================================================
stem = str(Path(input_path).parent / Path(input_path).stem)
csv_out = f"{stem}_positions.csv"

with open(csv_out, 'w') as f:
    f.write(f"# source: {Path(input_path).name}\n")
    if t0:
        f.write(f"# start: {t0.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("segment,start_sec,end_sec,start_time,end_time,duration_sec,"
            "position,angle_mean,angle_std\n")
    for i, seg in enumerate(seg_info):
        if t0:
            t_start = (t0 + timedelta(seconds=float(seg['start_sec']))).strftime('%H:%M:%S')
            t_end = (t0 + timedelta(seconds=float(seg['end_sec']))).strftime('%H:%M:%S')
        else:
            t_start = f"{seg['start_sec']:.1f}"
            t_end = f"{seg['end_sec']:.1f}"
        f.write(f"{i+1},{seg['start_sec']:.1f},{seg['end_sec']:.1f},"
                f"{t_start},{t_end},{seg['duration']:.0f},"
                f"{seg['position']},{seg['circ_mean']:.1f},{seg['circ_std']:.1f}\n")

print(f"\nSegments CSV: {csv_out}")

# =============================================================
# PLOT
# =============================================================
fig, ax = plt.subplots(figsize=(16, 5))

# Background: full roll trace in light gray
if t0:
    times = np.array([mdates.date2num(t0 + timedelta(seconds=float(s))) for s in seconds])
    ax.plot(times, roll_raw, color='lightgray', linewidth=0.4, zorder=1)

    # Color each segment
    for seg in seg_info:
        si, ei = seg['start_idx'], seg['end_idx']
        ax.fill_between(times[si:ei+1], roll_raw[si:ei+1],
                        alpha=0.3, color=seg['color'], zorder=2)
        ax.plot(times[si:ei+1], roll_raw[si:ei+1],
                color=seg['color'], linewidth=0.6, zorder=3)
        # Label at midpoint
        mid = (si + ei) // 2
        ax.text(times[mid], roll_raw[mid] + 8, seg['position'],
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=seg['color'] * 0.7)  # darken label

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=15))
    ax.set_xlabel('Time')
else:
    hrs = seconds / 3600
    ax.plot(hrs, roll_raw, color='lightgray', linewidth=0.4, zorder=1)
    for seg in seg_info:
        si, ei = seg['start_idx'], seg['end_idx']
        ax.fill_between(hrs[si:ei+1], roll_raw[si:ei+1],
                        alpha=0.3, color=seg['color'], zorder=2)
        ax.plot(hrs[si:ei+1], roll_raw[si:ei+1],
                color=seg['color'], linewidth=0.6, zorder=3)
        mid = (si + ei) // 2
        ax.text(hrs[mid], roll_raw[mid] + 8, seg['position'],
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('Hours')

ax.set_ylabel('Roll angle (°)')
title = f"Sleep Positions — {Path(input_path).name}"
if t0:
    title += f"  (start {t0.strftime('%Y-%m-%d %H:%M')})"
ax.set_title(title, fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

# Legend
legend_patches = [Patch(facecolor=colors[ci], alpha=0.4,
                        label=f"Pos {cl['label']} ({cl['mean']:.0f}°, "
                              f"{cl['total_dur']/60:.0f} min)")
                  for ci, cl in enumerate(clusters)]
ax.legend(handles=legend_patches, loc='best', fontsize=8)

fig.tight_layout()
out_png = f"{stem}_positions.png"
fig.savefig(out_png, dpi=150, bbox_inches='tight')

if not args.no_plot:
    plt.show()
