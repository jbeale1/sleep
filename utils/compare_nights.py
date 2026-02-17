#!/usr/bin/env python3

"""
Compare positional sleep statistics across multiple nights.

Reads positional_sleep_stats.csv files from multiple night directories,
matches sleep positions across nights by angle (circular distance),
and produces grouped bar charts for key metrics.

Usage:
  python compare_nights.py <dir1> <dir2> [dir3 ...]
  python compare_nights.py C:\\sleep\\20260214 C:\\sleep\\20260215 C:\\sleep\\20260216
  python compare_nights.py C:\\sleep\\202602*     (shell glob)

J. Beale  2026-02
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='Compare positional sleep stats across nights')
parser.add_argument('directories', nargs='+', help='Night directories containing positional_sleep_stats.csv')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNG but do not display')
parser.add_argument('--out', dest='out_dir', default=None,
                    help='Output directory for PNG (default: first input directory)')
parser.add_argument('--cluster-thresh', type=float, default=40, dest='cluster_thresh',
                    help='Max circular angle distance to merge positions (default: 40°)')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

CLUSTER_THRESH = args.cluster_thresh

# =============================================================
# LOAD ALL NIGHTS
# =============================================================
def find_stats_csv(directory):
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and f.name == 'positional_sleep_stats.csv':
            return str(f)
    return None

def circular_distance(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)

def circular_mean(angles, weights=None):
    rad = np.deg2rad(angles)
    if weights is None:
        weights = np.ones(len(angles))
    w = np.array(weights) / np.sum(weights)
    return np.rad2deg(np.arctan2(np.sum(w * np.sin(rad)),
                                  np.sum(w * np.cos(rad))))

# Load per-position aggregated stats from each night
nights = []

for d in args.directories:
    if not Path(d).is_dir():
        print(f"Warning: {d} is not a directory, skipping")
        continue

    csv_path = find_stats_csv(d)
    if csv_path is None:
        print(f"Warning: No positional_sleep_stats.csv in {d}, skipping")
        continue

    # Extract night date from directory name
    dir_name = Path(d).name
    m = re.search(r'(\d{8})', dir_name)
    night_label = m.group(1) if m else dir_name

    # Read segments
    segments = []
    with open(csv_path, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = line.split(',')
                continue
            parts = line.split(',')
            if len(parts) < len(header):
                continue
            row = {}
            for h, v in zip(header, parts):
                try:
                    row[h] = float(v) if v else np.nan
                except ValueError:
                    row[h] = v
            segments.append(row)

    if not segments:
        print(f"Warning: No segments in {csv_path}, skipping")
        continue

    # Aggregate by position label within this night
    pos_labels = sorted(set(str(s.get('position', '')) for s in segments))
    pos_stats = {}

    for pos in pos_labels:
        segs = [s for s in segments if str(s.get('position', '')) == pos]
        total_dur = sum(s.get('duration_min', 0) for s in segs)
        if total_dur <= 0:
            continue

        def wavg(key):
            vals = [(s.get(key, np.nan), s.get('duration_min', 0)) for s in segs
                    if not np.isnan(s.get(key, np.nan)) and s.get('duration_min', 0) > 0]
            if not vals:
                return np.nan
            v, w = zip(*vals)
            return np.average(v, weights=w)

        total_odi_events = sum(s.get('odi3_events', 0) for s in segs
                               if not np.isnan(s.get('odi3_events', np.nan)))

        pos_stats[pos] = {
            'angle': wavg('angle'),
            'total_min': total_dur,
            'hr_mean': wavg('hr_mean'),
            'hr_median': wavg('hr_median'),
            'sdnn': wavg('sdnn'),
            'rmssd': wavg('rmssd'),
            'pnn50': wavg('pnn50'),
            'spo2_mean': wavg('spo2_mean'),
            'spo2_min': min((s.get('spo2_min', np.nan) for s in segs
                             if not np.isnan(s.get('spo2_min', np.nan))),
                            default=np.nan),
            'odi3': total_odi_events / max(total_dur / 60, 0.01),
            'odi3_events': int(total_odi_events),
            't_below_90': wavg('t_below_90_pct'),
            'n_pvc': sum(s.get('n_pvc', 0) for s in segs
                         if not np.isnan(s.get('n_pvc', np.nan))),
        }

    nights.append({
        'label': night_label,
        'dir': d,
        'positions': pos_stats,
    })

if len(nights) < 1:
    print("Error: No valid night data found")
    sys.exit(1)

print(f"Loaded {len(nights)} nights: {', '.join(n['label'] for n in nights)}")

# =============================================================
# MATCH POSITIONS ACROSS NIGHTS BY ANGLE
# =============================================================
# Collect all (night_idx, pos_label, angle, duration) entries
all_positions = []
for ni, night in enumerate(nights):
    for pos, stats in night['positions'].items():
        if not np.isnan(stats['angle']):
            all_positions.append({
                'night_idx': ni,
                'pos_label': pos,
                'angle': stats['angle'],
                'duration': stats['total_min'],
            })

# Greedy clustering by circular distance
# Sort by duration descending so dominant positions anchor clusters first
all_positions.sort(key=lambda e: e['duration'], reverse=True)

clusters = []  # list of {'members': [...], 'mean': angle}
for entry in all_positions:
    best_cluster = None
    best_dist = CLUSTER_THRESH

    for ci, cl in enumerate(clusters):
        dist = circular_distance(entry['angle'], cl['mean'])
        if dist <= best_dist:
            best_dist = dist
            best_cluster = ci

    if best_cluster is not None:
        clusters[best_cluster]['members'].append(entry)
        # Update weighted mean
        angles = [m['angle'] for m in clusters[best_cluster]['members']]
        weights = [m['duration'] for m in clusters[best_cluster]['members']]
        clusters[best_cluster]['mean'] = circular_mean(angles, weights)
    else:
        clusters.append({'members': [entry], 'mean': entry['angle']})

# Sort clusters by total duration
for cl in clusters:
    cl['total_dur'] = sum(m['duration'] for m in cl['members'])
clusters.sort(key=lambda c: c['total_dur'], reverse=True)

# Assign descriptive labels based on angle
# Rough convention: 0°=face-up reference varies, but we can use relative naming
def angle_label(angle):
    """Assign a rough position name based on angle."""
    # Normalize to -180..180
    a = ((angle + 180) % 360) - 180
    # These are rough — user's sensor has arbitrary offset
    return f"{a:.0f}°"

for ci, cl in enumerate(clusters):
    cl['name'] = f"Pos {ci+1} ({angle_label(cl['mean'])})"
    cl['short'] = f"P{ci+1}"

# Build lookup: (night_idx, pos_label) → cluster index
pos_to_cluster = {}
for ci, cl in enumerate(clusters):
    for m in cl['members']:
        pos_to_cluster[(m['night_idx'], m['pos_label'])] = ci

print(f"\nMatched into {len(clusters)} position clusters:")
for cl in clusters:
    night_str = ', '.join(
        f"{nights[m['night_idx']]['label']}:{m['pos_label']}"
        for m in cl['members'])
    print(f"  {cl['name']:20s} {cl['total_dur']:6.0f} min total  [{night_str}]")

# =============================================================
# PREPARE PLOT DATA
# =============================================================
# Metrics to plot — (key, display_name, unit, higher_is_better or None)
metrics = [
    ('odi3',      'ODI-3',       '/hr',  False),
    ('hr_mean',   'Heart Rate',  'bpm',  None),
    ('sdnn',      'SDNN',        'ms',   None),
    ('rmssd',     'RMSSD',       'ms',   True),
    ('pnn50',     'pNN50',       '%',    True),
    ('spo2_mean', 'SpO2 Mean',   '%',    True),
]

n_metrics = len(metrics)
n_clusters = len(clusters)
n_nights = len(nights)

# Build data matrix: [metric][cluster][night] = value
data = np.full((n_metrics, n_clusters, n_nights), np.nan)

for ni, night in enumerate(nights):
    for pos, stats in night['positions'].items():
        key = (ni, pos)
        if key in pos_to_cluster:
            ci = pos_to_cluster[key]
            for mi, (mkey, _, _, _) in enumerate(metrics):
                data[mi, ci, ni] = stats.get(mkey, np.nan)

# =============================================================
# PLOT
# =============================================================
night_colors = plt.cm.Set1(np.linspace(0, 0.8, n_nights))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

bar_width = 0.7 / max(n_nights, 1)

for mi, (mkey, mname, munit, higher_good) in enumerate(metrics):
    ax = axes[mi]
    x = np.arange(n_clusters)

    for ni in range(n_nights):
        vals = data[mi, :, ni]
        offset = (ni - (n_nights - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.9,
                      color=night_colors[ni], alpha=0.8,
                      label=nights[ni]['label'])

    ax.set_xticks(x)
    ax.set_xticklabels([cl['name'] for cl in clusters], fontsize=8)
    ax.set_ylabel(munit, fontsize=9)
    ax.set_title(mname, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Ensure y-axis starts at 0 for rate metrics
    if mkey in ('odi3', 'pnn50', 't_below_90'):
        ax.set_ylim(bottom=0)
    # SpO2: narrow range to show clinically meaningful differences
    elif mkey == 'spo2_mean':
        valid_vals = data[mi][~np.isnan(data[mi])]
        if len(valid_vals) > 0:
            lo = max(85, np.min(valid_vals) - 2)
            ax.set_ylim(bottom=lo, top=100)

    # Value labels on bars (after axis limits are set)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for ni in range(n_nights):
        vals = data[mi, :, ni]
        offset = (ni - (n_nights - 1) / 2) * bar_width
        for xi, v in enumerate(vals):
            if not np.isnan(v):
                fmt = f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"
                ax.text(x[xi] + offset, v + 0.02 * y_range,
                        fmt, ha='center', va='bottom', fontsize=7)

# Single legend for all panels
handles = [Patch(facecolor=night_colors[ni], alpha=0.8,
                 label=nights[ni]['label'])
           for ni in range(n_nights)]
fig.legend(handles=handles, loc='upper center', ncol=n_nights,
           fontsize=10, bbox_to_anchor=(0.5, 1.0))

fig.suptitle('Positional Sleep Statistics — Night Comparison',
             fontsize=14, fontweight='bold', y=1.03)
fig.tight_layout()

# Save
out_dir = args.out_dir or args.directories[0]
out_path = str(Path(out_dir) / 'compare_nights.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")

if not args.no_plot:
    plt.show()

# =============================================================
# PRINT COMPARISON TABLE
# =============================================================
print("\n" + "="*100)
print("CROSS-NIGHT COMPARISON BY POSITION")
print("="*100)

col_w = 10
header_cols = [''] + [nights[ni]['label'] for ni in range(n_nights)]
for ci, cl in enumerate(clusters):
    print(f"\n  {cl['name']} ({cl['total_dur']:.0f} min total)")
    print(f"  {'Metric':<12s}", end='')
    for ni in range(n_nights):
        dur = 0
        for m in cl['members']:
            if m['night_idx'] == ni:
                dur = m['duration']
        print(f"  {nights[ni]['label']:>{col_w}s}", end='')
    print()
    print(f"  {'------':<12s}", end='')
    for ni in range(n_nights):
        print(f"  {'------':>{col_w}s}", end='')
    print()

    # Duration row
    print(f"  {'Duration':<12s}", end='')
    for ni in range(n_nights):
        dur = sum(m['duration'] for m in cl['members'] if m['night_idx'] == ni)
        if dur > 0:
            dur_str = f"{dur:.0f}m" if dur < 120 else f"{dur/60:.1f}h"
            print(f"  {dur_str:>{col_w}s}", end='')
        else:
            print(f"  {'—':>{col_w}s}", end='')
    print()

    for mi, (mkey, mname, munit, _) in enumerate(metrics):
        print(f"  {mname:<12s}", end='')
        for ni in range(n_nights):
            v = data[mi, ci, ni]
            if np.isnan(v):
                print(f"  {'—':>{col_w}s}", end='')
            else:
                fmt = f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"
                print(f"  {fmt:>{col_w}s}", end='')
        print()

print("\n" + "="*100)
