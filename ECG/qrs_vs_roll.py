#!/usr/bin/env python3
"""
QRS Amplitude vs Body Roll Angle

Aligns per-beat ECG metrics with body orientation data from an accelerometer
belt, then plots QRS amplitude against roll angle across the recording night.
Optionally fits a cosine curve to model the projection of the cardiac dipole
onto the recording electrode axis.

Input files (auto-discovered from a directory, or passed explicitly):
  MOT_YYYY-MM-DD_HHMMSS.csv   — motion/orientation data
  ECG_YYYYMMDD_HHMMSS_beats.csv — per-beat ECG metrics

Usage:
  python qrs_vs_roll.py <directory_or_mot_file> [beats_csv]
  python qrs_vs_roll.py 20260220/
  python qrs_vs_roll.py MOT_2026-02-20_224320.csv ECG_20260220_225940_beats.csv
  python qrs_vs_roll.py 20260220/ --no-fit --max-noise 10 --save

Options:
  --no-fit       Skip cosine curve fitting
  --max-noise N  Exclude beats with noise above N µV (default: 15)
  --min-rr N     Exclude beats with R-R below N ms (default: 400, filters artifacts)
  --save         Save plot to PNG instead of displaying

J. Beale  2026-02
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='QRS amplitude vs body roll angle')
parser.add_argument('input',  help='Directory containing both files, or MOT csv path')
parser.add_argument('beats',  nargs='?', default=None, help='Beats CSV (if input is MOT file)')
parser.add_argument('--no-fit',    action='store_true', dest='no_fit',
                    help='Skip cosine curve fit')
parser.add_argument('--max-noise', type=float, default=15.0, dest='max_noise',
                    help='Exclude beats with noise_rms above this µV (default: 15)')
parser.add_argument('--min-rr',    type=float, default=400.0, dest='min_rr',
                    help='Exclude beats with R-R below this ms (default: 400)')
parser.add_argument('--save',      action='store_true',
                    help='Save PNG instead of displaying')
args = parser.parse_args()

# =============================================================
# FILE DISCOVERY
# =============================================================
def find_files(input_path, beats_path):
    """Return (mot_path, beats_path) as strings, or raise."""
    p = Path(input_path)
    if p.is_dir():
        mot_files   = sorted(p.glob('MOT_*.csv'))
        beat_files  = sorted(p.glob('ECG_*_beats.csv'))
        if not mot_files:
            raise FileNotFoundError(f"No MOT_*.csv found in {p}")
        if not beat_files:
            raise FileNotFoundError(f"No ECG_*_beats.csv found in {p}")
        return str(mot_files[0]), str(beat_files[0])
    else:
        # p is the MOT file directly
        if beats_path is None:
            raise ValueError("Provide beats CSV as second argument when input is a file")
        return str(p), str(beats_path)

mot_path, beats_path = find_files(args.input, args.beats)
print(f"MOT  file : {mot_path}")
print(f"Beats file: {beats_path}")

# =============================================================
# LOAD MOTION FILE
# =============================================================
def load_mot(path):
    """
    Parse MOT CSV.  Header comment supplies the wall-clock start time:
      # start YYYY-MM-DD HH:MM:SS ...
    The msec column is a direct millisecond offset from that exact second:
    msec=0 corresponds to start_epoch exactly (counter reset at top of second).
    Each data row: msec, pitch, roll, rot, total, rms
    Returns arrays: epoch (float64), roll (float32), rms (float32)
    """
    start_epoch = None

    with open(path, 'r') as fh:
        lines = fh.readlines()

    for line in lines:
        if line.startswith('#'):
            m = re.search(r'start\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if m:
                start_epoch = datetime.strptime(
                    m.group(1), '%Y-%m-%d %H:%M:%S').timestamp()
                break

    if start_epoch is None:
        raise ValueError(f"Could not parse start time from {path}")

    # Skip comment and header lines
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    # First non-comment line is the CSV header
    header = data_lines[0].strip().split(',')
    col = {name: i for i, name in enumerate(header)}
    rows = []
    for line in data_lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < max(col['msec'], col['roll'], col['rms']) + 1:
            continue
        try:
            rows.append((float(parts[col['msec']]),
                         float(parts[col['roll']]),
                         float(parts[col['rms']])))
        except ValueError:
            continue

    rows = np.array(rows)
    msec_arr = rows[:, 0]
    roll_arr = rows[:, 1]
    rms_arr  = rows[:, 2]

    # msec=0 is exactly start_epoch; offset is simply msec/1000
    epoch_arr = start_epoch + msec_arr / 1000.0

    print(f"MOT : {len(epoch_arr)} samples, "
          f"{(epoch_arr[-1]-epoch_arr[0])/3600:.2f} h, "
          f"roll range [{roll_arr.min():.1f}, {roll_arr.max():.1f}] deg")
    return epoch_arr, roll_arr, rms_arr

mot_epoch, mot_roll, mot_rms = load_mot(mot_path)

# =============================================================
# LOAD BEATS FILE
# =============================================================
def load_beats(path):
    """
    Parse ECG beats CSV.  Comment lines start with #.
    Required columns: epoch_s, qrs_amp_uv, rr_ms, is_artifact
    Optional: noise_rms_uv (if absent, no noise filtering)
    Returns a dict of column arrays.
    """
    with open(path, 'r') as fh:
        lines = fh.readlines()

    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    header = data_lines[0].strip().split(',')
    col = {name.strip(): i for i, name in enumerate(header)}

    required = ['epoch_s', 'qrs_amp_uv', 'is_artifact']
    for c in required:
        if c not in col:
            raise ValueError(f"Column '{c}' not found in {path}. "
                             f"Available: {list(col.keys())}")

    rows = []
    for line in data_lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < len(header):
            continue
        try:
            rows.append([float(parts[col[c]]) if parts[col[c]] else np.nan
                         for c in header])
        except ValueError:
            continue

    arr = np.array(rows)
    result = {c: arr[:, i] for c, i in col.items()}

    print(f"Beats: {len(arr)} beats, "
          f"{(result['epoch_s'][-1]-result['epoch_s'][0])/3600:.2f} h")
    return result

beats = load_beats(beats_path)

# =============================================================
# ALIGN: INTERPOLATE ROLL ANGLE ONTO BEAT TIMESTAMPS
# =============================================================
beat_epoch = beats['epoch_s']

# Only interpolate within the time range covered by the MOT file
in_range = ((beat_epoch >= mot_epoch[0]) &
            (beat_epoch <= mot_epoch[-1]))

roll_at_beat  = np.full(len(beat_epoch), np.nan)
motRMS_at_beat = np.full(len(beat_epoch), np.nan)
roll_at_beat[in_range]   = np.interp(beat_epoch[in_range], mot_epoch, mot_roll)
motRMS_at_beat[in_range] = np.interp(beat_epoch[in_range], mot_epoch, mot_rms)

n_overlap = np.sum(in_range)
print(f"Time overlap: {n_overlap} beats have matching MOT data")
if n_overlap == 0:
    print("ERROR: No time overlap between files.  Check timestamps.")
    sys.exit(1)

# =============================================================
# QUALITY FILTERS
# =============================================================
qrs = beats['qrs_amp_uv']
artifact = beats['is_artifact'].astype(bool)
rr = beats.get('rr_ms', np.full(len(qrs), np.nan))

good = (in_range
        & ~artifact
        & ~np.isnan(qrs)
        & ~np.isnan(roll_at_beat))

# R-R filter (removes ectopics and detection errors)
if 'rr_ms' in beats:
    good &= (rr > args.min_rr)

# Motion filter: exclude beats where accelerometer RMS is elevated
# (body actively moving — axis geometry is changing rapidly)
MOT_RMS_THRESHOLD = 5.0   # deg/s or g units depending on sensor
good &= (motRMS_at_beat < MOT_RMS_THRESHOLD)

# Optional noise filter (column may not exist)
if 'noise_rms_uv' in beats:
    noise = beats['noise_rms_uv']
    good &= (noise < args.max_noise)

roll_plot = roll_at_beat[good]
qrs_plot  = qrs[good]
time_plot = beat_epoch[good]

print(f"After filtering: {np.sum(good)} beats used "
      f"({100*np.sum(good)/n_overlap:.0f}% of overlapping beats)")

if len(roll_plot) < 10:
    print("Too few points after filtering to plot.")
    sys.exit(1)

# =============================================================
# POSITION CLUSTER DETECTION
# =============================================================
# Sleep positions are a small number of discrete angles, not a
# continuous distribution. Detect clusters with 1D k-means starting
# from a histogram peak-finding seed, then report per-cluster stats.

from scipy.signal import find_peaks

def detect_position_clusters(roll_arr, bin_width=2.0, min_gap=15.0):
    """
    Find discrete sleep position clusters in roll angle data.
    Returns cluster centres (sorted) and per-sample cluster labels.
    """
    edges = np.arange(roll_arr.min() - bin_width,
                      roll_arr.max() + bin_width * 2, bin_width)
    counts, edges = np.histogram(roll_arr, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Find histogram peaks separated by at least min_gap degrees
    min_dist = max(1, int(min_gap / bin_width))
    peak_idx, props = find_peaks(counts, distance=min_dist,
                                 height=len(roll_arr) * 0.005)  # >0.5% of beats

    if len(peak_idx) == 0:
        # Fallback: single cluster at mean
        return np.array([np.mean(roll_arr)]), np.zeros(len(roll_arr), dtype=int)

    cluster_centres = centres[peak_idx]

    # Assign each beat to nearest cluster centre
    dists = np.abs(roll_arr[:, None] - cluster_centres[None, :])
    labels = np.argmin(dists, axis=1)

    # Refine centres to median of assigned beats (one iteration)
    for k in range(len(cluster_centres)):
        mask = labels == k
        if mask.sum() > 0:
            cluster_centres[k] = np.median(roll_arr[mask])

    # Re-assign after refinement
    dists = np.abs(roll_arr[:, None] - cluster_centres[None, :])
    labels = np.argmin(dists, axis=1)

    return cluster_centres, labels

cluster_centres, cluster_labels = detect_position_clusters(roll_plot)
n_clusters = len(cluster_centres)

print(f"\nDetected {n_clusters} sleep position cluster(s):")
cluster_stats = []
for k, centre in enumerate(cluster_centres):
    mask = cluster_labels == k
    vals = qrs_plot[mask]
    med  = np.median(vals)
    q25, q75 = np.percentile(vals, [25, 75])
    print(f"  Cluster {k+1}: roll={centre:.1f}°  n={mask.sum()}  "
          f"QRS median={med:.0f}µV  IQR=[{q25:.0f}, {q75:.0f}]")
    cluster_stats.append((centre, mask, med, q25, q75))

# =============================================================
# COSINE FIT (only if angle coverage is sufficient)
# =============================================================
fit_roll   = None
fit_qrs    = None
r_squared  = None
popt       = None

COVERAGE_THRESHOLD = 180.0  # degrees — need broad spread for a valid fit
angle_coverage = roll_plot.max() - roll_plot.min()

if not args.no_fit:
    if angle_coverage < COVERAGE_THRESHOLD:
        print(f"\nCosine fit skipped: angle coverage {angle_coverage:.0f}° "
              f"< {COVERAGE_THRESHOLD:.0f}° threshold. "
              f"Use cluster box plot instead.")
    else:
        try:
            from scipy.optimize import curve_fit

            def cosine_model(theta_deg, A, phi_deg, offset):
                theta = np.deg2rad(theta_deg)
                phi   = np.deg2rad(phi_deg)
                return A * np.cos(theta - phi) + offset

            p0 = [np.std(qrs_plot) * np.sqrt(2),
                  roll_plot[np.argmax(qrs_plot)],
                  np.mean(qrs_plot)]

            popt, pcov = curve_fit(cosine_model, roll_plot, qrs_plot,
                                   p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))
            fit_roll = np.linspace(roll_plot.min(), roll_plot.max(), 500)
            fit_qrs  = cosine_model(fit_roll, *popt)

            residuals = qrs_plot - cosine_model(roll_plot, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((qrs_plot - np.mean(qrs_plot))**2)
            r_squared = 1 - ss_res / ss_tot

            print(f"\nCosine fit:  A={popt[0]:.0f}µV  φ={popt[1]:.1f}°  "
                  f"offset={popt[2]:.0f}µV  R²={r_squared:.3f}")
            print(f"  Optimal axis angle (max QRS): {popt[1]:.1f} ± {perr[1]:.1f}°")
            print(f"  Amplitude of modulation:      {abs(popt[0]):.0f} ± {perr[0]:.0f} µV")

        except Exception as e:
            print(f"Cosine fit failed: {e}")

# =============================================================
# PLOT
# =============================================================
import matplotlib.dates as mdates

# Colour-code by time so we can see drift/hysteresis across the night
t_norm = (time_plot - time_plot.min()) / (time_plot.max() - time_plot.min() + 1e-9)
cmap   = plt.cm.plasma

# Layout: scatter (top-left), box plots (top-right), timeline (bottom)
fig = plt.figure(figsize=(13, 9))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1],
                       width_ratios=[3, 1], hspace=0.35, wspace=0.08)
ax     = fig.add_subplot(gs[0, 0])   # scatter
ax_box = fig.add_subplot(gs[0, 1])   # cluster box plots
ax_tl  = fig.add_subplot(gs[1, :])   # timeline (full width)

# --- Scatter: QRS vs roll, coloured by time ---
sc = ax.scatter(roll_plot, qrs_plot,
                c=t_norm, cmap=cmap,
                s=4, alpha=0.35, linewidths=0, zorder=2)

# Highlight cluster centres with vertical lines
cluster_colors = plt.cm.tab10(np.linspace(0, 0.9, n_clusters))
for k, (centre, mask, med, q25, q75) in enumerate(cluster_stats):
    ax.axvline(centre, color=cluster_colors[k], lw=1.2,
               linestyle='--', alpha=0.7,
               label=f'Pos {k+1}: {centre:.0f}°  med={med:.0f}µV  n={mask.sum()}')

if fit_roll is not None:
    ax.plot(fit_roll, fit_qrs, 'r-', lw=2,
            label=f'Cosine fit  R²={r_squared:.3f}  φ={popt[1]:.1f}°',
            zorder=4)

# Colorbar
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label('Time through night')
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
t_range = time_plot.max() - time_plot.min()
cb.set_ticklabels([
    datetime.fromtimestamp(time_plot.min() + f * t_range).strftime('%H:%M')
    for f in [0, 0.25, 0.5, 0.75, 1.0]
])

ax.set_xlabel('Body roll angle (degrees)')
ax.set_ylabel('QRS amplitude (µV)')
date_str = datetime.fromtimestamp(time_plot.min()).strftime('%Y-%m-%d')
ax.set_title(f'QRS Amplitude vs Body Roll Angle  —  {date_str}  '
             f'({np.sum(good)} beats)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# --- Box plots: one box per position cluster ---
box_data   = [qrs_plot[mask] for _, mask, *_ in cluster_stats]
box_labels = [f'Pos {k+1}\n{centre:.0f}°\nn={mask.sum()}'
              for k, (centre, mask, *_) in enumerate(cluster_stats)]

bp = ax_box.boxplot(box_data, patch_artist=True,
                    medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_box.set_xticklabels(box_labels, fontsize=8)
ax_box.set_ylabel('QRS amplitude (µV)')
ax_box.set_title('Per-position')
ax_box.grid(True, axis='y', alpha=0.3)
ax_box.yaxis.tick_right()
ax_box.yaxis.set_label_position('right')

# --- Timeline: roll angle through the night ---
beat_mpl = np.array([mdates.date2num(datetime.fromtimestamp(e))
                     for e in time_plot])
ax_tl.scatter(beat_mpl, roll_plot,
              c=t_norm, cmap=cmap, s=2, alpha=0.5, linewidths=0)
# Mark cluster centres as horizontal lines in timeline
for k, (centre, *_) in enumerate(cluster_stats):
    ax_tl.axhline(centre, color=cluster_colors[k], lw=0.8,
                  linestyle='--', alpha=0.6)
ax_tl.set_ylabel('Roll (°)')
ax_tl.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_tl.xaxis.set_major_locator(mdates.AutoDateLocator())
ax_tl.set_xlabel('Time')
ax_tl.grid(True, alpha=0.3)

if args.save:
    out_path = Path(beats_path).parent / (Path(beats_path).stem + '_roll_plot.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
else:
    plt.show()
