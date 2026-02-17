#!/usr/bin/env python3

"""
Full disclosure ECG display — continuous strips stacked vertically.

Renders the entire overnight ECG recording as rows of continuous
waveform, similar to a Holter monitor full-disclosure printout.
Optionally color-codes background by sleep position.

Output: one PNG per hour (or configurable page duration).

Usage:
  python ecg_disclosure.py <csv_file_or_directory> [options]
  python ecg_disclosure.py C:\\sleep\\20260215 --seconds-per-row 10
  python ecg_disclosure.py C:\\sleep\\20260215 --positions MOT_..._positions.csv

J. Beale  2026-02
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy import signal
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='ECG full disclosure display')
parser.add_argument('input_path', help='ECG CSV file or directory')
parser.add_argument('sample_rate', nargs='?', type=int, default=250,
                    help='Sample rate in sps (default: 250)')
parser.add_argument('--prefiltered', action='store_true',
                    help='Data is already filtered')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNGs but do not display')
parser.add_argument('--seconds-per-row', type=float, default=10, dest='sec_per_row',
                    help='Seconds per strip row (default: 10)')
parser.add_argument('--rows-per-page', type=int, default=60, dest='rows_per_page',
                    help='Rows per page/image (default: 60 = 10 min at 10s/row)')
parser.add_argument('--gain', type=float, default=None,
                    help='Fixed amplitude gain in µV per row height. '
                         'Default: auto (4x median QRS amplitude)')
parser.add_argument('--positions', default=None,
                    help='Position segments CSV for background coloring')
parser.add_argument('--study-end', type=int, default=7, dest='study_end',
                    help='Hour to end display (default: 7 = 7AM)')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

FS = args.sample_rate
SEC_PER_ROW = args.sec_per_row
ROWS_PER_PAGE = args.rows_per_page
SAMPLES_PER_ROW = int(SEC_PER_ROW * FS)

# Filtering
HP_FREQ = 0.5
NOTCH_FREQ = 60
LP_FREQ = 40

# =============================================================
# RESOLVE INPUT FILE
# =============================================================
def find_ecg_csv(directory):
    ecg_pattern = re.compile(r'^ECG_\d{8}_\d{6}\.csv$')
    if not Path(directory).is_dir():
        return None
    for filepath in sorted(Path(directory).iterdir()):
        if filepath.is_file() and ecg_pattern.match(filepath.name):
            return str(filepath)
    return None

def find_positions_csv(directory):
    for filepath in sorted(Path(directory).iterdir()):
        if filepath.is_file() and filepath.name.endswith('_positions.csv'):
            return str(filepath)
    return None

input_path = args.input_path
pos_file = args.positions

if Path(input_path).is_dir():
    directory = input_path
    found = find_ecg_csv(directory)
    if found is None:
        print(f"Error: No ECG_*.csv file found in {input_path}")
        sys.exit(1)
    input_path = found
    print(f"Found ECG file: {input_path}")
    if pos_file is None:
        pos_file = find_positions_csv(directory)
        if pos_file:
            print(f"Found positions: {Path(pos_file).name}")
else:
    directory = str(Path(input_path).parent)

# =============================================================
# LOAD & FILTER ECG
# =============================================================
data_raw = np.loadtxt(input_path, delimiter=",", skiprows=1).flatten()
N = len(data_raw)
print(f"Loaded {N} samples ({N/FS:.1f}s = {N/FS/3600:.1f}h) at {FS} sps")

if args.prefiltered:
    ecg = data_raw.copy()
    print("Using pre-filtered data")
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
        ts = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
        return ts
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

# Get start time
sync_result = load_sync_file(input_path)
if sync_result is not None:
    sync_idx, sync_epoch = sync_result
    t0 = datetime.fromtimestamp(sync_epoch[0])
    print(f"Sync file: start {t0.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    t0 = parse_filename_timestamp(input_path)
    if t0:
        print(f"Start from filename: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        t0 = datetime(2000, 1, 1, 0, 0, 0)
        print("No timestamp; using elapsed time")

# =============================================================
# LOAD POSITION SEGMENTS (optional)
# =============================================================
pos_segments = []
pos_colors_map = {}

if pos_file:
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
                    start_sec = float(parts[1])
                    end_sec = float(parts[2])
                    position = parts[6]
                    # Convert to epoch for matching with ECG timeline
                    seg_start_epoch = pos_t0.timestamp() + start_sec
                    seg_end_epoch = pos_t0.timestamp() + end_sec
                    # Convert to ECG sample indices
                    ecg_start_epoch = t0.timestamp()
                    s_start = int((seg_start_epoch - ecg_start_epoch) * FS)
                    s_end = int((seg_end_epoch - ecg_start_epoch) * FS)
                    s_start = max(0, min(N, s_start))
                    s_end = max(0, min(N, s_end))
                    pos_segments.append({
                        'start': s_start,
                        'end': s_end,
                        'position': position,
                    })
                    if position not in pos_colors_map:
                        pos_colors_map[position] = None

    # Assign colors
    base_colors = [
        (0.85, 0.92, 1.0, 0.5),   # light blue
        (1.0, 0.90, 0.85, 0.5),   # light salmon
        (0.85, 1.0, 0.85, 0.5),   # light green
        (1.0, 1.0, 0.85, 0.5),   # light yellow
        (0.92, 0.85, 1.0, 0.5),   # light purple
    ]
    for i, pos in enumerate(sorted(pos_colors_map.keys())):
        pos_colors_map[pos] = base_colors[i % len(base_colors)]

    print(f"Loaded {len(pos_segments)} position segments: "
          f"{', '.join(f'{k}' for k in sorted(pos_colors_map.keys()))}")

# =============================================================
# DETERMINE AMPLITUDE SCALING
# =============================================================
if args.gain is not None:
    row_amplitude = args.gain
else:
    # Auto-gain: use 4x the median absolute value in the middle of the recording
    # (avoid start/end where electrodes may be disconnecting)
    mid_start = N // 4
    mid_end = 3 * N // 4
    sample_chunk = ecg[mid_start:mid_end]
    # Median of rolling max amplitude (proxy for QRS peaks)
    chunk_len = min(len(sample_chunk), 100 * FS)  # sample up to 100s
    idx = np.random.default_rng(42).choice(len(sample_chunk) - SAMPLES_PER_ROW,
                                            min(200, len(sample_chunk) // SAMPLES_PER_ROW),
                                            replace=False)
    row_maxes = []
    for i in idx:
        seg = sample_chunk[i:i+SAMPLES_PER_ROW]
        row_maxes.append(np.max(seg) - np.min(seg))
    row_amplitude = np.median(row_maxes) * 1.6
    row_amplitude = max(row_amplitude, 200)  # floor at 200 µV

print(f"Row amplitude: {row_amplitude:.0f} µV (half-height = ±{row_amplitude/2:.0f} µV)")

# =============================================================
# DETERMINE STUDY END (sample index)
# =============================================================
study_end_sample = N
if args.study_end is not None:
    # Find sample index corresponding to study_end hour
    for s in range(0, N, FS):
        sample_time = t0 + timedelta(seconds=s / FS)
        h = sample_time.hour
        if h >= args.study_end and h < 20:
            study_end_sample = s
            break

total_rows = int(np.ceil(study_end_sample / SAMPLES_PER_ROW))
total_pages = int(np.ceil(total_rows / ROWS_PER_PAGE))
page_duration = SEC_PER_ROW * ROWS_PER_PAGE

print(f"Rendering {total_rows} rows across {total_pages} pages "
      f"({SEC_PER_ROW:.0f}s/row, {ROWS_PER_PAGE} rows/page = "
      f"{page_duration/60:.0f} min/page)")

# =============================================================
# GET POSITION FOR A SAMPLE RANGE
# =============================================================
def get_position_at(sample_start, sample_end):
    """Return position label for a sample range, or None."""
    mid = (sample_start + sample_end) // 2
    for seg in pos_segments:
        if seg['start'] <= mid <= seg['end']:
            return seg['position']
    return None

# =============================================================
# RENDER PAGES
# =============================================================
stem = str(Path(input_path).parent / Path(input_path).stem)
out_files = []

for page in range(total_pages):
    start_row = page * ROWS_PER_PAGE
    end_row = min(start_row + ROWS_PER_PAGE, total_rows)
    n_rows = end_row - start_row

    if n_rows <= 0:
        break

    # Figure dimensions: width=16" for 10s strips, height scales with rows
    row_height = 0.28  # inches per row
    fig_height = max(4, n_rows * row_height + 1.2)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    page_start_sample = start_row * SAMPLES_PER_ROW
    page_start_time = t0 + timedelta(seconds=page_start_sample / FS)
    page_end_time = t0 + timedelta(seconds=min((end_row) * SAMPLES_PER_ROW, N) / FS)

    ax.set_title(f"ECG Full Disclosure — {Path(input_path).name}    "
                 f"{page_start_time.strftime('%H:%M')}–{page_end_time.strftime('%H:%M')}    "
                 f"Page {page+1}/{total_pages}",
                 fontsize=11, fontweight='bold', loc='left')

    # X-axis: 0 to SEC_PER_ROW
    x = np.arange(SAMPLES_PER_ROW) / FS

    for ri in range(n_rows):
        row_idx = start_row + ri
        s_start = row_idx * SAMPLES_PER_ROW
        s_end = min(s_start + SAMPLES_PER_ROW, study_end_sample)

        if s_start >= study_end_sample:
            break

        # Y offset: row 0 at top, increasing downward
        y_center = -ri

        # Position background coloring
        pos = get_position_at(s_start, s_end)
        if pos and pos in pos_colors_map:
            bg_color = pos_colors_map[pos]
            ax.axhspan(y_center - 0.48, y_center + 0.48,
                       color=bg_color, zorder=0)

        # Extract and normalize signal for this row
        row_data = ecg[s_start:s_end]
        if len(row_data) < SAMPLES_PER_ROW:
            # Pad short final row
            row_data = np.pad(row_data, (0, SAMPLES_PER_ROW - len(row_data)),
                              constant_values=np.nan)

        # Scale to fit in row: signal / row_amplitude maps to ±0.5
        y_scaled = row_data / row_amplitude + y_center

        # Clip to row bounds to prevent bleed into adjacent rows
        y_clipped = np.clip(y_scaled, y_center - 0.48, y_center + 0.48)

        # Plot
        ax.plot(x[:len(y_clipped)], y_clipped, color='black',
                linewidth=0.3, alpha=0.85, zorder=2)

        # Time label on left margin
        row_time = t0 + timedelta(seconds=s_start / FS)
        time_str = row_time.strftime('%H:%M:%S')
        ax.text(-0.15, y_center, time_str, fontsize=5.5,
                va='center', ha='right', family='monospace', color='#444444')

        # Position label on right margin
        if pos:
            ax.text(SEC_PER_ROW + 0.1, y_center, pos, fontsize=5.5,
                    va='center', ha='left', family='monospace',
                    color='#666666')

    # Formatting
    ax.set_xlim(-0.05, SEC_PER_ROW + 0.05)
    ax.set_ylim(-n_rows + 0.5, 0.6)
    ax.set_xlabel('Seconds', fontsize=9)

    # Light grid at 1-second intervals
    for s in range(int(SEC_PER_ROW) + 1):
        lw = 0.5 if s % 5 == 0 else 0.15
        alpha = 0.4 if s % 5 == 0 else 0.15
        ax.axvline(s, color='#cc8888', linewidth=lw, alpha=alpha, zorder=1)

    # Horizontal separators every row
    for ri in range(n_rows + 1):
        ax.axhline(-ri + 0.5, color='#cc8888', linewidth=0.15,
                   alpha=0.15, zorder=1)

    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Position legend if we have segments
    if pos_colors_map:
        legend_patches = [mpatches.Patch(facecolor=pos_colors_map[p],
                                         edgecolor='gray', linewidth=0.5,
                                         label=f'Pos {p}')
                          for p in sorted(pos_colors_map.keys())]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=7,
                  framealpha=0.8)

    fig.tight_layout()
    out_path = f"{stem}_disclosure_p{page+1:02d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    out_files.append(out_path)
    plt.close(fig)

print(f"\nSaved {len(out_files)} pages:")
for f in out_files:
    print(f"  {Path(f).name}")

if not args.no_plot and out_files:
    # Reopen first page for display
    fig = plt.figure(figsize=(16, 10))
    img = plt.imread(out_files[0])
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Page 1 of {len(out_files)} — close to exit")
    plt.tight_layout()
    plt.show()
