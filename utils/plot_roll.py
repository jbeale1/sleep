#!/usr/bin/env python3

"""
Plot body roll angle vs time from MOT_*_breath.csv file.

Usage:
  python plot_roll.py <csv_file_or_directory> [--no-plot]
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

parser = argparse.ArgumentParser(description='Plot body roll angle vs time')
parser.add_argument('input_path', help='CSV file or directory containing MOT_*_breath.csv')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNG but do not display')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

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
# Parse start time from comment line
t0 = None
with open(input_path, 'r') as f:
    first_line = f.readline().strip()
    m = re.search(r'start\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', first_line)
    if m:
        t0 = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')

data = np.genfromtxt(input_path, delimiter=',', names=True, skip_header=1)
seconds = data['seconds']
roll = data['roll_deg']

N = len(seconds)
print(f"Loaded {N} samples ({seconds[-1]-seconds[0]:.0f}s) from {Path(input_path).name}")

# =============================================================
# PLOT
# =============================================================
if t0 is not None:
    # Convert to matplotlib datetime
    times = [mdates.date2num(t0 + timedelta(seconds=float(s))) for s in seconds]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(times, roll, color='steelblue', linewidth=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=15))
    ax.set_xlabel('Time')
    title_base = f"{Path(input_path).name}  (start {t0.strftime('%Y-%m-%d %H:%M')})"
else:
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(seconds / 3600, roll, color='steelblue', linewidth=0.6)
    ax.set_xlabel('Hours')
    title_base = Path(input_path).name

ax.set_ylabel('Roll angle (°)')
ax.set_title(f'Body Roll Angle — {title_base}', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

fig.tight_layout()
stem = str(Path(input_path).parent / Path(input_path).stem)
out_png = f"{stem}_roll.png"
fig.savefig(out_png, dpi=150, bbox_inches='tight')

if not args.no_plot:
    plt.show()
