#!/usr/bin/env python3

"""
Sleep overview — multi-panel time-aligned dashboard.

Reads per-beat CSV from analyze_ecg.py and optional sensor data files,
then produces a stacked time-aligned plot with wall-clock time axis.

Panels (enabled when data is available):
  - Heart Rate (from *_beats.csv)
  - Breath Envelope (from MOT_*_breath.csv)
  - Breathing Noise Envelope (from *_noise_envelope.csv)
  - SpO2 (from Checkme O2 or similar pulse oximeter CSV)
  - (future: body position, etc.)

Usage:
  python plot_sleep_overview.py <beats_csv> [--breath FILE] [--noise-env FILE] [--spo2 FILE] [--no-plot]
  python plot_sleep_overview.py ECG_20260213_231431_beats.csv --breath MOT_20260213_breath.csv

J. Beale  2026-02
"""

import numpy as np
import matplotlib
import argparse
import csv
import io
import re
import sys
from pathlib import Path
from datetime import datetime
import glob

# =============================================================
# PARSE ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='Sleep overview multi-panel dashboard')
parser.add_argument('beats_csv', help='Per-beat CSV from analyze_ecg.py (*_beats.csv)')
parser.add_argument('--breath', default=None,
                    help='Tilt-based breath CSV (MOT_*_breath.csv). '
                         'If omitted, auto-detects in same directory.')
parser.add_argument('--spo2', default=None,
                    help='Pulse oximeter CSV (e.g. Checkme O2 export). '
                         'If omitted, auto-detects in same directory.')
parser.add_argument('--noise-env', default=None, dest='noise_env',
                    help='Breathing noise envelope CSV (*_noise_envelope.csv). '
                         'If omitted, auto-detects in same directory.')
parser.add_argument('--no-plot', action='store_true', dest='no_plot',
                    help='Save PNG but do not display')
args = parser.parse_args()

if args.no_plot:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =============================================================
# RESOLVE INPUT FILE (support both file and directory)
# =============================================================
def find_beats_csv(directory):
    """
    Search directory for the first ECG_<YYYYMMDD_HHMMSS>_beats.csv file.
    Returns the full path to the matching file, or None if not found.
    """
    beats_pattern = re.compile(r'^ECG_\d{8}_\d{6}_beats\.csv$')
    if not Path(directory).is_dir():
        return None
    
    files = sorted(Path(directory).iterdir())
    for filepath in files:
        if filepath.is_file() and beats_pattern.match(filepath.name):
            return str(filepath)
    return None

# Check if input is a directory; if so, find the ECG beats CSV file
if Path(args.beats_csv).is_dir():
    found_file = find_beats_csv(args.beats_csv)
    if found_file is None:
        print(f"Error: No ECG_*_beats.csv file found in {args.beats_csv}")
        sys.exit(1)
    args.beats_csv = found_file
    print(f"Found beats CSV: {args.beats_csv}")

def load_beats_csv(filepath):
    """Load per-beat CSV exported by analyze_ecg.py.
    Returns dict with numpy arrays keyed by column name, plus metadata."""
    metadata = {}
    header = None
    rows = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                # Parse metadata comments: "# key: value"
                m = re.match(r'#\s*(\w+):\s*(.*)', line)
                if m:
                    metadata[m.group(1)] = m.group(2).strip()
                continue
            if header is None:
                header = line.strip().split(',')
                continue
            rows.append(line.strip().split(','))

    data = {}
    for col_idx, col_name in enumerate(header):
        vals = []
        for row in rows:
            v = row[col_idx] if col_idx < len(row) else ''
            if v == '':
                vals.append(np.nan)
            elif col_name in ('is_artifact', 'is_pvc'):
                vals.append(bool(int(v)))
            elif col_name == 'sample_idx':
                vals.append(int(v))
            else:
                vals.append(float(v))
        if col_name in ('is_artifact', 'is_pvc'):
            data[col_name] = np.array(vals, dtype=bool)
        elif col_name == 'sample_idx':
            data[col_name] = np.array(vals, dtype=int)
        else:
            data[col_name] = np.array(vals, dtype=float)

    data['_metadata'] = metadata
    return data


def load_breath_csv(filepath):
    """Load tilt-based breathing CSV (MOT_*_breath.csv).
    Returns dict with 'epoch_s', 'envelope_deg', and optionally
    'breaths_per_min', 'roll_deg' as numpy arrays."""
    # Parse start time from comment line
    start_time_str = None
    with open(filepath, 'r') as f:
        first = f.readline().strip()
        if first.startswith('# start '):
            start_time_str = first[8:].split('sync_millis')[0].strip()

    if not start_time_str:
        print(f"Warning: no start time in {filepath}")
        return None

    try:
        t0_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        t0_epoch = t0_dt.timestamp()
    except ValueError:
        print(f"Warning: cannot parse start time '{start_time_str}'")
        return None

    # Find header line (starts with 'seconds,')
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('seconds,'):
            header_idx = i
            break

    if header_idx is None:
        print(f"Warning: no header found in {filepath}")
        return None

    reader = csv.DictReader(io.StringIO(''.join(lines[header_idx:])))
    seconds = []
    envelope = []
    breaths = []
    roll = []

    for row in reader:
        try:
            sec = float(row['seconds'])
            env = float(row['envelope_deg'])
            seconds.append(sec)
            envelope.append(env)
            bpm = row.get('breaths_per_min', '').strip()
            breaths.append(float(bpm) if bpm else np.nan)
            r = row.get('roll_deg', '').strip()
            roll.append(float(r) if r else np.nan)
        except (ValueError, KeyError):
            continue

    result = {
        'epoch_s': t0_epoch + np.array(seconds),
        'envelope_deg': np.array(envelope),
    }
    breaths = np.array(breaths)
    if not np.all(np.isnan(breaths)):
        result['breaths_per_min'] = breaths
    roll = np.array(roll)
    if not np.all(np.isnan(roll)):
        result['roll_deg'] = roll

    print(f"Loaded breath data: {len(seconds)} samples, "
          f"start {start_time_str}")
    return result


def auto_find_breath(beats_path):
    """Look for MOT_*_breath.csv in the same directory as the beats CSV."""
    d = Path(beats_path).parent
    matches = sorted(d.glob('MOT_*_breath.csv'))
    if matches:
        if len(matches) > 1:
            print(f"Note: found {len(matches)} breath files, using {matches[-1].name}")
        return str(matches[-1])
    return None


def load_spo2_csv(filepath):
    """Load pulse oximeter CSV. Supports Checkme O2 format and any CSV with
    'Oxygen Level' or 'SpO2' column and a parseable time column.
    Returns dict with 'epoch_s' and 'spo2_pct' numpy arrays, or None."""

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Find the header line
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if 'oxygen level' in low or 'spo2' in low:
            header_idx = i
            break

    if header_idx is None:
        return None

    reader = csv.DictReader(io.StringIO(''.join(lines[header_idx:])))
    fieldnames = reader.fieldnames
    if fieldnames is None:
        return None

    # Identify columns (case-insensitive)
    fn_lower = {f.strip().lower(): f for f in fieldnames}
    time_col = None
    spo2_col = None

    for key, orig in fn_lower.items():
        if key == 'time':
            time_col = orig
        elif key in ('oxygen level', 'spo2', 'spo2_pct', 'spo2%'):
            spo2_col = orig

    if time_col is None or spo2_col is None:
        print(f"Warning: could not identify Time and SpO2 columns in {filepath}")
        return None

    # Checkme O2 format: "HH:MM:SS DD/MM/YYYY"
    # Try several common timestamp formats
    TIME_FMTS = [
        '%H:%M:%S %d/%m/%Y',    # Checkme O2: 23:10:23 13/02/2026
        '%Y-%m-%d %H:%M:%S',    # ISO
        '%m/%d/%Y %H:%M:%S',    # US
        '%d/%m/%Y %H:%M:%S',    # EU
        '%H:%M:%S %m/%d/%Y',
    ]

    epochs = []
    spo2_vals = []
    fmt_found = None

    for row in reader:
        try:
            time_str = row[time_col].strip()
            spo2_val = int(row[spo2_col].strip())

            # Parse timestamp
            if fmt_found:
                dt = datetime.strptime(time_str, fmt_found)
            else:
                dt = None
                for fmt in TIME_FMTS:
                    try:
                        dt = datetime.strptime(time_str, fmt)
                        fmt_found = fmt
                        break
                    except ValueError:
                        continue
                if dt is None:
                    continue

            epochs.append(dt.timestamp())
            # Treat 0 as invalid (finger off sensor)
            spo2_vals.append(spo2_val if spo2_val > 0 else np.nan)

        except (ValueError, KeyError):
            continue

    if len(epochs) < 2:
        print(f"Warning: too few valid rows in {filepath}")
        return None

    result = {
        'epoch_s': np.array(epochs),
        'spo2_pct': np.array(spo2_vals, dtype=float),
    }
    t0_str = datetime.fromtimestamp(epochs[0]).strftime('%Y-%m-%d %H:%M:%S')
    n_valid = np.sum(~np.isnan(result['spo2_pct']))
    n_total = len(epochs)
    print(f"Loaded SpO2 data: {n_total} samples ({n_valid} valid), start {t0_str}")
    return result


def auto_find_spo2(beats_path):
    """Look for pulse oximeter CSV in the same directory.
    First tries Checkme O2 pattern, then sniffs CSV headers."""
    d = Path(beats_path).parent

    # Try Checkme O2 pattern first
    matches = sorted(d.glob('Checkme*O2*.csv'))
    if matches:
        if len(matches) > 1:
            print(f"Note: found {len(matches)} Checkme O2 files, using {matches[-1].name}")
        return str(matches[-1])

    # Fall back: sniff all CSVs for 'Oxygen Level' or 'SpO2' header
    for csv_path in sorted(d.glob('*.csv')):
        # Skip known non-SpO2 files
        name = csv_path.name.lower()
        if '_beats' in name or '_sync' in name or 'mot_' in name or 'ecg_' in name:
            continue
        if 'breathing_analysis' in name:
            continue
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    low = line.lower()
                    if 'oxygen level' in low or 'spo2' in low:
                        print(f"Auto-detected SpO2 file by header: {csv_path.name}")
                        return str(csv_path)
                    break  # only check first non-comment line
        except Exception:
            continue

    return None


def load_noise_envelope_csv(filepath):
    """Load breathing noise envelope CSV (*_noise_envelope.csv).
    Returns dict with 'epoch_s' and 'noise_db' numpy arrays, or None."""
    epochs = []
    db_vals = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.strip().startswith('epoch_s'):
                continue  # header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    epochs.append(float(parts[0]))
                    db_vals.append(float(parts[1]))
                except ValueError:
                    continue

    if len(epochs) < 2:
        print(f"Warning: too few valid rows in {filepath}")
        return None

    result = {
        'epoch_s': np.array(epochs),
        'noise_db': np.array(db_vals),
    }
    t0_str = datetime.fromtimestamp(epochs[0]).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Loaded noise envelope: {len(epochs)} samples (1 Hz), start {t0_str}")
    return result


def auto_find_noise_envelope(beats_path):
    """Look for *_noise_envelope.csv in the same directory."""
    d = Path(beats_path).parent
    matches = sorted(d.glob('*_noise_envelope.csv'))
    if matches:
        if len(matches) > 1:
            print(f"Note: found {len(matches)} noise envelope files, using {matches[-1].name}")
        return str(matches[-1])
    return None


# =============================================================
# TIME CONVERSION UTILITIES
# =============================================================

def epoch_to_mpl(epoch_arr, ref_epoch):
    """Convert unix epoch array to matplotlib date numbers."""
    ref_mpl = mdates.date2num(datetime.fromtimestamp(ref_epoch))
    return ref_mpl + (np.asarray(epoch_arr) - ref_epoch) / 86400.0


class AdaptiveTimeTicker(mdates.ticker.Locator):
    """Tick locator that switches to second-level ticks when zoomed in < 5 min."""
    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        span_sec = (vmax - vmin) * 86400.0
        if span_sec <= 300:
            return self._second_ticks(vmin, vmax, span_sec)
        else:
            loc = mdates.AutoDateLocator(minticks=6, maxticks=15)
            loc.set_axis(self.axis)
            return loc()

    def _second_ticks(self, vmin, vmax, span_sec):
        from datetime import timedelta
        if span_sec <= 30:
            interval = 5
        elif span_sec <= 60:
            interval = 10
        elif span_sec <= 120:
            interval = 15
        elif span_sec <= 300:
            interval = 30
        else:
            interval = 60

        dt_min = mdates.num2date(vmin).replace(microsecond=0)
        dt_max = mdates.num2date(vmax)
        cur = dt_min.replace(second=(dt_min.second // interval) * interval)
        ticks = []
        while cur <= dt_max:
            ticks.append(mdates.date2num(cur))
            cur += timedelta(seconds=interval)
        return ticks


class AdaptiveTimeFormatter(mdates.ticker.Formatter):
    """Formatter that shows HH:MM:SS when zoomed in < 5 min, HH:MM otherwise."""
    def __init__(self):
        self._fmt_hms = mdates.DateFormatter('%H:%M:%S')
        self._fmt_hm  = mdates.DateFormatter('%H:%M')

    def __call__(self, x, pos=None):
        ax = self.axis
        vmin, vmax = ax.get_view_interval()
        span_sec = (vmax - vmin) * 86400.0
        if span_sec <= 300:
            return self._fmt_hms(x, pos)
        else:
            return self._fmt_hm(x, pos)


def format_time_axis(ax, duration_sec=None):
    """Apply adaptive time tick labels that switch to HH:MM:SS on zoom."""
    ax.xaxis.set_major_locator(AdaptiveTimeTicker())
    ax.xaxis.set_major_formatter(AdaptiveTimeFormatter())
    ax.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    ax.set_xlabel('Time')


# =============================================================
# PANEL DEFINITIONS
# =============================================================
# Each panel is a function: panel_func(ax, data_dict, ref_epoch)
# Returns True if it drew something, False to skip.

def panel_heart_rate(ax, data, ref_epoch):
    """Heart rate panel from beats CSV."""
    beats = data.get('beats')
    if beats is None:
        return False

    epoch = beats['epoch_s']
    hr = beats['hr_bpm']
    hr_avg = beats.get('hr_avg_bpm')
    is_art = beats.get('is_artifact', np.zeros(len(hr), dtype=bool))
    is_pvc = beats.get('is_pvc', np.zeros(len(hr), dtype=bool))

    t_wall = epoch_to_mpl(epoch, ref_epoch)
    valid = ~np.isnan(hr)
    clean = valid & ~is_art & ~is_pvc
    dirty = valid & is_art

    if np.any(dirty):
        ax.scatter(t_wall[dirty], hr[dirty], s=3, c='silver', alpha=0.7,
                   zorder=1, label='Artifact')
    ax.scatter(t_wall[clean], hr[clean], s=4, c='crimson', alpha=0.6,
               zorder=2, label='Heart Rate (ECG R-R)')

    # Rolling average
    if hr_avg is not None:
        v = ~np.isnan(hr_avg)
        if np.any(v):
            ax.plot(t_wall[v], hr_avg[v], color='navy', linewidth=1.5,
                    alpha=0.7, zorder=3, label='Avg')

    ax.set_ylabel('bpm', fontsize=9)
    ax.set_title('Heart Rate (ECG R-R)', fontsize=10, loc='left')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    return True


def panel_breath_envelope(ax, data, ref_epoch):
    """Breath envelope panel from tilt/IMU data."""
    breath = data.get('breath')
    if breath is None:
        return False

    t_wall = epoch_to_mpl(breath['epoch_s'], ref_epoch)
    env = breath['envelope_deg']

    ax.fill_between(t_wall, 0, env, color='teal', alpha=0.25, zorder=1)
    ax.plot(t_wall, env, color='teal', linewidth=0.5, alpha=0.7, zorder=2)

    ax.set_ylabel('degrees', fontsize=9)
    ax.set_title('Breathing Motion Envelope', fontsize=10, loc='left')
    ax.set_ylim(0, 6) # just clip on the larger peaks
    ax.grid(True, alpha=0.3)
    return True


def panel_spo2(ax, data, ref_epoch):
    """SpO2 panel from pulse oximeter data."""
    spo2 = data.get('spo2')
    if spo2 is None:
        return False

    t_wall = epoch_to_mpl(spo2['epoch_s'], ref_epoch)
    vals = spo2['spo2_pct']
    valid = ~np.isnan(vals)

    ax.plot(t_wall[valid], vals[valid], color='dodgerblue', linewidth=1.5,
            alpha=0.8, zorder=2)

    ax.set_ylabel('SpO₂ %', fontsize=9)
    ax.set_title('SpO₂', fontsize=10, loc='left')
    ax.set_ylim(max(75, np.nanmin(vals) - 2) if np.any(valid) else 75, 102)
    ax.axhline(90, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.grid(True, alpha=0.3)
    return True


def panel_noise_envelope(ax, data, ref_epoch):
    """Breathing noise envelope panel (dB relative to noise floor)."""
    nenv = data.get('noise_env')
    if nenv is None:
        return False

    t_wall = epoch_to_mpl(nenv['epoch_s'], ref_epoch)
    db = nenv['noise_db']

    ax.fill_between(t_wall, 0, db, color='mediumpurple', alpha=0.3, zorder=1)
    ax.plot(t_wall, db, color='mediumpurple', linewidth=0.6, alpha=0.8, zorder=2)

    ax.set_ylabel('dB', fontsize=9)
    ax.set_title('Breathing Noise Envelope (dB re noise floor)', fontsize=10, loc='left')
    ax.set_ylim(0, max(10, np.max(db) * 1.1))
    ax.grid(True, alpha=0.3)
    return True


# Registry of available panels in display order.
# Each entry: (name, panel_function, default_height_ratio)
PANEL_REGISTRY = [
    ('heart_rate',       panel_heart_rate,       3),
    ('breath_envelope',  panel_breath_envelope,  2),
    ('noise_envelope',   panel_noise_envelope,   2),
    ('spo2',             panel_spo2,             2),
    # Future panels:
    # ('body_position',  panel_body_position,    1),
]


# =============================================================
# MAIN
# =============================================================

# Load data sources
data = {}

print(f"Loading beats: {args.beats_csv}")
data['beats'] = load_beats_csv(args.beats_csv)

# Breath file: explicit, auto-detect, or skip
breath_path = args.breath
if breath_path is None:
    breath_path = auto_find_breath(args.beats_csv)
if breath_path:
    print(f"Loading breath: {breath_path}")
    data['breath'] = load_breath_csv(breath_path)
else:
    print("No breath file found (use --breath to specify)")

# SpO2 file: explicit, auto-detect, or skip
spo2_path = args.spo2
if spo2_path is None:
    spo2_path = auto_find_spo2(args.beats_csv)
if spo2_path:
    print(f"Loading SpO2: {spo2_path}")
    spo2_data = load_spo2_csv(spo2_path)
    if spo2_data is not None:
        data['spo2'] = spo2_data
else:
    print("No SpO2 file found (use --spo2 to specify)")

# Noise envelope file: explicit, auto-detect, or skip
noise_env_path = args.noise_env
if noise_env_path is None:
    noise_env_path = auto_find_noise_envelope(args.beats_csv)
if noise_env_path:
    print(f"Loading noise envelope: {noise_env_path}")
    nenv_data = load_noise_envelope_csv(noise_env_path)
    if nenv_data is not None:
        data['noise_env'] = nenv_data
else:
    print("No noise envelope file found (use --noise-env to specify)")

# Reference epoch for time conversion (from beats data)
ref_epoch = data['beats']['epoch_s'][0]

# Determine which panels have data
active_panels = []
for name, func, height in PANEL_REGISTRY:
    if name == 'heart_rate' and 'beats' in data:
        active_panels.append((name, func, height))
    elif name == 'breath_envelope' and 'breath' in data:
        active_panels.append((name, func, height))
    elif name == 'noise_envelope' and 'noise_env' in data:
        active_panels.append((name, func, height))
    elif name == 'spo2' and 'spo2' in data:
        active_panels.append((name, func, height))

if not active_panels:
    print("No data available for any panel.")
    import sys; sys.exit(1)

print(f"Panels: {', '.join(name for name, _, _ in active_panels)}")

# Build figure
n_panels = len(active_panels)
height_ratios = [h for _, _, h in active_panels]
fig_height = sum(height_ratios) * 1.5 + 1

fig, axes = plt.subplots(n_panels, 1, figsize=(16, fig_height), sharex=True,
                          gridspec_kw={'height_ratios': height_ratios})
if n_panels == 1:
    axes = [axes]

stem = str(Path(args.beats_csv).parent / Path(args.beats_csv).stem)
# Strip _beats suffix for cleaner output name
stem = re.sub(r'_beats$', '', stem)
source_name = data['beats']['_metadata'].get('source', Path(args.beats_csv).name)
fig.suptitle(f"Sleep Overview — {source_name}", fontsize=12, fontweight='bold')

# Calculate total duration for axis formatting decision
beats_start = data['beats']['epoch_s'][0]
beats_end = data['beats']['epoch_s'][-1]
duration_sec = beats_end - beats_start

for i, (name, func, _) in enumerate(active_panels):
    func(axes[i], data, ref_epoch)
    axes[i].fmt_xdata = mdates.DateFormatter('%H:%M:%S')

# Format the shared time axis on the bottom panel
format_time_axis(axes[-1], duration_sec)

# Lock all panels to the time range of the heartrate (beats) data
xlim_start = epoch_to_mpl(np.array([beats_start]), ref_epoch)[0]
xlim_end   = epoch_to_mpl(np.array([beats_end]),   ref_epoch)[0]
axes[0].set_xlim(xlim_start, xlim_end)

fig.tight_layout()



out_path = f"{stem}_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")

if not args.no_plot:
    plt.show()
