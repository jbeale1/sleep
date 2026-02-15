#!/usr/bin/env python3

"""
SpO2 overnight analysis — summary statistics from pulse oximetry CSV.

Reads CSV from Checkme O2 Ultra (or similar) with columns:
  Time, Oxygen Level, Pulse Rate, Motion

Metrics:
  - Mean, median, min SpO2
  - SpO2 distribution (1% bins)
  - ODI-3, ODI-4 (oxygen desaturation index)
  - Time below 90%, 88%
  - Desaturation event summary (count, mean depth, mean duration, total area)
  - Delta index (CT90-style variability)
  - Artifact/invalid sample count

Usage:
  python analyze_spo2.py <csv_file_or_directory> [--study-end 7]
  python analyze_spo2.py "Checkme O2 Ultra 2355_20260214220511.csv"
  python analyze_spo2.py C:\\Users\\beale\\Documents\\2026-sleep\\20260215

J. Beale  2026-02
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

# =============================================================
# PARSE ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='SpO2 overnight analysis')
parser.add_argument('input_path', help='CSV file or directory containing SpO2 data')
parser.add_argument('--study-end', type=int, default=7, dest='study_end',
                    help='Hour (0-23) to end study window (default: 7 = 7AM)')
parser.add_argument('--save-summary', action='store_true', dest='save_summary',
                    help='Save summary statistics to text file')
args = parser.parse_args()

STUDY_END_HOUR = args.study_end

# =============================================================
# RESOLVE INPUT FILE
# =============================================================
def find_spo2_csv(directory):
    """Search directory for first Checkme O2 Ultra CSV file."""
    pattern = re.compile(r'^Checkme O2 Ultra.*\.csv$', re.IGNORECASE)
    if not Path(directory).is_dir():
        return None
    files = sorted(Path(directory).iterdir())
    for filepath in files:
        if filepath.is_file() and pattern.match(filepath.name):
            return str(filepath)
    return None

input_path = args.input_path
if Path(input_path).is_dir():
    found = find_spo2_csv(input_path)
    if found is None:
        print(f"Error: No Checkme O2 Ultra CSV found in {input_path}")
        sys.exit(1)
    input_path = found
    print(f"Found SpO2 file: {input_path}")

# =============================================================
# LOAD CSV
# =============================================================
timestamps = []
spo2_raw = []
pulse_raw = []
motion_raw = []

with open(input_path, 'r') as f:
    header = f.readline().strip()
    for line_num, line in enumerate(f, start=2):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            dt = datetime.strptime(parts[0].strip(), '%H:%M:%S %d/%m/%Y')
            spo2_val = int(parts[1].strip())
            pulse_val = int(parts[2].strip())
            motion_val = int(parts[3].strip())
            timestamps.append(dt)
            spo2_raw.append(spo2_val)
            pulse_raw.append(pulse_val)
            motion_raw.append(motion_val)
        except (ValueError, IndexError):
            continue

timestamps = np.array(timestamps)
spo2 = np.array(spo2_raw, dtype=float)
pulse = np.array(pulse_raw, dtype=float)
motion = np.array(motion_raw, dtype=float)
N = len(spo2)

if N < 10:
    print(f"Error: Only {N} valid samples found.")
    sys.exit(1)

# Elapsed seconds from start
t0 = timestamps[0]
elapsed = np.array([(ts - t0).total_seconds() for ts in timestamps])
duration_sec = elapsed[-1] - elapsed[0]
duration_hr = duration_sec / 3600

print(f"Loaded {N} samples ({duration_sec:.0f}s = {duration_hr:.1f}h) from {Path(input_path).name}")
print(f"Start: {t0.strftime('%Y-%m-%d %H:%M:%S')}  "
      f"End: {timestamps[-1].strftime('%H:%M:%S')}")

# =============================================================
# STUDY WINDOW MASK (exclude post-study data)
# =============================================================
in_study = np.ones(N, dtype=bool)
for i in range(N):
    h = timestamps[i].hour
    if h >= STUDY_END_HOUR and h < 20:  # 7AM-8PM is outside overnight study
        in_study[i] = False

n_post_study = np.sum(~in_study)
if n_post_study > 0:
    print(f"Study window: {np.sum(in_study)} samples before {STUDY_END_HOUR}:00 AM "
          f"({n_post_study} post-study samples masked)")

# =============================================================
# ARTIFACT DETECTION
# =============================================================
# Flag samples where SpO2 is 0 or implausibly low (probe off),
# or where rate of change is impossibly fast.

is_artifact = np.zeros(N, dtype=bool)

# Zero or very low readings (probe disconnected)
is_artifact |= (spo2 <= 50)

# Impossible rate of change: >4%/sec sustained for >=1 sample
# (physiological desaturation is typically <1%/sec)
if N > 1:
    dspo2 = np.abs(np.diff(spo2))
    fast_change = np.zeros(N, dtype=bool)
    fast_change[1:] = dspo2 > 4
    # Also flag the adjacent sample
    fast_change[:-1] |= dspo2 > 4
    is_artifact |= fast_change

# Propagate: if SpO2 jumps from artifact to valid, give 3s settling time
artifact_settle = 3  # seconds
artifact_idx = np.where(is_artifact)[0]
for idx in artifact_idx:
    settle_end = min(N, idx + artifact_settle + 1)
    is_artifact[idx:settle_end] = True

n_artifact = np.sum(is_artifact)
is_valid = ~is_artifact & in_study

print(f"Artifact samples: {n_artifact}/{N} ({100*n_artifact/N:.1f}%)")

spo2_valid = spo2[is_valid]
n_valid = len(spo2_valid)

if n_valid < 10:
    print("Error: insufficient valid samples after artifact removal.")
    sys.exit(1)

study_duration_sec = np.sum(is_valid)  # 1 sample = 1 second
study_duration_hr = study_duration_sec / 3600

# =============================================================
# BASIC STATISTICS
# =============================================================
spo2_mean = np.mean(spo2_valid)
spo2_median = np.median(spo2_valid)
spo2_std = np.std(spo2_valid)
spo2_min = np.min(spo2_valid)

# Nadir: require >=2 consecutive seconds at the value
spo2_nadir = spo2_min
nadir_time_str = ""
if N > 1:
    # Find minimum value that persists for >=2 consecutive valid samples
    for test_val in range(int(spo2_min), int(spo2_min) + 5):
        # Find runs at this value
        at_val = (spo2 == test_val) & is_valid
        if np.sum(at_val) < 2:
            continue
        # Check for consecutive pairs
        consecutive = False
        idx_at = np.where(at_val)[0]
        for j in range(len(idx_at) - 1):
            if idx_at[j+1] - idx_at[j] == 1:
                consecutive = True
                nadir_idx = idx_at[j]
                break
        if consecutive:
            spo2_nadir = test_val
            nadir_time_str = f" at {timestamps[nadir_idx].strftime('%H:%M:%S')}"
            break

# =============================================================
# SpO2 DISTRIBUTION (1% bins)
# =============================================================
bin_edges = list(range(80, 101)) + [101]  # 80,81,...,99,100,101
bin_labels = [f"{v}" for v in range(80, 100)] + ['100']
counts, _ = np.histogram(spo2_valid, bins=bin_edges)
pcts = 100.0 * counts / n_valid

# Also count anything below 80
below_80 = np.sum(spo2_valid < 80)
below_80_pct = 100.0 * below_80 / n_valid

# =============================================================
# TIME BELOW THRESHOLDS
# =============================================================
def time_below(threshold):
    """Seconds and percentage of valid study time below threshold."""
    n_below = np.sum(spo2_valid <= threshold)
    return n_below, 100.0 * n_below / n_valid

t90_sec, t90_pct = time_below(90)
t88_sec, t88_pct = time_below(88)
t85_sec, t85_pct = time_below(85)

# =============================================================
# DESATURATION EVENT DETECTION (ODI)
# =============================================================
# Baseline: 120-second trailing maximum of valid SpO2
# (max better than median for detecting drops from recent stable level)
# Event: SpO2 drops >= threshold below baseline for >= 10 seconds

BASELINE_WINDOW = 120  # seconds
DESAT_DURATION_MIN = 10  # seconds minimum
DESAT_THRESHOLDS = [3, 4]  # for ODI-3 and ODI-4

# Compute rolling baseline (trailing max over valid samples)
baseline = np.full(N, np.nan)
# Use a simple approach: for each second, max of valid spo2 in trailing window
valid_spo2_filled = spo2.copy()
valid_spo2_filled[~is_valid] = np.nan

for i in range(N):
    if not is_valid[i]:
        continue
    win_start = max(0, i - BASELINE_WINDOW)
    win = valid_spo2_filled[win_start:i+1]
    win_clean = win[~np.isnan(win)]
    if len(win_clean) >= 5:
        baseline[i] = np.max(win_clean)
    elif i > 0 and not np.isnan(baseline[i-1]):
        baseline[i] = baseline[i-1]

# Detect desaturation events for each threshold
class DesatEvent:
    def __init__(self, start, end, depth, baseline_val):
        self.start = start       # sample index
        self.end = end           # sample index
        self.depth = depth       # max drop below baseline
        self.duration = end - start  # seconds (1 sps)
        self.baseline = baseline_val
        self.nadir = baseline_val - depth
        # Area: integral of (baseline - spo2) during event
        self.area = 0.0

def detect_desaturations(threshold):
    """Detect desaturation events with given threshold (% drop from baseline)."""
    events = []
    in_event = False
    event_start = 0
    event_max_drop = 0
    event_baseline = 0

    for i in range(N):
        if not is_valid[i] or np.isnan(baseline[i]):
            if in_event:
                # End event at artifact/invalid
                dur = i - event_start
                if dur >= DESAT_DURATION_MIN and event_max_drop >= threshold:
                    evt = DesatEvent(event_start, i, event_max_drop, event_baseline)
                    # Compute area
                    for j in range(event_start, i):
                        if is_valid[j] and not np.isnan(baseline[j]):
                            drop = baseline[j] - spo2[j]
                            if drop > 0:
                                evt.area += drop
                    events.append(evt)
                in_event = False
            continue

        drop = baseline[i] - spo2[i]

        if not in_event:
            if drop >= threshold:
                in_event = True
                event_start = i
                event_max_drop = drop
                event_baseline = baseline[i]
        else:
            if drop >= threshold:
                event_max_drop = max(event_max_drop, drop)
            else:
                # Event ended
                dur = i - event_start
                if dur >= DESAT_DURATION_MIN and event_max_drop >= threshold:
                    evt = DesatEvent(event_start, i, event_max_drop, event_baseline)
                    for j in range(event_start, i):
                        if is_valid[j] and not np.isnan(baseline[j]):
                            d = baseline[j] - spo2[j]
                            if d > 0:
                                evt.area += d
                    events.append(evt)
                in_event = False

    # Close any open event at end
    if in_event:
        dur = N - event_start
        if dur >= DESAT_DURATION_MIN and event_max_drop >= threshold:
            evt = DesatEvent(event_start, N, event_max_drop, event_baseline)
            for j in range(event_start, N):
                if j < N and is_valid[j] and not np.isnan(baseline[j]):
                    d = baseline[j] - spo2[j]
                    if d > 0:
                        evt.area += d
            events.append(evt)

    return events

events_3 = detect_desaturations(3)
events_4 = detect_desaturations(4)

odi_3 = len(events_3) / max(study_duration_hr, 0.01)
odi_4 = len(events_4) / max(study_duration_hr, 0.01)

# =============================================================
# DELTA INDEX (mean absolute difference of successive 12s averages)
# =============================================================
DELTA_EPOCH = 12  # seconds
n_epochs = int(np.sum(is_valid) // DELTA_EPOCH)
epoch_means = []
# Walk through valid samples in blocks of DELTA_EPOCH
valid_idx = np.where(is_valid)[0]
for e in range(n_epochs):
    block = spo2[valid_idx[e*DELTA_EPOCH : (e+1)*DELTA_EPOCH]]
    epoch_means.append(np.mean(block))

epoch_means = np.array(epoch_means)
if len(epoch_means) > 1:
    delta_index = np.mean(np.abs(np.diff(epoch_means)))
else:
    delta_index = 0.0

# =============================================================
# PULSE RATE SUMMARY (from oximeter, for cross-reference with ECG)
# =============================================================
pulse_valid = pulse[is_valid]
# Filter out zero/implausible pulse
pulse_ok = pulse_valid[(pulse_valid >= 30) & (pulse_valid <= 200)]

# =============================================================
# SUMMARY OUTPUT (optional save to file)
# =============================================================
stem = str(Path(input_path).parent / Path(input_path).stem)
_summary_file = None
_original_stdout = sys.stdout

if args.save_summary:
    summary_path = f"{stem}_summary.txt"
    _summary_file = open(summary_path, 'w')
    _summary_file.write(f"Source: {Path(input_path).name}\n")
    _summary_file.write(f"Start: {t0.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _summary_file.write(f"Samples: {N} ({duration_sec:.0f}s)\n\n")

    class _Tee:
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
        def flush(self):
            self.stream.flush()
            self.file.flush()

    sys.stdout = _Tee(_summary_file, _original_stdout)

# =============================================================
# SUMMARY
# =============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"  {'Study duration':25s}: {study_duration_hr:.1f} h ({study_duration_sec:.0f} s)")
print(f"  {'Valid samples':25s}: {n_valid} ({100*n_valid/N:.1f}%)")

print(f"\n  {'SpO2 Mean':25s}: {spo2_mean:5.1f} ± {spo2_std:.1f} %")
print(f"  {'SpO2 Median':25s}: {spo2_median:5.0f} %")
print(f"  {'SpO2 Nadir':25s}: {spo2_nadir:5.0f} %{nadir_time_str}")

# Distribution - compact vector, only show bins with >0%
parts = []
if below_80_pct >= 0.005:
    parts.append(f"<80:{below_80_pct:.2f}")
for lbl, pct in zip(bin_labels, pcts):
    if pct >= 0.005:
        parts.append(f"{lbl}:{pct:.1f}")
print(f"  {'SpO2 Distribution (%)':25s}: [{', '.join(parts)}]")

# Time below thresholds
print(f"\n  {'Time ≤90%':25s}: {t90_sec:5d} s ({t90_pct:.2f}%)"
      f"  = {t90_sec/60:.1f} min")
print(f"  {'Time ≤88%':25s}: {t88_sec:5d} s ({t88_pct:.2f}%)"
      f"  = {t88_sec/60:.1f} min")
if t85_sec > 0:
    print(f"  {'Time ≤85%':25s}: {t85_sec:5d} s ({t85_pct:.2f}%)"
          f"  = {t85_sec/60:.1f} min")

# ODI
print(f"\n  {'ODI-3 (≥3% drops)':25s}: {odi_3:5.1f} /hr  ({len(events_3)} events)")
print(f"  {'ODI-4 (≥4% drops)':25s}: {odi_4:5.1f} /hr  ({len(events_4)} events)")

# Desaturation event details (using 3% events)
if len(events_3) > 0:
    depths = np.array([e.depth for e in events_3])
    durations = np.array([e.duration for e in events_3])
    areas = np.array([e.area for e in events_3])
    total_area = np.sum(areas)
    print(f"  {'Desat depth (≥3%)':25s}: {np.mean(depths):5.1f} ± {np.std(depths):.1f} %"
          f"  [max {np.max(depths):.0f}%]")
    print(f"  {'Desat duration (≥3%)':25s}: {np.mean(durations):5.0f} ± {np.std(durations):.0f} s"
          f"  [max {np.max(durations):.0f}s]")
    print(f"  {'Desat area (≥3%)':25s}: {np.mean(areas):5.1f} mean, {total_area:.0f} total (%·s)")

# Delta index
print(f"\n  {'Delta index (12s)':25s}: {delta_index:5.2f} %")

# Pulse rate from oximeter
if len(pulse_ok) > 0:
    print(f"\n  {'Pulse rate (oximeter)':25s}: {np.mean(pulse_ok):5.1f} ± {np.std(pulse_ok):.1f} bpm"
          f"  [{np.min(pulse_ok):.0f} – {np.max(pulse_ok):.0f}]")
    print(f"  {'Pulse median':25s}: {np.median(pulse_ok):5.0f} bpm")

print(f"\n  Artifacts: {n_artifact} samples ({100*n_artifact/N:.1f}%)")
if n_post_study > 0:
    print(f"  Post-study (>{STUDY_END_HOUR}AM): {n_post_study} samples excluded")
print("="*60)

# =============================================================
# CLOSE SUMMARY FILE
# =============================================================
if _summary_file:
    sys.stdout = _original_stdout
    _summary_file.close()
    print(f"Summary saved: {summary_path}")
