#!/usr/bin/env python3
"""
Generate Sleep Dashboard HTML - 2026-02-05 J.Beale

Reads four CSV files from a directory and produces an interactive HTML
dashboard with synchronized panels for SpO2, Heart Rate, Respiratory Rate,
Apnea/Obstruction events, and Motion.

Input files (in specified directory):
  SleepU 6294_*.csv                              - SpO2, HR, Motion
  breathing_analysis_report_respiratory_rate.csv  - Breaths per minute
  breathing_analysis_report_apnea_events.csv     - Apnea gap events
  breathing_analysis_report_obstructions.csv     - Obstruction events
  MOT_*_breath.csv                               - Tilt-based breathing envelope (optional)

Usage: python generate_sleep_dashboard.py <input_directory>
"""

import os
import sys
import csv
import json
import bisect
import calendar
import glob
from datetime import datetime, timedelta

VERSION = "1.6.0"
BUILD_DATE = "2026-02-22"


def find_input_files(input_dir):
    """Locate the required CSV files. Returns dict of paths or raises.
    Prefers Checkme O2 Ultra over SleepU for SpO2/HR/Motion if available."""
    files = {}

    # Prefer Checkme O2 Ultra file if available, fall back to SleepU
    checkme_pattern = os.path.join(input_dir, "Checkme O2 Ultra 2355_*.csv")
    checkme_matches = sorted(glob.glob(checkme_pattern))

    sleepu_pattern = os.path.join(input_dir, "SleepU 6294_*.csv")
    sleepu_matches = sorted(glob.glob(sleepu_pattern))

    if checkme_matches:
        if len(checkme_matches) > 1:
            print(f"  Note: found {len(checkme_matches)} Checkme files, using {os.path.basename(checkme_matches[-1])}")
        files['oximeter'] = checkme_matches[-1]
        files['oximeter_type'] = 'checkme'
        if sleepu_matches:
            print(f"  Note: Checkme O2 Ultra found, using it instead of SleepU")
    elif sleepu_matches:
        if len(sleepu_matches) > 1:
            print(f"  Note: found {len(sleepu_matches)} SleepU files, using {os.path.basename(sleepu_matches[-1])}")
        files['oximeter'] = sleepu_matches[-1]
        files['oximeter_type'] = 'sleepu'
    else:
        raise FileNotFoundError(
            f"No file matching 'Checkme O2 Ultra 2355_*.csv' or 'SleepU 6294_*.csv' in {input_dir}")

    # Breathing analysis files (optional)
    exact = {
        'resp_rate': 'breathing_analysis_report_respiratory_rate.csv',
        'apnea':     'breathing_analysis_report_apnea_events.csv',
        'obstruct':  'breathing_analysis_report_obstructions.csv',
    }
    breathing_missing = []
    for key, name in exact.items():
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            files[key] = path
        else:
            breathing_missing.append(name)
    if breathing_missing:
        print(f"  Note: breathing analysis files not found, those panels will be blank:")
        for name in breathing_missing:
            print(f"    {name}")

    # Optional: tilt-based breathing envelope CSV
    breath_pattern = os.path.join(input_dir, "MOT_*_breath.csv")
    breath_matches = sorted(glob.glob(breath_pattern))
    if breath_matches:
        files['tilt_breath'] = breath_matches[-1]
        if len(breath_matches) > 1:
            print(f"  Note: found {len(breath_matches)} breath files, using {os.path.basename(breath_matches[-1])}")

    # Optional: ECG beats file (e.g. ECG_20260221_225814_beats.csv)
    ecg_pattern = os.path.join(input_dir, "ECG_*_beats.csv")
    ecg_matches = sorted(glob.glob(ecg_pattern))
    if ecg_matches:
        files['ecg_beats'] = ecg_matches[-1]
        if len(ecg_matches) > 1:
            print(f"  Note: found {len(ecg_matches)} ECG beats files, using {os.path.basename(ecg_matches[-1])}")

    return files


def parse_sleepu_time(ts_str):
    """Parse SleepU timestamp like '22:42:27 04/02/2026' -> datetime."""
    return datetime.strptime(ts_str.strip(), "%H:%M:%S %d/%m/%Y")


def parse_iso_time(ts_str):
    """Parse timestamp like '2026-02-04 22:45:00' -> datetime."""
    return datetime.strptime(ts_str.strip(), "%Y-%m-%d %H:%M:%S")


def dt_to_minutes(dt, ref_date):
    """Convert datetime to minutes since midnight of ref_date."""
    delta = dt - datetime.combine(ref_date, datetime.min.time())
    return round(delta.total_seconds() / 60.0, 1)


def read_sleepu(filepath, ref_date):
    """Read SleepU CSV at full resolution. Returns compact dict:
       {t0: start_minutes, dt: interval_minutes, spo2: [...], hr: [...], mot: [...]}
       Invalid samples ('--' or out-of-range) are stored as None."""
    spo2, hr, mot = [], [], []
    first_dt_obj = None
    second_dt_obj = None
    ref_midnight = datetime.combine(ref_date, datetime.min.time())
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                dt = parse_sleepu_time(row['Time'])
            except (ValueError, KeyError):
                continue

            if first_dt_obj is None:
                first_dt_obj = dt
            elif second_dt_obj is None:
                second_dt_obj = dt

            o2 = row.get('Oxygen Level', '').strip()
            pulse = row.get('Pulse Rate', '').strip()
            motion = row.get('Motion', '').strip()

            if o2 == '--' or pulse == '--':
                spo2.append(None); hr.append(None); mot.append(None)
                continue
            o2v = int(o2)
            pv = int(pulse)
            mv = int(motion) if motion != '--' else 0

            if o2v < 70 or pv < 30:
                spo2.append(None); hr.append(None); mot.append(None)
                continue

            spo2.append(o2v)
            hr.append(pv)
            mot.append(mv)

    # Compute t0 and dt from raw datetimes (avoid rounding loss)
    t0 = (first_dt_obj - ref_midnight).total_seconds() / 60.0
    interval_sec = (second_dt_obj - first_dt_obj).total_seconds()
    dt_min = interval_sec / 60.0
    return {'t0': round(t0, 4), 'dt': round(dt_min, 6), 'spo2': spo2, 'hr': hr, 'mot': mot}


def read_checkme(filepath, ref_date, bin_seconds=4):
    """Read Checkme O2 Ultra CSV (1-second sampling) and resample to bin_seconds.
    Within each bin: SpO2=min, HR=round(mean), Motion=max.
    Returns same format as read_sleepu:
       {t0: start_minutes, dt: interval_minutes, spo2: [...], hr: [...], mot: [...]}
    """
    # First pass: read all valid samples with timestamps
    ref_midnight = datetime.combine(ref_date, datetime.min.time())
    raw = []  # list of (seconds_from_midnight, o2, pulse, motion)
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dt = parse_sleepu_time(row['Time'])  # same timestamp format
            except (ValueError, KeyError):
                continue

            o2 = row.get('Oxygen Level', '').strip()
            pulse = row.get('Pulse Rate', '').strip()
            motion = row.get('Motion', '').strip()

            sec = (dt - ref_midnight).total_seconds()

            if o2 == '--' or pulse == '--':
                raw.append((sec, None, None, None))
                continue
            o2v = int(o2)
            pv = int(pulse)
            mv = int(motion) if motion != '--' else 0
            if o2v < 70 or pv < 30:
                raw.append((sec, None, None, None))
                continue
            raw.append((sec, o2v, pv, mv))

    if not raw:
        return {'t0': 0, 'dt': bin_seconds / 60.0, 'spo2': [], 'hr': [], 'mot': []}

    # Create fixed-width bins starting at first sample
    t_start = raw[0][0]
    t_end = raw[-1][0]
    dt_min = bin_seconds / 60.0
    spo2, hr, mot = [], [], []

    bin_start = t_start
    ri = 0  # index into raw
    while bin_start <= t_end:
        bin_end = bin_start + bin_seconds
        # Collect samples in [bin_start, bin_end)
        o2_vals, pr_vals, mot_vals = [], [], []
        while ri < len(raw) and raw[ri][0] < bin_end:
            _, o2, pr, mv = raw[ri]
            if o2 is not None:
                o2_vals.append(o2)
                pr_vals.append(pr)
                mot_vals.append(mv)
            ri += 1

        if o2_vals:
            spo2.append(min(o2_vals))
            hr.append(round(sum(pr_vals) / len(pr_vals)))
            mot.append(max(mot_vals))
        else:
            spo2.append(None)
            hr.append(None)
            mot.append(None)

        bin_start += bin_seconds

    t0 = t_start / 60.0
    return {'t0': round(t0, 4), 'dt': round(dt_min, 6), 'spo2': spo2, 'hr': hr, 'mot': mot}


def read_resp_rate(filepath, ref_date):
    """Read respiratory rate CSV. Returns list of [t, breaths_per_min]."""
    data = []
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            try:
                dt = parse_iso_time(row['timestamp'])
            except (ValueError, KeyError):
                continue
            t = dt_to_minutes(dt, ref_date)
            data.append([t, round(float(row['breaths_per_minute'].strip()), 1)])
    return data


def read_apnea(filepath, ref_date):
    """Read apnea events CSV. Returns list of [t, duration_sec]."""
    data = []
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            try:
                dt = parse_iso_time(row['wall_clock_time'])
            except (ValueError, KeyError):
                continue
            t = dt_to_minutes(dt, ref_date)
            data.append([t, float(row['duration_sec'].strip())])
    return data


def read_obstructions(filepath, ref_date):
    """Read obstructions CSV. Returns list of [t, duration_sec, severity_flag]."""
    data = []
    with open(filepath, newline='') as f:
        for row in csv.DictReader(f):
            try:
                dt = parse_iso_time(row['wall_clock_time'])
            except (ValueError, KeyError):
                continue
            t = dt_to_minutes(dt, ref_date)
            sev = 1 if row['severity'].strip() == 'severe' else 0
            data.append([t, float(row['duration_sec'].strip()), sev])
    return data


def read_tilt_breath(filepath, ref_date):
    """Read tilt-based breathing envelope CSV.
    Returns dict: {tiltRR: [[t, bpm], ...], tiltEnv: [[t, deg], ...], tiltRoll: [[t, deg], ...]}
    where t is minutes since midnight of ref_date."""
    start_time = None
    with open(filepath, 'r') as f:
        first = f.readline().strip()
        if first.startswith('# start '):
            start_time = first[8:].split('sync_millis')[0].strip()

    if not start_time:
        print(f"  Warning: no start time in {filepath}, skipping tilt breath data")
        return None

    try:
        t0_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"  Warning: cannot parse start time '{start_time}', skipping tilt breath data")
        return None

    ref_midnight = datetime.combine(ref_date, datetime.min.time())
    t0_minutes = (t0_dt - ref_midnight).total_seconds() / 60.0

    tilt_rr = []
    tilt_env = []
    tilt_roll = []
    with open(filepath, newline='') as f:
        # skip comment lines
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('seconds,'):
            header_idx = i
            break
    if header_idx is None:
        print(f"  Warning: no header found in {filepath}")
        return None

    import io
    reader = csv.DictReader(io.StringIO(''.join(lines[header_idx:])))
    for row in reader:
        try:
            sec = float(row['seconds'])
            t_min = t0_minutes + sec / 60.0
            env_val = float(row['envelope_deg'])
            tilt_env.append([round(t_min, 4), round(env_val, 4)])
            bpm_str = row.get('breaths_per_min', '').strip()
            if bpm_str:
                tilt_rr.append([round(t_min, 4), round(float(bpm_str), 1)])
            roll_str = row.get('roll_deg', '').strip()
            if roll_str:
                tilt_roll.append([round(t_min, 4), round(float(roll_str), 2)])
        except (ValueError, KeyError):
            continue

    return {'tiltRR': tilt_rr, 'tiltEnv': tilt_env, 'tiltRoll': tilt_roll}


def read_ecg_beats(filepath, ref_date, bin_seconds=2, pleth_delay_s=0.25):
    """Read ECG beats CSV and compute causal median-3 smoothed HR, resampled to bin_seconds.
    Beat timestamps are Unix epoch UTC; output t is minutes since local midnight (PST = UTC-8).
    Keeps artifact-flagged beats — they are real heartbeats, just unusual.
    Returns list of [t_minutes, hr_bpm] pairs, or [] on error."""
    # Epoch of PST (UTC-8) midnight for ref_date
    utc_midnight_epoch = calendar.timegm(ref_date.timetuple())
    local_midnight_epoch = utc_midnight_epoch + 8 * 3600  # PST midnight = UTC 08:00

    beats = []  # (pleth_arrival_epoch_s, rr_ms)
    try:
        with open(filepath, newline='') as f:
            lines = [l for l in f if not l.startswith('#')]
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                epoch_s = float(row['epoch_s']) + pleth_delay_s
                rr_ms = float(row['rr_ms'])
            except (ValueError, KeyError):
                continue
            if not (100 < rr_ms < 3000):
                continue
            beats.append((epoch_s, rr_ms))
    except OSError:
        return []

    if len(beats) < 4:
        return []

    beat_times = [b[0] for b in beats]

    # Step 1: 1-Hz grid of last-cycle instantaneous HR
    t_start = beat_times[0]
    t_end   = beat_times[-1]
    grid_1hz = []
    t = t_start + 1.0
    while t <= t_end:
        idx = bisect.bisect_right(beat_times, t) - 1
        if idx >= 1:
            iv = (beat_times[idx] - beat_times[idx - 1]) * 1000.0
            grid_1hz.append(60000.0 / iv if 300 < iv < 3000 else None)
        else:
            grid_1hz.append(None)
        t += 1.0

    if not grid_1hz:
        return []

    # Step 2: causal median-3 spike filter
    # A single corrupted pleth cycle (missed or doubled pulse) produces one outlier
    # sample; taking median of 3 consecutive values removes it without smoothing
    # genuine HR changes that persist across 2+ seconds.
    med3 = list(grid_1hz)
    for i in range(2, len(med3)):
        window = [v for v in [grid_1hz[i - 2], grid_1hz[i - 1], grid_1hz[i]] if v is not None]
        if len(window) >= 2:
            med3[i] = sorted(window)[len(window) // 2]

    # Step 3: resample to bin_seconds by averaging, to match the oximeter output rate
    result = []
    n = len(med3)
    for j in range(0, n, bin_seconds):
        chunk = med3[j:j + bin_seconds]
        vals = [v for v in chunk if v is not None]
        if vals:
            epoch_bin = t_start + 1.0 + j
            t_min = (epoch_bin - local_midnight_epoch) / 60.0
            result.append([round(t_min, 4), round(sum(vals) / len(vals), 1)])

    return result


def compute_time_range(sleepu, rr):
    """Return (t_min, t_max) with some padding, and formatted subtitle info."""
    t0 = sleepu['t0']
    n = len(sleepu['spo2'])
    t_end = t0 + (n - 1) * sleepu['dt']
    all_t = [t0, t_end] + [p[0] for p in rr]
    t_min = min(all_t)
    t_max = max(all_t)
    # Pad by 3 minutes on each side
    return t_min - 3, t_max + 3


def minutes_to_clock(m):
    """Convert minutes-since-midnight to 'HH:MM' string."""
    h = int(m / 60) % 24
    mn = int(m % 60)
    return f"{h:02d}:{mn:02d}"


def format_duration(t_min, t_max):
    """Format duration as '~Xh Ym'."""
    dur = t_max - t_min
    hours = int(dur // 60)
    mins = int(dur % 60)
    return f"~{hours}h {mins:02d}m"


# =============================================================================
# HTML Template
# =============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sleep Study — %%TITLE_DATE%%</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: #0a0e14;
    color: #c5cdd8;
    font-family: 'IBM Plex Sans', sans-serif;
    padding: 8px 32px 0;
    min-height: 100vh;
    overflow: hidden;
  }

  h1 {
    font-size: 20px;
    font-weight: 600;
    color: #e8ecf0;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
  }

  .subtitle {
    font-size: 13px;
    color: #6b7a8d;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
  }

  .chart-container {
    position: relative;
    width: 100%;
  }

  canvas {
    display: block;
    width: 100% !important;
  }

  canvas.zoomed { cursor: grab; }
  canvas.zoomed.panning { cursor: grabbing; }

  .legend {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 6px;
    padding: 6px 16px;
    background: #111822;
    border-radius: 6px;
    border: 1px solid #1e2a38;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
    color: #8899aa;
  }

  .legend-swatch {
    width: 14px;
    height: 4px;
    border-radius: 2px;
  }

  .legend-swatch.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .legend-swatch.bar {
    width: 14px;
    height: 10px;
    border-radius: 2px;
  }

  .tooltip-box {
    position: absolute;
    display: none;
    pointer-events: none;
    background: #161e28;
    border: 1px solid #2a3a4a;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #c5cdd8;
    z-index: 100;
    white-space: nowrap;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }

  .tooltip-box .tt-time {
    font-weight: 600;
    color: #e8ecf0;
    margin-bottom: 4px;
  }

  .tooltip-box .tt-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 2px 0;
  }

  .tooltip-box .tt-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .crosshair {
    position: absolute;
    top: 0;
    width: 1px;
    background: rgba(100,140,180,0.25);
    pointer-events: none;
    display: none;
  }

  .btn-bar {
    position: fixed;
    top: 12px;
    right: 16px;
    display: flex;
    gap: 8px;
    z-index: 200;
  }

  .theme-toggle {
    background: #1a2430;
    border: 1px solid #2a3a4a;
    color: #8899aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
  }
  .theme-toggle:hover { background: #243040; color: #b0c0d0; }

  .version-stamp {
    display: inline;
    margin-left: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #2a3a4a;
    vertical-align: middle;
  }
  body.light-theme .version-stamp { color: #c0c0c0; }

  body.light-theme { background: #ffffff; color: #333; }
  body.light-theme h1 { color: #111; }
  body.light-theme .subtitle { color: #555; }
  body.light-theme .legend { background: #f0f0f0; border-color: #ccc; }
  body.light-theme .legend-item { color: #555; }
  body.light-theme .tooltip-box { background: #fff; border-color: #bbb; color: #333; box-shadow: 0 4px 16px rgba(0,0,0,0.15); }
  body.light-theme .tooltip-box .tt-time { color: #111; }
  body.light-theme .crosshair { background: rgba(0,0,0,0.2); }
  body.light-theme .theme-toggle { background: #e8e8e8; border-color: #ccc; color: #555; }
  body.light-theme .theme-toggle:hover { background: #ddd; color: #333; }

  @media print {
    body { background: #fff !important; color: #333 !important; padding: 4px 16px !important; overflow: visible !important; }
    h1 { color: #111 !important; }
    .subtitle { color: #555 !important; }
    .legend { background: #f5f5f5 !important; border-color: #ccc !important; }
    .legend-item { color: #555 !important; }
    .btn-bar { display: none !important; }
    .version-stamp { display: none !important; }
    .tooltip-box, .crosshair { display: none !important; }
  }
</style>
</head>
<body>

<div class="btn-bar">
  <button class="theme-toggle" id="resetZoom" style="display:none;" onclick="resetZoom()">&#x21BA; Reset Zoom</button>
  <button class="theme-toggle" id="savePng" onclick="savePNG(2)">&#x1F4BE; Save PNG 2x</button>
  <button class="theme-toggle" id="savePng1x" onclick="savePNG(1)">&#x1F4BE; Save PNG 1x</button>
  <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()">&#x263C; Light</button>
</div>
<h1>Overnight Sleep Study <span class="version-stamp" id="versionStamp">v%%VERSION%% &middot; %%BUILD_DATE%%</span></h1>
<div class="subtitle">%%SUBTITLE%%</div>

<div class="legend" id="legendBar">
  <div class="legend-item"><div class="legend-swatch" id="lsw-spo2" style="background:#48b8e8"></div>SpO&#x2082; (%)</div>
  <div class="legend-item"><div class="legend-swatch" id="lsw-hr" style="background:rgba(232,88,120,0.45); border-top: 2px dashed rgba(232,88,120,0.55); height:0; width:18px;"></div>Pleth HR</div>
  <div class="legend-item"><div class="legend-swatch" id="lsw-hrma" style="background:rgba(255,220,80,0.85)"></div>HR Trend (~5m avg)</div>
  <div class="legend-item"><div class="legend-swatch" id="lsw-ecghr" style="background:#e83030"></div>ECG HR (med-3)</div>
  <div class="legend-item"><div class="legend-swatch" id="lsw-tiltenv" style="background:rgba(64,208,208,0.4)"></div>Breath Envelope (°)</div>
  <div class="legend-item"><div class="legend-swatch bar" id="lsw-apnea" style="background:rgba(255,80,60,0.5)"></div>Apnea</div>
  <div class="legend-item"><div class="legend-swatch bar" id="lsw-apnealong" style="background:rgba(255,220,40,0.8)"></div>Apnea &gt;50s</div>
  <div class="legend-item"><span style="display:inline-flex;align-items:center;gap:3px;">Obstr: <div class="legend-swatch dot" id="lsw-omod" style="background:#f0a030"></div>mod <div class="legend-swatch dot" id="lsw-osev" style="background:#ff4040"></div>sev</span></div>
  <div class="legend-item"><div class="legend-swatch bar" id="lsw-roll" style="background:linear-gradient(to right, hsl(240,75%,55%), hsl(180,75%,55%), hsl(120,75%,55%), hsl(60,75%,55%), hsl(0,75%,55%)); border-radius:2px;"></div>Body Roll (°)</div>
  <div class="legend-item"><div class="legend-swatch bar" id="lsw-mot" style="background:rgba(160,140,220,0.5)"></div>Motion</div>
  <div class="legend-item"><div class="legend-swatch" id="lsw-good" style="background:rgba(80,200,120,0.5); border-top: 2px dotted rgba(80,200,120,0.8); height:0; width:18px;"></div>Restful (&ge;15m)</div>
</div>

<div class="chart-container" id="chartArea">
  <canvas id="mainCanvas"></canvas>
  <div class="crosshair" id="crosshair"></div>
  <div class="tooltip-box" id="tooltip"></div>
</div>

<script>
// ============================================================
// DATA (minutes since midnight of recording start date)
// ============================================================
const RAW = %%DATA_JSON%%;
const T_MIN_FULL = %%T_MIN%%;
const T_MAX_FULL = %%T_MAX%%;
let T_MIN = T_MIN_FULL;
let T_MAX = T_MAX_FULL;

// ============================================================
// Color Themes
// ============================================================
const THEMES = {
  dark: {
    bodyBg: '#0a0e14',
    titleText: '#e8ecf0',
    subtitleText: '#6b7a8d',
    legendBg: '#111822',
    legendBorder: '#1e2a38',
    legendText: '#8899aa',
    versionText: '#2a3a4a',
    panelBg: '#0d1219',
    panelBorder: '#1a2430',
    gridLine: 'rgba(30,42,56,0.6)',
    gridLineX: 'rgba(30,42,56,0.4)',
    yAxisText: '#4a5a6a',
    xAxisText: '#5a6a7a',
    xAxisLine: '#1a2430',
    spo2: '#48b8e8',
    hr: 'rgba(232,88,120,0.45)',
    resp: '#58d888',
    respArea: 'rgba(88,216,136,0.12)',
    spo2Ref: 'rgba(255,80,60,0.3)',
    evtLabel: '#ff6050',
    apneaBar: 'rgba(255,70,50,0.35)',
    obstrSevere: '#ff4040',
    obstrMod: '#f0a030',
    sevLabel: 'rgba(255,70,50,0.4)',
    modLabel: 'rgba(240,160,48,0.4)',
    motionBar: 'rgba(160,140,220,0.7)',
    motionAxis: 'rgba(160,140,220,0.7)',
    tiltResp: '#40d0d0',
    tiltEnv: 'rgba(64,208,208,0.35)',
    tiltEnvLine: 'rgba(64,208,208,0.85)',
    hrMA: 'rgba(255,220,80,0.85)',
    ecgHR: '#e83030',
    apneaLong: 'rgba(255,220,40,0.8)',
    apneaLongText: '#ffe060',
    spo2DipText: '#ff6060',
    goodSleep: 'rgba(80,200,120,0.5)',
    ahiHourlyText: 'rgba(255,255,255,0.85)',
    rollAxis: 'rgba(200,180,140,0.7)',
  },
  light: {
    bodyBg: '#ffffff',
    titleText: '#111111',
    subtitleText: '#555555',
    legendBg: '#f0f0f0',
    legendBorder: '#cccccc',
    legendText: '#555555',
    versionText: '#c0c0c0',
    panelBg: '#f8f8f8',
    panelBorder: '#cccccc',
    gridLine: 'rgba(0,0,0,0.12)',
    gridLineX: 'rgba(0,0,0,0.10)',
    yAxisText: '#444444',
    xAxisText: '#444444',
    xAxisLine: '#999999',
    spo2: '#0077aa',
    hr: 'rgba(180,40,70,0.4)',
    resp: '#228844',
    respArea: 'rgba(34,136,68,0.10)',
    spo2Ref: 'rgba(200,50,30,0.4)',
    evtLabel: '#cc3322',
    apneaBar: 'rgba(200,50,30,0.4)',
    obstrSevere: '#cc2222',
    obstrMod: '#cc7700',
    sevLabel: 'rgba(200,50,30,0.5)',
    modLabel: 'rgba(200,120,0,0.5)',
    motionBar: 'rgba(80,60,160,0.65)',
    motionAxis: 'rgba(80,60,140,0.8)',
    tiltResp: '#008888',
    tiltEnv: 'rgba(0,136,136,0.28)',
    tiltEnvLine: 'rgba(0,136,136,0.80)',
    hrMA: 'rgba(180,120,0,0.85)',
    ecgHR: '#cc0000',
    apneaLong: 'rgba(200,160,0,0.75)',
    apneaLongText: '#806000',
    spo2DipText: '#cc0000',
    goodSleep: 'rgba(40,160,80,0.45)',
    ahiHourlyText: 'rgba(0,0,0,0.8)',
    rollAxis: 'rgba(120,100,60,0.8)',
  }
};
let theme = THEMES.dark;

function updateLegendColors() {
  document.getElementById('lsw-spo2').style.background = theme.spo2;
  const hrEl = document.getElementById('lsw-hr');
  hrEl.style.background = 'none';
  hrEl.style.borderTop = '2px dashed ' + theme.hr.replace(/[\d.]+\)$/, '0.8)');
  document.getElementById('lsw-hrma').style.background = theme.hrMA;
  const ecgHREl = document.getElementById('lsw-ecghr');
  if (ecgHREl) ecgHREl.style.background = theme.ecgHR;
  const tiltEnvEl = document.getElementById('lsw-tiltenv');
  if (tiltEnvEl) tiltEnvEl.style.background = theme.tiltEnvLine;
  document.getElementById('lsw-apnea').style.background = theme.apneaBar;
  document.getElementById('lsw-apnealong').style.background = theme.apneaLong;
  document.getElementById('lsw-omod').style.background = theme.obstrMod;
  document.getElementById('lsw-osev').style.background = theme.obstrSevere;
  document.getElementById('lsw-mot').style.background = theme.motionBar;
  const goodEl = document.getElementById('lsw-good');
  goodEl.style.background = 'none';
  goodEl.style.borderTop = '2px dotted ' + theme.goodSleep;
}

function toggleTheme() {
  const body = document.body;
  const btn = document.getElementById('themeToggle');
  if (body.classList.contains('light-theme')) {
    body.classList.remove('light-theme');
    theme = THEMES.dark;
    btn.innerHTML = '&#x263C; Light';
  } else {
    body.classList.add('light-theme');
    theme = THEMES.light;
    btn.innerHTML = '&#x263D; Dark';
  }
  draw();
  updateLegendColors();
}

// Auto-switch theme for printing
window.addEventListener('beforeprint', () => {
  if (!document.body.classList.contains('light-theme')) {
    document.body.classList.add('light-theme');
    document.body.dataset.wasdk = '1';
    theme = THEMES.light;
    draw();
    updateLegendColors();
  }
});
window.addEventListener('afterprint', () => {
  if (document.body.dataset.wasdk === '1') {
    document.body.classList.remove('light-theme');
    delete document.body.dataset.wasdk;
    theme = THEMES.dark;
    draw();
    updateLegendColors();
  }
});

// ============================================================
// Layout (dynamic — fills viewport)
// ============================================================
const canvas = document.getElementById('mainCanvas');
const ctx = canvas.getContext('2d');

const DPR = window.devicePixelRatio || 1;
// Panel proportional weights: SpO2, HR, Resp, Events, Motion
const PANEL_WEIGHTS = [0.25, 0.25, 0.22, 0.16, 0.12];
const PANEL_GAP = 4;
const MARGIN = { top: 6, right: 24, bottom: 28, left: 54 };

let W, TOTAL_H, PANEL_HEIGHTS, plotW;

function computeLayout() {
  W = canvas.parentElement.clientWidth;
  // Available height = viewport minus header elements
  const headerH = document.getElementById('chartArea').getBoundingClientRect().top;
  const availH = window.innerHeight - headerH - 4;
  TOTAL_H = Math.max(300, availH);
  const panelSpace = TOTAL_H - MARGIN.top - MARGIN.bottom - PANEL_GAP * (PANEL_WEIGHTS.length - 1);
  PANEL_HEIGHTS = PANEL_WEIGHTS.map(w => Math.round(w * panelSpace));
  plotW = W - MARGIN.left - MARGIN.right;

  canvas.width = W * DPR;
  canvas.height = TOTAL_H * DPR;
  canvas.style.height = TOTAL_H + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  const crosshair = document.getElementById('crosshair');
  crosshair.style.height = (TOTAL_H - MARGIN.bottom) + 'px';
}

computeLayout();

function tToX(t) { return MARGIN.left + (t - T_MIN) / (T_MAX - T_MIN) * plotW; }
function xToT(x) { return T_MIN + (x - MARGIN.left) / plotW * (T_MAX - T_MIN); }

function minToClockStr(m) {
  let h = Math.floor(m / 60) % 24;
  let mn = Math.floor(m % 60);
  let s = Math.round((m % 1) * 60);
  let str = `${h.toString().padStart(2,'0')}:${mn.toString().padStart(2,'0')}`;
  if (s !== 0) str += `:${s.toString().padStart(2,'0')}`;
  return str;
}

function minToClockStrFull(m) {
  let h = Math.floor(m / 60) % 24;
  let mn = Math.floor(m % 60);
  let s = Math.round((m % 1) * 60);
  return `${h.toString().padStart(2,'0')}:${mn.toString().padStart(2,'0')}:${s.toString().padStart(2,'0')}`;
}

function panelTop(i) {
  let y = MARGIN.top;
  for (let j = 0; j < i; j++) y += PANEL_HEIGHTS[j] + PANEL_GAP;
  return y;
}

function valToY(v, vMin, vMax, panelIdx) {
  const pt = panelTop(panelIdx);
  const ph = PANEL_HEIGHTS[panelIdx];
  const pad = 6;
  return pt + pad + (1 - (v - vMin) / (vMax - vMin)) * (ph - 2*pad);
}

// ============================================================
// Drawing helpers
// ============================================================
function drawPanelBg(idx) {
  const y = panelTop(idx);
  const h = PANEL_HEIGHTS[idx];
  ctx.fillStyle = theme.panelBg;
  ctx.fillRect(MARGIN.left, y, plotW, h);
  ctx.strokeStyle = theme.panelBorder;
  ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN.left, y, plotW, h);
}

function drawLine(data, tKey, vKey, vMin, vMax, panelIdx, color, lineWidth) {
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth || 1.2;
  ctx.lineJoin = 'round';
  let first = true;
  for (const pt of data) {
    const x = tToX(pt[tKey]);
    const y = valToY(pt[vKey], vMin, vMax, panelIdx);
    if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
    if (first) { ctx.moveTo(x, y); first = false; }
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawArea(data, tKey, vKey, vMin, vMax, panelIdx, fillColor) {
  const pt0 = panelTop(panelIdx);
  const ph = PANEL_HEIGHTS[panelIdx];
  const baseY = pt0 + ph - 6;
  ctx.beginPath();
  let first = true;
  let lastX;
  for (const pt of data) {
    const x = tToX(pt[tKey]);
    const y = valToY(pt[vKey], vMin, vMax, panelIdx);
    if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
    if (first) { ctx.moveTo(x, baseY); ctx.lineTo(x, y); first = false; }
    else ctx.lineTo(x, y);
    lastX = x;
  }
  if (lastX !== undefined) {
    ctx.lineTo(lastX, baseY);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();
  }
}

function drawYAxis(vMin, vMax, panelIdx, ticks, unit, color) {
  ctx.font = '10px IBM Plex Mono';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of ticks) {
    const y = valToY(v, vMin, vMax, panelIdx);
    ctx.strokeStyle = theme.gridLine;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(MARGIN.left, y);
    ctx.lineTo(MARGIN.left + plotW, y);
    ctx.stroke();
    ctx.fillStyle = theme.yAxisText;
    ctx.fillText(v.toString(), MARGIN.left - 6, y);
  }
  ctx.save();
  ctx.translate(14, panelTop(panelIdx) + PANEL_HEIGHTS[panelIdx]/2);
  ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center';
  ctx.font = '500 13px IBM Plex Mono';
  ctx.fillStyle = color;
  ctx.fillText(unit, 0, 0);
  ctx.restore();
}

function drawXAxis() {
  const y = TOTAL_H - MARGIN.bottom + 4;
  ctx.font = '11px IBM Plex Mono';
  ctx.fillStyle = theme.xAxisText;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  // Work in integer seconds to avoid floating-point modulo errors
  const rangeMin = T_MAX - T_MIN;
  const rangeSec = rangeMin * 60;

  // Adaptive intervals (all in seconds)
  let majorSec, minorSec, tickSec;
  // tickSec = smallest marks (short hairline at bottom only)
  // minorSec = intermediate (dashed line through panels)
  // majorSec = labelled (solid line through panels)
  if (rangeSec > 14400)     { majorSec = 3600; minorSec = 1800; tickSec = 0; }
  else if (rangeSec > 5400) { majorSec = 1800; minorSec = 900;  tickSec = 0; }
  else if (rangeSec > 2700) { majorSec = 900;  minorSec = 300;  tickSec = 60; }
  else if (rangeSec > 1200) { majorSec = 300;  minorSec = 60;   tickSec = 0; }
  else if (rangeSec > 480)  { majorSec = 120;  minorSec = 60;   tickSec = 10; }
  else if (rangeSec > 180)  { majorSec = 60;   minorSec = 10;   tickSec = 0; }
  else                      { majorSec = 30;   minorSec = 10;   tickSec = 0; }

  const finest = tickSec > 0 ? tickSec : minorSec;
  const tMinSec = Math.round(T_MIN * 60);
  const tMaxSec = Math.round(T_MAX * 60);
  const firstSec = Math.ceil(tMinSec / finest) * finest;

  for (let s = firstSec; s <= tMaxSec; s += finest) {
    const tMin = s / 60;
    const x = tToX(tMin);
    if (x < MARGIN.left - 1 || x > MARGIN.left + plotW + 1) continue;

    const isMajor = (s % majorSec) === 0;
    const isMinor = !isMajor && (s % minorSec) === 0;

    if (isMajor) {
      ctx.strokeStyle = theme.gridLineX;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, MARGIN.top);
      ctx.lineTo(x, TOTAL_H - MARGIN.bottom);
      ctx.stroke();
      // Major tick mark on axis
      ctx.strokeStyle = theme.xAxisText;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, TOTAL_H - MARGIN.bottom - 7);
      ctx.lineTo(x, TOTAL_H - MARGIN.bottom);
      ctx.stroke();
      ctx.fillText(minToClockStr(tMin), x, y);
    } else if (isMinor) {
      // Minor ticks: solid marks on axis, 5px tall
      ctx.strokeStyle = theme.xAxisText;
      ctx.lineWidth = 0.7;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, TOTAL_H - MARGIN.bottom - 5);
      ctx.lineTo(x, TOTAL_H - MARGIN.bottom);
      ctx.stroke();
    } else {
      // Finest tick marks: short solid marks, 3px tall
      ctx.strokeStyle = theme.xAxisText;
      ctx.lineWidth = 0.4;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x, TOTAL_H - MARGIN.bottom - 3);
      ctx.lineTo(x, TOTAL_H - MARGIN.bottom);
      ctx.stroke();
    }
  }
  ctx.strokeStyle = theme.xAxisLine;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(MARGIN.left, TOTAL_H - MARGIN.bottom);
  ctx.lineTo(MARGIN.left + plotW, TOTAL_H - MARGIN.bottom);
  ctx.stroke();
}

// ============================================================
// Prepare data arrays
// ============================================================
// SleepU stored compactly: {t0, dt, spo2:[], hr:[], mot:[]}
// Reconstruct time from index: t = t0 + i * dt
const _su = RAW.sleepU;
const spo2Data = [];
const hrData   = [];
const motData  = [];
for (let i = 0; i < _su.spo2.length; i++) {
  const t = _su.t0 + i * _su.dt;
  if (_su.spo2[i] !== null) spo2Data.push({t, v: _su.spo2[i]});
  if (_su.hr[i]   !== null) hrData.push({t, v: _su.hr[i]});
  if (_su.mot[i]  !== null) motData.push({t, v: _su.mot[i]});
}
const rrData    = RAW.rr.map(d => ({t: d[0], v: d[1]}));
const apneaData = RAW.apnea.map(d => ({t: d[0], dur: d[1]}));
const obstrData = RAW.obstr.map(d => ({t: d[0], dur: d[1], sev: d[2]}));
const tiltRRData  = (RAW.tiltRR  || []).map(d => ({t: d[0], v: d[1]}));
const tiltEnvData = (RAW.tiltEnv || []).map(d => ({t: d[0], v: d[1]}));
const tiltRollData = (RAW.tiltRoll || []).map(d => ({t: d[0], v: d[1]}));
const ecgHRData   = (RAW.ecgHR   || []).map(d => ({t: d[0], v: d[1]}));
const hasTilt  = tiltEnvData.length > 0;
const hasRoll  = tiltRollData.length > 0;
const hasEcgHR = ecgHRData.length > 0;

// Map roll angle to color using full rainbow, auto-scaled to data range.
// Computed range from actual data gives maximum contrast for small shifts.
const ROLL_MIN = hasRoll ? Math.min(...tiltRollData.map(d => d.v)) : 0;
const ROLL_MAX = hasRoll ? Math.max(...tiltRollData.map(d => d.v)) : 1;
// Use percentile-based range to ignore brief outliers (e.g. getting out of bed)
const rollSorted = hasRoll ? tiltRollData.map(d => d.v).sort((a,b) => a - b) : [0];
const ROLL_P05 = rollSorted[Math.floor(rollSorted.length * 0.03)];
const ROLL_P95 = rollSorted[Math.floor(rollSorted.length * 0.97)];
const ROLL_LO = ROLL_P05;
const ROLL_HI = ROLL_P95;
const ROLL_SPAN = Math.max(ROLL_HI - ROLL_LO, 0.5); // avoid div-by-zero

function rollToColor(angle, alpha) {
  const a = alpha || 0.7;
  // Clamp to percentile range, then map 0..1
  const t = Math.max(0, Math.min(1, (angle - ROLL_LO) / ROLL_SPAN));
  // Full rainbow hue sweep: 240 (blue) -> 180 (cyan) -> 120 (green) -> 60 (yellow) -> 0 (red)
  const h = 240 - t * 240;
  return `hsla(${Math.round(h)}, 75%, 55%, ${a})`;
}

// ============================================================
// Compute dynamic Y-axis ranges from data
// ============================================================
const spo2Vals = spo2Data.map(d => d.v);
const hrVals = hrData.map(d => d.v).concat(ecgHRData.map(d => d.v));
const rrVals = rrData.map(d => d.v);

const SPO2_MIN = Math.max(70, Math.min(...spo2Vals) - 2);
const SPO2_MAX = 100;
const HR_MIN = Math.max(30, Math.floor((Math.min(...hrVals) - 3) / 5) * 5);
const HR_MAX = Math.min(150, Math.ceil((Math.max(...hrVals) + 3) / 5) * 5);
const RR_MIN = 0;
const allRRVals = rrVals.concat(tiltRRData.map(d => d.v));
const RR_MAX = Math.ceil((Math.max(...allRRVals) + 2) / 5) * 5;
const MOT_MAX = Math.min(50, Math.max(10, Math.ceil(Math.max(...motData.map(d => d.v)) / 5) * 5));
const ENV_MAX = hasTilt ? Math.min(5, Math.ceil(Math.max(...tiltEnvData.map(d => d.v)) * 10) / 10) : 1;

// Compute apnea Y-axis range (seconds)
const APNEA_MAX_DUR = apneaData.length > 0
  ? Math.ceil(Math.max(...apneaData.map(d => d.dur)) / 10) * 10
  : 60;

// Compute HR moving average (window ~5 minutes centered)
const HR_MA_WINDOW = 75;  // samples each side (~5 min half-window at 4s interval)
const hrMA = [];
for (let i = 0; i < hrData.length; i++) {
  let lo = Math.max(0, i - HR_MA_WINDOW);
  let hi = Math.min(hrData.length - 1, i + HR_MA_WINDOW);
  let sum = 0;
  for (let j = lo; j <= hi; j++) sum += hrData[j].v;
  hrMA.push({t: hrData[i].t, v: Math.round(sum / (hi - lo + 1) * 10) / 10});
}

// Detect SpO2 dips below 85% (local minima with >=20 sample spacing)
const SPO2_DIP_THRESH = 85;
const spo2Dips = [];
const DIP_MIN_SEP = 20; // minimum samples between reported dips
for (let i = 1; i < spo2Data.length - 1; i++) {
  const v = spo2Data[i].v;
  if (v >= SPO2_DIP_THRESH) continue;
  // Check it's a local minimum within a window of ±5 samples
  let isMin = true;
  for (let j = Math.max(0, i-5); j <= Math.min(spo2Data.length-1, i+5); j++) {
    if (j !== i && spo2Data[j].v < v) { isMin = false; break; }
  }
  if (!isMin) continue;
  // Enforce spacing from previous dip
  if (spo2Dips.length > 0 && i - spo2Dips[spo2Dips.length-1].idx < DIP_MIN_SEP) {
    // Keep the deeper one
    if (v < spo2Dips[spo2Dips.length-1].v) spo2Dips[spo2Dips.length-1] = {idx: i, t: spo2Data[i].t, v};
    continue;
  }
  spo2Dips.push({idx: i, t: spo2Data[i].t, v});
}

// Compute HR 15-minute moving average (for good-sleep detection)
const HR_MA15_WINDOW = 112;  // samples each side (~7.5 min half-window → 15 min total at 4s)
const hrMA15 = [];
for (let i = 0; i < hrData.length; i++) {
  let lo = Math.max(0, i - HR_MA15_WINDOW);
  let hi = Math.min(hrData.length - 1, i + HR_MA15_WINDOW);
  let sum = 0;
  for (let j = lo; j <= hi; j++) sum += hrData[j].v;
  hrMA15.push({t: hrData[i].t, v: sum / (hi - lo + 1)});
}

// ============================================================
// Detect "good sleep" periods (≥20 min contiguous)
//   1. No apnea events
//   2. No obstructions (moderate or severe)
//   3. SpO2 ≥ threshold (96% for SleepU, 95% for Checkme)
//   4. 15-min HR MA non-increasing
// ============================================================
const GOOD_MIN_DURATION = 15; // minutes
const HR_RISE_TOL = 1.0;     // bpm tolerance for "non-increasing"
const HR_LOOKBACK = 30;      // samples (~2 min) for slope check
const GOOD_SPO2_MIN = %%GOOD_SPO2_MIN%%;  // SpO2 threshold for restful detection

// Build sorted event intervals for quick overlap checks
const evtIntervals = [];
for (const a of apneaData) evtIntervals.push({t0: a.t, t1: a.t + a.dur / 60});
for (const o of obstrData) evtIntervals.push({t0: o.t, t1: o.t + o.dur / 60});
evtIntervals.sort((a, b) => a.t0 - b.t0);

// Scan at HR data resolution
const goodMask = new Uint8Array(hrData.length); // 1 = good at this sample
for (let i = 0; i < hrData.length; i++) {
  const t = hrData[i].t;

  // Check 1 & 2: no apnea or obstruction overlapping this time
  let hasEvent = false;
  for (const ev of evtIntervals) {
    if (ev.t0 > t + 0.5) break; // events are sorted, no need to check further
    if (ev.t1 >= t - 0.5) { hasEvent = true; break; }
  }
  if (hasEvent) continue;

  // Check 3: SpO2 ≥ threshold (find nearest SpO2 sample)
  // Binary-ish search: SpO2 data is same resolution as HR from same source
  let spo2Ok = false;
  for (let si = Math.max(0, i - 2); si <= Math.min(spo2Data.length - 1, i + 2); si++) {
    if (Math.abs(spo2Data[si].t - t) < 0.2) {
      if (spo2Data[si].v >= GOOD_SPO2_MIN) spo2Ok = true;
      break;
    }
  }
  if (!spo2Ok) continue;

  // Check 4: 15-min HR MA avg HR below 75 bpm
  if (i >= HR_LOOKBACK) {
    if (hrMA15[i].v >= 75) continue;
  }

  
  goodMask[i] = 1;
}

// Extract contiguous runs ≥ GOOD_MIN_DURATION
const goodPeriods = [];
let runStart = -1;
for (let i = 0; i <= hrData.length; i++) {
  if (i < hrData.length && goodMask[i]) {
    if (runStart < 0) runStart = i;
  } else {
    if (runStart >= 0) {
      const tStart = hrData[runStart].t;
      const tEnd = hrData[i - 1].t;
      if (tEnd - tStart >= GOOD_MIN_DURATION) {
        goodPeriods.push({t0: tStart, t1: tEnd});
      }
      runStart = -1;
    }
  }
}

// Append restful total to subtitle
const restfulTotal = Math.round(goodPeriods.reduce((s, gp) => s + (gp.t1 - gp.t0), 0));
document.querySelector('.subtitle').insertAdjacentHTML('beforeend',
  ` \u00a0\u00a0\u2022 \u00a0Restful: ${restfulTotal}m`);

// Generate tick arrays
function makeTicks(lo, hi, step) {
  const ticks = [];
  for (let v = Math.ceil(lo/step)*step; v <= hi; v += step) ticks.push(v);
  return ticks;
}

const spo2Ticks = makeTicks(SPO2_MIN, SPO2_MAX, 5);
const hrTicks = makeTicks(HR_MIN, HR_MAX, (HR_MAX - HR_MIN > 40) ? 10 : 5);
const rrTicks = makeTicks(RR_MIN, RR_MAX, 5);
const apneaTicks = makeTicks(0, APNEA_MAX_DUR, APNEA_MAX_DUR > 80 ? 20 : 10);
const motTicks = makeTicks(0, MOT_MAX, MOT_MAX > 30 ? Math.ceil(MOT_MAX / 3 / 5) * 5 : 5);

// ============================================================
// DRAW
// ============================================================
function draw() {
  ctx.fillStyle = theme.bodyBg;
  ctx.fillRect(0, 0, W, TOTAL_H);
  for (let i = 0; i < 5; i++) drawPanelBg(i);
  drawXAxis();

  // Panel 0: SpO2
  drawYAxis(SPO2_MIN, SPO2_MAX, 0, spo2Ticks, 'SpO\u2082 %', theme.spo2);
  const y90 = valToY(90, SPO2_MIN, SPO2_MAX, 0);
  ctx.strokeStyle = theme.spo2Ref;
  ctx.lineWidth = 1;
  ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(MARGIN.left, y90); ctx.lineTo(MARGIN.left+plotW, y90); ctx.stroke();
  ctx.setLineDash([]);
  drawLine(spo2Data, 't', 'v', SPO2_MIN, SPO2_MAX, 0, theme.spo2, 1.0);

  // Label SpO2 dips below 85% (skip overlapping labels)
  ctx.font = 'bold 11px IBM Plex Mono';
  ctx.fillStyle = theme.spo2DipText;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  let lastLabelRight = -Infinity;
  const LABEL_PAD = 4; // minimum pixel gap between labels
  for (const dip of spo2Dips) {
    const dx = tToX(dip.t);
    const dy = valToY(dip.v, SPO2_MIN, SPO2_MAX, 0);
    const txt = dip.v + '%';
    const tw = ctx.measureText(txt).width;
    const labelLeft = dx - tw / 2;
    if (labelLeft < lastLabelRight + LABEL_PAD) continue; // skip overlap
    ctx.fillText(txt, dx, dy + 3);
    lastLabelRight = dx + tw / 2;
  }

  // Panel 1: HR
  drawYAxis(HR_MIN, HR_MAX, 1, hrTicks, 'HR bpm', theme.ecgHR);
  ctx.save();
  ctx.setLineDash([4, 3]);
  drawLine(hrData, 't', 'v', HR_MIN, HR_MAX, 1, theme.hr, 1.2);
  ctx.setLineDash([]);
  ctx.restore();
  drawLine(hrMA,   't', 'v', HR_MIN, HR_MAX, 1, theme.hrMA, 2.0);
  if (hasEcgHR) {
    drawLine(ecgHRData, 't', 'v', HR_MIN, HR_MAX, 1, theme.ecgHR, 1.0);
  }

  // Panel 2: Breath Envelope amplitude only
  if (hasTilt) {
    const envStep = ENV_MAX > 2 ? 1 : 0.5;
    drawYAxis(0, ENV_MAX, 2, makeTicks(0, ENV_MAX, envStep), 'Breath amp °', theme.tiltEnvLine);

    const P2_TOP = panelTop(2);
    const P2_H = PANEL_HEIGHTS[2];
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, P2_TOP, plotW, P2_H);
    ctx.clip();

    // Filled area under envelope
    ctx.fillStyle = theme.tiltEnv;
    ctx.beginPath();
    let started = false;
    const baseY2 = valToY(0, 0, ENV_MAX, 2);
    for (const d of tiltEnvData) {
      const x = tToX(d.t);
      if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
      const y = valToY(Math.min(d.v, ENV_MAX), 0, ENV_MAX, 2);
      if (!started) { ctx.moveTo(x, baseY2); started = true; }
      ctx.lineTo(x, y);
    }
    for (let i = tiltEnvData.length - 1; i >= 0; i--) {
      const x = tToX(tiltEnvData[i].t);
      if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
      ctx.lineTo(x, baseY2);
      break;
    }
    ctx.closePath();
    ctx.fill();

    // Envelope top line
    ctx.strokeStyle = theme.tiltEnvLine;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    started = false;
    for (const d of tiltEnvData) {
      const x = tToX(d.t);
      if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
      const y = valToY(Math.min(d.v, ENV_MAX), 0, ENV_MAX, 2);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // Panel 3: Events (apnea bars scaled to seconds Y-axis)
  const EVT_TOP = panelTop(3);
  const EVT_H = PANEL_HEIGHTS[3];
  drawYAxis(0, APNEA_MAX_DUR, 3, apneaTicks, 'Apnea (s)', theme.evtLabel);

  // Draw good-sleep indicators (green dotted line at panel midpoint)
  const goodY = EVT_TOP + EVT_H * 0.5;
  ctx.strokeStyle = theme.goodSleep;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  for (const gp of goodPeriods) {
    const x0 = Math.max(tToX(gp.t0), MARGIN.left);
    const x1 = Math.min(tToX(gp.t1), MARGIN.left + plotW);
    if (x1 <= x0) continue;
    ctx.beginPath();
    ctx.moveTo(x0, goodY);
    ctx.lineTo(x1, goodY);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Draw all apnea bars first (clipped to plot area)
  const apneaBars = [];
  const plotLeft = MARGIN.left;
  const plotRight = MARGIN.left + plotW;
  for (const a of apneaData) {
    let x = tToX(a.t);
    const w = Math.max(1.5, (a.dur / 60) / (T_MAX - T_MIN) * plotW);
    let x2 = x + w;
    // Clip to plot area
    if (x2 < plotLeft || x > plotRight) continue;
    x = Math.max(x, plotLeft);
    x2 = Math.min(x2, plotRight);
    const clampDur = Math.min(a.dur, APNEA_MAX_DUR);
    const barBot = EVT_TOP + EVT_H - 6;
    const barTop = valToY(clampDur, 0, APNEA_MAX_DUR, 3);
    const h = barBot - barTop;
    const isLong = a.dur > 50;
    ctx.fillStyle = isLong ? theme.apneaLong : theme.apneaBar;
    ctx.fillRect(x, barTop, x2 - x, h);
    apneaBars.push({x, w: x2 - x, barTop, barBot, dur: a.dur, isLong});
  }

  // Collect obstruction dot bounding boxes
  const obstrBoxes = [];
  for (const o of obstrData) {
    const ox = tToX(o.t);
    if (ox < plotLeft || ox > plotRight) continue;
    const oy = EVT_TOP + (o.sev === 1 ? 6 : 16);
    const r = o.sev === 1 ? 2.5 : 2;
    obstrBoxes.push({x: ox - r, y: oy - r, w: r*2, h: r*2});
    ctx.beginPath();
    ctx.arc(ox, oy, r, 0, Math.PI*2);
    ctx.fillStyle = o.sev === 1 ? theme.obstrSevere : theme.obstrMod;
    ctx.fill();
  }

  // Place >50s apnea labels: prefer left/right of bar, no leader lines
  ctx.font = 'bold 11px IBM Plex Mono';
  const LABEL_H = 12;
  const LABEL_GAP = 3;
  const placedLabels = []; // {x, y, w, h} bounding boxes

  function boxOverlaps(ax, ay, aw, ah, bx, by, bw, bh) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
  }

  function labelCollides(lx, ly, lw) {
    for (const ob of obstrBoxes) {
      if (boxOverlaps(lx, ly, lw, LABEL_H, ob.x - 2, ob.y - 2, ob.w + 4, ob.h + 4)) return true;
    }
    for (const pl of placedLabels) {
      if (boxOverlaps(lx, ly, lw, LABEL_H, pl.x, pl.y, pl.w, pl.h)) return true;
    }
    for (const b of apneaBars) {
      if (boxOverlaps(lx, ly, lw, LABEL_H, b.x - 1, b.barTop, b.w + 2, b.barBot - b.barTop)) return true;
    }
    return false;
  }

  // Collect long bars and group into clusters (nearby = within 40px)
  const longBars = apneaBars.filter(b => b.isLong);
  const CLUSTER_DIST = 40;
  const clusters = [];
  for (const b of longBars) {
    const cx = b.x + b.w / 2;
    if (clusters.length > 0) {
      const last = clusters[clusters.length - 1];
      const lastCx = last[last.length - 1].x + last[last.length - 1].w / 2;
      if (cx - lastCx < CLUSTER_DIST) {
        last.push(b);
        continue;
      }
    }
    clusters.push([b]);
  }

  for (const cluster of clusters) {
    for (let ci = 0; ci < cluster.length; ci++) {
      const b = cluster[ci];
      const label = Math.round(b.dur) + 's';
      const lw = ctx.measureText(label).width + 4;
      const midY = b.barTop + Math.min(12, (b.barBot - b.barTop) * 0.3);

      // Build ordered candidate positions
      const candidates = [];
      const leftX = b.x - lw - LABEL_GAP;
      const rightX = b.x + b.w + LABEL_GAP;
      const aboveX = b.x + b.w/2 - lw/2;
      const aboveY = b.barTop - LABEL_H - 2;

      if (cluster.length === 1) {
        // Isolated: try right, left, then above
        candidates.push({x: rightX, y: midY});
        candidates.push({x: leftX, y: midY});
        candidates.push({x: aboveX, y: aboveY});
      } else if (cluster.length === 2) {
        // Pair: first goes left, second goes right
        if (ci === 0) {
          candidates.push({x: leftX, y: midY});
          candidates.push({x: rightX, y: midY});
          candidates.push({x: aboveX, y: aboveY});
        } else {
          candidates.push({x: rightX, y: midY});
          candidates.push({x: leftX, y: midY});
          candidates.push({x: aboveX, y: aboveY});
        }
      } else {
        // 3+: outer go outward, middle ones go above then left/right
        if (ci === 0) {
          candidates.push({x: leftX, y: midY});
          candidates.push({x: aboveX, y: aboveY});
          candidates.push({x: rightX, y: midY});
        } else if (ci === cluster.length - 1) {
          candidates.push({x: rightX, y: midY});
          candidates.push({x: aboveX, y: aboveY});
          candidates.push({x: leftX, y: midY});
        } else {
          candidates.push({x: aboveX, y: aboveY});
          candidates.push({x: rightX, y: midY});
          candidates.push({x: leftX, y: midY});
        }
      }

      // Try each candidate, pick first that fits
      let placed = false;
      for (const c of candidates) {
        if (c.x < MARGIN.left - 4 || c.x + lw > MARGIN.left + plotW + 4) continue;
        if (c.y < EVT_TOP) continue;
        if (!labelCollides(c.x, c.y, lw)) {
          ctx.fillStyle = theme.apneaLongText;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillText(label, c.x, c.y);
          placedLabels.push({x: c.x, y: c.y, w: lw, h: LABEL_H});
          placed = true;
          break;
        }
      }
      // Fallback: draw to right regardless
      if (!placed) {
        const fx = rightX;
        const fy = midY;
        ctx.fillStyle = theme.apneaLongText;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(label, fx, fy);
        placedLabels.push({x: fx, y: fy, w: lw, h: LABEL_H});
      }
    }
  }

  // --- Hourly apnea count labels at bottom of panel ---
  {
    // Compute apnea counts per clock hour
    const ahiByHour = {};
    for (const a of apneaData) {
      // a.t is minutes since midnight of ref_date
      const hourKey = Math.floor(a.t / 60);
      ahiByHour[hourKey] = (ahiByHour[hourKey] || 0) + 1;
    }
    ctx.font = 'bold 12px IBM Plex Mono';
    ctx.fillStyle = theme.ahiHourlyText;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const labelY = EVT_TOP + EVT_H - 4;
    // Iterate visible whole hours
    const firstHour = Math.ceil(T_MIN / 60);
    const lastHour = Math.floor(T_MAX / 60);
    for (let h = firstHour; h <= lastHour; h++) {
      const count = ahiByHour[h] || 0;
      const hourMidT = h * 60 + 30; // center of the hour
      const x = tToX(hourMidT);
      if (x < plotLeft || x > plotRight) continue;
      ctx.fillText(String(count), x, labelY);
    }
  }

  // Panel 4: Motion
  const MOT_TOP = panelTop(4);
  const MOT_H = PANEL_HEIGHTS[4];
  drawYAxis(0, MOT_MAX, 4, motTicks, 'Motion', theme.motionAxis);
  for (const m of motData) {
    if (m.v === 0) continue;
    const x = tToX(m.t);
    if (x < MARGIN.left || x > MARGIN.left + plotW) continue;
    const barH = Math.max(1, (Math.min(m.v, MOT_MAX) / MOT_MAX) * (MOT_H - 8));
    const y = MOT_TOP + MOT_H - 3 - barH;
    ctx.fillStyle = theme.motionBar;
    ctx.fillRect(x - 0.5, y, 1.5, barH);
  }

  // Roll angle color strip at top of motion panel
  if (hasRoll) {
    const rollH = 6;
    const rollY = MOT_TOP + 1;
    for (let i = 0; i < tiltRollData.length - 1; i++) {
      const d = tiltRollData[i];
      const x1 = tToX(d.t);
      const x2 = tToX(tiltRollData[i + 1].t);
      if (x2 < MARGIN.left || x1 > MARGIN.left + plotW) continue;
      ctx.fillStyle = rollToColor(d.v);
      ctx.fillRect(x1, rollY, Math.max(1, x2 - x1), rollH);
    }
  }
}

draw();

// ============================================================
// Tooltip / Crosshair
// ============================================================
const tooltip = document.getElementById('tooltip');
const chartArea = document.getElementById('chartArea');

function findNearest(arr, t) {
  const maxDist = Math.max(0.5, (T_MAX - T_MIN) * 0.006); // ~0.6% of visible range
  let best = null, bestDist = Infinity;
  for (const pt of arr) {
    const d = Math.abs(pt.t - t);
    if (d < bestDist) { bestDist = d; best = pt; }
  }
  return bestDist < maxDist ? best : null;
}

chartArea.addEventListener('mousemove', (e) => {
  if (panState) {
    crosshair.style.display = 'none';
    tooltip.style.display = 'none';
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  if (mx < MARGIN.left || mx > MARGIN.left + plotW || my > TOTAL_H - MARGIN.bottom) {
    crosshair.style.display = 'none';
    tooltip.style.display = 'none';
    return;
  }

  const t = xToT(mx);
  crosshair.style.display = 'block';
  crosshair.style.left = mx + 'px';

  const sp = findNearest(spo2Data, t);
  const hr = findNearest(hrData, t);
  const hrm = findNearest(hrMA, t);
  const mo = findNearest(motData, t);

  let apnStr = '';
  for (const a of apneaData) {
    if (t >= a.t && t <= a.t + a.dur/60) {
      apnStr = `${a.dur.toFixed(0)}s apnea`;
      break;
    }
  }

  let html = `<div class="tt-time">${minToClockStrFull(t)}</div>`;
  if (sp) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.spo2}"></div>SpO\u2082: ${sp.v}%</div>`;
  if (hr) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.hr}"></div>Pleth HR: ${hr.v} bpm</div>`;
  if (hrm) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.hrMA}"></div>HR avg: ${hrm.v} bpm</div>`;
  if (hasEcgHR) {
    const ecg = findNearest(ecgHRData, t);
    if (ecg) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.ecgHR}"></div>ECG HR: ${ecg.v} bpm</div>`;
  }
  if (hasTilt) {
    const tenv = findNearest(tiltEnvData, t);
    if (tenv) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.tiltEnvLine}"></div>Breath amp: ${tenv.v.toFixed(2)}°</div>`;
  }
  if (mo && mo.v > 0) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.motionAxis}"></div>Motion: ${mo.v}</div>`;
  if (hasRoll) {
    const rl = findNearest(tiltRollData, t);
    if (rl) html += `<div class="tt-row"><div class="tt-dot" style="background:${rollToColor(rl.v, 1)}"></div>Roll: ${rl.v.toFixed(1)}°</div>`;
  }
  if (apnStr) html += `<div class="tt-row"><div class="tt-dot" style="background:${theme.obstrSevere}"></div>${apnStr}</div>`;

  tooltip.innerHTML = html;
  tooltip.style.display = 'block';

  let tx = mx + 16;
  let ty = my - 10;
  const tw = tooltip.offsetWidth;
  if (tx + tw > W - 8) tx = mx - tw - 16;
  if (ty < 4) ty = 4;
  tooltip.style.left = tx + 'px';
  tooltip.style.top = ty + 'px';
});

chartArea.addEventListener('mouseleave', () => {
  crosshair.style.display = 'none';
  tooltip.style.display = 'none';
});

// ============================================================
// Zoom & Pan
// ============================================================
const ZOOM_FACTOR = 0.8; // scroll-zoom sensitivity (smaller = zoom faster)
const MIN_RANGE = 2;     // minimum visible range in minutes

function updateZoomUI() {
  const btn = document.getElementById('resetZoom');
  const isZoomed = Math.abs(T_MIN - T_MIN_FULL) > 0.1 || Math.abs(T_MAX - T_MAX_FULL) > 0.1;
  btn.style.display = isZoomed ? 'block' : 'none';
  canvas.classList.toggle('zoomed', isZoomed);
}

function resetZoom() {
  T_MIN = T_MIN_FULL;
  T_MAX = T_MAX_FULL;
  updateZoomUI();
  draw();
}

// Wheel zoom: centered on cursor position
chartArea.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  if (mx < MARGIN.left || mx > MARGIN.left + plotW) return;

  const tCursor = xToT(mx);
  const range = T_MAX - T_MIN;
  const factor = e.deltaY > 0 ? 1 / ZOOM_FACTOR : ZOOM_FACTOR;
  const newRange = Math.max(MIN_RANGE, Math.min(T_MAX_FULL - T_MIN_FULL, range * factor));

  // Fraction of cursor within the current view
  const frac = (tCursor - T_MIN) / range;
  T_MIN = tCursor - frac * newRange;
  T_MAX = tCursor + (1 - frac) * newRange;

  // Clamp to full range
  if (T_MIN < T_MIN_FULL) { T_MAX += T_MIN_FULL - T_MIN; T_MIN = T_MIN_FULL; }
  if (T_MAX > T_MAX_FULL) { T_MIN -= T_MAX - T_MAX_FULL; T_MAX = T_MAX_FULL; }
  T_MIN = Math.max(T_MIN, T_MIN_FULL);
  T_MAX = Math.min(T_MAX, T_MAX_FULL);

  updateZoomUI();
  draw();
}, {passive: false});

// Click-drag pan
let panState = null;
chartArea.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  if (mx < MARGIN.left || mx > MARGIN.left + plotW) return;
  // Only pan when zoomed in
  if (Math.abs(T_MAX - T_MIN - (T_MAX_FULL - T_MIN_FULL)) < 0.1) return;
  panState = { startX: e.clientX, startTMin: T_MIN, startTMax: T_MAX };
  canvas.classList.add('panning');
});

window.addEventListener('mousemove', (e) => {
  if (!panState) return;
  const dx = e.clientX - panState.startX;
  const dtPerPx = (panState.startTMax - panState.startTMin) / plotW;
  let shift = -dx * dtPerPx;

  let newMin = panState.startTMin + shift;
  let newMax = panState.startTMax + shift;
  if (newMin < T_MIN_FULL) { newMax += T_MIN_FULL - newMin; newMin = T_MIN_FULL; }
  if (newMax > T_MAX_FULL) { newMin -= newMax - T_MAX_FULL; newMax = T_MAX_FULL; }

  T_MIN = Math.max(newMin, T_MIN_FULL);
  T_MAX = Math.min(newMax, T_MAX_FULL);
  updateZoomUI();
  draw();
});

window.addEventListener('mouseup', () => {
  if (panState) {
    panState = null;
    canvas.classList.remove('panning');
  }
});

// ============================================================
// Save PNG at specified resolution with header
// ============================================================
function savePNG(exportScale) {
  const EXPORT_SCALE = exportScale || 2;
  const isLight = document.body.classList.contains('light-theme');

  // --- Measure header content from DOM ---
  const titleText = document.querySelector('h1').childNodes[0].textContent.trim();
  const subtitleText = document.querySelector('.subtitle').textContent;
  const legendItems = [];
  document.querySelectorAll('.legend-item').forEach(item => {
    const swatches = item.querySelectorAll('.legend-swatch');
    const swatch = swatches[0];
    if (!swatch) return;
    const cs = getComputedStyle(swatch);
    const entry = {
      color: cs.backgroundColor,
      borderColor: cs.borderTopColor,
      isDot: swatch.classList.contains('dot'),
      isBar: swatch.classList.contains('bar'),
      text: item.textContent.trim()
    };
    // Collect extra swatches for combined items (e.g. "Obstr: mod sev")
    if (swatches.length > 1) {
      entry.extraSwatches = [];
      for (let s = 1; s < swatches.length; s++) {
        const ecs = getComputedStyle(swatches[s]);
        entry.extraSwatches.push({
          color: ecs.backgroundColor,
          isDot: swatches[s].classList.contains('dot'),
          isBar: swatches[s].classList.contains('bar'),
        });
      }
    }
    legendItems.push(entry);
  });

  // --- Header layout constants ---
  const PAD_TOP = 12;
  const TITLE_H = 24;
  const GAP1 = 4;
  const SUB_H = 16;
  const GAP2 = 8;
  const LEG_PAD = 8;
  const LEG_ROW_H = 16;
  const LEG_BOT = 8;

  // Pre-measure legend to determine if it wraps
  const tmpC = document.createElement('canvas');
  const tmpX = tmpC.getContext('2d');
  tmpX.font = '11px IBM Plex Mono';
  const LEG_ITEM_GAP = 12;
  const LEG_SWATCH_W = 20;
  const maxLegW = W - 32;
  let rows = [[]];
  let rowW = 0;
  for (const item of legendItems) {
    const iw = LEG_SWATCH_W + tmpX.measureText(item.text).width + LEG_ITEM_GAP;
    if (rowW > 0 && rowW + iw > maxLegW) {
      rows.push([]);
      rowW = 0;
    }
    rows[rows.length - 1].push(item);
    rowW += iw;
  }
  const legendBoxH = LEG_PAD + rows.length * LEG_ROW_H + LEG_BOT;
  const headerH = PAD_TOP + TITLE_H + GAP1 + SUB_H + GAP2 + legendBoxH + 8;

  // --- Create offscreen canvas ---
  const totalH = headerH + TOTAL_H;
  const off = document.createElement('canvas');
  off.width = W * EXPORT_SCALE;
  off.height = totalH * EXPORT_SCALE;
  const oc = off.getContext('2d');
  oc.setTransform(EXPORT_SCALE, 0, 0, EXPORT_SCALE, 0, 0);

  // Background
  oc.fillStyle = theme.bodyBg;
  oc.fillRect(0, 0, W, totalH);

  // Title
  let y = PAD_TOP;
  oc.font = '600 20px IBM Plex Sans, sans-serif';
  oc.fillStyle = theme.titleText;
  oc.textAlign = 'left';
  oc.textBaseline = 'top';
  oc.fillText(titleText, 0, y);

  // Version stamp (inline after title, subtle)
  const versionText = document.getElementById('versionStamp').textContent;
  const titleWidth = oc.measureText(titleText).width;
  oc.font = '9px IBM Plex Mono, monospace';
  oc.fillStyle = theme.versionText;
  oc.textAlign = 'left';
  oc.textBaseline = 'top';
  oc.fillText(versionText, titleWidth + 12, y + 6);

  y += TITLE_H + GAP1;

  // Subtitle
  oc.font = '13px IBM Plex Mono, monospace';
  oc.fillStyle = theme.subtitleText;
  oc.fillText(subtitleText, 0, y);
  y += SUB_H + GAP2;

  // Legend box
  oc.fillStyle = theme.legendBg;
  oc.strokeStyle = theme.legendBorder;
  oc.lineWidth = 1;
  const legBoxY = y;
  oc.beginPath();
  oc.roundRect(16, legBoxY, W - 32, legendBoxH, 6);
  oc.fill();
  oc.stroke();

  // Legend items
  oc.font = '11px IBM Plex Mono, monospace';
  let ly = legBoxY + LEG_PAD;
  for (const row of rows) {
    let lx = 32;
    for (const item of row) {
      // Combined items with multiple swatches (e.g. "Obstr: mod sev")
      if (item.extraSwatches) {
        oc.fillStyle = theme.legendText;
        oc.textAlign = 'left';
        oc.textBaseline = 'top';
        // Draw: "Obstr: " then dot1 "mod " then dot2 "sev"
        const label = 'Obstr: ';
        oc.fillText(label, lx, ly);
        lx += oc.measureText(label).width;
        // First dot (mod)
        oc.fillStyle = item.color;
        oc.beginPath(); oc.arc(lx + 4, ly + 6, 4, 0, Math.PI * 2); oc.fill();
        lx += 10;
        oc.fillStyle = theme.legendText;
        oc.fillText('mod ', lx, ly);
        lx += oc.measureText('mod ').width;
        // Second dot (sev)
        oc.fillStyle = item.extraSwatches[0].color;
        oc.beginPath(); oc.arc(lx + 4, ly + 6, 4, 0, Math.PI * 2); oc.fill();
        lx += 10;
        oc.fillStyle = theme.legendText;
        oc.fillText('sev', lx, ly);
        lx += oc.measureText('sev').width + LEG_ITEM_GAP;
        continue;
      }
      // Swatch
      oc.fillStyle = item.color;
      if (item.isDot) {
        oc.beginPath();
        oc.arc(lx + 4, ly + 6, 4, 0, Math.PI * 2);
        oc.fill();
      } else if (item.isBar) {
        // Special handling for gradient swatches (e.g. Roll)
        if (item.text.includes('Roll')) {
          const grad = oc.createLinearGradient(lx, 0, lx + 14, 0);
          grad.addColorStop(0, 'hsl(240,75%,55%)');
          grad.addColorStop(0.25, 'hsl(180,75%,55%)');
          grad.addColorStop(0.5, 'hsl(120,75%,55%)');
          grad.addColorStop(0.75, 'hsl(60,75%,55%)');
          grad.addColorStop(1, 'hsl(0,75%,55%)');
          oc.fillStyle = grad;
        }
        oc.fillRect(lx, ly + 1, 14, 10);
      } else {
        // Check if this is the "Restful" item (dotted line style)
        if (item.text.includes('Restful')) {
          oc.strokeStyle = item.borderColor || item.color;
          oc.lineWidth = 2;
          oc.setLineDash([6, 4]);
          oc.beginPath();
          oc.moveTo(lx, ly + 6);
          oc.lineTo(lx + 14, ly + 6);
          oc.stroke();
          oc.setLineDash([]);
        } else {
          // Regular solid line
          oc.fillRect(lx, ly + 4, 14, 4);
        }
      }
      lx += LEG_SWATCH_W;
      // Label
      oc.fillStyle = theme.legendText;
      oc.textAlign = 'left';
      oc.textBaseline = 'top';
      oc.fillText(item.text, lx, ly);
      lx += oc.measureText(item.text).width + LEG_ITEM_GAP;
    }
    ly += LEG_ROW_H;
  }

  // --- Render chart at export scale, then composite ---
  canvas.width = W * EXPORT_SCALE;
  canvas.height = TOTAL_H * EXPORT_SCALE;
  ctx.setTransform(EXPORT_SCALE, 0, 0, EXPORT_SCALE, 0, 0);
  draw();

  oc.setTransform(1, 0, 0, 1, 0, 0);
  oc.drawImage(canvas, 0, headerH * EXPORT_SCALE,
               W * EXPORT_SCALE, TOTAL_H * EXPORT_SCALE);

  // --- Download ---
  const link = document.createElement('a');
  link.download = `sleep_dashboard_%%TITLE_DATE%%_${EXPORT_SCALE}x.png`;
  link.href = off.toDataURL('image/png');
  link.click();

  // --- Restore main canvas ---
  canvas.width = W * DPR;
  canvas.height = TOTAL_H * DPR;
  canvas.style.height = TOTAL_H + 'px';
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  draw();
}

window.addEventListener('resize', () => {
  computeLayout();
  draw();
});
</script>
</body>
</html>"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_sleep_dashboard.py <input_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)

    # Find files
    try:
        files = find_input_files(input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Found input files:")
    for key, path in files.items():
        if key == 'oximeter_type':
            continue
        print(f"  {key}: {os.path.basename(path)}")

    oximeter_type = files['oximeter_type']

    # Determine reference date from the oximeter filename or first timestamp
    with open(files['oximeter'], newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        first_dt = parse_sleepu_time(first_row['Time'])
    ref_date = first_dt.date()
    print(f"  Reference date: {ref_date}")

    # Read all data
    if oximeter_type == 'checkme':
        sleepu = read_checkme(files['oximeter'], ref_date)
        oximeter_label = "Checkme O2 Ultra"
    else:
        sleepu = read_sleepu(files['oximeter'], ref_date)
        oximeter_label = "SleepU"
    rr    = read_resp_rate(files['resp_rate'], ref_date) if 'resp_rate' in files else []
    apnea = read_apnea(files['apnea'], ref_date)         if 'apnea'     in files else []
    obstr = read_obstructions(files['obstruct'], ref_date) if 'obstruct' in files else []

    # Optional tilt-based breathing data
    tilt_data = None
    if 'tilt_breath' in files:
        tilt_data = read_tilt_breath(files['tilt_breath'], ref_date)
        if tilt_data:
            # Clip to oximeter time range
            oxi_t0 = sleepu['t0']
            oxi_t1 = oxi_t0 + (len(sleepu['spo2']) - 1) * sleepu['dt']
            tilt_data['tiltRR'] = [p for p in tilt_data['tiltRR'] if oxi_t0 <= p[0] <= oxi_t1]
            tilt_data['tiltEnv'] = [p for p in tilt_data['tiltEnv'] if oxi_t0 <= p[0] <= oxi_t1]
            tilt_data['tiltRoll'] = [p for p in tilt_data.get('tiltRoll', []) if oxi_t0 <= p[0] <= oxi_t1]
            print(f"  Tilt resp rate samples: {len(tilt_data['tiltRR'])} (clipped to oximeter range)")
            print(f"  Tilt envelope samples: {len(tilt_data['tiltEnv'])}")
            print(f"  Tilt roll samples: {len(tilt_data['tiltRoll'])}")

    # Optional ECG-derived heart rate (median-3 smoothed, resampled to oximeter bin rate)
    ecg_hr = []
    if 'ecg_beats' in files:
        ecg_hr = read_ecg_beats(files['ecg_beats'], ref_date)
        print(f"  ECG beats HR samples: {len(ecg_hr)}")

    # -> New: compute apnea counts for dashboard (seconds)
    apnea_count_gt10 = sum(1 for a in apnea if a[1] > 10.0)
    apnea_count_gt50 = sum(1 for a in apnea if a[1] > 50.0)

    # Compute SpO2-derived metrics
    spo2_valid = [v for v in sleepu['spo2'] if v is not None]
    interval_min = sleepu['dt']  # minutes per sample

    # Time with SpO2 < 90%
    samples_below_90 = sum(1 for v in spo2_valid if v < 90)
    time_below_90_min = round(samples_below_90 * interval_min, 1)

    # ODI (Oxygen Desaturation Index): >=3% drops per hour of valid recording
    # Uses adaptive baseline: highest SpO2 in trailing window, count events
    # where SpO2 drops >=3% below that baseline
    def compute_odi(spo2_series, interval_minutes, desat_threshold=3,
                    baseline_window_min=2.0, min_event_gap_min=0.5):
        """Count >=desat_threshold% desaturation events per hour."""
        window_samples = max(1, int(baseline_window_min / interval_minutes))
        gap_samples = max(1, int(min_event_gap_min / interval_minutes))
        events = 0
        in_desat = False
        cooldown = 0
        valid_count = 0
        for i, v in enumerate(spo2_series):
            if v is None:
                continue
            valid_count += 1
            # Compute baseline from recent valid samples in trailing window
            start = max(0, i - window_samples)
            baseline_vals = [s for s in spo2_series[start:i+1] if s is not None]
            if not baseline_vals:
                continue
            baseline = max(baseline_vals)
            if cooldown > 0:
                cooldown -= 1
                if v >= baseline - 1:  # recovered
                    in_desat = False
                continue
            if not in_desat and baseline - v >= desat_threshold:
                events += 1
                in_desat = True
                cooldown = gap_samples
            elif in_desat and v >= baseline - 1:
                in_desat = False
        valid_hours = (valid_count * interval_minutes) / 60.0
        return round(events / valid_hours, 1) if valid_hours > 0 else 0.0

    odi_3pct = compute_odi(sleepu['spo2'], interval_min, desat_threshold=3)
    odi_4pct = compute_odi(sleepu['spo2'], interval_min, desat_threshold=4)

    print(f"\nData loaded:")
    print(f"  {oximeter_label} samples: {len(sleepu['spo2'])} (resampled to interval={sleepu['dt']*60:.1f}s)")
    print(f"  Resp rate epochs: {len(rr)}")
    print(f"  Apnea events: {len(apnea)}")
    print(f"  Obstructions: {len(obstr)}")
    print(f"  Apneas >10s: {apnea_count_gt10}, >50s: {apnea_count_gt50}")
    print(f"  ODI (≥3%): {odi_3pct}/hr, ODI (≥4%): {odi_4pct}/hr")
    print(f"  Time SpO2 <90%: {time_below_90_min} min")

    if not sleepu['spo2']:
        print("Error: no valid SleepU data found")
        sys.exit(1)

    # Compute time range
    t_min, t_max = compute_time_range(sleepu, rr)

    # Determine the end date for subtitle
    first_t = sleepu['t0']
    last_t = sleepu['t0'] + (len(sleepu['spo2']) - 1) * sleepu['dt']
    start_dt = first_dt
    end_minutes = last_t
    end_h = int(end_minutes / 60) % 24
    end_m = int(end_minutes % 60)
    # If recording crosses midnight, end date is next day
    end_date = ref_date
    if last_t >= 1440:
        end_date = ref_date + timedelta(days=1)

    subtitle = (f"{ref_date.strftime('%Y-%m-%d')} {minutes_to_clock(first_t)} "
                f"&rarr; {end_date.strftime('%Y-%m-%d')} {minutes_to_clock(last_t)} "
                f"&nbsp;&middot;&nbsp; {format_duration(first_t, last_t)} recording"
                f" &nbsp;&middot;&nbsp; {oximeter_label}")
    # append apnea summary to subtitle for the dashboard (only if breathing data available)
    has_breathing = bool(apnea or obstr or rr)
    if has_breathing:
        subtitle += f" &nbsp;&nbsp; • &nbsp;Apneas &gt;10s: {apnea_count_gt10}, &gt;50s: {apnea_count_gt50}"
        subtitle += f" &nbsp;&nbsp; • &nbsp;ODI(3%): {odi_3pct}/hr, ODI(4%): {odi_4pct}/hr"
    subtitle += f" &nbsp;&nbsp; • &nbsp;SpO\u2082&lt;90%: {time_below_90_min}m"

    title_date = end_date.strftime('%Y-%m-%d')

    # Build compact JSON
    data = {
        'sleepU': sleepu,
        'rr': rr,
        'apnea': apnea,
        'obstr': obstr,
        'ecgHR': ecg_hr,
    }
    if tilt_data:
        data['tiltRR'] = tilt_data['tiltRR']
        data['tiltEnv'] = tilt_data['tiltEnv']
        data['tiltRoll'] = tilt_data.get('tiltRoll', [])
    data_json = json.dumps(data, separators=(',', ':'))

    # SpO2 threshold for "Restful" detection: lower by 1% for Checkme
    good_spo2_min = 90 if oximeter_type == 'checkme' else 91

    # Generate HTML
    html = HTML_TEMPLATE
    html = html.replace('%%TITLE_DATE%%', title_date)
    html = html.replace('%%SUBTITLE%%', subtitle)
    html = html.replace('%%DATA_JSON%%', data_json)
    html = html.replace('%%T_MIN%%', str(round(t_min, 1)))
    html = html.replace('%%T_MAX%%', str(round(t_max, 1)))
    html = html.replace('%%VERSION%%', VERSION)
    html = html.replace('%%BUILD_DATE%%', BUILD_DATE)
    html = html.replace('%%GOOD_SPO2_MIN%%', str(good_spo2_min))

    # Write output
    output_path = os.path.join(input_dir, "sleep_dashboard.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nDashboard written to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path):,} bytes")


if __name__ == "__main__":
    main()