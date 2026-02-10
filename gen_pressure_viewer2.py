#!/usr/bin/env python3
"""
view_overnight.py — Generate interactive HTML viewer for overnight pressure recordings.

Reads CSV with columns: millis,pressure_hPa
Extracts start time from filename pattern: YYYYMMDD_HHMMSS_*
Applies highpass filter to remove barometric drift.
Outputs a zoomable HTML file using Plotly.

Usage:
  python view_overnight.py 20260208_224952_Overnight_20260209_070200.csv
  python view_overnight.py somefile.csv --start "2026-02-08 22:49:52"
  python view_overnight.py somefile.csv --hp-minutes 45
"""

import sys
import os
import re
import json
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
from scipy import signal
from datetime import datetime, timedelta
from pathlib import Path

FS_NOMINAL = 31.5


def parse_start_time(filename):
    """Extract start datetime from filename like 20260208_224952_..."""
    basename = Path(filename).stem
    match = re.match(r'(\d{8})_(\d{6})', basename)
    if match:
        return datetime.strptime(match.group(1) + match.group(2), '%Y%m%d%H%M%S')
    return None


def load_and_process(filepath, start_time, hp_period_min=30):
    """Load CSV, compute timestamps, highpass filter, respiratory band."""
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    N = len(df)
    print(f"  {N:,} samples")

    millis = df['millis'].values.astype(float)
    pressure = df['pressure_hPa'].values.astype(float)

    # Compute actual sample rate
    dt_ms = np.diff(millis)
    dt_clean = dt_ms[(dt_ms > 10) & (dt_ms < 100)]
    fs = 1000.0 / np.median(dt_clean) if len(dt_clean) > 100 else FS_NOMINAL
    print(f"  Sample rate: {fs:.1f} Hz")

    t_seconds = (millis - millis[0]) / 1000.0
    duration_h = t_seconds[-1] / 3600
    print(f"  Duration: {duration_h:.2f} hours")

    # Resample to uniform grid for filtering
    print("  Resampling to uniform grid...")
    t_uniform = np.arange(0, t_seconds[-1], 1.0 / fs)
    p_uniform = np.interp(t_uniform, t_seconds, pressure)
    N_uniform = len(t_uniform)

    # Highpass filter to remove barometric drift
    hp_cutoff = 1.0 / (hp_period_min * 60.0)
    print(f"  Highpass filter: {hp_cutoff:.6f} Hz ({hp_period_min} min)")
    sos_hp = signal.butter(2, hp_cutoff, btype='highpass', fs=fs, output='sos')
    p_hp = signal.sosfiltfilt(sos_hp, p_uniform)

    # Respiratory bandpass
    print("  Respiratory bandpass (0.05-0.5 Hz)...")
    sos_resp = signal.butter(3, [0.05, 0.5], btype='bandpass', fs=fs, output='sos')
    p_resp = signal.sosfiltfilt(sos_resp, p_uniform)

    # Decimate for HTML — target ~8 Hz (preserves respiratory fully)
    dec = max(1, int(round(fs / 8.0)))
    fs_dec = fs / dec
    print(f"  Decimating {dec}x: {fs:.1f} -> {fs_dec:.1f} Hz "
          f"({N_uniform:,} -> {N_uniform//dec:,} pts)")

    t_dec = t_uniform[::dec]
    hp_dec = p_hp[::dec]
    resp_dec = p_resp[::dec]
    raw_dec = p_uniform[::dec]

    # Compute wall-clock timestamps as epoch milliseconds (compact for JSON)
    print("  Computing timestamps...")
    epoch_start = start_time.timestamp() * 1000  # ms since unix epoch
    ts_epoch_ms = (epoch_start + t_dec * 1000).astype(np.int64)

    return {
        'ts_ms': ts_epoch_ms,
        'hp': np.round(hp_dec, 4),
        'resp': np.round(resp_dec, 4),
        'raw': np.round(raw_dec, 2),
        'fs_dec': fs_dec,
        'fs_orig': fs,
        'duration_h': duration_h,
        'n_orig': N,
        'dec_factor': dec,
        'start_time': start_time,
    }


def load_audio_envelopes(audio_dir, pressure_start_time, pressure_end_time,
                         envelope_rate=8.0):
    """Find mp3 files, decode with ffmpeg, compute RMS and peak envelopes.

    Looks for files matching *_YYYYMMDD_HHMMSS.mp3 in audio_dir.
    Returns dict with ts_ms, rms, peak arrays aligned to wall-clock time,
    or None if no audio files found.
    """
    # Find mp3 files with timestamp in name
    mp3_pattern = os.path.join(audio_dir, '*.mp3')
    mp3_files = sorted(glob.glob(mp3_pattern))
    if not mp3_files:
        print(f"  No mp3 files found in {audio_dir}")
        return None

    # Parse timestamps and sort
    file_times = []
    ts_re = re.compile(r'(\d{8})_(\d{6})\.mp3$', re.IGNORECASE)
    for f in mp3_files:
        m = ts_re.search(os.path.basename(f))
        if m:
            t = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            file_times.append((t, f))

    file_times.sort()
    if not file_times:
        print("  No mp3 files with parseable timestamps found")
        return None

    print(f"  Found {len(file_times)} mp3 files: "
          f"{file_times[0][0].strftime('%H:%M:%S')} - "
          f"{file_times[-1][0].strftime('%H:%M:%S')}")

    all_ts_ms = []
    all_rms = []
    all_peak = []

    for i, (file_start, filepath) in enumerate(file_times):
        fname = os.path.basename(filepath)
        # Decode mp3 to raw 16-bit mono PCM via ffmpeg
        try:
            result = subprocess.run(
                ['ffmpeg', '-i', filepath, '-f', 's16le', '-ac', '1',
                 '-ar', '24000', '-v', 'error', '-'],
                capture_output=True, timeout=30)
            if result.returncode != 0:
                print(f"    ffmpeg error on {fname}: {result.stderr.decode()[:100]}")
                continue
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"    ffmpeg failed on {fname}: {e}")
            continue

        pcm = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        pcm /= 32768.0  # normalize to -1..1
        mp3_sr = 24000

        if len(pcm) < mp3_sr:
            print(f"    {fname}: too short ({len(pcm)} samples), skipping")
            continue

        # Compute block-wise RMS and peak envelopes
        block_size = int(mp3_sr / envelope_rate)  # 3000 samples for 8 Hz
        n_blocks = len(pcm) // block_size
        if n_blocks == 0:
            continue

        pcm_trimmed = pcm[:n_blocks * block_size].reshape(n_blocks, block_size)
        rms = np.sqrt(np.mean(pcm_trimmed ** 2, axis=1))
        peak = np.max(np.abs(pcm_trimmed), axis=1)

        # Timestamps for each block center
        block_centers_s = (np.arange(n_blocks) + 0.5) * block_size / mp3_sr
        epoch_start_ms = file_start.timestamp() * 1000
        block_ts_ms = (epoch_start_ms + block_centers_s * 1000).astype(np.int64)

        all_ts_ms.append(block_ts_ms)
        all_rms.append(rms)
        all_peak.append(peak)

        if (i + 1) % 10 == 0 or i == len(file_times) - 1:
            print(f"    Processed {i+1}/{len(file_times)} files")

    if not all_ts_ms:
        print("  No audio data extracted")
        return None

    ts_ms = np.concatenate(all_ts_ms)
    rms = np.concatenate(all_rms)
    peak = np.concatenate(all_peak)

    # Sort by time (should already be sorted, but just in case)
    order = np.argsort(ts_ms)
    ts_ms = ts_ms[order]
    rms = rms[order]
    peak = peak[order]

    # Convert to dB (floor at -80 dB to avoid log(0))
    rms_db = 20 * np.log10(np.maximum(rms, 1e-4))
    peak_db = 20 * np.log10(np.maximum(peak, 1e-4))

    total_dur = (ts_ms[-1] - ts_ms[0]) / 3600000
    print(f"  Audio envelope: {len(ts_ms):,} points, {total_dur:.2f} hours, "
          f"RMS range: {rms_db.min():.0f} to {rms_db.max():.0f} dB")

    return {
        'ts_ms': ts_ms,
        'rms_db': np.round(rms_db, 1),
        'peak_db': np.round(peak_db, 1),
    }


def generate_html(data, filepath, hp_period_min, audio_data=None):
    basename = Path(filepath).stem
    start_time = data['start_time']
    end_time = start_time + timedelta(hours=data['duration_h'])
    title = f"{basename}  ({start_time.strftime('%H:%M')}\u2013{end_time.strftime('%H:%M')})"

    n_pts = len(data['ts_ms'])
    print(f"  Embedding {n_pts:,} points per trace...")

    # Encode as compact JSON — epoch ms is much smaller than ISO strings
    ts_json = json.dumps(data['ts_ms'].tolist())
    hp_json = json.dumps(data['hp'].tolist())
    resp_json = json.dumps(data['resp'].tolist())
    raw_json = json.dumps(data['raw'].tolist())

    has_audio = audio_data is not None
    if has_audio:
        audio_ts_json = json.dumps(audio_data['ts_ms'].tolist())
        audio_rms_json = json.dumps(audio_data['rms_db'].tolist())
        audio_peak_json = json.dumps(audio_data['peak_db'].tolist())
        print(f"  Embedding {len(audio_data['ts_ms']):,} audio envelope points...")

    # Audio-dependent CSS/HTML/JS sections
    audio_checkbox = ('<label><input type="checkbox" id="cbAudioRMS" checked>'
                      ' Audio RMS</label>'
                      '<label><input type="checkbox" id="cbAudioPeak">'
                      ' Audio Peak</label>') if has_audio else ''
    audio_div = '<div id="audioPlot"></div>' if has_audio else ''
    audio_plot_height = '42vh' if has_audio else '55vh'
    audio_div_css = '#audioPlot { width: 100%; height: 18vh; margin-top: 4px; }' if has_audio else ''

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; padding: 12px;
  }}
  h1 {{ font-size: 16px; padding: 8px 0; color: #a0c4ff; }}
  .info {{ font-size: 12px; color: #888; margin-bottom: 8px; }}
  .info span {{ color: #aaa; }}
  #mainPlot {{ width: 100%; height: {audio_plot_height}; }}
  #overviewPlot {{ width: 100%; height: 14vh; margin-top: 4px; }}
  {audio_div_css}
  .controls {{
    display: flex; gap: 14px; align-items: center;
    margin: 8px 0; flex-wrap: wrap;
  }}
  .controls label {{ font-size: 13px; color: #aaa; cursor: pointer; }}
  .controls input[type=checkbox] {{ margin-right: 3px; }}
  .controls button {{
    background: #2a2a4a; color: #a0c4ff; border: 1px solid #444;
    border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 13px;
  }}
  .controls button:hover {{ background: #3a3a5a; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="info">
  {data['n_orig']:,} samples at {data['fs_orig']:.1f} Hz
  &mdash; displayed at {data['fs_dec']:.1f} Hz ({data['dec_factor']}x decimation)
  &mdash; highpass period: {hp_period_min} min
  <span>&ensp;|&ensp; Drag to zoom &bull; double-click to reset
  &bull; click overview to jump</span>
</div>
<div class="controls">
  <label><input type="checkbox" id="cbHP" checked> Detrended</label>
  <label><input type="checkbox" id="cbResp"> Respiratory</label>
  <label><input type="checkbox" id="cbRaw"> Raw pressure</label>
  {audio_checkbox}
  <button onclick="resetZoom()">Full</button>
  <button onclick="zoomTo(2)">2h</button>
  <button onclick="zoomTo(1)">1h</button>
  <button onclick="zoomTo(0.25)">15m</button>
  <button onclick="zoomTo(1/60)">1m</button>
  <button onclick="nudge(-1)">&#9664;</button>
  <button onclick="nudge(1)">&#9654;</button>
</div>
<div id="mainPlot"></div>
{audio_div}
<div id="overviewPlot"></div>

<script>
// --- Pressure data (epoch ms timestamps) ---
const tsMs = {ts_json};
const hp = {hp_json};
const resp = {resp_json};
const raw = {raw_json};

const ts = tsMs.map(m => new Date(m));
const tStart = ts[0];
const tEnd = ts[ts.length - 1];

// --- Main pressure plot ---
const traceHP = {{
  x: ts, y: hp, type: 'scattergl', mode: 'lines',
  name: 'Detrended', line: {{ color: '#5dade2', width: 1 }},
  visible: true, hovertemplate: '%{{x|%H:%M:%S}}<br>%{{y:.3f}} hPa<extra>Detrended</extra>'
}};
const traceResp = {{
  x: ts, y: resp, type: 'scattergl', mode: 'lines',
  name: 'Respiratory', line: {{ color: '#e74c3c', width: 1.8 }},
  visible: 'legendonly',
  hovertemplate: '%{{x|%H:%M:%S}}<br>%{{y:.3f}} hPa<extra>Resp</extra>'
}};
const traceRaw = {{
  x: ts, y: raw, type: 'scattergl', mode: 'lines',
  name: 'Raw', line: {{ color: '#666', width: 0.5 }},
  yaxis: 'y2', visible: 'legendonly',
  hovertemplate: '%{{x|%H:%M:%S}}<br>%{{y:.2f}} hPa<extra>Raw</extra>'
}};

const mainLayout = {{
  paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
  font: {{ color: '#ccc', size: 11 }},
  margin: {{ l: 65, r: 65, t: 10, b: 10 }},
  xaxis: {{
    type: 'date', gridcolor: '#2a2a4a',
    tickformat: '%H:%M:%S',
    range: [tStart, tEnd],
  }},
  yaxis: {{
    title: 'hPa (detrended)', gridcolor: '#2a2a4a',
    zeroline: true, zerolinecolor: '#444',
  }},
  yaxis2: {{
    title: 'hPa (absolute)', overlaying: 'y', side: 'right',
    showgrid: false,
  }},
  legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.4)', font: {{size: 11}} }},
  hovermode: 'x unified',
  dragmode: 'zoom',
}};

Plotly.newPlot('mainPlot', [traceHP, traceResp, traceRaw], mainLayout,
  {{ responsive: true, displaylogo: false,
     modeBarButtonsToRemove: ['lasso2d','select2d'] }});
"""

    # Audio plot JS (conditional)
    if has_audio:
        html += f"""
// --- Audio data ---
const audioTsMs = {audio_ts_json};
const audioRms = {audio_rms_json};
const audioPeak = {audio_peak_json};
const audioTs = audioTsMs.map(m => new Date(m));

const traceAudioRms = {{
  x: audioTs, y: audioRms, type: 'scattergl', mode: 'lines',
  name: 'RMS', line: {{ color: '#2ecc71', width: 1 }},
  hovertemplate: '%{{x|%H:%M:%S}}<br>%{{y:.1f}} dB<extra>RMS</extra>'
}};
const traceAudioPeak = {{
  x: audioTs, y: audioPeak, type: 'scattergl', mode: 'lines',
  name: 'Peak', line: {{ color: '#f39c12', width: 0.7 }},
  visible: 'legendonly',
  hovertemplate: '%{{x|%H:%M:%S}}<br>%{{y:.1f}} dB<extra>Peak</extra>'
}};

const audioLayout = {{
  paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
  font: {{ color: '#ccc', size: 11 }},
  margin: {{ l: 65, r: 65, t: 5, b: 10 }},
  xaxis: {{
    type: 'date', gridcolor: '#2a2a4a', tickformat: '%H:%M:%S',
    range: [tStart, tEnd],
    matches: undefined,
  }},
  yaxis: {{
    title: 'dB', gridcolor: '#2a2a4a',
  }},
  legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.4)', font: {{size: 11}} }},
  hovermode: 'x unified',
  dragmode: 'zoom',
}};

Plotly.newPlot('audioPlot', [traceAudioRms, traceAudioPeak], audioLayout,
  {{ responsive: true, displaylogo: false,
     modeBarButtonsToRemove: ['lasso2d','select2d'] }});

// Sync guard to prevent infinite main<->audio relayout loop
let _syncing = false;

// Sync main -> audio zoom
document.getElementById('mainPlot').on('plotly_relayout', function(ed) {{
  if (_syncing) return;
  let x0 = ed['xaxis.range[0]'] || (ed['xaxis.range'] && ed['xaxis.range'][0]);
  let x1 = ed['xaxis.range[1]'] || (ed['xaxis.range'] && ed['xaxis.range'][1]);
  if (x0 !== undefined && x1 !== undefined) {{
    _syncing = true;
    Plotly.relayout('audioPlot', {{ 'xaxis.range': [x0, x1] }}).then(() => _syncing = false);
  }}
}});
// Sync audio -> main zoom
document.getElementById('audioPlot').on('plotly_relayout', function(ed) {{
  if (_syncing) return;
  let x0 = ed['xaxis.range[0]'] || (ed['xaxis.range'] && ed['xaxis.range'][0]);
  let x1 = ed['xaxis.range[1]'] || (ed['xaxis.range'] && ed['xaxis.range'][1]);
  if (x0 !== undefined && x1 !== undefined) {{
    _syncing = true;
    Plotly.relayout('mainPlot', {{ 'xaxis.range': [x0, x1] }}).then(() => _syncing = false);
  }}
}});

// Audio checkbox toggles
document.getElementById('cbAudioRMS').addEventListener('change', function() {{
  Plotly.restyle('audioPlot', {{ visible: this.checked ? true : 'legendonly' }}, [0]);
}});
document.getElementById('cbAudioPeak').addEventListener('change', function() {{
  Plotly.restyle('audioPlot', {{ visible: this.checked ? true : 'legendonly' }}, [1]);
}});
"""

    # Overview and navigation (always present) — need to sync to audio too
    audio_relayout_in_overview = ""
    audio_relayout_in_nav = ""
    if has_audio:
        audio_relayout_in_overview = "Plotly.relayout('audioPlot', { 'xaxis.range': [x0, x1] });"
        audio_relayout_in_nav = "Plotly.relayout('audioPlot', { 'xaxis.range': [x0, x1] });"

    html += f"""
// --- Overview plot ---
const ovStep = Math.max(1, Math.floor(ts.length / 8000));
const ovTs = [], ovHp = [];
for (let i = 0; i < ts.length; i += ovStep) {{ ovTs.push(ts[i]); ovHp.push(hp[i]); }}

const overviewTrace = {{
  x: ovTs, y: ovHp, type: 'scattergl', mode: 'lines',
  line: {{ color: '#5dade2', width: 0.7 }}, hoverinfo: 'x',
}};

const overviewLayout = {{
  paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
  font: {{ color: '#ccc', size: 10 }},
  margin: {{ l: 65, r: 65, t: 5, b: 25 }},
  xaxis: {{
    type: 'date', gridcolor: '#2a2a4a', tickformat: '%H:%M',
    range: [tStart, tEnd], fixedrange: false,
  }},
  yaxis: {{ gridcolor: '#2a2a4a', zeroline: true, zerolinecolor: '#444' }},
  shapes: [{{
    type: 'rect', xref: 'x', yref: 'paper',
    x0: tStart, x1: tEnd, y0: 0, y1: 1,
    fillcolor: 'rgba(93,173,226,0.15)',
    line: {{ color: '#5dade2', width: 1 }}
  }}],
  dragmode: 'select',
}};

Plotly.newPlot('overviewPlot', [overviewTrace], overviewLayout,
  {{ responsive: true, displaylogo: false, displayModeBar: false }});

// Sync main zoom -> overview highlight (and audio if present)
document.getElementById('mainPlot').on('plotly_relayout', function(ed) {{
  let x0 = ed['xaxis.range[0]'] || (ed['xaxis.range'] && ed['xaxis.range'][0]);
  let x1 = ed['xaxis.range[1]'] || (ed['xaxis.range'] && ed['xaxis.range'][1]);
  if (x0 !== undefined && x1 !== undefined) {{
    Plotly.relayout('overviewPlot', {{
      'shapes[0].x0': x0, 'shapes[0].x1': x1
    }});
  }}
}});

// Overview click -> center main view
document.getElementById('overviewPlot').on('plotly_click', function(ed) {{
  if (ed.points && ed.points.length > 0) {{
    const mainDiv = document.getElementById('mainPlot');
    const xr = mainDiv.layout.xaxis.range;
    const curSpan = new Date(xr[1]) - new Date(xr[0]);
    const half = curSpan / 2;
    const clickT = new Date(ed.points[0].x).getTime();
    const x0 = new Date(Math.max(clickT - half, tStart.getTime()));
    const x1 = new Date(Math.min(clickT + half, tEnd.getTime()));
    Plotly.relayout('mainPlot', {{ 'xaxis.range': [x0, x1] }});
    {audio_relayout_in_overview}
  }}
}});

// Overview drag-select -> zoom main
document.getElementById('overviewPlot').on('plotly_selected', function(ed) {{
  if (ed && ed.range && ed.range.x) {{
    const x0 = ed.range.x[0], x1 = ed.range.x[1];
    Plotly.relayout('mainPlot', {{ 'xaxis.range': [x0, x1] }});
    {audio_relayout_in_overview}
  }}
}});

// Pressure checkbox toggles
['cbHP','cbResp','cbRaw'].forEach((id, idx) => {{
  document.getElementById(id).addEventListener('change', function() {{
    Plotly.restyle('mainPlot', {{ visible: this.checked ? true : 'legendonly' }}, [idx]);
  }});
}});

// Navigation
function resetZoom() {{
  Plotly.relayout('mainPlot', {{ 'xaxis.range': [tStart, tEnd] }});
  {audio_relayout_in_nav.replace('[x0, x1]', '[tStart, tEnd]')}
}}

function zoomTo(hours) {{
  const mainDiv = document.getElementById('mainPlot');
  const xr = mainDiv.layout.xaxis.range;
  const center = (new Date(xr[0]).getTime() + new Date(xr[1]).getTime()) / 2;
  const halfMs = hours * 3600 * 1000 / 2;
  const x0 = new Date(Math.max(center - halfMs, tStart.getTime()));
  const x1 = new Date(Math.min(center + halfMs, tEnd.getTime()));
  Plotly.relayout('mainPlot', {{ 'xaxis.range': [x0, x1] }});
  {audio_relayout_in_nav}
}}

function nudge(direction) {{
  const mainDiv = document.getElementById('mainPlot');
  const xr = mainDiv.layout.xaxis.range;
  const span = new Date(xr[1]) - new Date(xr[0]);
  const shift = direction * span * 0.5;
  let x0 = new Date(new Date(xr[0]).getTime() + shift);
  let x1 = new Date(new Date(xr[1]).getTime() + shift);
  if (x0 < tStart) {{ x0 = tStart; x1 = new Date(tStart.getTime() + span); }}
  if (x1 > tEnd) {{ x1 = tEnd; x0 = new Date(tEnd.getTime() - span); }}
  Plotly.relayout('mainPlot', {{ 'xaxis.range': [x0, x1] }});
  {audio_relayout_in_nav}
}}
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML viewer for overnight pressure recordings.')
    parser.add_argument('input', help='Input CSV file (millis,pressure_hPa)')
    parser.add_argument('--start',
                        help='Start time override: "YYYY-MM-DD HH:MM:SS"')
    parser.add_argument('--hp-minutes', type=float, default=30,
                        help='Highpass filter period in minutes (default: 30)')
    parser.add_argument('--audio-dir',
                        help='Directory containing mp3 files with timestamps in names '
                             '(default: same directory as input CSV)')
    parser.add_argument('--no-audio', action='store_true',
                        help='Skip audio processing even if mp3 files are present')
    parser.add_argument('-o', '--output',
                        help='Output HTML path (default: <input>_viewer.html)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        sys.exit(1)

    # Determine start time
    if args.start:
        start_time = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
    else:
        start_time = parse_start_time(args.input)
    if not start_time:
        print("Cannot determine start time. Use --start 'YYYY-MM-DD HH:MM:SS'")
        sys.exit(1)
    print(f"  Start time: {start_time}")

    data = load_and_process(args.input, start_time, args.hp_minutes)

    # Process audio if available
    audio_data = None
    if not args.no_audio:
        audio_dir = args.audio_dir or str(Path(args.input).parent)
        print(f"\n  Scanning for audio: {audio_dir}")
        end_time = start_time + timedelta(hours=data['duration_h'])
        audio_data = load_audio_envelopes(audio_dir, start_time, end_time)

    html = generate_html(data, args.input, args.hp_minutes, audio_data)

    if args.output:
        out_path = args.output
    else:
        out_path = str(Path(args.input).with_suffix('')) + '_viewer.html'

    try:
        with open(out_path, 'w') as f:
            f.write(html)
    except OSError:
        out_path = str(Path('.') / Path(out_path).name)
        with open(out_path, 'w') as f:
            f.write(html)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Output: {out_path} ({size_mb:.1f} MB)")
    print(f"  Open in browser to view.")


if __name__ == '__main__':
    main()
