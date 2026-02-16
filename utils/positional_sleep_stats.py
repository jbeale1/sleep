#!/usr/bin/env python3

"""
Positional sleep statistics — correlate ECG, SpO2, and body position.

Reads three data files (auto-detected from a directory or specified explicitly):
  1. ECG per-beat CSV (*_beats.csv from analyze_ecg.py --csv-out)
  2. SpO2 CSV (Checkme O2 Ultra *.csv as exported from the "Checkme O2 Ultra" device)
  3. Position segments CSV (*_positions.csv from analyze_position.py)

Computes per-position-segment statistics for HR, HRV, SpO2, and
desaturation events. Produces a summary table and CSV output.

Usage:
  python positional_sleep_stats.py <directory>
  python positional_sleep_stats.py --ecg beats.csv --spo2 o2.csv --pos segments.csv

Outputs:
    HR — Mean heart rate (bpm)
    HRmd — Median heart rate (bpm), less sensitive to outlier beats
    SDNN — Standard deviation of R-R intervals (ms), overall HRV including slow oscillations
    RMSSD — Root-mean-square of successive R-R differences (ms), beat-to-beat vagal/parasympathetic HRV
    pNN50 — Percentage of successive R-R differences >50 ms, another parasympathetic index
    SpO2 — Mean blood oxygen saturation (%)
    O2md — Median blood oxygen saturation (%)
    O2mn — Minimum blood oxygen saturation (%)
    ODI3 — Oxygen Desaturation Index: ≥3% drops per hour, primary apnea severity metric
    ≤90% — Percentage of time with SpO2 at or below 90%, hypoxemia burden
    PVC — Count of premature ventricular contractions in that segment

J. Beale  2026-02
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import re
import argparse

# =============================================================
# ARGUMENTS
# =============================================================
parser = argparse.ArgumentParser(description='Positional sleep statistics')
parser.add_argument('directory', nargs='?', default=None,
                    help='Directory containing all three data files')
parser.add_argument('--ecg', dest='ecg_file', default=None,
                    help='ECG per-beat CSV (*_beats.csv)')
parser.add_argument('--spo2', dest='spo2_file', default=None,
                    help='SpO2 CSV (Checkme O2 Ultra *.csv)')
parser.add_argument('--pos', dest='pos_file', default=None,
                    help='Position segments CSV (*_positions.csv)')
parser.add_argument('--save-summary', action='store_true', dest='save_summary',
                    help='Save summary to text file')
args = parser.parse_args()

# =============================================================
# FIND FILES
# =============================================================
def find_file(directory, pattern):
    """Find first file matching regex pattern in directory."""
    pat = re.compile(pattern, re.IGNORECASE)
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and pat.search(f.name):
            return str(f)
    return None

ecg_file = args.ecg_file
spo2_file = args.spo2_file
pos_file = args.pos_file

if args.directory:
    d = args.directory
    if not Path(d).is_dir():
        print(f"Error: {d} is not a directory")
        sys.exit(1)
    if ecg_file is None:
        ecg_file = find_file(d, r'_beats\.csv$')
    if spo2_file is None:
        spo2_file = find_file(d, r'^Checkme O2 Ultra.*\.csv$')
    if pos_file is None:
        pos_file = find_file(d, r'_positions\.csv$')

missing = []
if ecg_file is None: missing.append('ECG beats CSV (*_beats.csv)')
if spo2_file is None: missing.append('SpO2 CSV (Checkme O2 Ultra*.csv)')
if pos_file is None: missing.append('Position CSV (*_positions.csv)')
if missing:
    print("Missing files:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)

print(f"ECG beats: {Path(ecg_file).name}")
print(f"SpO2:      {Path(spo2_file).name}")
print(f"Positions: {Path(pos_file).name}")

# =============================================================
# LOAD POSITION SEGMENTS
# =============================================================
pos_start_epoch = None
segments = []

with open(pos_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('# start:'):
            m = re.search(r'start:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if m:
                pos_t0 = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                pos_start_epoch = pos_t0.timestamp()
        elif line.startswith('#') or line.startswith('segment,'):
            continue
        else:
            parts = line.split(',')
            if len(parts) >= 9:
                seg = {
                    'segment': int(parts[0]),
                    'start_sec': float(parts[1]),
                    'end_sec': float(parts[2]),
                    'start_time': parts[3],
                    'end_time': parts[4],
                    'duration': float(parts[5]),
                    'position': parts[6],
                    'angle_mean': float(parts[7]),
                    'angle_std': float(parts[8]),
                }
                # Compute epoch times for this segment
                if pos_start_epoch is not None:
                    seg['epoch_start'] = pos_start_epoch + seg['start_sec']
                    seg['epoch_end'] = pos_start_epoch + seg['end_sec']
                segments.append(seg)

if not segments:
    print("Error: No segments found in position file")
    sys.exit(1)

print(f"Loaded {len(segments)} position segments")

# =============================================================
# LOAD ECG BEATS
# =============================================================
ecg_epoch = []
ecg_hr = []
ecg_rr = []
ecg_qrs_amp = []
ecg_st = []
ecg_is_artifact = []
ecg_is_pvc = []

with open(ecg_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        if line.startswith('epoch_s'):
            continue  # header
        parts = line.split(',')
        if len(parts) < 17:
            continue
        try:
            epoch = float(parts[0])
            hr = float(parts[2]) if parts[2] else np.nan
            rr = float(parts[4]) if parts[4] else np.nan
            qrs_a = float(parts[5]) if parts[5] else np.nan
            st = float(parts[12]) if parts[12] else np.nan
            is_art = int(parts[13]) if parts[13] else 0
            is_pvc = int(parts[14]) if parts[14] else 0
            ecg_epoch.append(epoch)
            ecg_hr.append(hr)
            ecg_rr.append(rr)
            ecg_qrs_amp.append(qrs_a)
            ecg_st.append(st)
            ecg_is_artifact.append(is_art)
            ecg_is_pvc.append(is_pvc)
        except (ValueError, IndexError):
            continue

ecg_epoch = np.array(ecg_epoch)
ecg_hr = np.array(ecg_hr)
ecg_rr = np.array(ecg_rr)
ecg_qrs_amp = np.array(ecg_qrs_amp)
ecg_st = np.array(ecg_st)
ecg_is_artifact = np.array(ecg_is_artifact, dtype=bool)
ecg_is_pvc = np.array(ecg_is_pvc, dtype=bool)
ecg_is_clean = ~ecg_is_artifact & ~ecg_is_pvc

print(f"Loaded {len(ecg_epoch)} ECG beats")

# =============================================================
# LOAD SpO2
# =============================================================
spo2_epoch = []
spo2_val = []
spo2_pulse = []

with open(spo2_file, 'r') as f:
    header = f.readline()  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
        try:
            dt = datetime.strptime(parts[0].strip(), '%H:%M:%S %d/%m/%Y')
            sv = int(parts[1].strip())
            pv = int(parts[2].strip())
            spo2_epoch.append(dt.timestamp())
            spo2_val.append(sv)
            spo2_pulse.append(pv)
        except (ValueError, IndexError):
            continue

spo2_epoch = np.array(spo2_epoch)
spo2_val = np.array(spo2_val, dtype=float)
spo2_pulse = np.array(spo2_pulse, dtype=float)

# Basic artifact mask for SpO2
spo2_valid = spo2_val > 50

print(f"Loaded {len(spo2_epoch)} SpO2 samples")

# =============================================================
# PER-SEGMENT ANALYSIS
# =============================================================
DESAT_BASELINE_WINDOW = 120  # seconds for rolling max baseline
DESAT_THRESHOLD = 3          # % drop for ODI-3
DESAT_MIN_DURATION = 10      # seconds

def compute_segment_odi3(spo2_seg):
    """Simplified ODI-3 for a segment: count ≥3% drops from rolling max baseline."""
    n = len(spo2_seg)
    if n < DESAT_MIN_DURATION:
        return 0, 0.0

    # Rolling max baseline (trailing window)
    baseline = np.full(n, np.nan)
    for i in range(n):
        win_start = max(0, i - DESAT_BASELINE_WINDOW)
        win = spo2_seg[win_start:i+1]
        valid = win[win > 50]
        if len(valid) >= 3:
            baseline[i] = np.max(valid)
        elif i > 0 and not np.isnan(baseline[i-1]):
            baseline[i] = baseline[i-1]

    # Count desaturation events
    events = 0
    in_event = False
    event_start = 0

    for i in range(n):
        if np.isnan(baseline[i]) or spo2_seg[i] <= 50:
            if in_event and (i - event_start) >= DESAT_MIN_DURATION:
                events += 1
            in_event = False
            continue

        drop = baseline[i] - spo2_seg[i]
        if not in_event and drop >= DESAT_THRESHOLD:
            in_event = True
            event_start = i
        elif in_event and drop < DESAT_THRESHOLD:
            if (i - event_start) >= DESAT_MIN_DURATION:
                events += 1
            in_event = False

    # Close open event
    if in_event and (n - event_start) >= DESAT_MIN_DURATION:
        events += 1

    duration_hr = n / 3600.0
    odi = events / max(duration_hr, 0.01)
    return events, odi


seg_results = []

for seg in segments:
    e_start = seg['epoch_start']
    e_end = seg['epoch_end']
    dur_min = seg['duration'] / 60.0

    result = {
        'segment': seg['segment'],
        'start_time': seg['start_time'],
        'end_time': seg['end_time'],
        'duration_min': dur_min,
        'position': seg['position'],
        'angle': seg['angle_mean'],
    }

    # --- ECG beats in this segment ---
    ecg_mask = (ecg_epoch >= e_start) & (ecg_epoch <= e_end)
    ecg_clean_mask = ecg_mask & ecg_is_clean
    n_ecg = np.sum(ecg_mask)
    n_ecg_clean = np.sum(ecg_clean_mask)

    result['n_beats'] = n_ecg

    if n_ecg_clean >= 5:
        hr_seg = ecg_hr[ecg_clean_mask]
        hr_seg = hr_seg[~np.isnan(hr_seg)]
        rr_seg = ecg_rr[ecg_clean_mask]
        rr_seg = rr_seg[~np.isnan(rr_seg)]

        result['hr_mean'] = np.mean(hr_seg) if len(hr_seg) > 0 else np.nan
        result['hr_median'] = np.median(hr_seg) if len(hr_seg) > 0 else np.nan
        result['hr_std'] = np.std(hr_seg) if len(hr_seg) > 0 else np.nan

        if len(rr_seg) >= 5:
            result['sdnn'] = np.std(rr_seg)
            rr_diffs = np.abs(np.diff(rr_seg))
            result['rmssd'] = np.sqrt(np.mean(rr_diffs**2))
            result['pnn50'] = 100.0 * np.sum(rr_diffs > 50) / len(rr_diffs)
        else:
            result['sdnn'] = np.nan
            result['rmssd'] = np.nan
            result['pnn50'] = np.nan

        qrs_seg = ecg_qrs_amp[ecg_clean_mask]
        qrs_seg = qrs_seg[~np.isnan(qrs_seg)]
        result['qrs_amp'] = np.mean(qrs_seg) if len(qrs_seg) > 0 else np.nan

        st_seg = ecg_st[ecg_clean_mask]
        st_seg = st_seg[~np.isnan(st_seg)]
        result['st_level'] = np.mean(st_seg) if len(st_seg) > 0 else np.nan
    else:
        for k in ['hr_mean','hr_median','hr_std','sdnn','rmssd','pnn50',
                   'qrs_amp','st_level']:
            result[k] = np.nan

    # PVC count (all beats in segment, not just clean)
    pvc_mask = ecg_mask & ecg_is_pvc
    result['n_pvc'] = int(np.sum(pvc_mask))
    result['pvc_rate'] = result['n_pvc'] / max(dur_min, 0.01)

    # --- SpO2 samples in this segment ---
    spo2_mask = (spo2_epoch >= e_start) & (spo2_epoch <= e_end) & spo2_valid
    n_spo2 = np.sum(spo2_mask)
    result['n_spo2'] = n_spo2

    if n_spo2 >= 10:
        s_seg = spo2_val[spo2_mask]
        result['spo2_mean'] = np.mean(s_seg)
        result['spo2_median'] = np.median(s_seg)
        result['spo2_min'] = np.min(s_seg)
        result['spo2_std'] = np.std(s_seg)
        result['t_below_90'] = np.sum(s_seg <= 90) / n_spo2 * 100

        events, odi = compute_segment_odi3(s_seg)
        result['odi3_events'] = events
        result['odi3'] = odi
    else:
        for k in ['spo2_mean','spo2_median','spo2_min','spo2_std',
                   't_below_90','odi3_events','odi3']:
            result[k] = np.nan

    seg_results.append(result)

# =============================================================
# AGGREGATE BY POSITION TYPE
# =============================================================
positions = sorted(set(r['position'] for r in seg_results))

pos_agg = {}
for pos in positions:
    segs = [r for r in seg_results if r['position'] == pos]
    total_dur = sum(r['duration_min'] for r in segs)

    # Duration-weighted averages for rates; simple pooling for counts
    def wavg(key):
        vals = [(r[key], r['duration_min']) for r in segs
                if not np.isnan(r.get(key, np.nan))]
        if not vals:
            return np.nan
        v, w = zip(*vals)
        return np.average(v, weights=w)

    total_odi_events = sum(r.get('odi3_events', 0) for r in segs
                           if not np.isnan(r.get('odi3_events', np.nan)))
    total_pvc = sum(r['n_pvc'] for r in segs)

    pos_agg[pos] = {
        'n_segments': len(segs),
        'total_min': total_dur,
        'angle': wavg('angle'),
        'hr_mean': wavg('hr_mean'),
        'hr_median': wavg('hr_median'),
        'sdnn': wavg('sdnn'),
        'rmssd': wavg('rmssd'),
        'pnn50': wavg('pnn50'),
        'qrs_amp': wavg('qrs_amp'),
        'st_level': wavg('st_level'),
        'spo2_mean': wavg('spo2_mean'),
        'spo2_min': min((r['spo2_min'] for r in segs
                         if not np.isnan(r.get('spo2_min', np.nan))),
                        default=np.nan),
        'odi3': total_odi_events / max(total_dur / 60, 0.01),
        'odi3_events': int(total_odi_events),
        'pvc_total': total_pvc,
        'pvc_rate': total_pvc / max(total_dur, 0.01),
        't_below_90': wavg('t_below_90'),
    }

    # ODI clustering: how concentrated are events across segments?
    seg_odi_vals = [r.get('odi3', 0) for r in segs
                    if not np.isnan(r.get('odi3', np.nan))]
    seg_odi_events = [r.get('odi3_events', 0) for r in segs
                      if not np.isnan(r.get('odi3_events', np.nan))]
    if len(seg_odi_vals) > 0:
        pos_agg[pos]['odi3_range'] = (min(seg_odi_vals), max(seg_odi_vals))
    else:
        pos_agg[pos]['odi3_range'] = (0, 0)

    if total_odi_events > 0 and len(seg_odi_events) > 1:
        max_seg_events = max(seg_odi_events)
        # Concentration: fraction of events in the single highest segment
        pos_agg[pos]['odi3_concentration'] = max_seg_events / total_odi_events
        # Number of segments that have any events at all
        pos_agg[pos]['odi3_active_segs'] = sum(1 for e in seg_odi_events if e > 0)
    else:
        pos_agg[pos]['odi3_concentration'] = np.nan
        pos_agg[pos]['odi3_active_segs'] = sum(1 for e in seg_odi_events if e > 0)

# =============================================================
# SUMMARY OUTPUT
# =============================================================
# Optional Tee for --save-summary
_summary_file = None
_original_stdout = sys.stdout

if args.save_summary:
    if args.directory:
        stem = str(Path(args.directory) / 'positional_sleep_stats')
    else:
        stem = str(Path(pos_file).parent / 'positional_sleep_stats')
    summary_path = f"{stem}_summary.txt"
    _summary_file = open(summary_path, 'w')

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

print("\n" + "="*100)
print("PER-SEGMENT STATISTICS")
print("="*100)
print(f"  {'Seg':>3s} {'Start':>7s} {'End':>7s} {'Dur':>5s} {'Pos':>3s} "
      f"{'HR':>5s} {'HRmd':>5s} {'SDNN':>5s} {'RMSSD':>5s} {'pNN50':>5s} "
      f"{'SpO2':>5s} {'O2md':>4s} {'O2mn':>4s} "
      f"{'ODI3':>5s} {'≤90%':>5s} {'PVC':>4s}")
print(f"  {'---':>3s} {'-----':>7s} {'---':>7s} {'---':>5s} {'---':>3s} "
      f"{'--':>5s} {'----':>5s} {'----':>5s} {'-----':>5s} {'-----':>5s} "
      f"{'----':>5s} {'----':>4s} {'----':>4s} "
      f"{'----':>5s} {'----':>5s} {'---':>4s}")

for r in seg_results:
    def fmt(val, w=5, d=1):
        if np.isnan(val): return ' ' * w
        return f"{val:{w}.{d}f}"

    print(f"  {r['segment']:3d} {r['start_time']:>7s} {r['end_time']:>7s} "
          f"{r['duration_min']:4.0f}m {r['position']:>3s} "
          f"{fmt(r['hr_mean'])} {fmt(r['hr_median'])} "
          f"{fmt(r['sdnn'])} {fmt(r['rmssd'])} {fmt(r['pnn50'])} "
          f"{fmt(r['spo2_mean'])} {fmt(r['spo2_median'],4,0)} {fmt(r['spo2_min'],4,0)} "
          f"{fmt(r.get('odi3', np.nan))} "
          f"{fmt(r.get('t_below_90', np.nan))} "
          f"{r['n_pvc']:4d}")

# Position type summary
print("\n" + "="*100)
print("PER-POSITION SUMMARY (duration-weighted)")
print("="*100)
print(f"  {'Pos':>3s} {'Segs':>4s} {'Total':>6s} {'Angle':>6s} "
      f"{'HR':>5s} {'HRmd':>5s} {'SDNN':>5s} {'RMSSD':>5s} {'pNN50':>5s} "
      f"{'SpO2':>5s} {'O2mn':>4s} "
      f"{'ODI3':>5s} {'≤90%':>5s} {'PVC':>4s} {'PVC/m':>5s}")
print(f"  {'---':>3s} {'----':>4s} {'-----':>6s} {'-----':>6s} "
      f"{'--':>5s} {'----':>5s} {'----':>5s} {'-----':>5s} {'-----':>5s} "
      f"{'----':>5s} {'----':>4s} "
      f"{'----':>5s} {'----':>5s} {'---':>4s} {'-----':>5s}")

for pos in positions:
    a = pos_agg[pos]
    dur_str = f"{a['total_min']:.0f}m" if a['total_min'] < 120 else f"{a['total_min']/60:.1f}h"
    def fmt(val, w=5, d=1):
        if np.isnan(val): return ' ' * w
        return f"{val:{w}.{d}f}"

    print(f"  {pos:>3s} {a['n_segments']:4d} {dur_str:>6s} {a['angle']:5.0f}° "
          f"{fmt(a['hr_mean'])} {fmt(a['hr_median'])} "
          f"{fmt(a['sdnn'])} {fmt(a['rmssd'])} {fmt(a['pnn50'])} "
          f"{fmt(a['spo2_mean'])} {fmt(a['spo2_min'],4,0)} "
          f"{fmt(a['odi3'])} {fmt(a['t_below_90'])} "
          f"{a['pvc_total']:4d} {a['pvc_rate']:5.2f}")

print("="*100)

# ODI clustering notes
has_clustering_notes = False
for pos in positions:
    a = pos_agg[pos]
    if a['odi3_events'] > 0 and a['n_segments'] > 1:
        if not has_clustering_notes:
            print("\n  ODI-3 distribution across segments:")
            has_clustering_notes = True
        odi_lo, odi_hi = a['odi3_range']
        active = a['odi3_active_segs']
        conc = a['odi3_concentration']
        seg_details = [(r['segment'], r['start_time'], r.get('odi3', 0),
                        r.get('odi3_events', 0), r['duration_min'])
                       for r in seg_results if r['position'] == pos
                       and not np.isnan(r.get('odi3', np.nan))]
        print(f"    Pos {pos}: {a['odi3_events']} events across "
              f"{active}/{a['n_segments']} segments "
              f"(ODI range {odi_lo:.1f}–{odi_hi:.1f}/hr"
              f"{f', {conc:.0%} in top segment' if not np.isnan(conc) else ''})")
        for seg_num, seg_time, seg_odi, seg_evt, seg_dur in seg_details:
            if seg_evt > 0:
                print(f"      Seg {seg_num:2d} ({seg_time}, {seg_dur:.0f}m): "
                      f"ODI {seg_odi:.1f}/hr ({seg_evt} events)")

# =============================================================
# CSV OUTPUT
# =============================================================
if args.directory:
    csv_stem = str(Path(args.directory) / 'positional_sleep_stats')
else:
    csv_stem = str(Path(pos_file).parent / 'positional_sleep_stats')

csv_path = f"{csv_stem}.csv"
with open(csv_path, 'w') as f:
    f.write("segment,start_time,end_time,duration_min,position,angle,"
            "hr_mean,hr_median,hr_std,sdnn,rmssd,pnn50,"
            "qrs_amp,st_level,"
            "spo2_mean,spo2_median,spo2_min,spo2_std,t_below_90_pct,"
            "odi3_events,odi3,n_pvc,pvc_per_min,n_beats,n_spo2\n")
    for r in seg_results:
        def fv(val):
            if np.isnan(val): return ''
            return f"{val:.2f}"
        f.write(f"{r['segment']},{r['start_time']},{r['end_time']},"
                f"{r['duration_min']:.1f},{r['position']},{r['angle']:.1f},"
                f"{fv(r['hr_mean'])},{fv(r['hr_median'])},{fv(r['hr_std'])},"
                f"{fv(r['sdnn'])},{fv(r['rmssd'])},{fv(r['pnn50'])},"
                f"{fv(r['qrs_amp'])},{fv(r['st_level'])},"
                f"{fv(r.get('spo2_mean',np.nan))},{fv(r.get('spo2_median',np.nan))},"
                f"{fv(r.get('spo2_min',np.nan))},{fv(r.get('spo2_std',np.nan))},"
                f"{fv(r.get('t_below_90',np.nan))},"
                f"{r.get('odi3_events','')},{fv(r.get('odi3',np.nan))},"
                f"{r['n_pvc']},{r['pvc_rate']:.2f},"
                f"{r['n_beats']},{r.get('n_spo2','')}\n")

print(f"\nCSV saved: {csv_path}")

# =============================================================
# CLOSE SUMMARY FILE
# =============================================================
if _summary_file:
    sys.stdout = _original_stdout
    _summary_file.close()
    print(f"Summary saved: {summary_path}")
