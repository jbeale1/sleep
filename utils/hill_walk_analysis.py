#!/usr/bin/env python3
"""
hill_walk_analysis.py  —  Repeated hill walk HR comparison tool

Reads a GPX track file and a Polar H10 R-R interval file from the same
outing, detects walk phases from GPS speed and elevation, filters ectopic
beats, and produces:
  • A two-panel PNG: calibrated GPS elevation + HR vs time with phase shading
  • A console table of mean HR and SD per sub-phase

Designed for a specific out-and-back route with the following profile:
  level approach → 6 m dip to foot of hill → 40 m climb → summit pause
  → 40 m descent → 6 m re-ascent → indoor rest

Usage:
    python3 hill_walk_analysis.py  <gpx_file>  <rr_file>  [options]

Options:
    --baro-height M     Known hill height in metres (default: 40.0)
    --tz-offset H       RR file timezone offset from UTC, e.g. -8 for PST
                        (default: -8)
    --ectopic-thresh F  Max fractional RR deviation before flagging as ectopic
                        (default: 0.20)
    --out FILE          Output PNG filename (default: hill_walk_<timestamp>.png)

Dependencies:  numpy, matplotlib  (both standard in most Python environments)

J. Beale  v1.0  2026-02-25
"""

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input",
                   help="Directory containing the GPX and RR files, "
                        "OR explicit GPX file path (then rr_file is also required)")
    p.add_argument("rr_file", nargs="?", default=None,
                   help="Polar H10 RR .txt file (only needed when input is a GPX file)")
    p.add_argument("--baro-height",     type=float, default=40.0,
                   metavar="M",  help="Barometric hill height in metres (default 40.0)")
    p.add_argument("--tz-offset",       type=float, default=-8.0,
                   metavar="H",  help="RR file UTC offset in hours (default -8 = PST)")
    p.add_argument("--ectopic-thresh",  type=float, default=0.20,
                   metavar="F",  help="Ectopic RR deviation threshold 0-1 (default 0.20)")
    p.add_argument("--out",             type=str,   default=None,
                   metavar="FILE", help="Output PNG path")
    return p.parse_args()


def find_files_in_dir(directory):
    """
    Auto-detect the GPX track file and the Polar RR text file inside
    a directory.  Expects exactly one .gpx file and one file whose name
    contains '_RR' and ends in .txt (the standard Polar export format).
    """
    d = Path(directory)
    gpx_files = list({f.resolve(): f for f in list(d.glob("*.gpx")) + list(d.glob("*.GPX"))}.values())
    if len(gpx_files) == 0:
        sys.exit(f"ERROR: No .gpx file found in '{directory}'.")
    if len(gpx_files) > 1:
        sys.exit(f"ERROR: Multiple .gpx files found in '{directory}': "
                 + ", ".join(f.name for f in gpx_files)
                 + " -- pass file paths explicitly instead of a directory.")

    rr_files = [f for f in d.iterdir()
                if f.suffix.lower() == ".txt" and "_RR" in f.name]
    if len(rr_files) == 0:
        sys.exit(f"ERROR: No Polar RR file (*_RR.txt) found in '{directory}'.")
    if len(rr_files) > 1:
        sys.exit(f"ERROR: Multiple RR files found in '{directory}': "
                 + ", ".join(f.name for f in rr_files)
                 + " -- pass file paths explicitly instead of a directory.")

    return str(gpx_files[0]), str(rr_files[0])


# ── GPX parsing ───────────────────────────────────────────────────────────────

def load_gpx(path):
    """Return list of dicts with keys: t (UTC datetime), spd (m/s), ele (m)."""
    ns = "http://www.topografix.com/GPX/1/0"
    tree = ET.parse(path)
    root = tree.getroot()
    points = []
    for pt in root.findall(f".//{{{ns}}}trkpt"):
        ele_el = pt.find(f"{{{ns}}}ele")
        spd_el = pt.find(f"{{{ns}}}speed")
        t_str  = pt.find(f"{{{ns}}}time").text.rstrip("Z")
        if ele_el is None or spd_el is None:
            continue
        t_utc = datetime.fromisoformat(t_str).replace(tzinfo=timezone.utc)
        points.append({"t": t_utc, "spd": float(spd_el.text),
                       "ele": float(ele_el.text)})
    if not points:
        sys.exit("ERROR: No trackpoints with speed and elevation found in GPX file.")
    return points


# ── RR file parsing ───────────────────────────────────────────────────────────

def load_rr(path, tz_offset_h):
    """Return list of dicts with keys: t (UTC datetime), rr (ms int)."""
    records = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            key_ts = next((k for k in row if "timestamp" in k.lower()), None)
            key_rr = next((k for k in row if "rr" in k.lower()), None)
            if key_ts is None or key_rr is None:
                sys.exit("ERROR: Could not find timestamp/RR columns in RR file. "
                         "Expected semicolon-delimited with 'timestamp' and 'RR' headers.")
            t_local = datetime.fromisoformat(row[key_ts].strip())
            t_utc   = t_local.replace(tzinfo=timezone.utc) - \
                      timedelta(hours=tz_offset_h)
            records.append({"t": t_utc, "rr": int(row[key_rr].strip())})
    if not records:
        sys.exit("ERROR: No R-R records found in RR file.")
    return records


# ── Elevation smoothing and calibration ──────────────────────────────────────

def smooth_and_calibrate(eles, baro_height_m, half=7):
    """Running median smooth then scale to barometric hill height."""
    ele_smooth = np.array([
        np.median(eles[max(0, i-half):i+half+1])
        for i in range(len(eles))
    ])
    gps_range = ele_smooth.max() - ele_smooth.min()
    if gps_range < 1.0:
        sys.exit("ERROR: GPS elevation range is less than 1 m; cannot calibrate.")
    scale    = baro_height_m / gps_range
    ele_mean = ele_smooth.mean()
    ele_cal  = ele_mean + (ele_smooth - ele_mean) * scale
    return ele_cal, scale


# ── Phase boundary detection ─────────────────────────────────────────────────

def find_zero_speed_runs(spds, min_run=20, threshold=0.15):
    """Return list of (start_idx, end_idx) for runs of speed < threshold."""
    zero = spds < threshold
    runs = []
    run_start = None
    for i in range(len(spds)):
        if zero[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if i - run_start >= min_run:
                    runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None and len(spds) - run_start >= min_run:
        runs.append((run_start, len(spds) - 1))
    return runs


def detect_phases(points, ele_cal):
    """
    Returns a dict of boundary indices into `points`:
      gpx_start      first outdoor point
      asc_foot       elevation minimum on approach (start of 6m dip)
      asc_climb      where elevation returns to start level (foot of main climb)
      summit_start   start of summit pause
      summit_end     end of summit pause (walking resumes)
      desc_foot      elevation minimum on return (start of 6m re-ascent)
      outdoor_end    first point of indoor rest (sustained zero speed)
    """
    spds = np.array([p["spd"] for p in points])
    runs = find_zero_speed_runs(spds)

    # Runs after idx 100 are: [0] summit pause, [1] indoor rest (if GPS covers it)
    outdoor_runs = [(s, e) for s, e in runs if s > 100]
    if len(outdoor_runs) < 1:
        sys.exit("ERROR: Could not detect a summit pause from GPS speed data. "
                 "Check that the track covers at least the climb and summit.")

    summit_start_i = outdoor_runs[0][0]

    if len(outdoor_runs) >= 2:
        outdoor_end_i = outdoor_runs[1][0]
    else:
        # GPS logging stopped before reaching home — treat the last trackpoint
        # as the transition to indoor rest; RR data beyond this will be binned
        # into the Indoor rest phase automatically.
        outdoor_end_i = len(spds) - 1
        print("NOTE: Only one zero-speed run found after idx 100 (summit pause). "
              "Assuming GPS was stopped at house entry; indoor rest will use "
              "remaining RR data after the final trackpoint.")

    # Summit end = first sample after the pause where speed recovers
    summit_end_i = outdoor_runs[0][1] + 1
    for i in range(outdoor_runs[0][1], min(outdoor_runs[0][1] + 60, len(spds))):
        if spds[i] > 0.3:
            summit_end_i = i
            break

    # Ascent: elevation minimum between start and summit
    asc_foot_i  = int(np.argmin(ele_cal[:summit_start_i]))
    asc_climb_i = asc_foot_i
    start_ele   = ele_cal[0]
    for i in range(asc_foot_i, summit_start_i):
        if ele_cal[i] >= start_ele:
            asc_climb_i = i
            break

    # Descent: elevation minimum between summit end and outdoor end
    desc_foot_i = summit_end_i + int(
        np.argmin(ele_cal[summit_end_i:outdoor_end_i])
    )

    return {
        "asc_foot":     asc_foot_i,
        "asc_climb":    asc_climb_i,
        "summit_start": summit_start_i,
        "summit_end":   summit_end_i,
        "desc_foot":    desc_foot_i,
        "outdoor_end":  outdoor_end_i,
    }


# ── Ectopic / PVC filtering ───────────────────────────────────────────────────

def filter_ectopic(rr_arr, thresh=0.20, half=5):
    """Return boolean array: True = physiologically plausible beat."""
    local_med = np.array([
        np.median(rr_arr[max(0, i-half):i+half+1])
        for i in range(len(rr_arr))
    ])
    return np.abs(rr_arr - local_med) / local_med <= thresh


# ── Rolling median of clean HR ────────────────────────────────────────────────

def rolling_median_hr(t_arr, hr_clean, valid, half=15):
    clean_idx = np.where(valid)[0]
    roll_t, roll_hr = [], []
    for i in range(len(clean_idx)):
        w = clean_idx[max(0, i-half):i+half+1]
        roll_t.append(t_arr[clean_idx[i]])
        roll_hr.append(np.median(hr_clean[w]))
    return np.array(roll_t), np.array(roll_hr)


# ── Stats table ───────────────────────────────────────────────────────────────

def print_stats(phases, t_arr, hr_all, valid, output_lines=None):
    def out(msg):
        print(msg)
        if output_lines is not None:
            output_lines.append(msg)
    
    out(f"\n{'Sub-phase':<20} {'N':>5} {'Mean HR':>8} {'Min':>6} {'Max':>6} {'SD':>6}")
    out("-" * 54)
    for label, t0, t1 in phases:
        mask = valid & (t_arr >= t0) & (t_arr < t1)
        hr_p = hr_all[mask]
        if len(hr_p) >= 2:
            out(f"{label:<20} {len(hr_p):>5} {np.mean(hr_p):>8.1f} "
                  f"{np.min(hr_p):>6.1f} {np.max(hr_p):>6.1f} {np.std(hr_p):>6.1f}")
    out("")


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_plot(gpx_t_min, ele_cal, outdoor_mask, phases, phase_style,
              t_min, hr_all, hr_clean, valid, roll_t_min, roll_hr,
              n_flagged, n_total, out_path):

    fig, (ax_ele, ax_hr) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08}
    )
    fig.patch.set_facecolor("#1a1a2e")

    x_max = t_min[-1]

    for ax in (ax_ele, ax_hr):
        ax.set_facecolor("#0f0f23")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc", labelsize=9)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        ax.set_xlim(0, x_max)

    # Phase shading and boundary lines
    for label, t0_s, t1_s in phases:
        bg, lc = phase_style.get(label, ("#252535", "#aaaaaa"))
        t0m, t1m = t0_s / 60, t1_s / 60
        for ax in (ax_ele, ax_hr):
            ax.axvspan(t0m, t1m, color=bg, alpha=1.0, zorder=0)
            ax.axvline(t1m, color="#444466", lw=0.7, ls="--", zorder=1)

    # Phase labels on elevation panel
    ele_top = ele_cal[outdoor_mask].max() + 3
    for label, t0_s, t1_s in phases:
        _, lc = phase_style.get(label, ("#252535", "#aaaaaa"))
        mid_m = (t0_s + t1_s) / 2 / 60
        ax_ele.text(mid_m, ele_top, label, color=lc, fontsize=7.5,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e",
                              ec="none", alpha=0.7))

    # Elevation line
    ax_ele.plot(gpx_t_min[outdoor_mask], ele_cal[outdoor_mask],
                color="#44aaff", lw=0.9, alpha=0.5, zorder=2)

    # Highlight the two 6 m elevation sections in orange
    for label in ("6 m\ndescent", "6 m\nre-ascent"):
        t0_s, t1_s = next((t0, t1) for lbl, t0, t1 in phases if lbl == label)
        gpx_t_sec = gpx_t_min * 60
        seg = outdoor_mask & (gpx_t_sec >= t0_s) & (gpx_t_sec <= t1_s)
        ax_ele.plot(gpx_t_min[seg], ele_cal[seg],
                    color="#ffaa44", lw=2.0, alpha=0.9, zorder=3)

    ax_ele.set_ylabel("Elevation, cal. (m)", color="#aaaacc")
    ax_ele.yaxis.grid(True, color="#2a2a4a", lw=0.5)

    gpx_fname = Path(out_path).stem
    ax_ele.set_title(f"Hill Walk Analysis -- {gpx_fname}",
                     color="#ddddff", fontsize=12, pad=8)

    # HR scatter + line
    ax_hr.scatter(t_min[~valid], hr_all[~valid],
                  s=14, color="#ff4466", alpha=0.4, zorder=2,
                  label=f"Flagged ectopic ({n_flagged})")
    ax_hr.plot(t_min, hr_clean,
               color="#44ccaa", lw=0.8, alpha=0.6, zorder=3,
               label="HR (beat-by-beat)")
    ax_hr.plot(roll_t_min, roll_hr,
               color="#ffe066", lw=1.8, alpha=0.9, zorder=4,
               label="30-beat rolling median")

    ax_hr.set_ylim(40, 180)
    ax_hr.set_ylabel("Heart rate (bpm)", color="#aaaacc")
    ax_hr.set_xlabel("Time from RR recording start (min)", color="#aaaacc")
    ax_hr.yaxis.grid(True, color="#2a2a4a", lw=0.5)
    ax_hr.legend(fontsize=8, facecolor="#22224a", labelcolor="white",
                 framealpha=0.85, loc="upper right")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_lines = []  # Collect all output for file logging
    
    def out(msg=""):
        """Print to console and collect in output_lines."""
        print(msg)
        output_lines.append(msg)

    # ── Resolve input: directory or explicit file paths ──
    input_path = Path(args.input)
    if input_path.is_dir():
        gpx_file, rr_file = find_files_in_dir(args.input)
    elif input_path.suffix.lower() == ".gpx":
        if args.rr_file is None:
            sys.exit("ERROR: When passing a GPX file directly, "
                     "also provide the RR file as the second argument.")
        gpx_file, rr_file = args.input, args.rr_file
    else:
        sys.exit(f"ERROR: '{args.input}' is neither a directory nor a .gpx file.")

    # ── Load data ──
    out(f"Loading GPX : {gpx_file}")
    points = load_gpx(gpx_file)
    out(f"  {len(points)} trackpoints  "
          f"{points[0]['t'].strftime('%Y-%m-%d %H:%M:%S')} -> "
          f"{points[-1]['t'].strftime('%H:%M:%S')} UTC")

    out(f"Loading RR  : {rr_file}")
    rr_records = load_rr(rr_file, args.tz_offset)
    out(f"  {len(rr_records)} beats  "
          f"{rr_records[0]['t'].strftime('%Y-%m-%d %H:%M:%S')} -> "
          f"{rr_records[-1]['t'].strftime('%H:%M:%S')} UTC")

    # ── Elevation calibration ──
    eles     = np.array([p["ele"] for p in points])
    ele_cal, scale = smooth_and_calibrate(eles, args.baro_height)
    out(f"\nElevation calibration: GPS range {(eles.max()-eles.min()):.1f} m raw -> "
          f"scale factor {scale:.3f} -> {args.baro_height:.0f} m barometric")

    # ── Determine output directory early ──
    if args.out:
        output_dir = Path(args.out).parent
    else:
        output_dir = Path(gpx_file).parent

    # ── Phase detection ──
    idx = detect_phases(points, ele_cal)

    rr_start_utc  = rr_records[0]["t"]
    gpx_start_utc = points[0]["t"]
    gpx_offset_s  = (gpx_start_utc - rr_start_utc).total_seconds()
    rr_end_s      = (rr_records[-1]["t"] - rr_start_utc).total_seconds()

    def to_rr_s(pt_idx):
        return gpx_offset_s + (points[pt_idx]["t"] - gpx_start_utc).total_seconds()

    # Sub-phase time spans in RR-relative seconds
    phases = [
        ("Pre-walk\nrest",   0,                   gpx_offset_s),
        ("Level\napproach",  gpx_offset_s,         to_rr_s(idx["asc_foot"])),
        ("6 m\ndescent",     to_rr_s(idx["asc_foot"]),   to_rr_s(idx["asc_climb"])),
        ("40 m\nclimb",      to_rr_s(idx["asc_climb"]),  to_rr_s(idx["summit_start"])),
        ("Summit\npause",    to_rr_s(idx["summit_start"]),to_rr_s(idx["summit_end"])),
        ("40 m\ndescent",    to_rr_s(idx["summit_end"]),  to_rr_s(idx["desc_foot"])),
        ("6 m\nre-ascent",   to_rr_s(idx["desc_foot"]),  to_rr_s(idx["outdoor_end"])),
        ("Indoor\nrest",     to_rr_s(idx["outdoor_end"]), rr_end_s),
    ]

    phase_style = {
        "Pre-walk\nrest":  ("#252535", "#8888aa"),
        "Level\napproach": ("#1e2a1e", "#88bb88"),
        "6 m\ndescent":    ("#2a1e1e", "#dd8888"),
        "40 m\nclimb":     ("#1a2a1a", "#e06c1b"),
        "Summit\npause":   ("#25152a", "#9b59b6"),
        "40 m\ndescent":   ("#1a1a2a", "#2eaae1"),
        "6 m\nre-ascent":  ("#2a201a", "#ffaa44"),
        "Indoor\nrest":    ("#252535", "#8888aa"),
    }

    # Print timeline
    out(f"\nPhase timeline:")
    for label, t0, t1 in phases:
        lbl = label.replace('\n', ' ')
        out(f"  {lbl:<20}  {t0/60:5.1f} - {t1/60:5.1f} min  "
              f"({(t1-t0)/60:.1f} min)")

    # ── RR arrays ──
    rr_arr = np.array([r["rr"] for r in rr_records], dtype=float)
    t_arr  = np.array([(r["t"] - rr_start_utc).total_seconds()
                       for r in rr_records])
    t_min  = t_arr / 60.0

    # ── Ectopic filter ──
    valid      = filter_ectopic(rr_arr, thresh=args.ectopic_thresh)
    hr_all     = 60000.0 / rr_arr
    hr_clean   = np.where(valid, hr_all, np.nan)
    n_flagged  = int(np.sum(~valid))
    out(f"\nEctopic filter: {n_flagged}/{len(rr_arr)} beats flagged "
          f"({100*n_flagged/len(rr_arr):.1f}%)")

    # ── Rolling median ──
    roll_t_s, roll_hr = rolling_median_hr(t_arr, hr_clean, valid)
    roll_t_min = roll_t_s / 60.0

    # ── Stats ──
    # Use flat labels for the stats table
    phases_flat = [(lbl.replace('\n', ' '), t0, t1) for lbl, t0, t1 in phases]
    print_stats(phases_flat, t_arr, hr_all, valid, output_lines)

    # ── Plot ──
    gpx_t_s = np.array([
        gpx_offset_s + (p["t"] - gpx_start_utc).total_seconds()
        for p in points
    ])
    gpx_t_min    = gpx_t_s / 60.0
    outdoor_mask = gpx_t_s <= to_rr_s(idx["outdoor_end"])

    if args.out:
        out_path = args.out
    else:
        ts = points[0]["t"].strftime("%Y%m%d_%H%M%S")
        out_path = str(output_dir / f"hill_walk_{ts}.png")

    make_plot(gpx_t_min, ele_cal, outdoor_mask,
              phases, phase_style,
              t_min, hr_all, hr_clean, valid,
              roll_t_min, roll_hr,
              n_flagged, len(rr_arr), out_path)
    
    # ── Save output log ──
    ts = points[0]["t"].strftime("%Y%m%d")
    log_path = output_dir / f"{ts}_hill_summary.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(output_lines))
    out(f"Summary saved: {log_path}")


if __name__ == "__main__":
    main()
