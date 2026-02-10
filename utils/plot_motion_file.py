#!/usr/bin/env python3
"""
Plot MPU-6050 motion sensor CSV data from SD card.
Symlog y-axes show both subtle breathing/cardiac and large motion events.
Usage: python3 plot_motion.py MOT_006.csv [output.png]
If no output file given, displays interactively.
J.Beale 2026-02-09
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# --- Configuration ---
BREATH_LO = 0.1    # Hz (6 breaths/min)
BREATH_HI = 0.5    # Hz (30 breaths/min)
CARDIAC_LO = 0.8   # Hz (48 bpm)
CARDIAC_HI = 2.0   # Hz (120 bpm)

# symlog linear thresholds (adjust to taste)
LINTHRESH_ANGLE   = 0.1    # degrees
LINTHRESH_BREATH  = 0.02   # degrees
LINTHRESH_CARDIAC = 0.1    # mG
LINTHRESH_MOTION  = 5.0    # mG / mG·s

def load_and_plot(csv_path, save_path=None):
    # check for timestamp comment in first line
    start_time = None
    skip = 1  # default: skip just the header
    with open(csv_path, 'r') as f:
        first = f.readline().strip()
        if first.startswith('# start ') and 'unknown' not in first:
            start_time = first[8:]
            skip = 2  # skip comment + header

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=skip)
    if data.ndim != 2 or data.shape[1] != 6:
        print(f"Error: expected 6 columns (msec,pitch,roll,rot,total,rms), got {data.shape}")
        sys.exit(1)

    t = (data[:, 0] - data[0, 0]) / 1000.0
    pitch = data[:, 1]
    roll  = data[:, 2]
    rot   = data[:, 3]
    total = data[:, 4]
    rms   = data[:, 5]

    dt_median = np.median(np.diff(t))
    fs = 1.0 / dt_median
    duration_min = t[-1] / 60.0

    print(f"File: {csv_path}")
    print(f"Samples: {len(t)},  Duration: {duration_min:.1f} min,  Fs: {fs:.2f} Hz")

    # design filters
    breath_sos  = butter(2, [BREATH_LO, BREATH_HI], btype='bandpass', fs=fs, output='sos')
    cardiac_sos = butter(2, [CARDIAC_LO, CARDIAC_HI], btype='bandpass', fs=fs, output='sos')

    breath_filt  = sosfiltfilt(breath_sos, pitch)
    cardiac_filt = sosfiltfilt(cardiac_sos, rms)

    # use minutes for long recordings, seconds for short
    # if we have a start time, show real clock time on x-axis for long recordings
    use_clock = False
    if duration_min > 5:
        if start_time:
            from datetime import datetime, timedelta
            import matplotlib.dates as mdates
            try:
                t0_dt = datetime.fromisoformat(start_time)
                t_plot = [t0_dt + timedelta(seconds=s) for s in t]
                use_clock = True
                t_label = "Time"
            except ValueError:
                t_plot = t / 60.0
                t_label = "Time (minutes)"
        else:
            t_plot = t / 60.0
            t_label = "Time (minutes)"
    else:
        t_plot = t
        t_label = "Time (seconds)"

    # extract filename for title
    import os
    base = os.path.splitext(os.path.basename(csv_path))[0]
    title = f"MPU-6050 Motion — {base}  ({len(t)} samples, {duration_min:.1f} min)"
    if start_time:
        title += f"\nStarted: {start_time}"

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(hspace=0.30)

    ax1, ax2, ax3, ax4 = axes

    # 1: Body angle
    ax1.plot(t_plot, pitch, 'b-', linewidth=0.6, alpha=0.8, label='Pitch')
    ax1.plot(t_plot, roll, 'r-', linewidth=0.6, alpha=0.8, label='Roll')
    ax1.set_yscale('symlog', linthresh=LINTHRESH_ANGLE)
    ax1.set_ylabel("Degrees")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title(f"Body Angle (symlog, linear < {LINTHRESH_ANGLE}°)", fontsize=10)

    # 2: Breathing
    ax2.plot(t_plot, breath_filt, 'darkgreen', linewidth=0.8)
    ax2.set_yscale('symlog', linthresh=LINTHRESH_BREATH)
    ax2.set_ylabel("Degrees")
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_title(f"Breathing — pitch BP {BREATH_LO}–{BREATH_HI} Hz (symlog, linear < {LINTHRESH_BREATH}°)", fontsize=10)

    # 3: Cardiac
    ax3.plot(t_plot, cardiac_filt, 'firebrick', linewidth=0.6)
    ax3.set_yscale('symlog', linthresh=LINTHRESH_CARDIAC)
    ax3.set_ylabel("mG")
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_title(f"Cardiac — RMS BP {CARDIAC_LO}–{CARDIAC_HI} Hz (symlog, linear < {LINTHRESH_CARDIAC} mG)", fontsize=10)

    # 4: Raw motion
    ax4.plot(t_plot, total, 'purple', linewidth=0.6, alpha=0.8, label='Total impulse')
    ax4.plot(t_plot, rms, 'darkorange', linewidth=0.6, alpha=0.8, label='Accel RMS')
    ax4.set_yscale('symlog', linthresh=LINTHRESH_MOTION)
    ax4.set_ylabel("mG / mG·s")
    ax4.set_xlabel(t_label)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_title(f"Raw Motion (symlog, linear < {LINTHRESH_MOTION} mG)", fontsize=10)

    plt.tight_layout()

    if use_clock:
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=0, ha='center')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_motion.py <input.csv> [output.png]")
        sys.exit(1)
    csv_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    load_and_plot(csv_path, save_path)
