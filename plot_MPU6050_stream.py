#!/usr/bin/env python3
"""
Live serial plotter for RP2040 MPU-6050 motion sensor (v2).
Reads 6-column CSV from serial port, displays scrolling strip chart.
Bandpass-filtered pitch for breathing, bandpass-filtered RMS for cardiac.
Usage: python3 mpu6050_plot.py [/dev/ttyACM0]
J.Beale 2026-02-09
"""

import sys
import serial
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, sosfiltfilt

# --- Configuration ---
#DEFAULT_SERIAL = "/dev/ttyACM0"
DEFAULT_SERIAL = "COM28"

SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERIAL
BAUD_RATE = 115200
WINDOW_SEC = 40        # seconds of history to display
SAMPLE_INTERVAL = 0.2   # expected seconds between samples (5 Hz)
MAX_POINTS = int(WINDOW_SEC / SAMPLE_INTERVAL) + 20
FS = 1.0 / SAMPLE_INTERVAL  # 5 Hz sample rate

# Breathing bandpass: 0.1 - 0.5 Hz (6-30 breaths/min)
BREATH_LO = 0.1
BREATH_HI = 0.5
breath_sos = butter(2, [BREATH_LO, BREATH_HI], btype='bandpass', fs=FS, output='sos')

# Cardiac bandpass: 0.8 - 2.0 Hz (48-120 bpm)
# Nyquist = 2.5 Hz, so 2.0 Hz is safe
CARDIAC_LO = 0.8
CARDIAC_HI = 2.0
cardiac_sos = butter(2, [CARDIAC_LO, CARDIAC_HI], btype='bandpass', fs=FS, output='sos')

MIN_FILT_SAMPLES = 40  # need this many before filtering is meaningful

# --- Data buffers ---
t_buf     = deque(maxlen=MAX_POINTS)
pitch_buf = deque(maxlen=MAX_POINTS)
roll_buf  = deque(maxlen=MAX_POINTS)
rot_buf   = deque(maxlen=MAX_POINTS)
total_buf = deque(maxlen=MAX_POINTS)
rms_buf   = deque(maxlen=MAX_POINTS)

t0 = None

# --- Serial setup ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
print(f"Opened {SERIAL_PORT} at {BAUD_RATE} baud")
ser.readline()  # discard partial line

def read_serial():
    """Read and parse all available lines from serial."""
    global t0
    while ser.in_waiting:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw or raw.startswith("msec") or raw.startswith("=") or raw.startswith("["):
                continue
            parts = raw.split(',')
            if len(parts) != 6:
                continue
            msec  = int(parts[0])
            pitch = float(parts[1])
            roll  = float(parts[2])
            rot   = float(parts[3])
            total = float(parts[4])
            rms   = float(parts[5])

            if t0 is None:
                t0 = msec
            t_sec = (msec - t0) / 1000.0

            t_buf.append(t_sec)
            pitch_buf.append(pitch)
            roll_buf.append(roll)
            rot_buf.append(rot)
            total_buf.append(total)
            rms_buf.append(rms)

        except (ValueError, UnicodeDecodeError):
            continue

def apply_bandpass(data_deque, sos):
    """Apply zero-phase bandpass filter with mirror padding. Returns filtered array or empty."""
    n = len(data_deque)
    if n < MIN_FILT_SAMPLES:
        return np.array([])
    arr = np.array(data_deque)
    pad_len = min(n - 1, 40)
    padded = np.concatenate([arr[pad_len:0:-1], arr, arr[-2:-pad_len-2:-1]])
    filtered = sosfiltfilt(sos, padded)
    return filtered[pad_len:-pad_len]

# --- Plot setup: 4 subplots ---
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
fig.suptitle("MPU-6050 Live Motion (5 Hz)", fontsize=13)
fig.subplots_adjust(hspace=0.30)

ax_angle, ax_breath, ax_cardiac, ax_motion = axes

# 1: Body angle
line_pitch, = ax_angle.plot([], [], 'b-', linewidth=1, label='Pitch')
line_roll,  = ax_angle.plot([], [], 'r-', linewidth=1, label='Roll')
ax_angle.set_ylabel("Degrees")
ax_angle.legend(loc='upper left', fontsize=9)
ax_angle.grid(True, alpha=0.3)
ax_angle.set_title("Body Angle", fontsize=10)

# 2: Breathing (bandpass-filtered pitch)
line_breath, = ax_breath.plot([], [], 'darkgreen', linewidth=1.5,
                              label=f'Pitch BP {BREATH_LO}-{BREATH_HI} Hz')
ax_breath.set_ylabel("Degrees")
ax_breath.legend(loc='upper left', fontsize=9)
ax_breath.grid(True, alpha=0.3)
ax_breath.set_title("Breathing (bandpass-filtered pitch)", fontsize=10)

# 3: Cardiac (bandpass-filtered accel RMS)
line_cardiac, = ax_cardiac.plot([], [], 'firebrick', linewidth=1.2,
                                label=f'RMS BP {CARDIAC_LO}-{CARDIAC_HI} Hz')
ax_cardiac.set_ylabel("mG")
ax_cardiac.legend(loc='upper left', fontsize=9)
ax_cardiac.grid(True, alpha=0.3)
ax_cardiac.set_title("Cardiac (bandpass-filtered accel RMS)", fontsize=10)

# 4: Raw motion (total impulse + RMS)
line_total, = ax_motion.plot([], [], 'purple', linewidth=1, alpha=0.8, label='Total impulse')
line_rms,   = ax_motion.plot([], [], 'darkorange', linewidth=1, alpha=0.8, label='Accel RMS')
ax_motion.set_ylabel("mG / mG·s")
ax_motion.set_xlabel("Time (seconds)")
ax_motion.legend(loc='upper left', fontsize=9)
ax_motion.grid(True, alpha=0.3)
ax_motion.set_title("Raw Motion Metrics", fontsize=10)

def update(frame):
    read_serial()

    if len(t_buf) < 2:
        return []

    t = np.array(t_buf)
    t_max = t[-1]
    t_min = t_max - WINDOW_SEC
    visible = t >= t_min

    # 1: Angle
    line_pitch.set_data(t, np.array(pitch_buf))
    line_roll.set_data(t, np.array(roll_buf))

    # 2: Breathing
    bp = apply_bandpass(pitch_buf, breath_sos)
    if len(bp) > 0:
        line_breath.set_data(t, bp)

    # 3: Cardiac
    cp = apply_bandpass(rms_buf, cardiac_sos)
    if len(cp) > 0:
        line_cardiac.set_data(t, cp)

    # 4: Raw motion
    line_total.set_data(t, np.array(total_buf))
    line_rms.set_data(t, np.array(rms_buf))

    # Scroll x-axis
    for ax in axes:
        ax.set_xlim(t_min, t_max)

    # Auto-scale y axes on visible data
    if np.any(visible):
        # Angle
        p = np.array(pitch_buf)[visible]
        r = np.array(roll_buf)[visible]
        if len(p) > 0:
            ylo = min(p.min(), r.min())
            yhi = max(p.max(), r.max())
            margin = max(0.5, (yhi - ylo) * 0.1)
            ax_angle.set_ylim(ylo - margin, yhi + margin)

        # Breathing: symmetric around zero
        if len(bp) == len(t):
            bp_vis = bp[visible]
            if len(bp_vis) > 0:
                bp_max = max(0.005, np.abs(bp_vis).max())
                ax_breath.set_ylim(-bp_max * 1.3, bp_max * 1.3)

        # Cardiac: symmetric around zero
        if len(cp) == len(t):
            cp_vis = cp[visible]
            if len(cp_vis) > 0:
                cp_max = max(0.1, np.abs(cp_vis).max())
                ax_cardiac.set_ylim(-cp_max * 1.3, cp_max * 1.3)

        # Raw motion
        tot_vis = np.array(total_buf)[visible]
        rms_vis = np.array(rms_buf)[visible]
        if len(tot_vis) > 0:
            yhi = max(1.0, tot_vis.max(), rms_vis.max())
            ax_motion.set_ylim(-0.5, yhi * 1.1)

    return [line_pitch, line_roll, line_breath, line_cardiac,
            line_total, line_rms]

ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()

ser.close()