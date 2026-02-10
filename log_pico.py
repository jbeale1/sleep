# This script reads pressure data from a Pico microcontroller over serial,
# plots it live, and allows saving the data to a CSV log file with timestamps.
# The plot updates in real-time, showing the most recent 500 samples.
# Heart rate is estimated using a hybrid autocorrelation + derivative method.
# The "Save Log" button toggles logging on and off.
#
# Dependencies (install with ~\AppData\Local\Programs\Python\Python312\python.exe -m pip install ...):
#   matplotlib, numpy, scipy, pyserial

# External absolute pressure sensor is located on suprasternal notch,
# sealed with a 2mm thick foam pad. Pressure signal is a function
# of heartbeat, breathing, swallowing and any overall neck muscle tension changes.

import serial
import serial.tools.list_ports
import sys
import collections
import csv
import datetime
import os
import time
import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

BUFFER_SIZE = 500
FS = 31.5  # approximate sample rate from Pico
LOG_DIR = r"C:\Users\beale\Documents\2026-sleep\plog"

# HR analysis parameters
HR_MIN_BPM = 40
HR_MAX_BPM = 180
MIN_AUTOCORR_QUALITY = 0.20  # minimum autocorrelation peak to consider valid
MIN_BEATS = 4               # minimum detected beats in window to report HR
MAX_IBI_CV = 0.25           # max coefficient of variation of inter-beat intervals
HR_STALE_SECONDS = 5        # seconds before marking HR as stale

# Pre-compute filter coefficients (avoid recomputing every frame)
SOS_NARROW = sig.butter(4, [0.7, 2.5], btype='bandpass', fs=FS, output='sos')
SOS_LP = sig.butter(4, 8.0, btype='lowpass', fs=FS, output='sos')
SOS_HP = sig.butter(2, 0.5, btype='highpass', fs=FS, output='sos')
SOS_RESP = sig.butter(3, [0.05, 0.5], btype='bandpass', fs=FS, output='sos')


def find_pico():
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x2E8A:
            return port.device
    return None


def compute_hr(data_array):
    """Compute heart rate using hybrid autocorrelation + derivative method.
    Returns (hr_bpm, quality) where quality is 0-1, or (None, 0) if unreliable."""

    N = len(data_array)
    if N < int(4 * FS):
        return None, 0

    # Step 1: Autocorrelation on narrow bandpass to estimate beat period
    filtered = sig.sosfiltfilt(SOS_NARROW, data_array)
    x = filtered - np.mean(filtered)
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr) // 2:]
    if corr[0] <= 0:
        return None, 0
    corr = corr / corr[0]

    min_lag = int(60.0 / HR_MAX_BPM * FS)
    max_lag = min(int(60.0 / HR_MIN_BPM * FS), len(corr) - 1)
    if min_lag >= max_lag:
        return None, 0

    peaks, _ = sig.find_peaks(corr[min_lag:max_lag], distance=int(0.2 * FS))
    peaks = peaks + min_lag
    if len(peaks) == 0:
        return None, 0

    best_peak = peaks[np.argmax(corr[peaks])]
    ac_quality = corr[best_peak]
    if ac_quality < MIN_AUTOCORR_QUALITY:
        return None, 0

    est_period = best_peak / FS

    # Step 2: Derivative peak detection with autocorr-guided minimum distance
    smoothed = sig.sosfiltfilt(SOS_LP, data_array)
    detrended = sig.sosfiltfilt(SOS_HP, smoothed)
    deriv = np.diff(detrended) * FS
    deriv_std = np.std(deriv)
    if deriv_std == 0:
        return None, 0

    min_dist = max(int(0.6 * est_period * FS), 3)
    d_peaks, _ = sig.find_peaks(deriv, height=deriv_std * 0.8,
                                distance=min_dist, prominence=deriv_std * 0.5)

    if len(d_peaks) < MIN_BEATS:
        return None, 0

    ibi = np.diff(d_peaks) / FS

    # Reject outlier IBIs using median absolute deviation
    med_ibi = np.median(ibi)
    mad = np.median(np.abs(ibi - med_ibi))
    if mad > 0:
        ibi_good = ibi[np.abs(ibi - med_ibi) < 3 * mad]
    else:
        ibi_good = ibi

    if len(ibi_good) < 3:
        return None, 0

    # Assess consistency
    cv = np.std(ibi_good) / np.mean(ibi_good)
    if cv > MAX_IBI_CV:
        return None, 0

    quality = ac_quality * (1.0 - cv)
    hr = 60.0 / np.median(ibi_good)

    if hr < HR_MIN_BPM or hr > HR_MAX_BPM:
        return None, 0

    return hr, quality


# --- Find and open Pico serial port ---
port_name = find_pico()
if not port_name:
    print("No Pico found.")
    sys.exit(1)

print(f"Pico found on {port_name}")
ser = serial.Serial(port_name, baudrate=115200, timeout=0.05)

# --- Data buffers ---
data = collections.deque(maxlen=BUFFER_SIZE)
hr_times = collections.deque(maxlen=300)   # ~5 min at 1/sec
hr_values = collections.deque(maxlen=300)
hr_fresh = collections.deque(maxlen=300)   # True = fresh measurement, False = stale

# --- HR state ---
last_good_hr = None
last_good_time = 0
hr_update_counter = 0
HR_UPDATE_INTERVAL = 3  # recompute HR every N animation frames (~300ms)

# --- Logging state ---
log_file = None
log_writer = None
logging_active = False

# --- Set up figure with two subplots ---
fig, (ax_pressure, ax_hr) = plt.subplots(2, 1, figsize=(10, 7),
                                          gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(bottom=0.12, hspace=0.35)

# Pressure plot
pressure_line, = ax_pressure.plot([], [], linewidth=0.5)
resp_line, = ax_pressure.plot([], [], linewidth=2, color='red', alpha=0.7, label='Respiratory')
ax_pressure.legend(loc='upper left', fontsize=9)
ax_pressure.set_xlabel("Sample")
ax_pressure.set_ylabel("Pressure (hPa)")
ax_pressure.set_title("Live Pressure")

# HR readout text
hr_text = ax_pressure.text(0.98, 0.95, "HR: ---", transform=ax_pressure.transAxes,
                           fontsize=16, fontweight='bold', ha='right', va='top',
                           color='gray',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor='gray', alpha=0.8))

# HR trend plot
hr_line, = ax_hr.plot([], [], 'o-', markersize=3, color='steelblue')
hr_stale_line, = ax_hr.plot([], [], 'o', markersize=3, color='lightgray')
ax_hr.set_ylabel("HR (bpm)")
ax_hr.set_xlabel("Time (s ago)")
ax_hr.set_title("Heart Rate Trend")

# Save Log button
btn_ax = fig.add_axes([0.4, 0.01, 0.2, 0.04])
btn = Button(btn_ax, "Save Log")


def toggle_log(event):
    global log_file, log_writer, logging_active
    if not logging_active:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(LOG_DIR, f"pressure_log_{timestamp}.csv")
        log_file = open(filename, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp", "pressure_hPa"])
        logging_active = True
        btn.label.set_text("Stop Log")
        print(f"Logging to {filename}")
    else:
        logging_active = False
        log_file.close()
        log_file = None
        log_writer = None
        btn.label.set_text("Save Log")
        print("Logging stopped.")


btn.on_clicked(toggle_log)


def update(frame):
    global last_good_hr, last_good_time, hr_update_counter

    # --- Read serial data ---
    while ser.in_waiting:
        raw = ser.readline().decode('utf-8', errors='replace').strip()
        if raw:
            try:
                value = float(raw)
                data.append(value)
                if logging_active and log_writer:
                    log_writer.writerow([datetime.datetime.now().isoformat(), value])
            except ValueError:
                print(f"[Pico] {raw}")

    now = time.time()

    # --- Update pressure plot ---
    if data:
        ydata = list(data)
        xdata = list(range(len(ydata)))
        pressure_line.set_data(xdata, ydata)
        ax_pressure.set_xlim(0, BUFFER_SIZE)
        ymin, ymax = min(ydata), max(ydata)
        margin = max((ymax - ymin) * 0.1, 0.01)
        ax_pressure.set_ylim(ymin - margin, ymax + margin)

        # Overlay respiratory-band filtered signal
        if len(ydata) > int(2 * FS):
            arr = np.array(ydata)
            try:
                resp_filtered = sig.sosfiltfilt(SOS_RESP, arr)
                # Shift to sit on the raw data's mean level
                resp_display = resp_filtered - np.mean(resp_filtered) + np.mean(arr)
                resp_line.set_data(xdata, resp_display)
            except Exception:
                resp_line.set_data([], [])
        else:
            resp_line.set_data([], [])

    # --- Compute HR periodically ---
    hr_update_counter += 1
    if hr_update_counter >= HR_UPDATE_INTERVAL and len(data) >= int(4 * FS):
        hr_update_counter = 0
        arr = np.array(data)
        hr, quality = compute_hr(arr)

        if hr is not None:
            last_good_hr = hr
            last_good_time = now
            hr_times.append(now)
            hr_values.append(hr)
            hr_fresh.append(True)
            hr_text.set_text(f"HR: {hr:.0f} bpm")
            hr_text.set_color('green')
            hr_text.get_bbox_patch().set_edgecolor('green')
        else:
            if last_good_hr is not None:
                stale_sec = now - last_good_time
                hr_text.set_text(f"HR: {last_good_hr:.0f} bpm ({stale_sec:.0f}s ago)")
                hr_text.set_color('#CC8800')
                hr_text.get_bbox_patch().set_edgecolor('#CC8800')
                hr_times.append(now)
                hr_values.append(last_good_hr)
                hr_fresh.append(False)
            else:
                hr_text.set_text("HR: ---")
                hr_text.set_color('gray')
                hr_text.get_bbox_patch().set_edgecolor('gray')

    # --- Update HR trend plot ---
    if hr_times:
        times_arr = np.array(hr_times)
        vals_arr = np.array(hr_values)
        fresh_arr = np.array(hr_fresh)
        x_ago = -(now - times_arr)  # negative seconds = time ago

        fresh_mask = fresh_arr
        if np.any(fresh_mask):
            hr_line.set_data(x_ago[fresh_mask], vals_arr[fresh_mask])
        else:
            hr_line.set_data([], [])

        stale_mask = ~fresh_arr
        if np.any(stale_mask):
            hr_stale_line.set_data(x_ago[stale_mask], vals_arr[stale_mask])
        else:
            hr_stale_line.set_data([], [])

        ax_hr.set_xlim(min(x_ago[0], -60), 0)
        if len(vals_arr) > 0:
            hr_min, hr_max = vals_arr.min(), vals_arr.max()
            hr_margin = max((hr_max - hr_min) * 0.15, 3)
            ax_hr.set_ylim(hr_min - hr_margin, hr_max + hr_margin)

    return (pressure_line, resp_line, hr_line, hr_stale_line, hr_text)


ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

try:
    plt.show()
finally:
    if log_file:
        log_file.close()
    ser.close()
