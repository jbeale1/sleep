#!/usr/bin/env python3
"""
Time sync helper for RP2040 MPU-6050 logger.
Waits for serial port to appear, then responds to "TIME?" with current time.
Passes through serial output so you can watch the startup.
Delays response until the next integer second for sub-second sync to host time.
Logs RP2040-vs-PC clock drift to a CSV file for crystal characterization.

Usage: python3 time_sync_logger_start.py [PORT]
Press Ctrl+C to exit.
J.Beale 2026-02-10
"""

import sys
import os
import time
import serial
import serial.tools.list_ports
from datetime import datetime

pt = "COM28" if os.name == "nt" else "/dev/ttyACM0"
PORT = sys.argv[1] if len(sys.argv) > 1 else pt
BAUD = 115200

def port_exists(port):
    if os.name == "nt":
        return any(p.device == port for p in serial.tools.list_ports.comports())
    else:
        return os.path.exists(port)

# --- Wait for port to appear ---
print(f"Waiting for {PORT} ...")
while not port_exists(PORT):
    time.sleep(0.1)
print(f"Port found, connecting...")

# short delay to let the OS finish setting up the device
time.sleep(0.3)

ser = serial.Serial(PORT, BAUD, timeout=0.5)
print(f"Opened {PORT} — listening for TIME? request")

synced = False
sync_epoch = None
drift_file = None
drift_count = 0

try:
    while True:
        try:
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
        except serial.SerialException:
            print("\nDevice disconnected.")
            break            
        if not raw:
            continue

        pc_time = time.time()  # capture immediately on receipt
        print(raw)

        if raw == "TIME?" and not synced:
            # compute the next integer second, then sleep until it arrives
            now = time.time()
            next_sec = int(now) + 1
            time.sleep(next_sec - now)
            ts = datetime.fromtimestamp(next_sec).strftime("%Y-%m-%dT%H:%M:%S")
            response = f"TIME={ts}\n"
            ser.write(response.encode())
            print(f">>> Sent: {response.strip()}")
            sync_epoch = float(next_sec)
            synced = True

            # open drift log file
            drift_fname = f"drift_{datetime.fromtimestamp(next_sec).strftime('%Y%m%d_%H%M%S')}.csv"
            drift_file = open(drift_fname, 'w')
            drift_file.write("elapsed_s,offset_ms\n")
            print(f">>> Drift log: {drift_fname}")
            continue

        # parse CSV data lines: "msec,pitch,roll,rot,total,rms"
        if synced and sync_epoch and ',' in raw:
            try:
                msec_str = raw.split(',')[0]
                rp2040_msec = int(msec_str)
                rp2040_sec = rp2040_msec / 1000.0
                expected_pc_time = sync_epoch + rp2040_sec
                offset_ms = (pc_time - expected_pc_time) * 1000.0
                drift_count += 1
                # log every 50th sample (~1/10sec at 5Hz)
                if drift_count % 50 == 0:
                    drift_file.write(f"{rp2040_sec:.3f},{offset_ms:.1f}\n")
                    drift_file.flush()
            except (ValueError, IndexError):
                pass  # skip non-data lines (headers, comments, etc.)

except KeyboardInterrupt:
    print("\nDone.")
finally:
    ser.close()
    if drift_file:
        drift_file.close()
        print(f"Drift log saved ({drift_count} samples captured)")
