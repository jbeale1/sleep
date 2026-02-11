#!/usr/bin/env python3
"""
Time sync helper for RP2040 MPU-6050 logger.
Waits for serial port to appear, then responds to "TIME?" with current time.
Passes through serial output so you can watch the startup.
Delays response until the next integer second for sub-second sync to host time.

Usage: python3 time_sync.py [/dev/ttyACM0]
Press Ctrl+C to exit.
J.Beale 2026-02-10
"""

import sys
import os
import time
import serial
from datetime import datetime

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = 115200

# --- Wait for port to appear ---
print(f"Waiting for {PORT} ...")
while not os.path.exists(PORT):
    time.sleep(0.1)
print(f"Port found, connecting...")

# short delay to let the OS finish setting up the device
time.sleep(0.3)

ser = serial.Serial(PORT, BAUD, timeout=0.5)
print(f"Opened {PORT} — listening for TIME? request")

synced = False
try:
    while True:
        raw = ser.readline().decode('utf-8', errors='ignore').strip()
        if not raw:
            continue

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
            synced = True

except KeyboardInterrupt:
    print("\nDone.")
finally:
    ser.close()
