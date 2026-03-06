#!/usr/bin/env python3

# log data from serial port to file

import serial
import os
import time
from datetime import datetime

device = '/dev/ttyACM0'
print(f"Waiting for {device}...")

while not os.path.exists(device):
    time.sleep(1)

print(f"{device} found. Waiting for data...")

ser = serial.Serial(device, 115200, timeout=1)

csv_file = None
writer = None

while True:
    line = ser.readline()
    if line:
        text = line.decode('utf-8', errors='replace')
        print(text, end='')

        if csv_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"temp_{timestamp}.csv"
            csv_file = open(filename, 'w')
            print(f"Logging to {filename}")

        csv_file.write(text)
        csv_file.flush()
