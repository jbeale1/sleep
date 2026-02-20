#!/usr/bin/env python3
"""
ecg_blink.py — Reads HR from ECG log and blinks it in Morse code on GPIO 6.
LED wired: +3.3V -> 10k -> LED -> GPIO6 (active LOW = LED on)
"""

import os
import re
import time
import RPi.GPIO as GPIO

LOG_FILE    = "/home/pi/ECG/ecg_service.log"
GPIO_PIN    = 6          # BCM numbering (physical pin 31)
MAX_AGE_S   = 60         # File must be updated within this many seconds
INTERVAL_S  = 10.0        # Seconds between Morse transmissions
UNIT_S      = 0.12       # Morse dit duration in seconds

MORSE = {
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..',  '9': '----.'
}

# Active-low: LED ON = GPIO LOW, LED OFF = GPIO HIGH
LED_ON  = GPIO.LOW
LED_OFF = GPIO.HIGH

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_PIN, GPIO.OUT, initial=LED_OFF)

def led(state):
    GPIO.output(GPIO_PIN, state)

def dit():
    led(LED_ON);  time.sleep(UNIT_S)
    led(LED_OFF); time.sleep(UNIT_S)       # intra-character gap

def dah():
    led(LED_ON);  time.sleep(UNIT_S * 3)
    led(LED_OFF); time.sleep(UNIT_S)       # intra-character gap

def send_char(ch):
    for symbol in MORSE[ch]:
        dit() if symbol == '.' else dah()
    # Replace intra-char trailing gap with full inter-character gap (3 units total,
    # 1 already elapsed, so add 2 more)
    time.sleep(UNIT_S * 2)

def send_number(n):
    """Send an integer as individual Morse digit characters."""
    digits = str(n)
    for i, ch in enumerate(digits):
        send_char(ch)
    # Ensure LED is off at end
    led(LED_OFF)

def get_hr():
    """Return HR as int, or None if file missing/stale/unparseable."""
    if not os.path.isfile(LOG_FILE):
        return None
    age = time.time() - os.path.getmtime(LOG_FILE)
    if age > MAX_AGE_S:
        return None
    try:
        with open(LOG_FILE, 'rb') as f:
            # Efficiently grab last non-empty line
            f.seek(0, 2)
            size = f.tell()
            buf = b''
            pos = size
            while pos > 0:
                read = min(256, pos)
                pos -= read
                f.seek(pos)
                buf = f.read(read) + buf
                lines = buf.split(b'\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line:
                        last_line = line.decode('utf-8', errors='replace')
                        m = re.search(r'HR\s+([\d.]+)', last_line)
                        if m:
                            return int(float(m.group(1)))
                        return None
    except Exception:
        return None

def main():
    setup()
    print("ECG Morse blinker running. Ctrl-C to exit.")
    try:
        while True:
            t_start = time.monotonic()
            hr = get_hr()
            if hr is not None:
                print(f"HR={hr}, sending Morse...")
                send_number(hr)
            else:
                print("No recent data, skipping.")
            elapsed = time.monotonic() - t_start
            remaining = INTERVAL_S - elapsed
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        led(LED_OFF)
        GPIO.cleanup()

if __name__ == "__main__":
    main()
