#!/usr/bin/env python3
"""
Read data from Innovo IP900BPB pulse oximeter via BLE.
Captures pleth waveform (~20 Hz) and summary packets (SpO2, HR, PI).

Requires: pip install bleak
Usage: python innovo_ble_reader.py [--csv output.csv] [--duration 60]
"""

import asyncio
import argparse
import time
import csv
from datetime import datetime
from bleak import BleakClient, BleakScanner

# Correct characteristic UUID for this device
NOTIFY_UUID = "0000fff1-0000-1000-8000-00805f9b34fb"

DEVICE_NAME = "IP900BPB"
DEVICE_ADDR = "ED:9B:47:2E:0F:68"


class InnovoPulseOx:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None

        # Latest summary values
        self.spo2 = None
        self.heart_rate = None
        self.pi = None

        # Counters
        self.pleth_count = 0
        self.summary_count = 0

    def open_csv(self):
        if self.csv_path:
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'elapsed_s', 'type',
                'pleth', 'spo2', 'hr', 'pi'
            ])

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()

    def handle_notification(self, sender, data: bytearray):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if len(data) == 2 and data[0] == 0x01:
            # Pleth waveform sample
            pleth = data[1]
            self.pleth_count += 1

            if self.csv_writer:
                self.csv_writer.writerow([
                    timestamp, f'{elapsed:.3f}', 'pleth',
                    pleth, '', '', ''
                ])

        elif len(data) >= 6 and data[0] == 0x3E:
            # Summary packet:
            #   [0] = 0x3E (marker)
            #   [1] = SpO2 (0-100)
            #   [3] = Heart rate (bpm)
            #   [5] = Respiration rate (RR)
            #   [11]= Perfusion Index (PI)

            spo2 = data[1]
            hr = data[3]
            rr = data[5]
            pi = data[11] / 10.0
            print(f'[{timestamp}]  SpO2: {spo2}%  HR: {hr} bpm  RR: {rr}  PI: {pi:.1f}%')

            self.spo2 = spo2
            self.heart_rate = hr
            self.pi = pi
            self.summary_count += 1

            # print(f'[{timestamp}]  SpO2: {spo2}%  HR: {hr} bpm  PI: {pi:.1f}%')
            #print(f'[{timestamp}]  SpO2: {spo2}%  HR: {hr} bpm  PI: {pi:.1f}%  raw: {data.hex(" ")}')

            if self.csv_writer:
                self.csv_writer.writerow([
                    timestamp, f'{elapsed:.3f}', 'summary',
                    '', spo2, hr, pi
                ])
                self.csv_file.flush()


async def find_device():
    """Scan for the Innovo device."""
    print("Scanning for pulse oximeter...")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        if d.name and DEVICE_NAME in d.name:
            print(f"Found: {d.name} [{d.address}]")
            return d.address
        if d.address and d.address.upper() == DEVICE_ADDR.upper():
            print(f"Found by address: {d.name or 'Unknown'} [{d.address}]")
            return d.address
    return None


async def main(csv_path=None, duration=None):
    address = await find_device()
    if not address:
        print(f"Device not found. Make sure {DEVICE_NAME} is on with finger inserted.")
        return

    oximeter = InnovoPulseOx(csv_path=csv_path)
    oximeter.open_csv()

    try:
        async with BleakClient(address) as client:
            print(f"Connected to {address}")

            # May need to enable indications on fff0 to trigger data stream
            await client.start_notify("0000fff0-0000-1000-8000-00805f9b34fb",
                                       lambda s, d: None)
            
            await client.start_notify(NOTIFY_UUID, oximeter.handle_notification)
            print("Receiving data... (Ctrl+C to stop)\n")

            if duration:
                await asyncio.sleep(duration)
            else:
                while client.is_connected:
                    await asyncio.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        oximeter.close_csv()
        elapsed = time.time() - oximeter.start_time if oximeter.start_time else 0
        print(f"\nCaptured {oximeter.pleth_count} pleth samples, "
              f"{oximeter.summary_count} summary packets over {elapsed:.1f}s")
        if oximeter.pleth_count > 0 and elapsed > 0:
            print(f"Pleth rate: {oximeter.pleth_count / elapsed:.1f} Hz")
        if csv_path:
            print(f"Data saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovo IP900BPB Pulse Oximeter Reader')
    parser.add_argument('--csv', type=str, help='Output CSV file path')
    parser.add_argument('--duration', type=float, help='Recording duration in seconds')
    args = parser.parse_args()

    asyncio.run(main(csv_path=args.csv, duration=args.duration))
