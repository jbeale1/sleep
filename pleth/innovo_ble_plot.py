#!/usr/bin/env python3
"""
Read data from Innovo IP900BPB pulse oximeter via BLE.
Live pleth waveform plot + summary display (SpO2, HR, RR, PI).
Saves pleth and summary data to separate CSV files.

Requires: pip install bleak matplotlib
Usage: python innovo_ble_plot.py [--csv prefix] [--seconds 10]
  --csv prefix  creates prefix_pleth.csv and prefix_summary.csv
"""

import asyncio
import argparse
import os
import time
import csv
import threading
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bleak import BleakClient, BleakScanner

# BLE config
NOTIFY_UUID = "0000fff1-0000-1000-8000-00805f9b34fb"
INDICATE_UUID = "0000fff0-0000-1000-8000-00805f9b34fb"
DEVICE_NAME = "IP900BPB"
# DEVICE_ADDR = "ED:9B:47:2E:0F:68"
DEVICE_ADDRS = {"ED:9B:47:2E:0F:68", "D3:67:B7:93:27:00"}

# Plot config
PLOT_SECONDS = 10


class InnovoPulseOx:
    def __init__(self, plot_seconds=10, csv_prefix=None):
        self.csv_prefix = csv_prefix
        self.pleth_file = None
        self.pleth_writer = None
        self.summary_file = None
        self.summary_writer = None
        self.start_time = None

        # Pleth deque for CSV (keeps recent values)
        self.max_samples = plot_seconds * 25
        self.pleth_values = deque(maxlen=self.max_samples)

        # Sweep buffer for display (circular, written by handler)
        self.sweep_len = int(plot_seconds * 24)
        self.sweep_buf = [float('nan')] * self.sweep_len
        self.sweep_gap = max(6, self.sweep_len // 20)

        # Latest summary values
        self.spo2 = 0
        self.heart_rate = 0
        self.resp_rate = 0
        self.pi = 0.0

        self.pleth_count = 0
        self.summary_count = 0
        self.connected = False
        self.running = True

    def open_csv(self):
        if self.csv_prefix:
            self.pleth_file = open(f'{self.csv_prefix}_pleth.csv', 'w', newline='')
            self.pleth_writer = csv.writer(self.pleth_file)
            self.pleth_writer.writerow(['timestamp', 'elapsed_s', 'pleth'])

            self.summary_file = open(f'{self.csv_prefix}_summary.csv', 'w', newline='')
            self.summary_writer = csv.writer(self.summary_file)
            self.summary_writer.writerow(['timestamp', 'elapsed_s', 'spo2', 'hr', 'rr', 'pi'])

    def close_csv(self):
        if self.pleth_file:
            self.pleth_file.close()
        if self.summary_file:
            self.summary_file.close()

    def handle_notification(self, sender, data: bytearray):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        if len(data) == 2 and data[0] == 0x01:
            pleth = data[1]
            self.pleth_count += 1
            self.pleth_values.append(pleth)

            # Write into sweep buffer and clear gap ahead
            pos = (self.pleth_count - 1) % self.sweep_len
            self.sweep_buf[pos] = pleth
            for g in range(1, self.sweep_gap + 1):
                self.sweep_buf[(pos + g) % self.sweep_len] = float('nan')

            if self.pleth_writer:
                self.pleth_writer.writerow([timestamp, f'{elapsed:.3f}', pleth])

        elif len(data) >= 12 and data[0] == 0x3E:
            self.spo2 = data[1]
            self.heart_rate = data[3]
            self.resp_rate = data[5]
            self.pi = data[11] / 10.0
            self.summary_count += 1

            if self.summary_writer:
                self.summary_writer.writerow([
                    timestamp, f'{elapsed:.3f}',
                    self.spo2, self.heart_rate, self.resp_rate, self.pi
                ])
                self.summary_file.flush()


async def ble_task(oximeter):
    """BLE connection and data reception."""
    print("Scanning for pulse oximeter...")
    
    devices = await BleakScanner.discover(timeout=10.0)
    address = None
    for d in devices:
        if d.name and DEVICE_NAME in d.name:
            address = d.address
            break        
        if d.address and d.address.upper() in DEVICE_ADDRS:            
            address = d.address
            break

    if not address:
        print(f"Device not found. Make sure {DEVICE_NAME} is on with finger inserted.")
        oximeter.running = False
        return

    try:
        async with BleakClient(address) as client:
            print(f"Connected to {address}")
            oximeter.connected = True

            await client.start_notify(INDICATE_UUID, lambda s, d: None)
            await client.start_notify(NOTIFY_UUID, oximeter.handle_notification)
            print("Receiving data...\n")

            while oximeter.running and client.is_connected:
                await asyncio.sleep(0.5)

    except Exception as e:
        print(f"BLE Error: {e}")
    finally:
        oximeter.connected = False
        print("BLE disconnected.")


def run_ble_thread(oximeter):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ble_task(oximeter))


def format_elapsed(seconds):
    """Format elapsed seconds as H:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h}:{m:02d}:{s:02d}'


def main(csv_prefix=None, plot_seconds=PLOT_SECONDS):
    oximeter = InnovoPulseOx(plot_seconds=plot_seconds, csv_prefix=csv_prefix)
    oximeter.open_csv()
    if csv_prefix:
        print(f"Output: {csv_prefix}_pleth.csv")
        print(f"        {csv_prefix}_summary.csv")
    else:
        print("CSV output disabled.")

    ble_thread = threading.Thread(target=run_ble_thread, args=(oximeter,), daemon=True)
    ble_thread.start()

    # Set up matplotlib plot
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_ylabel('Pleth', color='#00ff00')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_color('gray')

    line, = ax.plot([], [], color='#00ff00', linewidth=1.5)

    # Top row: vitals
    txt_spo2 = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                       color='cyan', fontfamily='monospace', verticalalignment='top')
    txt_hr = ax.text(0.18, 0.95, '', transform=ax.transAxes, fontsize=14,
                     color='#ff4444', fontfamily='monospace', verticalalignment='top')
    txt_rr = ax.text(0.34, 0.95, '', transform=ax.transAxes, fontsize=14,
                     color='yellow', fontfamily='monospace', verticalalignment='top')
    txt_pi = ax.text(0.48, 0.95, '', transform=ax.transAxes, fontsize=14,
                     color='#ff88ff', fontfamily='monospace', verticalalignment='top')

    # Top right: clock and elapsed
    txt_clock = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=11,
                        color='#aaaaaa', fontfamily='monospace',
                        verticalalignment='top', horizontalalignment='right')
    txt_elapsed = ax.text(0.98, 0.87, '', transform=ax.transAxes, fontsize=11,
                          color='#888888', fontfamily='monospace',
                          verticalalignment='top', horizontalalignment='right')

    def update(frame):
        # Sweep display: just read the buffer maintained by the handler
        buf = oximeter.sweep_buf
        buf_len = oximeter.sweep_len
        if buf_len > 0:
            x_vals = [i / 24.0 for i in range(buf_len)]
            line.set_data(x_vals, buf)
            ax.set_xlim(0, plot_seconds)

            # Auto-scale Y from valid (non-NaN) values
            valid = [v for v in buf if v == v]
            if len(valid) > 1:
                vmin = min(valid)
                vmax = max(valid)
                vrange = vmax - vmin
                margin_bot = max(vrange * 0.4, 5)
                margin_top = max(vrange * 0.8, 10)
                ax.set_ylim(vmin - margin_bot, vmax + margin_top)

        txt_spo2.set_text(f'SpO2:{oximeter.spo2}%')
        txt_hr.set_text(f'HR:{oximeter.heart_rate}')
        txt_rr.set_text(f'RR:{oximeter.resp_rate}')
        txt_pi.set_text(f'PI:{oximeter.pi:.1f}%')

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt_clock.set_text(now_str)

        if oximeter.start_time:
            elapsed = time.time() - oximeter.start_time
            txt_elapsed.set_text(f'Rec: {format_elapsed(elapsed)}')
        else:
            txt_elapsed.set_text('Connecting...')

        return line, txt_spo2, txt_hr, txt_rr, txt_pi, txt_clock, txt_elapsed

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        oximeter.running = False
        ble_thread.join(timeout=3.0)
        oximeter.close_csv()
        elapsed = time.time() - oximeter.start_time if oximeter.start_time else 0
        print(f"\nCaptured {oximeter.pleth_count} pleth samples, "
              f"{oximeter.summary_count} summary packets over {elapsed:.1f}s")
        if oximeter.pleth_count > 0 and elapsed > 0:
            print(f"Pleth rate: {oximeter.pleth_count / elapsed:.1f} Hz")
        if csv_prefix:
            print(f"Data saved to {csv_prefix}_pleth.csv and {csv_prefix}_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Innovo IP900BPB Pulse Oximeter - Live Plot')
    parser.add_argument('--csv', type=str, metavar='PREFIX', default=None,
                        help='CSV output prefix (creates PREFIX_pleth.csv and PREFIX_summary.csv). '
                             'Default: YYYYMMDD_HHMMSS in script directory.')
    parser.add_argument('--no-csv', action='store_true',
                        help='Disable CSV file output')
    parser.add_argument('--seconds', type=int, default=PLOT_SECONDS,
                        help=f'Seconds of waveform visible (default {PLOT_SECONDS})')
    args = parser.parse_args()

    if args.no_csv:
        csv_prefix = None
    elif args.csv:
        csv_prefix = args.csv
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_prefix = os.path.join(script_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))

    main(csv_prefix=csv_prefix, plot_seconds=args.seconds)
