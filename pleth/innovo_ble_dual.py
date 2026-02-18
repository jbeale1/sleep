#!/usr/bin/env python3
"""
Read data from TWO Innovo IP900BPB pulse oximeters via BLE simultaneously.
Stacked pleth waveform plots + summary display (SpO2, HR, RR, PI) for each unit.
Saves pleth and summary data to CSV files with columns for both units.

Requires: pip install bleak matplotlib
Usage: python innovo_ble_dual.py [--csv prefix] [--seconds 10]
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
DEVICE_ADDRS = {"ED:9B:47:2E:0F:68", "D3:67:B7:93:27:00"}

# Plot config
PLOT_SECONDS = 10

UNIT_COLORS = {
    'Unit1': {'pleth': '#00ff00', 'spo2': 'cyan', 'hr': '#ff4444', 'rr': 'yellow', 'pi': '#ff88ff'},
    'Unit2': {'pleth': '#00ccff', 'spo2': '#66ffcc', 'hr': '#ff8844', 'rr': '#ffcc00', 'pi': '#cc88ff'},
}


class OximeterUnit:
    """State for a single pulse oximeter."""

    def __init__(self, label, plot_seconds=10):
        self.label = label
        self.address = None

        # Sweep buffer for display
        self.sweep_len = int(plot_seconds * 24)
        self.sweep_buf = [float('nan')] * self.sweep_len
        self.sweep_gap = max(6, self.sweep_len // 20)

        # Latest summary values
        self.spo2 = 0
        self.heart_rate = 0
        self.resp_rate = 0
        self.pi = 0.0

        self.pleth_count = 0
        self.sweep_offset = 0
        self.summary_count = 0
        self.connected = False

    def handle_notification(self, sender, data: bytearray, writer_callback, check_sync):
        if len(data) == 2 and data[0] == 0x01:
            pleth = data[1]
            self.pleth_count += 1
            check_sync()

            pos = (self.pleth_count - self.sweep_offset - 1) % self.sweep_len
            self.sweep_buf[pos] = pleth
            for g in range(1, self.sweep_gap + 1):
                self.sweep_buf[(pos + g) % self.sweep_len] = float('nan')

            writer_callback('pleth', self.label, pleth)

        elif len(data) >= 12 and data[0] == 0x3E:
            self.spo2 = data[1]
            self.heart_rate = data[3]
            self.resp_rate = data[5]
            self.pi = data[11] / 10.0
            self.summary_count += 1

            writer_callback('summary', self.label, {
                'spo2': self.spo2, 'hr': self.heart_rate,
                'rr': self.resp_rate, 'pi': self.pi
            })


class DualPulseOx:
    def __init__(self, plot_seconds=10, csv_prefix=None):
        self.csv_prefix = csv_prefix
        self.pleth_file = None
        self.pleth_writer = None
        self.summary_file = None
        self.summary_writer = None
        self.start_time = None
        self.plot_seconds = plot_seconds

        self.unit1 = OximeterUnit('Unit1', plot_seconds)
        self.unit2 = OximeterUnit('Unit2', plot_seconds)
        self.units = [self.unit1, self.unit2]

        self.running = True
        self.both_connected = False
        self._synced = False

        # CSV write lock
        self._csv_lock = threading.Lock()

        # Last pleth values for interleaved CSV
        self._last_pleth = {'Unit1': '', 'Unit2': ''}
        self._last_summary = {
            'Unit1': {'spo2': '', 'hr': '', 'rr': '', 'pi': ''},
            'Unit2': {'spo2': '', 'hr': '', 'rr': '', 'pi': ''},
        }

    def open_csv(self):
        if self.csv_prefix:
            self.pleth_file = open(f'{self.csv_prefix}_pleth.csv', 'w', newline='')
            self.pleth_writer = csv.writer(self.pleth_file)
            self.pleth_writer.writerow([
                'timestamp', 'elapsed_s', 'unit',
                'pleth_unit1', 'pleth_unit2'
            ])

            self.summary_file = open(f'{self.csv_prefix}_summary.csv', 'w', newline='')
            self.summary_writer = csv.writer(self.summary_file)
            self.summary_writer.writerow([
                'timestamp', 'elapsed_s', 'unit',
                'spo2_unit1', 'hr_unit1', 'rr_unit1', 'pi_unit1',
                'spo2_unit2', 'hr_unit2', 'rr_unit2', 'pi_unit2'
            ])

    def close_csv(self):
        if self.pleth_file:
            self.pleth_file.close()
        if self.summary_file:
            self.summary_file.close()

    def check_sync(self):
        """Once both units are streaming, reset sweep buffers so they start together."""
        if self._synced:
            return
        if self.unit1.pleth_count > 0 and self.unit2.pleth_count > 0:
            self._synced = True
            for u in self.units:
                u.sweep_buf[:] = [float('nan')] * u.sweep_len
                u.sweep_offset = u.pleth_count  # counter value at sync point

    def csv_callback(self, data_type, label, value):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        with self._csv_lock:
            if data_type == 'pleth' and self.pleth_writer:
                self._last_pleth[label] = value
                self.pleth_writer.writerow([
                    timestamp, f'{elapsed:.3f}', label,
                    self._last_pleth['Unit1'], self._last_pleth['Unit2']
                ])

            elif data_type == 'summary' and self.summary_writer:
                self._last_summary[label] = value
                s1 = self._last_summary['Unit1']
                s2 = self._last_summary['Unit2']
                self.summary_writer.writerow([
                    timestamp, f'{elapsed:.3f}', label,
                    s1.get('spo2', ''), s1.get('hr', ''), s1.get('rr', ''), s1.get('pi', ''),
                    s2.get('spo2', ''), s2.get('hr', ''), s2.get('rr', ''), s2.get('pi', ''),
                ])
                self.summary_file.flush()


async def connect_unit(unit, client_address, dual):
    """Connect to one oximeter and receive data on its own event loop."""
    try:
        async with BleakClient(client_address) as client:
            print(f"  {unit.label} connected to {client_address}")
            unit.connected = True
            unit.address = client_address

            await client.start_notify(INDICATE_UUID, lambda s, d: None)
            await client.start_notify(
                NOTIFY_UUID,
                lambda s, d: unit.handle_notification(s, d, dual.csv_callback, dual.check_sync)
            )

            while dual.running and client.is_connected:
                await asyncio.sleep(0.5)

    except Exception as e:
        print(f"  {unit.label} BLE Error: {e}")
    finally:
        unit.connected = False
        print(f"  {unit.label} disconnected.")


def run_unit_thread(unit, address, dual):
    """Run a single unit connection on its own event loop/thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(connect_unit(unit, address, dual))


def scan_and_start(dual):
    """Scan for devices, then launch a separate thread per unit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    devices = loop.run_until_complete(BleakScanner.discover(timeout=15.0))

    found = []
    for d in devices:
        if d.name and DEVICE_NAME in d.name:
            found.append(d.address)
        elif d.address and d.address.upper() in {a.upper() for a in DEVICE_ADDRS}:
            found.append(d.address)

    found = list(dict.fromkeys(found))

    if len(found) < 2:
        print(f"Found only {len(found)} device(s). Need 2 pulse oximeters powered on with fingers inserted.")
        if found:
            print(f"  Found: {found[0]}")
        dual.running = False
        return []

    print(f"Found 2 devices: {found[0]}, {found[1]}")
    print("Connecting to both...")

    addrs = sorted(found[:2])
    dual.unit1.address = addrs[0]
    dual.unit2.address = addrs[1]
    dual.both_connected = True

    # Launch each unit on its own thread with its own event loop
    t1 = threading.Thread(target=run_unit_thread, args=(dual.unit1, addrs[0], dual), daemon=True)
    t2 = threading.Thread(target=run_unit_thread, args=(dual.unit2, addrs[1], dual), daemon=True)
    t1.start()
    t2.start()
    return [t1, t2]


def format_elapsed(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h}:{m:02d}:{s:02d}'


def main(csv_prefix=None, plot_seconds=PLOT_SECONDS):
    dual = DualPulseOx(plot_seconds=plot_seconds, csv_prefix=csv_prefix)
    dual.open_csv()
    if csv_prefix:
        print(f"Output: {csv_prefix}_pleth.csv")
        print(f"        {csv_prefix}_summary.csv")
    else:
        print("CSV output disabled.")

    ble_threads = []

    def start_ble():
        nonlocal ble_threads
        ble_threads = scan_and_start(dual)

    scan_thread = threading.Thread(target=start_ble, daemon=True)
    scan_thread.start()

    # Set up matplotlib with two stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor('black')

    axes = [ax1, ax2]
    units = [dual.unit1, dual.unit2]
    labels = ['Unit1', 'Unit2']
    lines = []
    txt_objs = []

    for i, (ax, unit, label) in enumerate(zip(axes, units, labels)):
        colors = UNIT_COLORS[label]
        ax.set_facecolor('black')
        ax.set_ylabel(f'{label} Pleth', color=colors['pleth'], fontsize=10)
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('gray')

        ln, = ax.plot([], [], color=colors['pleth'], linewidth=1.5)
        lines.append(ln)

        # Vitals text
        t_spo2 = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=12,
                         color=colors['spo2'], fontfamily='monospace', verticalalignment='top')
        t_hr = ax.text(0.18, 0.92, '', transform=ax.transAxes, fontsize=12,
                       color=colors['hr'], fontfamily='monospace', verticalalignment='top')
        t_rr = ax.text(0.34, 0.92, '', transform=ax.transAxes, fontsize=12,
                       color=colors['rr'], fontfamily='monospace', verticalalignment='top')
        t_pi = ax.text(0.48, 0.92, '', transform=ax.transAxes, fontsize=12,
                       color=colors['pi'], fontfamily='monospace', verticalalignment='top')
        t_addr = ax.text(0.65, 0.92, '', transform=ax.transAxes, fontsize=9,
                         color='#666666', fontfamily='monospace', verticalalignment='top')

        txt_objs.append({'spo2': t_spo2, 'hr': t_hr, 'rr': t_rr, 'pi': t_pi, 'addr': t_addr})

    # Clock and elapsed on top subplot only
    txt_clock = ax1.text(0.98, 0.92, '', transform=ax1.transAxes, fontsize=10,
                         color='#aaaaaa', fontfamily='monospace',
                         verticalalignment='top', horizontalalignment='right')
    txt_elapsed = ax1.text(0.98, 0.78, '', transform=ax1.transAxes, fontsize=10,
                           color='#888888', fontfamily='monospace',
                           verticalalignment='top', horizontalalignment='right')

    ax2.set_xlabel('Time (s)', color='gray', fontsize=10)

    def update(frame):
        artists = []
        for i, (ax, unit, ln, txts) in enumerate(zip(axes, units, lines, txt_objs)):
            buf = unit.sweep_buf
            buf_len = unit.sweep_len
            if buf_len > 0:
                x_vals = [j / 24.0 for j in range(buf_len)]
                ln.set_data(x_vals, buf)
                ax.set_xlim(0, plot_seconds)

                valid = [v for v in buf if v == v]
                if len(valid) > 1:
                    vmin, vmax = min(valid), max(valid)
                    vrange = vmax - vmin
                    margin_bot = max(vrange * 0.4, 5)
                    margin_top = max(vrange * 0.8, 10)
                    ax.set_ylim(vmin - margin_bot, vmax + margin_top)

            txts['spo2'].set_text(f'SpO2:{unit.spo2}%')
            txts['hr'].set_text(f'HR:{unit.heart_rate}')
            txts['rr'].set_text(f'RR:{unit.resp_rate}')
            txts['pi'].set_text(f'PI:{unit.pi:.1f}%')
            txts['addr'].set_text(unit.address or 'scanning...')

            artists.extend([ln, txts['spo2'], txts['hr'], txts['rr'], txts['pi'], txts['addr']])

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        txt_clock.set_text(now_str)

        if dual.start_time:
            elapsed = time.time() - dual.start_time
            txt_elapsed.set_text(f'Rec: {format_elapsed(elapsed)}')
        else:
            txt_elapsed.set_text('Connecting...')

        artists.extend([txt_clock, txt_elapsed])
        return artists

    ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        dual.running = False
        scan_thread.join(timeout=3.0)
        for t in ble_threads:
            t.join(timeout=3.0)
        dual.close_csv()
        elapsed = time.time() - dual.start_time if dual.start_time else 0
        print(f"\nUnit1: {dual.unit1.pleth_count} pleth, {dual.unit1.summary_count} summary")
        print(f"Unit2: {dual.unit2.pleth_count} pleth, {dual.unit2.summary_count} summary")
        print(f"Duration: {elapsed:.1f}s")
        for u in dual.units:
            if u.pleth_count > 0 and elapsed > 0:
                print(f"  {u.label} pleth rate: {u.pleth_count / elapsed:.1f} Hz")
        if csv_prefix:
            print(f"Data saved to {csv_prefix}_pleth.csv and {csv_prefix}_summary.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual Innovo IP900BPB Pulse Oximeter - Live Plot')
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
