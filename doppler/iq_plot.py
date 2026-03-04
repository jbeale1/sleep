import argparse
import os
import sys
import time
from datetime import datetime

# --- Configuration ---
PORT = '/dev/ttyACM0'
#PORT = '/dev/ttyACM2'
BAUD = 115200
SAMPLE_RATE = 142.34       # approximate samples per second
HISTORY = 1000             # number of points to display
PKPK_WINDOW = 200          # samples over which to compute pk-pk readout
COUNTS_PER_VOLT = 8388608 / 5.0   # ADS1256 PGA=1, Vref=2.5V, single-ended
CLIP_HIGH = int(1.570 * COUNTS_PER_VOLT)  # 2,634,024 counts
CLIP_LOW  = int(0.005 * COUNTS_PER_VOLT)  #     8,389 counts
AXIS_MAX  = int(1.7   * COUNTS_PER_VOLT)  # 2,852,127 counts
REC_DELAY = 10.0             # seconds of countdown for delayed start
AUTO_REC_DELAY = 2.0    # seconds of countdown when --autostart is used
MAX_RECORD_SEC = 180000        # auto-stop recording after this many seconds
TRAIL_COLOR = 'dodgerblue'
BG_COLOR = '#0a0a0a'
OUTPUT_DIR = '/media/john/B06D-61C8/source/doppler'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--autostart', action='store_true',
                    help=f'Auto-start recording after {AUTO_REC_DELAY:.0f}s countdown (text-only, no GUI)')
parser.add_argument('--csv', '-c', help='Path to CSV file to playback instead of reading serial')
args = parser.parse_args()

if args.autostart and args.csv:
    print('Error: --autostart is for recording from serial; --csv is for playback.')
    sys.exit(1)

# --- Text-only mode: runs entirely before any GUI imports ---
if args.autostart:
    import serial
    ser = serial.Serial(PORT, BAUD, timeout=1)
    start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'IQ_{start_ts}.csv')

    print(f"--autostart: recording in {AUTO_REC_DELAY:.0f} s")
    print(f"Output: {output_path}")
    time.sleep(AUTO_REC_DELAY)
    ser.flushInput()  # discard any old data before starting

    print('seq,ms,I,Q')
    log_file = open(output_path, 'w', buffering=1)
    log_file.write('seq,ms,I,Q\n')
    print(f"Recording started: {output_path}")

    rec_start = time.monotonic()
    stdout_end = rec_start + REC_DELAY
    sample_count = 0
    last_seq = None
    total_dropped = 0

    # Print 20 raw lines from the serial port to STDOUT (no interpretation),
    # for debugging/inspection. Each line is written as-received (bytes).
    for _ in range(20):
        try:
            raw = ser.readline()
            if not raw:
                break
            try:
                # Write bytes directly to stdout buffer to avoid any decoding.
                sys.stdout.buffer.write(raw)
            except AttributeError:
                # Fallback: decode as ASCII if stdout.buffer not available.
                sys.stdout.write(raw.decode('ascii', errors='ignore'))
            sys.stdout.flush()
        except serial.SerialException:
            break

    # ser.flushInput()  # discard any old data before starting
    
    try:
        while True:
            if time.monotonic() - rec_start >= MAX_RECORD_SEC:
                print(f"Auto-stop: MAX_RECORD_SEC ({MAX_RECORD_SEC} s) reached.")
                break
            line = ser.readline().decode('ascii', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) != 4:
                continue
            try:
                seq, ms, i, q = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue

            #    print(f'{seq},{ms},{i},{q}')

            if last_seq is not None:
                gap = seq - last_seq - 1
                if gap < 0:
                    gap = (seq + 0x100000000 - last_seq - 1)
                if 0 < gap < 100000:
                    total_dropped += gap
                    print(f"WARNING: {gap} samples dropped at seq {seq}", file=sys.stderr)
            last_seq = seq

            log_file.write(f'{seq},{ms},{i},{q}\n')
            sample_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()
        elapsed = time.monotonic() - rec_start
        drop_str = f", {total_dropped} dropped" if total_dropped else ""
        print(f"Recording stopped. {sample_count} samples, {elapsed:.1f} s{drop_str}")
        ser.close()
    sys.exit(0)


# --- GUI mode only: these imports are never reached with --autostart ---
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque
import csv

# --- Recording state ---
recording = False
log_file = None
rec_start_time = None      # monotonic time when current recording began
rec_sample_count = 0
countdown_until = None     # monotonic time when delayed recording should start
last_seq = None            # track sequence number for gap detection
total_dropped = 0
stdout_until = None        # monotonic time until which raw data is echoed to stdout

# --- Sample rate measurement (from RP2040 seq + ms counters) ---
rate_seq_start = None       # seq number at start of current 1s window
rate_ms_start = None        # unwrapped ms at start of current 1s window
rate_last_update = None     # host monotonic time of last display update
measured_rate = 0.0         # displayed Hz value
unwrapped_ms = 0            # accumulated ms tracking rollovers
prev_ms_val = None          # previous raw ms%1000 value

# --- Live display buffers (always running) ---
xs = deque(maxlen=HISTORY)
ys = deque(maxlen=HISTORY)

# --- Serial / CSV setup ---
csv_mode = bool(args.csv)
csv_samples = []
csv_index = 0
playback_origin_monotonic = None
csv_base_ms = 0

if csv_mode:
    # Preload CSV and compute unwrapped absolute ms timestamps
    try:
        with open(args.csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            prev_ms_val = None
            unwrapped = 0
            for r in reader:
                try:
                    seq = int(r.get('seq') or r.get('seq\n'))
                    ms = int(r.get('ms'))
                    i = int(r.get('I') or r.get('i'))
                    q = int(r.get('Q') or r.get('q'))
                except Exception:
                    continue
                if prev_ms_val is not None and ms < prev_ms_val - 500:
                    unwrapped += 1000
                prev_ms_val = ms
                abs_ms = unwrapped + ms
                csv_samples.append((seq, abs_ms, i, q))
        if csv_samples:
            csv_base_ms = csv_samples[0][1]
    except FileNotFoundError:
        print(f'CSV file not found: {args.csv}')
        sys.exit(1)
else:
    ser = serial.Serial(PORT, BAUD, timeout=1)

# --- Wall-clock reference (for elapsed display even when not recording) ---
t_run_origin = None

# ---- Layout: main axes + two always-visible button axes ----
fig = plt.figure(figsize=(7, 7.8), facecolor=BG_COLOR)
ax     = fig.add_axes([0.10, 0.18, 0.85, 0.78])   # IQ plot
ax_btn_left  = fig.add_axes([0.08, 0.04, 0.38, 0.08])  # left button
ax_btn_right = fig.add_axes([0.54, 0.04, 0.38, 0.08])  # right button

ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, AXIS_MAX)
ax.set_ylim(0, AXIS_MAX)
ax.set_xlabel('I (counts)', color='white')
ax.set_ylabel('Q (counts)', color='white')
ax.set_title('I/Q Real-Time Plot', color='white')
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444444')
ax.grid(True, color='#222222', linestyle='--', linewidth=0.5)
ax.set_aspect('equal')

trail_scatter = ax.scatter([], [], s=3, c=TRAIL_COLOR, alpha=0.3, linewidths=0)
head_scatter  = ax.scatter([], [], s=30, c='white', zorder=5, linewidths=0)

status_text = ax.text(
    0.02, 0.98, '', transform=ax.transAxes,
    color='white', fontsize=10, fontfamily='monospace',
    verticalalignment='top', zorder=10,
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a1a', edgecolor='#444444', alpha=0.85)
)

# --- Buttons (always visible; relabelled/recoloured to reflect state) ---
btn_left  = Button(ax_btn_left,  'Start Now',           color='#1a3a1a', hovercolor='#2a5a2a')
btn_right = Button(ax_btn_right, f'Start +{REC_DELAY:.0f}s', color='#1a2a3a', hovercolor='#2a4a5a')
for b in (btn_left, btn_right):
    b.label.set_color('white')
    b.label.set_fontsize(11)

STYLE_IDLE_LEFT  = ('#1a3a1a', '#2a5a2a')
STYLE_IDLE_RIGHT = ('#1a2a3a', '#2a4a5a')
STYLE_STOP       = ('#3a1a1a', '#5a2a2a')
STYLE_DISABLED   = ('#1a1a1a', '#1a1a1a')

def _set_idle():
    btn_left.label.set_text('Start Now')
    btn_right.label.set_text(f'Start +{REC_DELAY:.0f}s')
    ax_btn_left.set_facecolor(STYLE_IDLE_LEFT[0]);   btn_left.hovercolor  = STYLE_IDLE_LEFT[1]
    ax_btn_right.set_facecolor(STYLE_IDLE_RIGHT[0]); btn_right.hovercolor = STYLE_IDLE_RIGHT[1]
    fig.canvas.draw_idle()

def _set_active():
    btn_left.label.set_text('Stop Recording')
    btn_right.label.set_text('')
    ax_btn_left.set_facecolor(STYLE_STOP[0]);        btn_left.hovercolor  = STYLE_STOP[1]
    ax_btn_right.set_facecolor(STYLE_DISABLED[0]);   btn_right.hovercolor = STYLE_DISABLED[1]
    fig.canvas.draw_idle()

def _begin_recording():
    global recording, log_file, rec_start_time, rec_sample_count, last_seq, total_dropped, stdout_until
    start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'IQ_{start_ts}.csv')
    log_file = open(output_path, 'w', buffering=1)
    log_file.write('seq,ms,I,Q\n')
    rec_start_time = time.monotonic()
    rec_sample_count = 0
    last_seq = None
    total_dropped = 0
    recording = True
    if args.autostart:
        stdout_until = rec_start_time + REC_DELAY
    print(f"Recording started: {output_path}")

def on_left(_event):
    global countdown_until
    if recording or countdown_until is not None:
        _do_stop()
    else:
        countdown_until = None
        _begin_recording()
        _set_active()

def on_right(_event):
    global countdown_until
    if recording or countdown_until is not None:
        return
    countdown_until = time.monotonic() + REC_DELAY
    _set_active()
    print(f"Countdown started — recording in {REC_DELAY:.0f} s")

def _do_stop():
    global recording, log_file, countdown_until
    countdown_until = None
    if recording:
        log_file.close()
        log_file = None
        elapsed = time.monotonic() - rec_start_time
        recording = False
        drop_str = f", {total_dropped} dropped" if total_dropped else ""
        print(f"Recording stopped. {rec_sample_count} samples, {elapsed:.1f} s{drop_str}")
    else:
        print("Countdown cancelled.")
    _set_idle()

btn_left.on_clicked(on_left)
btn_right.on_clicked(on_right)

def read_serial_line():
    """Read one line from serial, return (seq, ms, i, q) or None."""
    try:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) == 4:
            return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except (ValueError, serial.SerialException):
        pass
    return None


def _handle_sample(seq_num, ms_val, i_val, q_val, now, source_csv=False):
    """Common handling for a single sample (serial or CSV).
    If source_csv is True, ms_val is treated as absolute unwrapped ms."""
    global t_run_origin, rec_sample_count, last_seq, total_dropped
    global rate_seq_start, rate_ms_start, rate_last_update, measured_rate
    global unwrapped_ms, prev_ms_val

    if t_run_origin is None:
        t_run_origin = now

    # Sequence gap detection
    if last_seq is not None:
        gap = seq_num - last_seq - 1
        if gap < 0:
            gap = (seq_num + 0x100000000 - last_seq - 1)  # uint32 wrap
        if gap > 0 and gap < 100000:
            total_dropped += gap
            print(f"WARNING: {gap} samples dropped at seq {seq_num}")
    last_seq = seq_num

    if recording and log_file is not None:
        log_file.write(f'{seq_num},{ms_val},{i_val},{q_val}\n')
        rec_sample_count += 1
        if stdout_until is not None:
            if time.monotonic() <= stdout_until:
                print(f'{seq_num},{ms_val},{i_val},{q_val}')
            else:
                # stop echoing
                pass

    xs.append(i_val)
    ys.append(q_val)

    # Determine current milliseconds value for rate calculation
    if source_csv:
        cur_ms = ms_val
    else:
        if prev_ms_val is not None and ms_val < prev_ms_val - 500:
            unwrapped_ms += 1000  # rollover
        prev_ms_val = ms_val
        cur_ms = unwrapped_ms + ms_val

    # Update measured rate ~once per second (using RP2040 timebase)
    if rate_seq_start is None:
        rate_seq_start = seq_num
        rate_ms_start = cur_ms
        rate_last_update = now
    elif now - rate_last_update >= 1.0:
        d_seq = seq_num - rate_seq_start
        d_ms = cur_ms - rate_ms_start
        if d_ms > 0 and d_seq > 0:
            measured_rate = 1000.0 * d_seq / d_ms
        rate_seq_start = seq_num
        rate_ms_start = cur_ms
        rate_last_update = now

def update(_frame):
    global t_run_origin, rec_sample_count, countdown_until, last_seq, total_dropped
    global rate_seq_start, rate_ms_start, rate_last_update, measured_rate
    global unwrapped_ms, prev_ms_val
    global csv_mode, csv_index, playback_origin_monotonic, csv_samples, csv_base_ms
    now = time.monotonic()

    if csv_mode:
        # Playback from preloaded CSV at recorded real-time rate
        if playback_origin_monotonic is None and csv_index < len(csv_samples):
            # start playback aligned to now
            globals()['playback_origin_monotonic'] = now
        # emit any samples whose recorded time has arrived
        while csv_index < len(csv_samples):
            rel_ms = csv_samples[csv_index][1] - csv_base_ms
            if now - playback_origin_monotonic >= rel_ms / 1000.0:
                seq_num, ms_val, i_val, q_val = csv_samples[csv_index]
                globals()['csv_index'] = csv_index + 1
                _handle_sample(seq_num, ms_val, i_val, q_val, now, source_csv=True)
            else:
                break
        lines_to_read = 0
    else:
        waiting = ser.in_waiting
        lines_to_read = max(1, waiting // 20)

    if recording and rec_start_time is not None:
        if time.monotonic() - rec_start_time >= MAX_RECORD_SEC:
            print(f"Auto-stop: MAX_RECORD_SEC ({MAX_RECORD_SEC} s) reached.")
            _do_stop()

    if countdown_until is not None and time.monotonic() >= countdown_until:
        countdown_until = None
        _begin_recording()

    if not csv_mode:
        for _ in range(lines_to_read):
            sample = read_serial_line()
            if sample:
                seq_num, ms_val, i_val, q_val = sample
                _handle_sample(seq_num, ms_val, i_val, q_val, time.monotonic(), source_csv=False)

    if len(xs) < 2:
        return trail_scatter, head_scatter, status_text

    trail_scatter.set_offsets(list(zip(xs, ys)))
    head_scatter.set_offsets([[xs[-1], ys[-1]]])

    # --- Readout over last PKPK_WINDOW samples ---
    win_i = list(xs)[-PKPK_WINDOW:]
    win_q = list(ys)[-PKPK_WINDOW:]
    i_min, i_max = min(win_i), max(win_i)
    q_min, q_max = min(win_q), max(win_q)
    i_pkpk = i_max - i_min
    q_pkpk = q_max - q_min

    clip_i = i_max >= CLIP_HIGH or i_min <= CLIP_LOW
    clip_q = q_max >= CLIP_HIGH or q_min <= CLIP_LOW

    # Convert to volts for display readout
    def to_v(counts):
        return counts / COUNTS_PER_VOLT

    if recording and rec_start_time is not None:
        elapsed_s = time.monotonic() - rec_start_time
        h = int(elapsed_s // 3600)
        m = int((elapsed_s % 3600) // 60)
        s = int(elapsed_s % 60)
        rec_str = f'REC  {h:02d}:{m:02d}:{s:02d}  [{rec_sample_count}]'
        if total_dropped:
            rec_str += f'  \u26a0 {total_dropped} dropped'
    elif countdown_until is not None:
        remaining = countdown_until - time.monotonic()
        rec_str = f'starting in {remaining:.1f} s ...'
    else:
        rec_str = 'not recording'

    clip_str  = '  *** CLIP I ***' if clip_i else ''
    clip_str += '  *** CLIP Q ***' if clip_q else ''

    readout = (
        f'{rec_str}\n'
        f'I pk-pk  {to_v(i_pkpk):.4f} V  [{to_v(i_min):.4f} \u2013 {to_v(i_max):.4f}]\n'
        f'Q pk-pk  {to_v(q_pkpk):.4f} V  [{to_v(q_min):.4f} \u2013 {to_v(q_max):.4f}]\n'
        f'{measured_rate:.1f} Hz'
    )
    if clip_str:
        readout += f'\n{clip_str.strip()}'

    status_text.set_text(readout)
    status_text.set_color('red' if (clip_i or clip_q) else
                          'tomato' if recording else
                          'yellow' if countdown_until is not None else 'white')

    return trail_scatter, head_scatter, status_text

ani = animation.FuncAnimation(
    fig, update,
    interval=30,
    blit=True,
    cache_frame_data=False
)



try:
    plt.show()
finally:
    if not csv_mode:
        ser.close()
    if log_file is not None:
        log_file.close()
        drop_str = f", {total_dropped} dropped" if total_dropped else ""
        print(f"Recording closed on exit. {rec_sample_count} samples{drop_str}.")
