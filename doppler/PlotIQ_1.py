import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque
from datetime import datetime
import os
import time

# --- Configuration ---
PORT = '/dev/ttyACM0'
BAUD = 115200
SAMPLE_RATE = 200          # approximate samples per second
HISTORY = 1000             # number of points to display (5 seconds @ 200 sps)
PKPK_WINDOW = 200          # samples over which to compute pk-pk readout
CLIP_HIGH = 1.570          # V — warn if any sample in window exceeds this
CLIP_LOW  = 0.005          # V — warn if any sample in window falls below this
REC_DELAY = 10.0             # seconds of countdown for delayed start
MAX_RECORD_SEC = 1800        # auto-stop recording after this many seconds
TRAIL_COLOR = 'dodgerblue'
BG_COLOR = '#0a0a0a'
AXIS_MAX = 1.7             # slightly above 1.6V max
OUTPUT_DIR = '/home/jbeale/Documents/snore'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Recording state ---
recording = False
log_file = None
rec_start_time = None      # monotonic time when current recording began
rec_sample_count = 0
countdown_until = None     # monotonic time when delayed recording should start

# --- Live display buffers (always running) ---
xs = deque(maxlen=HISTORY)
ys = deque(maxlen=HISTORY)

# --- Serial setup ---
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
ax.set_xlabel('I (V)', color='white')
ax.set_ylabel('Q (V)', color='white')
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
    global recording, log_file, rec_start_time, rec_sample_count
    start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'IQ_{start_ts}.csv')
    log_file = open(output_path, 'w', buffering=1)
    log_file.write('elapsed_ms,I,Q\n')
    rec_start_time = time.monotonic()
    rec_sample_count = 0
    recording = True
    print(f"Recording started: {output_path}")

def on_left(_event):
    global countdown_until
    if recording or countdown_until is not None:
        # Act as Stop
        _do_stop()
    else:
        countdown_until = None
        _begin_recording()
        _set_active()

def on_right(_event):
    global countdown_until
    if recording or countdown_until is not None:
        return   # disabled — stop is on the left button
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
        print(f"Recording stopped. {rec_sample_count} samples, {elapsed:.1f} s")
    else:
        print("Countdown cancelled.")
    _set_idle()

btn_left.on_clicked(on_left)
btn_right.on_clicked(on_right)

def read_serial_line():
    """Read one line from serial, return (i, q) or None."""
    try:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        parts = line.split(',')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, serial.SerialException):
        pass
    return None

def update(_frame):
    global t_run_origin, rec_sample_count, countdown_until
    waiting = ser.in_waiting
    lines_to_read = max(1, waiting // 20)

    # Auto-stop when MAX_RECORD_SEC is reached
    if recording and rec_start_time is not None:
        if time.monotonic() - rec_start_time >= MAX_RECORD_SEC:
            print(f"Auto-stop: MAX_RECORD_SEC ({MAX_RECORD_SEC} s) reached.")
            _do_stop()

    # Fire delayed recording if countdown has elapsed
    if countdown_until is not None and time.monotonic() >= countdown_until:
        countdown_until = None
        _begin_recording()

    for _ in range(lines_to_read):
        sample = read_serial_line()
        if sample:
            i, q = sample
            now = time.monotonic()
            if t_run_origin is None:
                t_run_origin = now
            if recording and log_file is not None:
                elapsed_ms = (now - rec_start_time) * 1000.0
                log_file.write(f'{elapsed_ms:.1f},{i},{q}\n')
                rec_sample_count += 1
            xs.append(i)
            ys.append(q)

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

    if recording and rec_start_time is not None:
        elapsed_s = time.monotonic() - rec_start_time
        h = int(elapsed_s // 3600)
        m = int((elapsed_s % 3600) // 60)
        s = int(elapsed_s % 60)
        rec_str = f'REC  {h:02d}:{m:02d}:{s:02d}  [{rec_sample_count} samples]'
    elif countdown_until is not None:
        remaining = countdown_until - time.monotonic()
        rec_str = f'starting in {remaining:.1f} s ...'
    else:
        rec_str = 'not recording'

    clip_str  = '  *** CLIP I ***' if clip_i else ''
    clip_str += '  *** CLIP Q ***' if clip_q else ''

    readout = (
        f'{rec_str}\n'
        f'I pk-pk  {i_pkpk:.4f} V  [{i_min:.4f} \u2013 {i_max:.4f}]\n'
        f'Q pk-pk  {q_pkpk:.4f} V  [{q_min:.4f} \u2013 {q_max:.4f}]'
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
    ser.close()
    if log_file is not None:
        log_file.close()
        print(f"Recording closed on exit. {rec_sample_count} samples.")
