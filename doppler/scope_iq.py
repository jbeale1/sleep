"""
scope_iq.py — Triggered oscilloscope display for pendulum IQ data.

At 61 GHz with ~10 mm swing amplitude, I and Q each oscillate at roughly
4-8x the pendulum frequency (one IQ rotation per lambda/2 = 2.45 mm of travel).
The trigger must therefore operate on the ENVELOPE of the IQ signal, not on
I or Q directly.

Trigger: local MINIMUM of the smoothed IQ amplitude envelope = turnaround point
         (velocity = 0).  This is the most repeatable and noise-robust event.

One sweep = one full pendulum period (~3.1 s), showing raw I and Q waveforms
with multiple overlaid sweeps fading oldest to newest.

Usage:
    python scope_iq.py                   # default port /dev/ttyACM0
    python scope_iq.py --port COM3       # Windows
    python scope_iq.py --sweeps 6 --window 3.2 --f0 0.322

Controls (click the plot window first):
    +  /  -     increase / decrease sweep window (seconds, step 0.2)
    n  /  N     more / fewer overlaid sweeps
    q           quit
"""

import argparse
import collections
import sys
import threading
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as scipy_signal

# ── defaults ──────────────────────────────────────────────────────────────────
PORT          = '/dev/ttyACM0'
BAUD          = 115200
FS_NOM        = 142.34      # nominal sample rate (Hz)
F0_NOM        = 0.322       # nominal pendulum frequency (Hz)
N_SWEEPS      = 6
WINDOW_S      = 3.2         # slightly more than one pendulum period (1/0.322 = 3.1 s)
PRETRIG_S     = 0.15        # seconds before trigger shown
BUF_S         = 45.0        # ring buffer duration (seconds)
ENV_LP_FC     = 1.5         # Hz — low-pass cutoff for |IQ| envelope smoothing
MIN_SEP_FRAC  = 0.65        # min trigger separation as fraction of 1/f0

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--port',    default=PORT,      help='Serial port')
parser.add_argument('--baud',    default=BAUD,      type=int)
parser.add_argument('--sweeps',  default=N_SWEEPS,  type=int)
parser.add_argument('--window',  default=WINDOW_S,  type=float, help='Sweep window (s)')
parser.add_argument('--pretrig', default=PRETRIG_S, type=float)
parser.add_argument('--f0',      default=F0_NOM,    type=float, help='Pendulum freq (Hz)')
args = parser.parse_args()

# ── ring buffers (written by serial thread) ───────────────────────────────────
BUF_N   = int(BUF_S * FS_NOM) + 500
buf_I   = collections.deque(maxlen=BUF_N)
buf_Q   = collections.deque(maxlen=BUF_N)
buf_seq = collections.deque(maxlen=BUF_N)
buf_ms  = collections.deque(maxlen=BUF_N)
lock    = threading.Lock()
paused = False

# ── serial reader thread ──────────────────────────────────────────────────────
def serial_reader():
    import serial
    last_seq = None
    last_paused = False
    while True:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
            print(f"Connected: {args.port} @ {args.baud} baud", flush=True)
            while True:
                # Respect pause: when paused, do not read or append data.
                # On transition from paused->running, clear serial input buffer
                # to discard data we did not display.
                if paused:
                    try:
                        # clear any accumulated input in device/OS buffer
                        if hasattr(ser, 'reset_input_buffer'):
                            ser.reset_input_buffer()
                        elif hasattr(ser, 'flushInput'):
                            ser.flushInput()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    last_paused = True
                    continue
                else:
                    if last_paused:
                        # we just resumed; drop any serial-side buffered bytes
                        try:
                            if hasattr(ser, 'reset_input_buffer'):
                                ser.reset_input_buffer()
                            elif hasattr(ser, 'flushInput'):
                                ser.flushInput()
                        except Exception:
                            pass
                        last_paused = False
                try:
                    raw = ser.readline().decode('ascii', errors='ignore').strip()
                except Exception:
                    break
                parts = raw.split(',')
                if len(parts) != 4:
                    continue
                try:
                    seq, ms, i_val, q_val = (int(p) for p in parts)
                except ValueError:
                    continue
                if last_seq is not None:
                    gap = seq - last_seq - 1
                    if gap < 0:
                        gap = (seq + 0x100000000 - last_seq - 1)
                    if 0 < gap < 100000:
                        print(f"  gap: {gap} at seq {seq}", file=sys.stderr)
                last_seq = seq
                with lock:
                    # Only append when not paused (guard already checked above)
                    buf_I.append(i_val)
                    buf_Q.append(q_val)
                    buf_seq.append(seq)
                    buf_ms.append(ms)
        except Exception as e:
            print(f"Serial error: {e} — retrying in 2 s", file=sys.stderr)
            time.sleep(2)

threading.Thread(target=serial_reader, daemon=True).start()

# ── envelope filter (LP on |IQ|) ──────────────────────────────────────────────
def make_lp(fs, fc):
    return scipy_signal.butter(4, fc / (fs / 2), btype='low', output='sos')

sos_env = make_lp(FS_NOM, ENV_LP_FC)

def compute_envelope(arr_I, arr_Q):
    I = arr_I - arr_I.mean()
    Q = arr_Q - arr_Q.mean()
    amp = np.sqrt(I**2 + Q**2)
    return scipy_signal.sosfiltfilt(sos_env, amp), I, Q

def compute_phase_velocity(I_dc, Q_dc, fs):
    """Return (phi_unwrapped, vphi) where vphi = d(phi)/dt in rad/s.

    Uses magnitude gating + Savitzky-Golay smoothing to keep zero-crossings stable.
    """
    z = I_dc.astype(np.float64) + 1j * Q_dc.astype(np.float64)
    mag = np.abs(z)
    # Gate very low magnitude points (phase becomes noisy near origin)
    floor = np.percentile(mag, 10)
    valid = mag > floor

    phi = np.angle(z)
    idx = np.arange(len(phi))
    if valid.sum() < 3:
        return None, None

    # Fill invalid phase samples by interpolation before unwrap
    phi_filled = np.interp(idx, idx[valid], phi[valid])
    phi_u = np.unwrap(phi_filled)

    # dphi/dt -> rad/s
    vphi = np.gradient(phi_u) * fs

    # Smooth derivative while preserving timing (zero crossings)
    win = int(round(0.35 * fs))
    win = max(9, win | 1)  # odd, >=9
    vphi = scipy_signal.savgol_filter(vphi, win, 3)

    return phi_u, vphi


def find_vphi_zero_crossings(vphi, fs, f0, pretrig_n, window_n, direction='neg_to_pos'):
    """Return fractional-sample trigger positions for one turnaround per period.

    direction:
        'neg_to_pos' triggers on vphi crossing upward through 0.
        'pos_to_neg' triggers on vphi crossing downward through 0.
    """
    min_sep = int(MIN_SEP_FRAC / f0 * fs)  # prevents double-triggers
    triggers = []
    i = pretrig_n
    end = len(vphi) - window_n - 2
    while i < end:
        if direction == 'neg_to_pos':
            crossed = (vphi[i] < 0) and (vphi[i+1] >= 0)
        else:
            crossed = (vphi[i] > 0) and (vphi[i+1] <= 0)

        if crossed:
            denom = (vphi[i+1] - vphi[i])
            frac = 0.0 if denom == 0 else (-vphi[i] / denom)  # linear interp within [0,1)
            triggers.append(i + frac)
            i += min_sep
        else:
            i += 1
    return triggers

# ── plot ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams['toolbar'] = 'None'
fig = plt.figure(figsize=(12, 7.5), facecolor='#080c0c')
fig.canvas.manager.set_window_title('Pendulum IQ Scope — 61 GHz')

gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 2, 1],
                       hspace=0.40, left=0.08, right=0.97, top=0.91, bottom=0.08)
ax_I   = fig.add_subplot(gs[0])
ax_Q   = fig.add_subplot(gs[1])
ax_env = fig.add_subplot(gs[2])

BG      = '#080c0c'
GRID_C  = '#162020'
TRIG_C  = '#ff8f00'
COLORS  = ['#006064','#00838f','#0097a7','#00acc1','#00bcd4',
           '#26c6da','#4dd0e1','#80deea','#b2ebf2','#e0f7fa',
           '#a7ffeb','#64ffda','#1de9b6','#00bfa5','#00897b','#00695c']
MAX_SW  = 16

for ax, lbl in [(ax_I,'I  (counts, DC removed)'),
                (ax_Q,'Q  (counts, DC removed)'),
                (ax_env,'|IQ| envelope')]:
    ax.set_facecolor(BG)
    ax.set_ylabel(lbl, color='#78909c', fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#455a64', labelsize=8)
    for sp in ax.spines.values():
        sp.set_color('#1c2d2d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, color=GRID_C, linewidth=0.5)

ax_I.axvline(0, color=TRIG_C, lw=0.8, alpha=0.55, zorder=2)
ax_Q.axvline(0, color=TRIG_C, lw=0.8, alpha=0.55, zorder=2)
ax_env.set_xlabel('Time relative to latest trigger (s)',
                  color='#78909c', fontsize=9)

lines_I = [ax_I.plot([], [], lw=0.9, color=COLORS[k % len(COLORS)])[0]
           for k in range(MAX_SW)]
lines_Q = [ax_Q.plot([], [], lw=0.9, color=COLORS[k % len(COLORS)])[0]
           for k in range(MAX_SW)]
env_line,  = ax_env.plot([], [], color='#26c6da', lw=1.0)
trig_dots  = ax_env.scatter([], [], color=TRIG_C, s=28, zorder=5, marker='v')

title_obj = fig.suptitle('Waiting for data…', color='#cfd8dc', fontsize=10,
                          fontfamily='monospace', x=0.5, y=0.975)
fig.text(0.97, 0.005, '+/- window   n/N sweeps   q quit',
         color='#37474f', fontsize=8, ha='right', fontfamily='monospace')

scope = dict(window_s=args.window, pretrig_s=args.pretrig,
             n_sweeps=min(args.sweeps, MAX_SW))

def on_key(ev):
    k = ev.key
    if   k == 'q':           plt.close('all'); sys.exit(0)
    elif k in ('+', '='):    scope['window_s']  = min(scope['window_s']  + 0.2, 30.0)
    elif k == '-':           scope['window_s']  = max(scope['window_s']  - 0.2, 0.3)
    elif k == 'n':           scope['n_sweeps']  = min(scope['n_sweeps']  + 1, MAX_SW)
    elif k == 'N':           scope['n_sweeps']  = max(scope['n_sweeps']  - 1, 1)
    elif k == ' ':
        # Toggle pause: when paused, stop reading and freeze display.
        # When unpausing, discard any buffered samples collected while paused.
        global paused
        paused = not paused
        if paused:
            title_obj.set_text('PAUSED — press space to resume')
            print('Scope paused')
        else:
            # clear in-memory buffers to drop stale data
            with lock:
                buf_I.clear(); buf_Q.clear(); buf_seq.clear(); buf_ms.clear()
            title_obj.set_text('Resumed — waiting for fresh data...')
            print('Scope resumed (old data discarded)')
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key)

def extract_sweep(y, center_idx, pretrig_n, window_n):
    """Extract a sweep centered at fractional-sample center_idx using linear interpolation."""
    n = len(y)
    L = pretrig_n + window_n
    start = center_idx - pretrig_n
    end = start + (L - 1)
    if start < 0 or end > (n - 1):
        return None
    x = np.arange(n, dtype=np.float64)
    xi = start + np.arange(L, dtype=np.float64)
    return np.interp(xi, x, y)


def update(_frame):

    global _meas_fs

    if paused:
        return

    with lock:
        n = len(buf_I)
        if n < int(FS_NOM * 4):
            return
        arr_I   = np.array(buf_I,   dtype=np.float64)
        arr_Q   = np.array(buf_Q,   dtype=np.float64)
        arr_seq = np.array(buf_seq, dtype=np.float64)
        arr_ms  = np.array(buf_ms,  dtype=np.float64)
    # Sampling rate is crystal-derived; treat as fixed.
    fs = FS_NOM

    window_n  = int(scope['window_s'] * fs)
    pretrig_n = int(scope['pretrig_s'] * fs)
    n_sw      = scope['n_sweeps']

    if n < pretrig_n + window_n + 20:
        return

    env, I_dc, Q_dc = compute_envelope(arr_I, arr_Q)
    _phi, vphi = compute_phase_velocity(I_dc, Q_dc, fs)
    if vphi is None:
        return
    triggers = find_vphi_zero_crossings(vphi, fs, args.f0, pretrig_n, window_n, direction='neg_to_pos')

    t_axis = (np.arange(pretrig_n + window_n) - pretrig_n) / fs
    recent = triggers[-n_sw:]

    # update sweep lines
    segs_I, segs_Q = [], []
    for k in range(MAX_SW):
        if k >= len(recent):
            lines_I[k].set_data([], [])
            lines_Q[k].set_data([], [])
            continue
        frac  = k / max(len(recent) - 1, 1)
        alpha = 0.10 + 0.82 * frac
        lines_I[k].set_alpha(alpha)
        lines_Q[k].set_alpha(alpha)
        trig = recent[k]
        segI = extract_sweep(I_dc, trig, pretrig_n, window_n)
        segQ = extract_sweep(Q_dc, trig, pretrig_n, window_n)
        if segI is None or segQ is None:
            lines_I[k].set_data([], [])
            lines_Q[k].set_data([], [])
            continue
        lines_I[k].set_data(t_axis, segI)
        lines_Q[k].set_data(t_axis, segQ)
        segs_I.append(segI)
        segs_Q.append(segQ)

    def nice_lim(segs, margin=0.08):
        if not segs: return -1, 1
        cat = np.concatenate(segs)
        lo, hi = np.percentile(cat, 0.5), np.percentile(cat, 99.5)
        span = max(hi - lo, 1.0)
        return lo - margin*span, hi + margin*span

    for ax, segs in [(ax_I, segs_I), (ax_Q, segs_Q)]:
        ax.set_ylim(*nice_lim(segs))
        ax.set_xlim(t_axis[0], t_axis[-1])

    # envelope panel: last 4× window of history
    env_show_n = min(int(scope['window_s'] * 4 * fs), n)
    t_env = (np.arange(env_show_n) - env_show_n) / fs   # ends at 0
    env_line.set_data(t_env, env[-env_show_n:])
    ax_env.set_xlim(t_env[0], 0)
    env_slice = env[-env_show_n:]
    ax_env.set_ylim(0, env_slice.max() * 1.12 + 1)

    # trigger markers on envelope panel
    trig_times = [((t - n) / fs) for t in triggers if t >= n - env_show_n]
    # interpolate envelope at fractional trigger positions
    x_env = np.arange(len(env), dtype=np.float64)
    trig_pos = [t for t in triggers if t >= n - env_show_n]
    trig_vals = list(np.interp(np.array(trig_pos, dtype=np.float64), x_env, env)) if trig_pos else []
    if trig_times:
        trig_dots.set_offsets(np.c_[trig_times, trig_vals])
    else:
        trig_dots.set_offsets(np.empty((0, 2)))

    title_obj.set_text(
        f'Pendulum IQ Scope  |  window={scope["window_s"]:.1f}s  '
        f'pretrig={scope["pretrig_s"]:.2f}s  sweeps={n_sw}  '
        f'fs={fs:.2f}Hz (fixed)  triggers={len(triggers)}'
    )
    fig.canvas.draw_idle()

import matplotlib.animation as animation
_ani = animation.FuncAnimation(fig, update, interval=300,
                                blit=False, cache_frame_data=False)

print(f"Scope running.  Trigger = phase-velocity zero crossing (neg→pos) (one turnaround per period), f0={args.f0}Hz")
print("Click window then: +/- adjust window width, n/N adjust sweep count, q quit")
plt.show()
