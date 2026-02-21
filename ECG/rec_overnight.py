#!/home/pi/pieeg-env/bin/python

"""
PiEEG v2.1 — Overnight ECG recorder with live metrics
Record CH8 (E8) via REF (SRB1), filter in real-time, and log to CSV with timestamps.

3 electrodes from PiEEG board as follows:
  EEG Ref     → RA (right infraclavicular, just below the right clavicle)
  EEG BiasOut → RL (lower right abdominal quadrant)
  EEG e8      → LL (lower left abdominal quadrant)

Live display (every ~5s):
  - QRS amplitude, R-R interval, HR
  - RMS variation of QRS amp and R-R
  - T-wave amplitude and estimated QTc

Filters (causal IIR, applied sample-by-sample):
  - HP 0.5 Hz  (2nd-order Butterworth)
  - Notch 60 Hz (Q=30)
  - LP 40 Hz   (4th-order Butterworth)

Timestamps: wall-clock time logged every second for drift correction.
Output: two files —
  ECG_<timestamp>.csv        (filtered µV at 125 sps)
  ECG_<timestamp>_sync.csv   (output_sample_index, unix_time)

Ctrl-C for clean shutdown.

Note: first do:
     source  ~/pieeg-env/bin/activate

J. Beale  v1.3 2026-02-13
"""

VERSION = "1.4"
VERSION_DATE = "2026-02-20"

import spidev
import time
import signal as sig
import sys
import subprocess
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi
from RPi import GPIO
from datetime import datetime
from collections import deque

# =========================
# Graceful shutdown
# =========================
running = True

def handle_sigint(signum, frame):
    global running
    running = False
    print("\nStopping...")

sig.signal(sig.SIGINT, handle_sigint)

# =========================
# GPIO (DRDY)
# =========================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.IN)

# =========================
# SPI
# =========================
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 600000
spi.lsbfirst = False
spi.mode = 0b01
spi.bits_per_word = 8

# =========================
# ADS1299 registers & commands
# =========================
CONFIG1 = 0x01;  CONFIG2 = 0x02;  CONFIG3 = 0x03
CH8SET  = 0x0C;  MISC1   = 0x15

CMD_WAKEUP = 0x02;  CMD_STOP   = 0x0A;  CMD_RESET  = 0x06
CMD_SDATAC = 0x11;  CMD_RDATAC = 0x10;  CMD_START  = 0x08

def send_command(cmd):
    spi.xfer([cmd])
    time.sleep(0.002)

def write_reg(reg, val):
    spi.xfer([0x40 | reg, 0x00, val])
    time.sleep(0.002)

# =========================
# ADS1299 init
# =========================
send_command(CMD_WAKEUP)
send_command(CMD_STOP)
send_command(CMD_RESET)
send_command(CMD_SDATAC)

write_reg(CONFIG1, 0x96)    # 250 SPS, high-res
write_reg(CONFIG2, 0xD4)
write_reg(CONFIG3, 0xEC)    # internal ref ON, bias amp ON

write_reg(0x0D, 0x00)       # BIAS_SENSP off
write_reg(0x0E, 0x00)       # BIAS_SENSN off

write_reg(MISC1, 0x20)      # SRB1 on
write_reg(CH8SET, 0x30)     # CH8: gain=6, normal input

for ch in range(0x05, 0x05 + 7):
    write_reg(ch, 0x81)     # power down CH1-CH7

send_command(CMD_RDATAC)
send_command(CMD_START)

# =========================
# Parameters
# =========================
FS_IN  = 250     # ADS1299 sample rate
FS_OUT = 250     # final output sample rate
# DECIMATE = FS_IN // FS_OUT
DECIMATE = 1

GAIN = 6
VREF = 4.5
SCALE = 1e6 * VREF / (GAIN * (2**23))

SYNC_INTERVAL = FS_IN       # log timestamp every 1 sec of input
FLUSH_EVERY = FS_OUT * 60   # flush to disk every minute
STATS_INTERVAL = FS_OUT * 5 # print live stats every 5 sec
DISCARD_IN = FS_IN * 2      # discard first 2 sec (filter settling)

# =========================
# Design filters (SOS cascade)
# =========================
sos_hp    = butter(2, 0.5, 'highpass', fs=FS_IN, output='sos')
b_n, a_n  = iirnotch(60, Q=30, fs=FS_IN)
sos_notch = tf2sos(b_n, a_n)
sos_lp    = butter(4, 40, 'lowpass', fs=FS_IN, output='sos')

sos_all = np.vstack([sos_hp, sos_notch, sos_lp])
zi = sosfilt_zi(sos_all) * 0.0

# =========================
# R-peak detector state
# =========================
REFRACT_SAMPLES = int(0.25 * FS_OUT)  # 250ms — supports up to ~220 bpm
peak_threshold = 400.0       # µV initial guess, adapts
PEAK_THRESHOLD_MIN = 100.0   # µV floor — never decay below this
PEAK_THRESHOLD_MAX = 5000.0  # µV ceiling — cap artifact inflation
last_peak_idx = -REFRACT_SAMPLES
peak_adapt_alpha = 0.1       # EMA smoothing for threshold upward
peak_decay_alpha = 0.002     # per-sample decay rate when no peak seen

# Peak-tracking state machine
in_peak = False              # True while tracking a candidate R-peak
candidate_amp = 0.0          # max value seen during this peak
candidate_idx = 0            # output sample index of the max
candidate_buf_pos = 0        # analysis buffer position of the max

# Rolling analysis buffer (5 sec) for T-wave lookback
ANALYSIS_BUF_LEN = FS_OUT * 5
analysis_buf = np.zeros(ANALYSIS_BUF_LEN, dtype=np.float32)
abuf_write = 0               # write pointer (circular)
abuf_filled = False

# Recent beat measurements for stats
MAX_BEATS = 30
recent_qrs_amp = deque(maxlen=MAX_BEATS)
recent_rr_ms   = deque(maxlen=MAX_BEATS)
recent_twave   = deque(maxlen=MAX_BEATS)
recent_qt_ms   = deque(maxlen=MAX_BEATS)

prev_peak_out_idx = None
pending_twave_pos = None     # buffer position of most recent R-peak

# =========================
# Signal / recording state
# =========================
ACQ_WAITING   = 0   # waiting for initial valid ECG signal; not yet writing
ACQ_RECORDING = 1   # signal valid; writing to disk
ACQ_PAUSED    = 2   # signal lost mid-recording; counting samples, not writing

acq_state = ACQ_WAITING

NO_SIGNAL_RR_MULTIPLE  = 5    # declare loss after this many missed expected beats
NO_SIGNAL_FALLBACK_S   = 60   # fallback if RR not yet established (seconds)
NO_SIGNAL_MIN_S        = 120  # floor: never trigger pause sooner than this

def no_signal_timeout_samples():
    if len(recent_rr_ms) >= 2:
        mean_rr_samp = np.mean(recent_rr_ms) / 1000.0 * FS_OUT
        rr_based = int(NO_SIGNAL_RR_MULTIPLE * mean_rr_samp)
        return max(rr_based, int(NO_SIGNAL_MIN_S * FS_OUT))
    return int(NO_SIGNAL_FALLBACK_S * FS_OUT)

# =========================
# Output buffers
# =========================
CHUNK = 8192
ecg_buf = np.empty(CHUNK, dtype=np.float32)
ecg_idx = 0

sync_buf = []

out_sample_count = 0
in_sample_count  = 0
last_stats_out   = 0
samples_since_rpeak = 0      # for delayed T-wave measurement

# =========================
# File setup
# =========================
ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
ecg_fname  = f"ECG_{ts_str}.csv"
sync_fname = f"ECG_{ts_str}_sync.csv"

f_ecg  = open(ecg_fname, 'w')
f_sync = open(sync_fname, 'w')
f_ecg.write("ecg_raw_uV\n")
f_sync.write("sample_idx,unix_time\n")

t_start = time.time()
sync_buf.append((0, t_start))

print(f"rec_overnight.py v{VERSION} ({VERSION_DATE})")
print(f"Recording to {ecg_fname} at {FS_OUT} sps  (Ctrl-C to stop)")
print(f"Started: {datetime.now().isoformat()}")
print()

# =========================
# Helper: measure T-wave from circular buffer
# =========================
T_SEARCH_START = int(0.16 * FS_OUT)   # 160ms after R (skip QRS)
T_SEARCH_END   = int(0.55 * FS_OUT)   # 550ms after R (allow for T-end)
Q_LOOKBACK     = int(0.06 * FS_OUT)   # 60ms before R to find Q onset

def measure_twave_from_ring(r_buf_pos):
    """Measure T-wave amplitude and QT interval (Q-onset to T-end)."""

    # --- Baseline: median of 40-60ms before R-peak ---
    bl_indices = [(r_buf_pos - i) % ANALYSIS_BUF_LEN
                  for i in range(int(0.04 * FS_OUT), int(0.06 * FS_OUT) + 1)]
    baseline = np.median(analysis_buf[bl_indices])

    # --- Q-onset: find where signal first dips below baseline
    #     in the 60ms before R-peak ---
    q_offset = 0  # samples before R-peak
    for i in range(1, Q_LOOKBACK + 1):
        idx = (r_buf_pos - i) % ANALYSIS_BUF_LEN
        if analysis_buf[idx] <= baseline:
            q_offset = i
            break

    # --- T-wave peak ---
    indices = [(r_buf_pos + i) % ANALYSIS_BUF_LEN
               for i in range(T_SEARCH_START, T_SEARCH_END)]
    segment = np.array([analysis_buf[j] for j in indices])

    t_peak_rel = np.argmax(segment)
    t_amp = segment[t_peak_rel] - baseline
    if t_amp < 10:
        return None, None

    # --- T-wave end: tangent method ---
    # Find steepest downslope after T-peak, project to baseline
    t_end_rel = t_peak_rel  # fallback: T-peak position
    if t_peak_rel + 5 < len(segment):
        # Compute slope (first derivative) after T-peak
        post_peak = segment[t_peak_rel:]
        slopes = np.diff(post_peak)
        if len(slopes) > 2:
            steepest = np.argmin(slopes)  # most negative slope
            # Tangent line: y = segment[t_peak_rel+steepest] + slope*(x)
            # Find where it crosses baseline
            slope_val = slopes[steepest]
            if slope_val < -0.5:  # meaningful downslope
                y_at_tangent = post_peak[steepest]
                # samples from tangent point to baseline crossing
                dx = (baseline - y_at_tangent) / slope_val
                t_end_rel = t_peak_rel + steepest + max(0, dx)

    # QT = Q-onset to T-end
    qt_samples = q_offset + T_SEARCH_START + t_end_rel
    qt_ms = qt_samples / FS_OUT * 1000

    return t_amp, qt_ms

# =========================
# Helper: print stats
# =========================
email_sent = False
SEND_EMAIL_DIR = "/home/pi/ECG"
EMAIL_AFTER_RECORDING_SAMPLES = FS_OUT * 60  # 1 min of valid recording
recording_sample_count = 0   # increments only while ACQ_RECORDING

def build_summary():
    """Build a summary string from recent stats."""
    if len(recent_rr_ms) < 3:
        return None

    rr = np.array(recent_rr_ms)
    qa = np.array(recent_qrs_amp)

    hr = 60000.0 / np.mean(rr)
    rr_mean = np.mean(rr)
    rr_sd = np.std(rr)
    qa_mean = np.mean(qa)
    qa_sd = np.std(qa)

    lines = []
    lines.append(f"rec_overnight.py v{VERSION} ({VERSION_DATE})")
    lines.append(f"ECG Recorder started: {datetime.now().isoformat()}")
    lines.append(f"Output file: {ecg_fname}")
    lines.append(f"Sample rate: {FS_OUT} sps")
    lines.append(f"")
    lines.append(f"First minute summary ({len(recent_rr_ms)} beats):")
    lines.append(f"  HR:  {hr:.1f} bpm")
    lines.append(f"  RR:  {rr_mean:.0f} +/- {rr_sd:.1f} ms")
    lines.append(f"  QRS: {qa_mean:.0f} +/- {qa_sd:.0f} uV")

    if len(recent_qt_ms) >= 3:
        qt = np.mean(np.array(recent_qt_ms))
        tw = np.mean(np.array(recent_twave))
        rr_sec = rr_mean / 1000.0
        qtc = qt / np.sqrt(rr_sec)
        lines.append(f"  T:   {tw:+.0f} uV")
        lines.append(f"  QTc: {qtc:.0f} ms (Bazett)")

    lines.append(f"")
    lines.append(f"Recording in progress...")
    return "\n".join(lines)

def send_email_summary():
    """Send summary email via send_email.py."""
    global email_sent
    summary = build_summary()
    if summary is None:
        return
    try:
        subprocess.Popen(
            [sys.executable, f"{SEND_EMAIL_DIR}/send_email.py", summary],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("  >> Email summary sent", flush=True)
    except Exception as e:
        print(f"  >> Email failed: {e}", flush=True)
    email_sent = True
def print_stats():
    elapsed = time.time() - t_start

    if len(recent_rr_ms) < 3:
        print(f"  {elapsed/60:5.1f}m | Waiting for beats...", flush=True)
        return

    rr = np.array(recent_rr_ms)
    qa = np.array(recent_qrs_amp)

    hr = 60000.0 / np.mean(rr)
    rr_mean = np.mean(rr)
    rr_sd   = np.std(rr)
    qa_mean = np.mean(qa)
    qa_sd   = np.std(qa)

    # Bazett-corrected QTc
    qtc_str = ""
    if len(recent_qt_ms) >= 3:
        qt  = np.mean(np.array(recent_qt_ms))
        tw  = np.mean(np.array(recent_twave))
        rr_sec = rr_mean / 1000.0
        qtc = qt / np.sqrt(rr_sec)
        qtc_str = f" | T:{tw:+5.0f}µV  QTc:{qtc:.0f}ms"

    print(
        f"  {elapsed/60:5.1f}m | "
        f"HR {hr:4.1f}  "
        f"RR {rr_mean:5.0f}±{rr_sd:4.1f}ms  "
        f"QRS {qa_mean:5.0f}±{qa_sd:4.0f}µV"
        f"{qtc_str}",
        flush=True
    )

# =========================
# Acquisition loop
# =========================
try:
    while running:
        while GPIO.input(37) == 1:
            if not running:
                break
            time.sleep(0.00005)
        if not running:
            break

        frame = spi.readbytes(27)

        # Wait for DRDY to deassert (go high) before next sample.
        # Prevents re-reading on the same DRDY pulse if the SPI transaction
        # completed while DRDY was still low (causes all-zero frames).
        t_deassert = time.time() + 0.020
        while GPIO.input(37) == 0:
            if time.time() > t_deassert:
                break

        # STATUS[23:20] are hardwired to 0b1100 by the ADS1299.
        # A wrong value means this is a stale or mis-timed read; discard it.
        if (frame[0] & 0xF0) != 0xC0:
            continue

        raw = (frame[24] << 16) | (frame[25] << 8) | frame[26]
        if raw & 0x800000:
            raw -= 1 << 24

        sample_uv = raw * SCALE
        in_sample_count += 1

        # --- Causal IIR filter ---
        x = np.array([sample_uv])
        y, zi = sosfilt(sos_all, x, zi=zi)

        # --- Skip startup transient ---
        if in_sample_count <= DISCARD_IN:
            if in_sample_count == DISCARD_IN:
                # Reset counters and log start time now
                t_start = time.time()
                sync_buf.clear()
                sync_buf.append((0, t_start))
                print(f"Filter settled, waiting for ECG signal: "
                      f"{datetime.now().isoformat()}")
            continue

        # --- Decimation: keep every Nth sample ---
        if in_sample_count % DECIMATE == 0:
            val = y[0]          # filtered — for beat detection & stats

            # Always increment so R-peak indices stay coherent and
            # gaps during PAUSED/WAITING are preserved in the sync file.
            out_sample_count += 1

            # Write RAW to file buffer only while signal is valid
            if acq_state == ACQ_RECORDING:
                ecg_buf[ecg_idx] = sample_uv
                ecg_idx += 1
                recording_sample_count += 1

            # Write FILTERED to circular analysis buffer
            analysis_buf[abuf_write] = val
            r_buf_pos_now = abuf_write
            abuf_write = (abuf_write + 1) % ANALYSIS_BUF_LEN
            if abuf_write == 0:
                abuf_filled = True

            # --- Delayed T-wave measurement ---
            # Measure T-wave 440ms after the R-peak was detected,
            # so the full T-wave window is in the buffer
            if pending_twave_pos is not None:
                samples_since_rpeak += 1
                if samples_since_rpeak >= T_SEARCH_END + 4:
                    if abuf_filled or out_sample_count > ANALYSIS_BUF_LEN:
                        t_amp, qt_ms = measure_twave_from_ring(
                            pending_twave_pos)
                        if t_amp is not None and t_amp > 10:
                            recent_twave.append(t_amp)
                            recent_qt_ms.append(qt_ms)
                    pending_twave_pos = None

            # --- R-peak detection (true peak) ---
            since_peak = out_sample_count - last_peak_idx

            if not in_peak:
                # Decay threshold toward floor when no peak seen for > 1 RR
                decay_start = int(np.mean(recent_rr_ms) / 1000.0 * FS_OUT
                                  ) if len(recent_rr_ms) >= 2 else REFRACT_SAMPLES
                if since_peak > decay_start:
                    peak_threshold = max(
                        PEAK_THRESHOLD_MIN,
                        peak_threshold * (1.0 - peak_decay_alpha)
                    )

                # Look for threshold crossing after refractory
                if val > peak_threshold and since_peak > REFRACT_SAMPLES:
                    in_peak = True
                    candidate_amp = val
                    candidate_idx = out_sample_count
                    candidate_buf_pos = r_buf_pos_now
            else:
                # Tracking: update if still rising
                if val > candidate_amp:
                    candidate_amp = val
                    candidate_idx = out_sample_count
                    candidate_buf_pos = r_buf_pos_now

                # Commit peak when signal drops to 60% of candidate
                if val < candidate_amp * 0.6:
                    in_peak = False
                    last_peak_idx = candidate_idx

                    # Adapt threshold upward, capped to avoid artifact lock-up
                    new_thresh = (
                        peak_adapt_alpha * 0.4 * candidate_amp +
                        (1 - peak_adapt_alpha) * peak_threshold
                    )
                    peak_threshold = min(new_thresh, PEAK_THRESHOLD_MAX)

                    recent_qrs_amp.append(candidate_amp)

                    # R-R interval
                    if prev_peak_out_idx is not None:
                        rr_ms = ((candidate_idx - prev_peak_out_idx)
                                 / FS_OUT * 1000)
                        if 300 < rr_ms < 2000:
                            recent_rr_ms.append(rr_ms)

                    prev_peak_out_idx = candidate_idx

                    # Schedule T-wave measurement
                    pending_twave_pos = candidate_buf_pos
                    samples_since_rpeak = 0

            # --- Signal / recording state machine ---
            timeout_samp = no_signal_timeout_samples()
            samples_since_peak = out_sample_count - last_peak_idx

            if acq_state == ACQ_WAITING:
                if len(recent_rr_ms) >= 2:
                    acq_state = ACQ_RECORDING
                    print(f"Signal acquired, recording started: "
                          f"{datetime.now().isoformat()}", flush=True)

            elif acq_state == ACQ_RECORDING:
                if samples_since_peak > timeout_samp:
                    acq_state = ACQ_PAUSED
                    print(f"  No signal — recording paused at sample "
                          f"{out_sample_count} "
                          f"({datetime.now().isoformat()})", flush=True)

            elif acq_state == ACQ_PAUSED:
                # A peak was just committed if last_peak_idx is very recent
                if samples_since_peak < REFRACT_SAMPLES:
                    acq_state = ACQ_RECORDING
                    print(f"  Signal restored — recording resumed at sample "
                          f"{out_sample_count} "
                          f"({datetime.now().isoformat()})", flush=True)

            # --- Flush ECG file buffer ---
            if ecg_idx >= CHUNK:
                for v in ecg_buf[:ecg_idx]:
                    f_ecg.write(f"{v:.2f}\n")
                ecg_idx = 0

            # --- Periodic stats ---
            if out_sample_count - last_stats_out >= STATS_INTERVAL:
                if acq_state == ACQ_RECORDING:
                    recent_peak_samples = out_sample_count - last_peak_idx
                    if recent_peak_samples <= 6 * FS_OUT:
                        print_stats()
                last_stats_out = out_sample_count

                # Send email after first full minute of valid recording
                if not email_sent and recording_sample_count >= EMAIL_AFTER_RECORDING_SAMPLES:
                    send_email_summary()

        # --- Periodic timestamp sync ---
        if in_sample_count % SYNC_INTERVAL == 0:
            sync_buf.append((out_sample_count, time.time()))

        # --- Periodic disk flush ---
        if (out_sample_count % FLUSH_EVERY == 0 and
                out_sample_count > 0 and
                in_sample_count % DECIMATE == 0):
            if ecg_idx > 0:
                for v in ecg_buf[:ecg_idx]:
                    f_ecg.write(f"{v:.2f}\n")
                ecg_idx = 0
            f_ecg.flush()

            for idx, t in sync_buf:
                f_sync.write(f"{idx},{t:.6f}\n")
            sync_buf.clear()
            f_sync.flush()

except Exception as e:
    print(f"Error: {e}")

# =========================
# Cleanup
# =========================
send_command(CMD_STOP)
send_command(CMD_SDATAC)
spi.close()
GPIO.cleanup()

if ecg_idx > 0:
    for v in ecg_buf[:ecg_idx]:
        f_ecg.write(f"{v:.2f}\n")

for idx, t in sync_buf:
    f_sync.write(f"{idx},{t:.6f}\n")

f_ecg.close()
f_sync.close()

elapsed = time.time() - t_start
print(f"\nDone. {out_sample_count} samples in {elapsed/3600:.2f} hours")
print(f"Saved: {ecg_fname}, {sync_fname}")
