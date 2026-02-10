#!/usr/bin/env python3
"""
Plot MPU-6050 motion sensor CSV data from SD card.
Symlog y-axes show both subtle breathing/cardiac and large motion events.
Usage: python3 plot_motion.py MOT_006.csv [output.png]
If no output file given, displays interactively.
J.Beale 2026-02-09
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import butter, sosfiltfilt, find_peaks

# --- Configuration ---
BREATH_LO = 0.1    # Hz (6 breaths/min)
BREATH_HI = 0.5    # Hz (30 breaths/min)
CARDIAC_LO = 0.8   # Hz (48 bpm)
CARDIAC_HI = 2.0   # Hz (120 bpm)

# symlog linear thresholds (adjust to taste)
LINTHRESH_ANGLE   = 0.1    # degrees
LINTHRESH_BREATH  = 0.02   # degrees
LINTHRESH_CARDIAC = 0.1    # mG
LINTHRESH_MOTION  = 5.0    # mG / mG·s

def load_and_plot(csv_path, save_path=None):
    # check for timestamp comment in first line
    start_time = None
    skip = 1  # default: skip just the header
    with open(csv_path, 'r') as f:
        first = f.readline().strip()
        if first.startswith('# start ') and 'unknown' not in first:
            start_time = first[8:]
            skip = 2  # skip comment + header

    data = np.genfromtxt(csv_path, delimiter=',', skip_header=skip)
    if data.ndim != 2 or data.shape[1] != 6:
        print(f"Error: expected 6 columns (msec,pitch,roll,rot,total,rms), got {data.shape}")
        sys.exit(1)

    t = (data[:, 0] - data[0, 0]) / 1000.0
    pitch = data[:, 1]
    roll  = data[:, 2]
    rot   = data[:, 3]
    total = data[:, 4]
    rms   = data[:, 5]

    dt_median = np.median(np.diff(t))
    fs = 1.0 / dt_median
    duration_min = t[-1] / 60.0

    print(f"File: {csv_path}")
    print(f"Samples: {len(t)},  Duration: {duration_min:.1f} min,  Fs: {fs:.2f} Hz")

    # design filters
    breath_sos  = butter(2, [BREATH_LO, BREATH_HI], btype='bandpass', fs=fs, output='sos')
    cardiac_sos = butter(2, [CARDIAC_LO, CARDIAC_HI], btype='bandpass', fs=fs, output='sos')

    breath_filt  = sosfiltfilt(breath_sos, pitch)
    cardiac_filt = sosfiltfilt(cardiac_sos, rms)

    # use minutes for long recordings, seconds for short
    # if we have a start time, show real clock time on x-axis for long recordings
    use_clock = False
    if duration_min > 5:
        if start_time:
            from datetime import datetime, timedelta
            import matplotlib.dates as mdates
            try:
                t0_dt = datetime.fromisoformat(start_time)
                t_plot = [t0_dt + timedelta(seconds=s) for s in t]
                use_clock = True
                t_label = "Time"
            except ValueError:
                t_plot = t / 60.0
                t_label = "Time (minutes)"
        else:
            t_plot = t / 60.0
            t_label = "Time (minutes)"
    else:
        t_plot = t
        t_label = "Time (seconds)"

    # extract filename for title
    import os
    base = os.path.splitext(os.path.basename(csv_path))[0]
    title = f"MPU-6050 Motion — {base}  ({len(t)} samples, {duration_min:.1f} min)"
    if start_time:
        title += f"\nStarted: {start_time}"

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(hspace=0.30, bottom=0.08)

    ax1, ax2, ax3, ax4 = axes

    # track titles for both modes
    titles_symlog = [
        f"Body Angle (symlog, linear < {LINTHRESH_ANGLE}°)",
        f"Breathing — pitch BP {BREATH_LO}–{BREATH_HI} Hz (symlog, linear < {LINTHRESH_BREATH}°)",
        f"Cardiac — RMS BP {CARDIAC_LO}–{CARDIAC_HI} Hz (symlog, linear < {LINTHRESH_CARDIAC} mG)",
        f"Raw Motion (symlog, linear < {LINTHRESH_MOTION} mG)",
    ]
    titles_linear = [
        "Body Angle (linear)",
        f"Breathing — pitch BP {BREATH_LO}–{BREATH_HI} Hz (linear)",
        f"Cardiac — RMS BP {CARDIAC_LO}–{CARDIAC_HI} Hz (linear)",
        "Raw Motion (linear)",
    ]
    linthresh_vals = [LINTHRESH_ANGLE, LINTHRESH_BREATH, LINTHRESH_CARDIAC, LINTHRESH_MOTION]

    # 1: Body angle
    ax1.plot(t_plot, pitch, 'b-', linewidth=0.6, alpha=0.8, label='Pitch')
    ax1.plot(t_plot, roll, 'r-', linewidth=0.6, alpha=0.8, label='Roll')
    ax1.set_yscale('symlog', linthresh=LINTHRESH_ANGLE)
    ax1.set_ylabel("Degrees")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title(f"Body Angle (symlog, linear < {LINTHRESH_ANGLE}°)", fontsize=10)

    # 2: Breathing
    ax2.plot(t_plot, breath_filt, 'darkgreen', linewidth=0.8)
    ax2.set_yscale('symlog', linthresh=LINTHRESH_BREATH)
    ax2.set_ylabel("Degrees")
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_title(f"Breathing — pitch BP {BREATH_LO}–{BREATH_HI} Hz (symlog, linear < {LINTHRESH_BREATH}°)", fontsize=10)

    # 3: Cardiac
    ax3.plot(t_plot, cardiac_filt, 'firebrick', linewidth=0.6)
    ax3.set_yscale('symlog', linthresh=LINTHRESH_CARDIAC)
    ax3.set_ylabel("mG")
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_title(f"Cardiac — RMS BP {CARDIAC_LO}–{CARDIAC_HI} Hz (symlog, linear < {LINTHRESH_CARDIAC} mG)", fontsize=10)

    # 4: Raw motion
    ax4.plot(t_plot, total, 'purple', linewidth=0.6, alpha=0.8, label='Total impulse')
    ax4.plot(t_plot, rms, 'darkorange', linewidth=0.6, alpha=0.8, label='Accel RMS')
    ax4.set_yscale('symlog', linthresh=LINTHRESH_MOTION)
    ax4.set_ylabel("mG / mG·s")
    ax4.set_xlabel(t_label)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_title(f"Raw Motion (symlog, linear < {LINTHRESH_MOTION} mG)", fontsize=10)

    plt.tight_layout()

    # --- Rate stats annotations (visible when zoomed to ≤5 min) ---
    breath_text = ax2.text(0.98, 0.95, '', transform=ax2.transAxes,
                           fontsize=9, fontfamily='monospace',
                           ha='right', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    cardiac_text = ax3.text(0.98, 0.95, '', transform=ax3.transAxes,
                            fontsize=9, fontfamily='monospace',
                            ha='right', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    span_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes,
                         fontsize=9, fontfamily='monospace',
                         ha='right', va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    def get_visible_seconds():
        """Return (t_start, t_end) in seconds-from-start for current x-axis view."""
        xlim = ax2.get_xlim()
        if use_clock:
            # xlim is in matplotlib date numbers
            from datetime import datetime, timedelta
            x0 = mdates.num2date(xlim[0]).replace(tzinfo=None)
            x1 = mdates.num2date(xlim[1]).replace(tzinfo=None)
            t0_dt = datetime.fromisoformat(start_time)
            s0 = (x0 - t0_dt).total_seconds()
            s1 = (x1 - t0_dt).total_seconds()
        elif duration_min > 5 and not use_clock:
            # xlim is in minutes
            s0 = xlim[0] * 60.0
            s1 = xlim[1] * 60.0
        else:
            # xlim is in seconds
            s0 = xlim[0]
            s1 = xlim[1]
        return s0, s1

    def compute_rate(signal, t_arr, t_start, t_end, min_distance_sec):
        """Find peaks in visible region, return (mean_rate, std_rate) in per-minute, or None."""
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        idx = np.where(mask)[0]
        if len(idx) < 3:
            return None
        seg = signal[idx]
        t_seg = t_arr[idx]
        min_dist = max(1, int(min_distance_sec * fs))
        peaks, _ = find_peaks(seg, distance=min_dist, prominence=np.std(seg) * 0.3)
        if len(peaks) < 2:
            return None
        intervals = np.diff(t_seg[peaks])
        if len(intervals) < 1:
            return None
        rates = 60.0 / intervals  # per minute
        return np.mean(rates), np.std(rates), len(peaks)

    def on_xlim_changed(event_ax):
        try:
            s0, s1 = get_visible_seconds()
        except Exception:
            breath_text.set_text('')
            cardiac_text.set_text('')
            span_text.set_text('')
            return

        span_min = (s1 - s0) / 60.0

        # update displayed time span
        span_sec = max(0, s1 - s0)
        h = int(span_sec // 3600)
        m = int((span_sec % 3600) // 60)
        s = int(span_sec % 60)
        span_text.set_text(f'Span {h:d}:{m:02d}:{s:02d}')

        autoscale_y_to_visible()

        if span_min > 5.0:
            breath_text.set_text('')
            cardiac_text.set_text('')
            fig.canvas.draw_idle()
            return

        # Breathing: min peak spacing ~1.5s (40 breaths/min max)
        br = compute_rate(breath_filt, t, s0, s1, 1.5)
        if br:
            mean_r, std_r, n_peaks = br
            breath_text.set_text(f'Resp: {mean_r:.1f} ± {std_r:.1f} /min  (n={n_peaks})')
        else:
            breath_text.set_text('Resp: —')

        # Cardiac: min peak spacing ~0.4s (150 bpm max)
        cr = compute_rate(cardiac_filt, t, s0, s1, 0.4)
        if cr:
            mean_r, std_r, n_peaks = cr
            cardiac_text.set_text(f'HR: {mean_r:.0f} ± {std_r:.0f} bpm  (n={n_peaks})')
        else:
            cardiac_text.set_text('HR: —')

        fig.canvas.draw_idle()

    # connect to all shared x-axes (only need one, they're linked)
    ax2.callbacks.connect('xlim_changed', on_xlim_changed)

    # --- Symlog / Linear toggle button ---
    is_symlog = [True]  # mutable so callback can modify

    def autoscale_y_to_visible():
        """In linear mode, rescale Y axes to fit only the visible data."""
        if is_symlog[0]:
            return
        try:
            s0, s1 = get_visible_seconds()
        except Exception:
            return
        mask = (t >= s0) & (t <= s1)
        if not mask.any():
            return
        ax_data = [
            [pitch[mask], roll[mask]],
            [breath_filt[mask]],
            [cardiac_filt[mask]],
            [total[mask], rms[mask]],
        ]
        for ax, datasets in zip(axes, ax_data):
            if not ax.get_visible():
                continue
            all_visible = np.concatenate(datasets)
            ymin, ymax = np.nanmin(all_visible), np.nanmax(all_visible)
            margin = (ymax - ymin) * 0.05
            if margin == 0:
                margin = 0.1
            ax.set_ylim(ymin - margin, ymax + margin)

    btn_ax = fig.add_axes([0.92, 0.96, 0.07, 0.03])
    btn = Button(btn_ax, 'Y: symlog', color='lightgoldenrodyellow', hovercolor='khaki')
    btn.label.set_fontsize(8)

    def toggle_scale(event):
        is_symlog[0] = not is_symlog[0]
        if is_symlog[0]:
            for ax, lt, t_s in zip(axes, linthresh_vals, titles_symlog):
                ax.set_yscale('symlog', linthresh=lt)
                ax.set_title(t_s, fontsize=10)
                ax.grid(True, alpha=0.3, which='both')
            btn.label.set_text('Y: symlog')
        else:
            for ax, t_l in zip(axes, titles_linear):
                ax.set_yscale('linear')
                ax.set_title(t_l, fontsize=10)
                ax.grid(True, alpha=0.3)
            autoscale_y_to_visible()
            btn.label.set_text('Y: linear')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_scale)

    # --- Raw Motion panel toggle button ---
    show_raw = [True]
    btn2_ax = fig.add_axes([0.82, 0.96, 0.09, 0.03])
    btn2 = Button(btn2_ax, 'Raw: on', color='lightgoldenrodyellow', hovercolor='khaki')
    btn2.label.set_fontsize(8)

    def toggle_raw(event):
        show_raw[0] = not show_raw[0]
        if show_raw[0]:
            ax4.set_visible(True)
            ax3.set_xlabel('')
            ax3.tick_params(labelbottom=False)
            ax4.set_xlabel(t_label)
            fig.subplots_adjust(hspace=0.30, bottom=0.08)
            btn2.label.set_text('Raw: on')
        else:
            ax4.set_visible(False)
            ax4.set_xlabel('')
            ax3.set_xlabel(t_label)
            ax3.tick_params(labelbottom=True)
            fig.subplots_adjust(hspace=0.30, bottom=0.08)
            btn2.label.set_text('Raw: off')
        fig.canvas.draw_idle()

    btn2.on_clicked(toggle_raw)

    # trigger initial computation (must be after autoscale_y_to_visible is defined)
    on_xlim_changed(ax2)

    if use_clock:
        class AdaptiveTimeFormatter(plt.Formatter):
            def __init__(self, ax):
                self._ax = ax
            def __call__(self, x, pos=None):
                dt = mdates.num2date(x)
                # get all visible major tick locations
                locs = self._ax.xaxis.get_major_locator()()
                # check if any adjacent pair shares the same HH:MM
                need_seconds = False
                for i in range(1, len(locs)):
                    a = mdates.num2date(locs[i - 1])
                    b = mdates.num2date(locs[i])
                    if a.strftime('%H:%M') == b.strftime('%H:%M'):
                        need_seconds = True
                        break
                if need_seconds:
                    return dt.strftime('%H:%M:%S')
                else:
                    return dt.strftime('%H:%M')

        for ax in axes:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(AdaptiveTimeFormatter(ax))
        fig.autofmt_xdate(rotation=0, ha='center')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_motion.py <input.csv> [output.png]")
        sys.exit(1)
    csv_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    load_and_plot(csv_path, save_path)
