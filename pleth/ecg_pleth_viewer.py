#!/usr/bin/env python3
"""
ECG + Pleth Waveform Viewer
----------------------------
Displays HP-filtered ECG and pleth waveforms on a shared time axis (PST).

Usage:
    python ecg_pleth_viewer.py <data_directory>
"""

import sys
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime

from sleep_common import (PST, ECG_FS, PLETH_FS, PLETH_UP_FS, 
                          load_and_process, highpass_filter, upsample_pleth,
                          detect_pulse_feet)

DEFAULT_WINDOW_S = 10
DEFAULT_HP_CUTOFF = 0.5

def main():
    if len(sys.argv) != 2:
        print("Usage: python ecg_pleth_viewer.py <data_directory>")
        sys.exit(1)

    d = load_and_process(sys.argv[1])
    pleth_t, pleth_v = d['pleth_t'], d['pleth_v']
    ecg_t = d['ecg_t']
    ecg_clean = d['ecg_clean']
    t_start, t_end, overlap_s = d['t_start'], d['t_end'], d['overlap_s']

    # Upsample pleth for precise trough detection
    print("Upsampling pleth for trough detection...")
    pleth_t_up, pleth_v_up = upsample_pleth(pleth_t, pleth_v)

    ecg_filt = d['ecg_filt']

    # ── Plot setup ──────────────────────────────────────────────────────
    fig, (ax_ecg, ax_pleth) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.subplots_adjust(bottom=0.28, top=0.90, hspace=0.08)

    state = {
        'center': (t_start + t_end) / 2,
        'window': DEFAULT_WINDOW_S,
        'offset': 0.0,
        'hp_cutoff': DEFAULT_HP_CUTOFF,
        'ecg_filt': ecg_filt,
    }

    line_ecg, = ax_ecg.plot([], [], 'b-', linewidth=0.5)
    line_pleth, = ax_pleth.plot([], [], 'r-', linewidth=0.7)
    rpeak_dots, = ax_ecg.plot([], [], 'rv', markersize=6, linestyle='none')
    ptrough_dots, = ax_pleth.plot([], [], 'g^', markersize=6, linestyle='none')
    ptt_texts = []
    rr_texts = []

    ax_ecg.set_ylabel('ECG (µV, HP filtered)')
    ax_pleth.set_ylabel('Pleth')
    ax_pleth.set_xlabel('Time (PST)')

    def format_time(t, pos):
        try:
            return datetime.fromtimestamp(t, PST).strftime('%H:%M:%S')
        except:
            return ''
    ax_pleth.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

    title = fig.suptitle('', fontsize=9)

    def update_plot():
        c = state['center']
        w = state['window']
        off = state['offset']
        t0, t1 = c - w/2, c + w/2

        for obj in ptt_texts:
            obj.remove()
        ptt_texts.clear()
        for obj in rr_texts:
            obj.remove()
        rr_texts.clear()
        ptt_values = []

        # ECG
        emask = (ecg_t >= t0) & (ecg_t <= t1)
        r_times = np.array([])
        if emask.any():
            et, ev = ecg_t[emask], state['ecg_filt'][emask]
            line_ecg.set_data(et, ev)
            ax_ecg.set_xlim(t0, t1)
            margin = (ev.max() - ev.min()) * 0.05 + 1
            ax_ecg.set_ylim(ev.min() - margin, ev.max() + margin)
            if len(ev) > ECG_FS * 0.5:
                rpk_idx, _ = find_peaks(ev, height=ev.max() * 0.4,
                                        distance=int(ECG_FS * 0.4))
                r_times, r_vals = et[rpk_idx], ev[rpk_idx]
                rpeak_dots.set_data(r_times, r_vals)
                # Annotate R-R intervals between consecutive peaks
                if len(r_times) > 1:
                    for i in range(len(r_times) - 1):
                        rr_ms = (r_times[i+1] - r_times[i]) * 1000
                        mid_t = (r_times[i] + r_times[i+1]) / 2
                        # Place text near top of ECG panel
                        # Place text just below the R-peak markers, between them
                        y_pos = min(r_vals[i], r_vals[i+1]) * 0.95
                        txt = ax_ecg.text(
                            mid_t, y_pos, f'{rr_ms:.0f}',
                            fontsize=7, color='darkred', ha='center', va='top')
                        rr_texts.append(txt)
            else:
                rpeak_dots.set_data([], [])
        else:
            rpeak_dots.set_data([], [])

        # Pleth — draw original samples, detect pulse feet on upsampled
        pleth_t_shifted = pleth_t + off
        pmask = (pleth_t_shifted >= t0) & (pleth_t_shifted <= t1)
        p_foot_times = np.array([])
        p_foot_vals = np.array([])
        if pmask.any():
            pt, pv = pleth_t_shifted[pmask], pleth_v[pmask]
            line_pleth.set_data(pt, pv)
            margin = (pv.max() - pv.min()) * 0.05 + 1
            ax_pleth.set_ylim(pv.min() - margin, pv.max() + margin)
            # Pulse foot detection on upsampled pleth
            pt_up_shifted = pleth_t_up + off
            upmask = (pt_up_shifted >= t0) & (pt_up_shifted <= t1)
            if upmask.any():
                pt_up = pt_up_shifted[upmask]
                pv_up = pleth_v_up[upmask]
                if len(pv_up) > PLETH_UP_FS * 0.5:
                    p_foot_times, p_foot_vals = detect_pulse_feet(
                        pt_up, pv_up, PLETH_UP_FS)
                    ptrough_dots.set_data(p_foot_times, p_foot_vals)
                else:
                    ptrough_dots.set_data([], [])
            else:
                ptrough_dots.set_data([], [])
        else:
            ptrough_dots.set_data([], [])

        # PTT measurement
        if len(r_times) > 0 and len(p_foot_times) > 0:
            for ft_time, ft_val in zip(p_foot_times, p_foot_vals):
                preceding = r_times[r_times < ft_time]
                if len(preceding) > 0:
                    ptt_ms = (ft_time - preceding[-1]) * 1000
                    if 50 < ptt_ms < 800:
                        ptt_values.append(ptt_ms)
                        txt = ax_pleth.text(
                            ft_time, ft_val + (pv.max() - pv.min()) * 0.08,
                            f'{ptt_ms:.0f}',
                            fontsize=7, color='green', ha='center', va='bottom')
                        ptt_texts.append(txt)

        # R-R stats
        rr_str = ""
        if len(r_times) > 1:
            rr_ms = np.diff(r_times) * 1000
            rr_ms = rr_ms[rr_ms > 300]
            if len(rr_ms) > 0:
                hr_bpm = 60000.0 / np.mean(rr_ms)
                rr_str = (f"R-R: {np.mean(rr_ms):.0f}±{np.std(rr_ms):.0f} ms  "
                          f"({hr_bpm:.0f} bpm, n={len(rr_ms)})")

        ptt_str = ""
        if ptt_values:
            ptt_str = f"PTT: {np.mean(ptt_values):.0f}±{np.std(ptt_values):.0f} ms (n={len(ptt_values)})"

        dt_center = datetime.fromtimestamp(c, PST)
        line1 = (f"{dt_center:%Y-%m-%d %H:%M:%S} PST  |  "
                 f"Window: {w:.1f}s  |  "
                 f"Offset: {off:+.3f}s  |  "
                 f"HP: {state['hp_cutoff']:.2f} Hz")
        stats = "  |  ".join(filter(None, [rr_str, ptt_str]))
        title.set_text(f"{line1}\n{stats}" if stats else line1)
        fig.canvas.draw_idle()

    # ── Sliders ─────────────────────────────────────────────────────────
    ax_pos = fig.add_axes([0.15, 0.15, 0.70, 0.025])
    ax_win = fig.add_axes([0.15, 0.11, 0.70, 0.025])
    ax_off = fig.add_axes([0.15, 0.07, 0.70, 0.025])
    ax_hp  = fig.add_axes([0.15, 0.03, 0.70, 0.025])

    sl_pos = Slider(ax_pos, 'Position', t_start, t_end,
                    valinit=state['center'], valstep=0.1)
    sl_win = Slider(ax_win, 'Window (s)', 1, min(120, overlap_s),
                    valinit=DEFAULT_WINDOW_S, valstep=0.5)
    sl_off = Slider(ax_off, 'Offset (s)', -10, 10,
                    valinit=0.0, valstep=0.001)
    sl_hp  = Slider(ax_hp, 'HP cutoff', 0.05, 5.0,
                    valinit=DEFAULT_HP_CUTOFF, valstep=0.05)

    sl_pos.valtext.set_text(datetime.fromtimestamp(state['center'], PST).strftime('%H:%M:%S'))

    def on_pos(val):
        state['center'] = val
        sl_pos.valtext.set_text(datetime.fromtimestamp(val, PST).strftime('%H:%M:%S'))
        update_plot()
    def on_win(val):
        state['window'] = val
        update_plot()
    def on_off(val):
        state['offset'] = val
        update_plot()
    def on_hp(val):
        state['hp_cutoff'] = val
        print(f"Re-filtering ECG at {val:.2f} Hz...")
        state['ecg_filt'] = highpass_filter(ecg_clean, ECG_FS, val)
        update_plot()

    sl_pos.on_changed(on_pos)
    sl_win.on_changed(on_win)
    sl_off.on_changed(on_off)
    sl_hp.on_changed(on_hp)

    # ── Keyboard / scroll navigation ────────────────────────────────────
    def on_scroll(event):
        step = state['window'] * 0.25
        if event.button == 'up':
            state['center'] = min(state['center'] + step, t_end)
        else:
            state['center'] = max(state['center'] - step, t_start)
        sl_pos.set_val(state['center'])

    def on_key(event):
        step = state['window'] * 0.25
        if event.key == 'right':
            state['center'] = min(state['center'] + step, t_end)
            sl_pos.set_val(state['center'])
        elif event.key == 'left':
            state['center'] = max(state['center'] - step, t_start)
            sl_pos.set_val(state['center'])
        elif event.key == 'up':
            state['window'] = max(1, state['window'] / 1.5)
            sl_win.set_val(state['window'])
        elif event.key == 'down':
            state['window'] = min(overlap_s, state['window'] * 1.5)
            sl_win.set_val(state['window'])

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    update_plot()
    print("\n── Controls ──")
    print("  Left/Right arrows or scroll: pan in time")
    print("  Up/Down arrows: zoom in/out")
    print("  Sliders: position, window size, pleth offset, HP cutoff")
    plt.show()

if __name__ == '__main__':
    main()
