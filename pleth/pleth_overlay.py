#!/usr/bin/env python3

"""
Pleth waveform overlay — superimposed pulse beats aligned at upstroke.

Loads pleth CSV files from left and right hands, detects individual beats,
aligns them at the systolic upstroke (max dV/dt), and overlays them for
direct visual comparison of waveform morphology.

Usage:
  python pleth_overlay.py <left_pleth.csv> <right_pleth.csv>
  python pleth_overlay.py left_pleth.csv right_pleth.csv --sample-rate 24

J. Beale  2026-02
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import argparse
import sys
from pathlib import Path


def load_pleth(path):
    """Load pleth CSV, return values array."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    values = np.array([int(r['pleth']) for r in rows], dtype=float)
    times = np.array([float(r['elapsed_s']) for r in rows])
    return values, times


def detect_beats(values, sample_rate=24, min_amplitude=30):
    """Detect beats by finding troughs before sharp upstrokes.
    Returns list of (trough_index, peak_index) tuples."""
    n = len(values)
    beats = []

    for i in range(2, n - 2):
        if (values[i] <= values[i-1] and values[i] <= values[i-2] and
                values[i+1] > values[i] and values[i+2] > values[i+1]):
            # Look ahead for peak
            search_end = min(i + int(sample_rate * 0.5), n)
            seg = values[i:search_end]
            peak_offset = np.argmax(seg)
            amplitude = seg[peak_offset] - values[i]
            if amplitude >= min_amplitude and peak_offset >= 1:
                beats.append((i, i + peak_offset))

    return beats


def find_upstroke(values, trough_i, peak_i):
    """Find the point of maximum derivative (steepest rise) between trough and peak."""
    if peak_i - trough_i < 2:
        return trough_i
    seg = values[trough_i:peak_i + 1]
    deriv = np.diff(seg)
    return trough_i + np.argmax(deriv)


def filter_slow_beats(beats, sample_rate=24, slowest_pct=30):
    """Keep only beats in the slowest percentile (longest inter-beat interval).
    Returns filtered list of (trough_i, peak_i) tuples."""
    if len(beats) < 3:
        return beats

    # Compute inter-beat intervals (trough to trough)
    intervals = []
    for i in range(len(beats) - 1):
        dt = (beats[i + 1][0] - beats[i][0]) / sample_rate
        intervals.append((dt, i))

    # Find threshold for slowest N%
    intervals_sorted = sorted(intervals, key=lambda x: x[0], reverse=True)
    n_keep = max(1, int(len(intervals) * slowest_pct / 100))
    keep_indices = set(idx for _, idx in intervals_sorted[:n_keep])

    filtered = [beats[i] for i in keep_indices]
    kept_intervals = [dt for dt, idx in intervals if idx in keep_indices]
    mean_hr = 60.0 / (sum(kept_intervals) / len(kept_intervals))
    print(f"  Slowest {slowest_pct}%: kept {len(filtered)}/{len(beats)} beats "
          f"(avg {mean_hr:.0f} bpm, interval {sum(kept_intervals)/len(kept_intervals)*1000:.0f}ms)")

    return filtered


def extract_aligned_beats(values, beats, sample_rate=24,
                           pre_ms=150, post_ms=800, all_beats=None):
    """Extract beat windows aligned at upstroke point.
    Returns (time_axis_ms, array of shape [n_beats, window_len], aligned normalized)."""
    pre_samp = int(pre_ms / 1000 * sample_rate)
    post_samp = int(post_ms / 1000 * sample_rate)
    window_len = pre_samp + post_samp
    n = len(values)

    extracted = []
    for trough_i, peak_i in beats:
        align_i = find_upstroke(values, trough_i, peak_i)
        start = align_i - pre_samp
        end = align_i + post_samp
        if start < 0 or end >= n:
            continue

        seg = values[start:end].copy()

        # Normalize: 0 at trough, 1 at peak (removes auto-gain/offset)
        trough_val = values[trough_i]
        peak_val = values[peak_i]
        amp = peak_val - trough_val
        if amp < 10:
            continue
        seg = (seg - trough_val) / amp

        extracted.append(seg)

    time_ms = np.linspace(-pre_ms, post_ms, window_len)
    return time_ms, np.array(extracted) if extracted else np.empty((0, window_len))


def main():
    parser = argparse.ArgumentParser(description='Pleth waveform overlay comparison')
    parser.add_argument('left_csv', help='Left hand pleth CSV file')
    parser.add_argument('right_csv', help='Right hand pleth CSV file')
    parser.add_argument('--sample-rate', type=float, default=24,
                        help='Sample rate in Hz (default: 24)')
    parser.add_argument('--pre', type=float, default=150,
                        help='Window before alignment point in ms (default: 150)')
    parser.add_argument('--post', type=float, default=800,
                        help='Window after alignment point in ms (default: 800)')
    parser.add_argument('--max-beats', type=int, default=200,
                        help='Max beats to overlay per hand (default: 200)')
    parser.add_argument('--no-median', action='store_true',
                        help='Hide median template lines')
    parser.add_argument('--slowest', type=int, default=None, metavar='PCT',
                        help='Use only the slowest N%% of beats (e.g. --slowest 30)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to file instead of displaying')
    args = parser.parse_args()

    sr = args.sample_rate

    # Load data
    print(f"Loading left:  {args.left_csv}")
    left_vals, left_times = load_pleth(args.left_csv)
    print(f"  {len(left_vals)} samples, {left_times[-1]:.1f}s")

    print(f"Loading right: {args.right_csv}")
    right_vals, right_times = load_pleth(args.right_csv)
    print(f"  {len(right_vals)} samples, {right_times[-1]:.1f}s")

    # Detect beats
    left_beats = detect_beats(left_vals, sr)
    right_beats = detect_beats(right_vals, sr)
    print(f"Detected: {len(left_beats)} left beats, {len(right_beats)} right beats")

    # Filter for slowest beats if requested
    if args.slowest:
        print(f"Filtering slowest {args.slowest}%:")
        print(f"  Left:", end="")
        left_beats = filter_slow_beats(left_beats, sr, args.slowest)
        print(f"  Right:", end="")
        right_beats = filter_slow_beats(right_beats, sr, args.slowest)

    # Extract aligned windows
    post = args.post
    t_ms, left_waves = extract_aligned_beats(left_vals, left_beats, sr,
                                              args.pre, post)
    _, right_waves = extract_aligned_beats(right_vals, right_beats, sr,
                                            args.pre, post)

    # Limit beats if requested
    if len(left_waves) > args.max_beats:
        step = len(left_waves) // args.max_beats
        left_waves = left_waves[::step][:args.max_beats]
    if len(right_waves) > args.max_beats:
        step = len(right_waves) // args.max_beats
        right_waves = right_waves[::step][:args.max_beats]

    print(f"Overlaying: {len(left_waves)} left, {len(right_waves)} right")

    # Compute medians
    left_median = np.median(left_waves, axis=0) if len(left_waves) > 0 else None
    right_median = np.median(right_waves, axis=0) if len(right_waves) > 0 else None

    # ---- Plotting ----
    fig = plt.figure(figsize=(14, 8), facecolor='black')
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    # Colors
    left_color = '#4488ff'
    right_color = '#ff6644'
    left_med_color = '#88ccff'
    right_med_color = '#ffaa88'
    bg_color = '#0a0a0a'

    left_name = Path(args.left_csv).stem
    right_name = Path(args.right_csv).stem

    # --- Top left: Left hand overlay ---
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_facecolor(bg_color)
    for wave in left_waves:
        ax_left.plot(t_ms, wave, color=left_color, alpha=0.08, linewidth=0.7)
    if left_median is not None and not args.no_median:
        ax_left.plot(t_ms, left_median, color=left_med_color, linewidth=2.5,
                     label=f'Median (n={len(left_waves)})')
        ax_left.legend(loc='upper right', fontsize=9, facecolor='#222222',
                       edgecolor='gray', labelcolor='white')
    ax_left.set_title(f'LEFT — {left_name}', color=left_color, fontsize=12)
    ax_left.set_ylabel('Normalized amplitude', color='gray', fontsize=10)
    ax_left.set_xlabel('Time from upstroke (ms)', color='gray', fontsize=10)
    ax_left.tick_params(colors='gray')
    ax_left.axvline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    ax_left.set_ylim(-0.3, 1.3)
    for spine in ax_left.spines.values():
        spine.set_color('#333333')

    # --- Top right: Right hand overlay ---
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.set_facecolor(bg_color)
    for wave in right_waves:
        ax_right.plot(t_ms, wave, color=right_color, alpha=0.08, linewidth=0.7)
    if right_median is not None and not args.no_median:
        ax_right.plot(t_ms, right_median, color=right_med_color, linewidth=2.5,
                      label=f'Median (n={len(right_waves)})')
        ax_right.legend(loc='upper right', fontsize=9, facecolor='#222222',
                        edgecolor='gray', labelcolor='white')
    ax_right.set_title(f'RIGHT — {right_name}', color=right_color, fontsize=12)
    ax_right.set_ylabel('Normalized amplitude', color='gray', fontsize=10)
    ax_right.set_xlabel('Time from upstroke (ms)', color='gray', fontsize=10)
    ax_right.tick_params(colors='gray')
    ax_right.axvline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    ax_right.set_ylim(-0.3, 1.3)
    for spine in ax_right.spines.values():
        spine.set_color('#333333')

    # --- Bottom left: Median comparison ---
    ax_comp = fig.add_subplot(gs[1, 0])
    ax_comp.set_facecolor(bg_color)
    if left_median is not None:
        ax_comp.plot(t_ms, left_median, color=left_color, linewidth=2, label='Left')
    if right_median is not None:
        ax_comp.plot(t_ms, right_median, color=right_color, linewidth=2, label='Right')
    ax_comp.legend(loc='upper right', fontsize=9, facecolor='#222222',
                   edgecolor='gray', labelcolor='white')
    ax_comp.set_title('Median comparison', color='white', fontsize=11)
    ax_comp.set_ylabel('Normalized', color='gray', fontsize=10)
    ax_comp.set_xlabel('Time from upstroke (ms)', color='gray', fontsize=10)
    ax_comp.tick_params(colors='gray')
    ax_comp.axvline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    ax_comp.set_ylim(-0.3, 1.3)
    for spine in ax_comp.spines.values():
        spine.set_color('#333333')

    # --- Bottom right: Difference + stats ---
    ax_diff = fig.add_subplot(gs[1, 1])
    ax_diff.set_facecolor(bg_color)
    if left_median is not None and right_median is not None:
        diff = left_median - right_median
        ax_diff.fill_between(t_ms, 0, diff, where=(diff > 0),
                             color=left_color, alpha=0.3, label='Left > Right')
        ax_diff.fill_between(t_ms, 0, diff, where=(diff < 0),
                             color=right_color, alpha=0.3, label='Right > Left')
        ax_diff.plot(t_ms, diff, color='white', linewidth=1.5)
        ax_diff.axhline(0, color='gray', linewidth=0.5)
        ax_diff.legend(loc='upper right', fontsize=8, facecolor='#222222',
                       edgecolor='gray', labelcolor='white')

        # Compute and display stats
        # Rise phase: -50ms to +50ms around alignment
        rise_mask = (t_ms >= -50) & (t_ms <= 50)
        # Fall phase: +100ms to +600ms
        fall_mask = (t_ms >= 100) & (t_ms <= 600)

        rise_diff = np.mean(np.abs(diff[rise_mask]))
        fall_diff = np.mean(diff[fall_mask])

        stats_text = (f"Rise phase diff: {rise_diff:.3f}\n"
                      f"Fall phase (L−R): {fall_diff:+.3f}")
        ax_diff.text(0.03, 0.95, stats_text, transform=ax_diff.transAxes,
                     fontsize=9, color='white', fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(facecolor='#222222', edgecolor='gray', alpha=0.8))

    ax_diff.set_title('Difference (Left − Right)', color='white', fontsize=11)
    ax_diff.set_ylabel('Δ Normalized', color='gray', fontsize=10)
    ax_diff.set_xlabel('Time from upstroke (ms)', color='gray', fontsize=10)
    ax_diff.tick_params(colors='gray')
    ax_diff.axvline(0, color='white', alpha=0.3, linewidth=0.5, linestyle='--')
    for spine in ax_diff.spines.values():
        spine.set_color('#333333')

    fig.suptitle('Pleth Waveform Comparison — Left vs Right Hand',
                 color='white', fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        plt.savefig(args.save, dpi=150, facecolor='black')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
