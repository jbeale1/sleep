#!/usr/bin/env python3
"""
micro_doppler.py — Micro-Doppler spectrogram from 60 GHz CW radar IQ data.

Usage:
    python micro_doppler.py <input.csv> [options]

Input CSV format (auto-detected):
    4-column:  seq,ms,I,Q       (integer counts, ms is modulo 1000)
    3-column:  elapsed_ms,I,Q   (float volts)
    2-column:  I,Q              (assumes uniform sample rate)

Options:
    --fs RATE         Override sample rate in Hz (default: auto from timestamps,
                      or 141 Hz if no timestamps)
    --fmax FREQ       Max Doppler frequency to display in Hz (default: 5)
    --nfft SIZE       FFT size for spectrogram (default: 1024)
    --overlap FRAC    Overlap fraction 0-0.99 (default: 0.95)
    --dc-window SEC   Adaptive DC removal window in seconds (default: 10)
    --cmap NAME       Matplotlib colormap (default: magma)
    --output FILE     Save plot to file instead of displaying
    --dpi DPI         Output resolution (default: 150)
"""

import argparse
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.signal import spectrogram
from scipy.ndimage import uniform_filter1d


def parse_start_time(filepath):
    """Extract wall-clock start time from filename like IQ_20260228_231702.csv.
    Returns datetime or None."""
    basename = filepath.replace('\\', '/').split('/')[-1]
    m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', basename)
    if m:
        return datetime(*[int(g) for g in m.groups()])
    return None


def load_iq_csv(filepath):
    """Load IQ data from CSV, auto-detecting format.
    Supported formats:
        4-col: seq, ms%1000, I, Q   (new integer format)
        3-col: elapsed_ms, I, Q     (old float format)
        2-col: I, Q                 (no timestamps)
    Returns (I, Q, fs) where fs is None if no timestamps present."""

    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    has_header = not first_line[0].isdigit() and not first_line[0] == '-'

    data = np.genfromtxt(filepath, delimiter=',',
                         skip_header=1 if has_header else 0)

    if data.ndim == 1:
        sys.exit("Error: CSV must have at least 2 columns (I, Q)")

    ncols = data.shape[1]

    if ncols >= 4:
        # seq, ms%1000, I, Q format
        seq = data[:, 0].astype(int)
        ms_raw = data[:, 1].astype(int)
        I = data[:, 2].astype(np.float64)
        Q = data[:, 3].astype(np.float64)

        # Check for dropped samples
        seq_diffs = np.diff(seq)
        gaps = np.where(seq_diffs != 1)[0]
        if len(gaps):
            total_dropped = int(np.sum(seq_diffs[gaps] - 1))
            print(f"WARNING: {total_dropped} dropped samples at {len(gaps)} gap(s)")

        # Reconstruct elapsed time from ms%1000 with rollover tracking
        elapsed_ms = np.zeros(len(ms_raw), dtype=np.float64)
        epoch = 0
        for i in range(1, len(ms_raw)):
            if ms_raw[i] < ms_raw[i-1] - 500:
                epoch += 1000
            elapsed_ms[i] = epoch + ms_raw[i] - ms_raw[0]

        duration_s = elapsed_ms[-1] / 1000.0
        if duration_s > 0:
            fs = (len(I) - 1) / duration_s
        else:
            fs = None
        print(f"Loaded {len(I)} samples, {duration_s:.1f}s, "
              f"measured rate: {fs:.1f} Hz")

    elif ncols >= 3 and has_header:
        # elapsed_ms, I, Q format
        t_ms = data[:, 0]
        I = data[:, 1]
        Q = data[:, 2]
        duration_s = (t_ms[-1] - t_ms[0]) / 1000.0
        if duration_s > 0:
            fs = (len(t_ms) - 1) / duration_s
        else:
            fs = None
        print(f"Loaded {len(I)} samples, {duration_s:.1f}s, "
              f"measured rate: {fs:.1f} Hz")
    elif ncols >= 2:
        I = data[:, 0]
        Q = data[:, 1]
        fs = None
        print(f"Loaded {len(I)} samples (no timestamps)")
    else:
        sys.exit("Error: CSV must have at least 2 columns")

    return I, Q, fs


def make_spectrogram(I, Q, fs, fmax=10, nfft=1024, overlap_frac=0.95,
                     dc_window_sec=10, cmap='magma', start_time=None):
    """Compute and return figure with micro-Doppler spectrogram."""

    N = len(I)
    nfft = min(nfft, N // 4)
    noverlap = int(nfft * overlap_frac)

    # Adaptive DC removal
    win = int(dc_window_sec * fs)
    win = max(win, 3)  # minimum 3 samples
    I_dc = I - uniform_filter1d(I, win)
    Q_dc = Q - uniform_filter1d(Q, win)

    # IQ gain balance
    i_std = np.std(I_dc)
    q_std = np.std(Q_dc)
    if q_std > 0:
        Q_dc = Q_dc * (i_std / q_std)

    S = I_dc + 1j * Q_dc

    f_sp, t_sp, Sxx = spectrogram(S, fs=fs, nperseg=nfft,
                                   noverlap=noverlap,
                                   return_onesided=False,
                                   window='hann')
    Sxx_sh = np.fft.fftshift(Sxx, axes=0)
    f_sh = np.fft.fftshift(f_sp)

    freq_mask = np.abs(f_sh) <= fmax
    Sxx_db = 10 * np.log10(Sxx_sh[freq_mask, :] + 1e-12)
    vmin, vmax = np.percentile(Sxx_db, [5, 97])

    fig, ax = plt.subplots(figsize=(14, 5))

    if start_time is not None:
        # Convert spectrogram time offsets to datetime objects
        t_dt = np.array([start_time + timedelta(seconds=float(s)) for s in t_sp])
        im = ax.pcolormesh(t_dt, f_sh[freq_mask], Sxx_db,
                           shading='gouraud', cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=0, ha='center')
        ax.set_xlabel('Time', fontsize=12)
        time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        im = ax.pcolormesh(t_sp, f_sh[freq_mask], Sxx_db,
                           shading='gouraud', cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('Time (s)', fontsize=12)
        time_str = f'{t_sp[-1]:.0f}s'

    ax.set_ylabel('Doppler Frequency (Hz)', fontsize=12)
    ax.set_title(f'Micro-Doppler Spectrogram — {fs:.1f} Hz, '
                 f'±{fmax} Hz, nfft={nfft}  [{time_str}]', fontsize=13)
    plt.colorbar(im, ax=ax, label='dB')
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Micro-Doppler spectrogram from CW radar IQ data')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('--fs', type=float, default=None,
                        help='Sample rate override (Hz)')
    parser.add_argument('--fmax', type=float, default=5,
                        help='Max Doppler frequency to display (Hz)')
    parser.add_argument('--nfft', type=int, default=1024,
                        help='FFT size')
    parser.add_argument('--overlap', type=float, default=0.95,
                        help='Overlap fraction (0-0.99)')
    parser.add_argument('--dc-window', type=float, default=10,
                        help='DC removal window (seconds)')
    parser.add_argument('--cmap', default='magma',
                        help='Colormap name')
    parser.add_argument('--output', default=None,
                        help='Save to file instead of displaying')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output DPI')
    args = parser.parse_args()

    I, Q, fs_measured = load_iq_csv(args.input)

    fs = args.fs or fs_measured or 141.0
    if args.fs:
        print(f"Using override sample rate: {fs:.1f} Hz")
    elif fs_measured:
        print(f"Using measured sample rate: {fs:.1f} Hz")
    else:
        print(f"No timestamps found, using default: {fs:.1f} Hz")

    start_time = parse_start_time(args.input)
    if start_time:
        print(f"Start time from filename: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    fig = make_spectrogram(I, Q, fs,
                           fmax=args.fmax,
                           nfft=args.nfft,
                           overlap_frac=args.overlap,
                           dc_window_sec=args.dc_window,
                           cmap=args.cmap,
                           start_time=start_time)

    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved: {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
