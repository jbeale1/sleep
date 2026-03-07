#!/usr/bin/env python3
"""
pendulum_iq_deviation.py

Read an IQ CSV (seq, ms, I, Q), estimate Doppler velocity v(t) robustly
(no phase unwrap), then plot slow deviations in frequency and amplitude
relative to nominal f0 and A0.

Usage:
  python pendulum_iq_deviation.py path/to/IQ_*.csv --f0 0.32192 --A0 0.573

Notes:
- Assumes sample rate is fixed; default fs=142.34 sps (your stated value).
- Uses "seq" to build time: t = (seq - seq0)/fs (ignores ms except for info).
- Works best if IQ has decent SNR; points near origin are gated/downweighted.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def butter_sos_lowpass(fc_hz: float, fs: float, order: int = 4):
    if fc_hz <= 0 or fc_hz >= fs / 2:
        raise ValueError(f"Lowpass cutoff fc={fc_hz} must be in (0, fs/2).")
    return signal.butter(order, fc_hz / (fs / 2), btype="low", output="sos")


def whiten_iq(I: np.ndarray, Q: np.ndarray):
    """Ellipse-correct by whitening the 2D covariance (DC removed)."""
    X = np.vstack([I - np.mean(I), Q - np.mean(Q)]).T
    cov = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(cov)
    # Protect against tiny eigenvalues
    w = np.maximum(w, 1e-12)
    Winv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    Xw = X @ Winv_sqrt.T
    return Xw[:, 0], Xw[:, 1]


def robust_omega_from_complex(z: np.ndarray, fs: float, eps_scale: float = 1e-3):
    """
    omega(t) = Im(conj(z)*dz/dt) / (|z|^2 + eps)
    Uses eps regularization to reduce blow-up near the origin.
    """
    dzdt = np.gradient(z, 1.0 / fs)
    power = np.abs(z) ** 2
    eps = eps_scale * np.median(power)
    omega = np.imag(np.conj(z) * dzdt) / (power + eps)
    return omega


def interp_nans(y: np.ndarray):
    """Linear-interpolate NaNs."""
    y = y.copy()
    n = len(y)
    x = np.arange(n)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.zeros_like(y)
    y[~good] = np.interp(x[~good], x[good], y[good])
    return y

def robust_ln_fit(t, A, tmin=30.0, tmax=None, mad_k=4.0):
    """
    Fit ln(A) = b + m t on a selected interval, with optional MAD outlier rejection.
    Returns (m, b, tau, mask_used).
    """
    A = np.asarray(A)
    t = np.asarray(t)

    mask = np.isfinite(A) & (A > 0)
    mask &= (t >= tmin)
    if tmax is not None:
        mask &= (t <= tmax)

    tt = t[mask]
    yy = np.log(A[mask])

    if len(tt) < 10:
        raise ValueError("Not enough samples for a reliable decay fit.")

    # Robust outlier rejection in ln(A) (one pass)
    med = np.median(yy)
    mad = np.median(np.abs(yy - med)) + 1e-12
    good = np.abs(yy - med) < (mad_k * 1.4826 * mad)

    tt2 = tt[good]
    yy2 = yy[good]

    # Linear fit
    m, b = np.polyfit(tt2, yy2, 1)
    tau = -1.0 / m
    return m, b, tau, mask, good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="IQ CSV with columns: seq, ms, I, Q")
    ap.add_argument("--fs", type=float, default=142.34, help="Sample rate (sps)")
    ap.add_argument("--f0", type=float, default=0.32192, help="Nominal frequency (Hz)")
    ap.add_argument("--A0", type=float, default=5.3, help="Nominal displacement amplitude (mm)")
    ap.add_argument("--rf_ghz", type=float, default=61.0, help="Radar RF center frequency (GHz)")
    ap.add_argument("--iq_lp_hz", type=float, default=5.0,
                    help="Lowpass cutoff for complex IQ before differentiating (Hz)")
    ap.add_argument("--env_lp_hz", type=float, default=0.05,
                    help="Lowpass cutoff for complex envelope V(t) after demod (Hz). "
                         "Set based on how slow you expect drift (e.g., 0.02–0.2).")
    ap.add_argument("--last-hours", type=float, default=None,
                    help="If set, only use the last X hours of data from the input CSV (floating-point hours).")
    ap.add_argument("--first-hours", type=float, default=None,
                    help="If set, only use the first X hours of data from the input CSV (floating-point hours). Mutually exclusive with --last-hours.")
    ap.add_argument("--gate_abs", type=float, default=0.20,
                    help="Gate threshold on |z| (after whitening+IQ LP). Below this, "
                         "omega is marked NaN then interpolated.")
    ap.add_argument("--show_v_preview", action="store_true",
                    help="Also show a quick plot of v(t) with nominal sinusoid.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Optionally trim to first/last X hours of data (based on sample rate fs)
    if args.first_hours is not None and args.last_hours is not None:
        raise ValueError("--first-hours and --last-hours are mutually exclusive")

    if args.first_hours is not None:
        if args.first_hours <= 0:
            raise ValueError("--first-hours must be positive")
        n_samples = int(round(float(args.first_hours) * 3600.0 * float(args.fs)))
        if n_samples < 2:
            raise ValueError("--first-hours yields fewer than 2 samples; increase value")
        if len(df) > n_samples:
            df = df.head(n_samples).reset_index(drop=True)
            print(f"Using first {args.first_hours} hours -> {n_samples} samples")

    elif args.last_hours is not None:
        if args.last_hours <= 0:
            raise ValueError("--last-hours must be positive")
        # compute number of samples corresponding to requested hours
        n_samples = int(round(float(args.last_hours) * 3600.0 * float(args.fs)))
        if n_samples < 2:
            raise ValueError("--last-hours yields fewer than 2 samples; increase value")
        if len(df) > n_samples:
            df = df.tail(n_samples).reset_index(drop=True)
            print(f"Using last {args.last_hours} hours -> {n_samples} samples")

    for col in ("seq", "I", "Q"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.csv}")

    seq = df["seq"].to_numpy()
    fs = float(args.fs)
    t = (seq - seq[0]) / fs

    I = df["I"].to_numpy(dtype=float)
    Q = df["Q"].to_numpy(dtype=float)

    # --- 1) ellipse correct / whiten ---
    Iw, Qw = whiten_iq(I, Q)
    z = Iw + 1j * Qw

    # --- 2) lowpass complex IQ to reduce HF noise ---
    sos_iq = butter_sos_lowpass(args.iq_lp_hz, fs, order=4)
    zf = signal.sosfiltfilt(sos_iq, z)

    # --- Estimate average carrier frequency f0 using an FFT peak (robust, positive) ---
    try:
        N = len(zf)
        if N < 16:
            raise RuntimeError("Too few samples for FFT-based f0 estimation")
        # Window the data to reduce spectral leakage
        win = np.hanning(N)
        Z = np.fft.fft(zf * win)
        freqs = np.fft.fftfreq(N, d=1.0 / fs)
        # consider only positive frequencies
        pos = freqs > 0
        fpos = freqs[pos]
        mag = np.abs(Z[pos])

        # Limit search to a band around the nominal args.f0 (avoid DC and very high noise)
        low = max(0.001, args.f0 * 0.5)
        high = max(0.1, args.f0 * 2.0)
        band_mask = (fpos >= low) & (fpos <= high)
        if not np.any(band_mask):
            raise RuntimeError("No frequency bins in search band for f0 estimation")

        mag_band = mag[band_mask]
        f_band = fpos[band_mask]

        # Prefer the peak closest to the nominal args.f0 to avoid picking harmonics
        idx_closest = int(np.argmin(np.abs(f_band - args.f0)))

        # Quadratic interpolation around the closest bin (if possible)
        if 0 < idx_closest < (len(mag_band) - 1):
            alpha = mag_band[idx_closest - 1]
            beta = mag_band[idx_closest]
            gamma = mag_band[idx_closest + 1]
            denom = (alpha - 2 * beta + gamma)
            if denom == 0:
                delta = 0.0
            else:
                delta = 0.5 * (alpha - gamma) / denom
            df = f_band[1] - f_band[0]
            f0_est = f_band[idx_closest] + delta * df
        else:
            f0_est = f_band[idx_closest]

        # If the chosen closest peak is suspiciously far from args.f0, warn and optionally
        # fall back to the strongest peak in the band. Here we simply warn.
        if abs(f0_est - args.f0) > 0.5 * args.f0:
            # find the strongest band peak as fallback
            peak_idx = int(np.argmax(mag_band))
            f_peak = f_band[peak_idx]
            print(f"Warning: closest peak {f0_est:.6f} Hz far from nominal {args.f0:.6f} Hz; strongest band peak at {f_peak:.6f} Hz")

        if f0_est <= 0:
            raise RuntimeError(f"Non-positive f0_est={f0_est}")

        print(f"Estimated f0 from FFT peak: {f0_est:.8f} Hz")
        f0 = float(f0_est)
    except Exception as e:
        print(f"f0 estimation failed, using --f0={args.f0}: {e}")
        f0 = float(args.f0)

    # --- 3) omega(t) without unwrapping, with gating near origin ---
    omega = robust_omega_from_complex(zf, fs, eps_scale=1e-3)

    abs_z = np.abs(zf)
    omega_g = omega.copy()
    omega_g[abs_z < args.gate_abs] = np.nan
    omega_i = interp_nans(omega_g)

    # --- 4) convert omega -> velocity (mm/s) ---
    c = 299_792_458.0
    f_hz = args.rf_ghz * 1e9
    lam_mm = (c / f_hz) * 1000.0
    v_mm_s = omega_i * (lam_mm / (4.0 * np.pi))

    # --- 5) Complex demodulation at f0 to get slow envelope of v(t) ---
    # v(t) ~ Re{ V(t) * exp(+j*2π f0 t) }
    # so V(t) ~ 2 * LPF( v(t) * exp(-j*2π f0 t) )
    carrier = np.exp(-1j * 2.0 * np.pi * f0 * t)
    vbb = v_mm_s * carrier

    sos_env = butter_sos_lowpass(args.env_lp_hz, fs, order=4)
    V = 2.0 * signal.sosfiltfilt(sos_env, vbb)  # complex envelope of velocity

    # amplitude of velocity & convert to displacement amplitude
    Vamp = np.abs(V)                      # mm/s
    w0 = 2.0 * np.pi * f0
    A_est_mm = Vamp / w0                  # mm

    # instantaneous frequency deviation from envelope phase:
    # f_inst = f0 + (1/2π) d/dt arg(V)
    phi = np.unwrap(np.angle(V))
    dphi_dt = np.gradient(phi, 1.0 / fs)
    df_hz = dphi_dt / (2.0 * np.pi)
    # instantaneous frequency is nominal f0 (estimated above) plus deviation
    f_inst = f0 + df_hz

    # deviations relative to nominal
    dA_pct = 100.0 * (A_est_mm - args.A0) / args.A0
    df_mHz = 1000.0 * (f_inst - f0)

    # Optional: fit an exponential decay to A_est_mm to estimate a time constant tau
    # --- Fit amplitude decay and compute Q ---
    fit_tmin = 30.0      # or expose as --fit_tmin
    fit_tmax = None      # or expose as --fit_tmax

    m, b, tau, mask_fit, good_fit = robust_ln_fit(t, A_est_mm, tmin=fit_tmin, tmax=fit_tmax, mad_k=4.0)

    # f0 is set to estimated value above (or fallback to args.f0)
    Q_val = np.pi * f0 * tau

    print(f"Decay fit window: t >= {fit_tmin:.1f} s" + ("" if fit_tmax is None else f", t <= {fit_tmax:.1f} s"))
    print(f"Fit slope m = {m:.6e} 1/s  =>  tau = {tau:.1f} s")
    print(f"Q = pi * f0 * tau = {Q_val:.0f}   (f0={f0:.7f} Hz)",end="")
    print(f" (A = {A_est_mm[mask_fit][good_fit].max():.4f} -> {A_est_mm[mask_fit][good_fit].min():.4f} mm)")



    # --- Plots ---
    # --- Frequency deviation plot with linear trend line ---
    plt.figure()
    plt.plot(t, df_mHz, linewidth=1.0, label='Δf (mHz)')
    # linear trend fit (ignore NaNs)
    good = np.isfinite(df_mHz)
    if good.sum() >= 2:
        slope, intercept = np.polyfit(t[good], df_mHz[good], 1)
        trend = slope * t + intercept
        plt.plot(t, trend, '--', color='red', linewidth=1.2, label=f'Linear trend: {slope:.3e} mHz/s')
        total_drift_mHz = slope * (t[-1] - t[0])
        plt.annotate(f'Trend {slope:.3e} mHz/s\nTotal Δ {total_drift_mHz:.2e} mHz', xy=(0.02, 0.95), xycoords='axes fraction', color='red', fontsize=9, verticalalignment='top')

    plt.title(f"Frequency deviation vs nominal f0={f0:.8f} Hz")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Δf (mHz)")
    plt.ylim(-np.nanmax(np.abs(df_mHz))*1.1, np.nanmax(np.abs(df_mHz))*1.1)  # symmetric
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.plot(t, dA_pct, linewidth=1.0)
    plt.title(f"Amplitude deviation vs nominal A0={args.A0:.3f} mm")
    plt.xlabel("Time (s)")
    plt.ylabel("ΔA (%)")
    plt.ylim(-np.nanmax(np.abs(dA_pct))*1.1, np.nanmax(np.abs(dA_pct))*1.1)  # symmetric
    plt.grid(True, alpha=0.3)

    # Optional preview of v(t) against the nominal sinusoid amplitude
    if args.show_v_preview:
        # nominal velocity amplitude from A0
        Vamp0 = args.A0 * w0
        v_nom = Vamp0 * np.sin(2.0 * np.pi * args.f0 * t)
        plt.figure()
        plt.plot(t, v_mm_s, linewidth=0.8, label="v(t) estimated")
        plt.plot(t, v_nom, linewidth=1.0, label="nominal v(t) from f0,A0")
        plt.title("Velocity estimate vs nominal")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (mm/s)")
        plt.legend()
        plt.grid(True, alpha=0.3)


    # Plot amplitude and fitted exponential
    tt = t[mask_fit][good_fit]
    A_fit = np.exp(b + m * tt)

    plt.figure()
    plt.plot(t, A_est_mm, linewidth=0.8, label="A_est(t) from envelope (mm)")
    plt.plot(tt, A_fit, linewidth=2.0, label=f"exp fit (tau={tau:.0f} s, Q={Q_val:.0f})")
    plt.axvline(fit_tmin, linestyle="--", linewidth=1.0)
    plt.title("Amplitude envelope and exponential decay fit")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mm)")
    plt.legend()
    plt.grid(True, alpha=0.3)    

    # Optional: plot ln(A) with fit line (useful diagnostic)
    plt.figure()
    plt.plot(t[mask_fit], np.log(A_est_mm[mask_fit]), ".", markersize=2, label="ln(A) used (pre-outlier)")
    plt.plot(tt, b + m * tt, linewidth=2.0, label="linear fit in ln(A)")
    plt.axvline(fit_tmin, linestyle="--", linewidth=1.0)
    plt.title("ln(Amplitude) linearized decay fit")
    plt.xlabel("Time (s)")
    plt.ylabel("ln(A)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
    