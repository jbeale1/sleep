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
    ap.add_argument("--A0", type=float, default=0.573, help="Nominal displacement amplitude (mm)")
    ap.add_argument("--rf_ghz", type=float, default=61.0, help="Radar RF center frequency (GHz)")
    ap.add_argument("--iq_lp_hz", type=float, default=5.0,
                    help="Lowpass cutoff for complex IQ before differentiating (Hz)")
    ap.add_argument("--env_lp_hz", type=float, default=0.05,
                    help="Lowpass cutoff for complex envelope V(t) after demod (Hz). "
                         "Set based on how slow you expect drift (e.g., 0.02–0.2).")
    ap.add_argument("--gate_abs", type=float, default=0.20,
                    help="Gate threshold on |z| (after whitening+IQ LP). Below this, "
                         "omega is marked NaN then interpolated.")
    ap.add_argument("--show_v_preview", action="store_true",
                    help="Also show a quick plot of v(t) with nominal sinusoid.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

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
    carrier = np.exp(-1j * 2.0 * np.pi * args.f0 * t)
    vbb = v_mm_s * carrier

    sos_env = butter_sos_lowpass(args.env_lp_hz, fs, order=4)
    V = 2.0 * signal.sosfiltfilt(sos_env, vbb)  # complex envelope of velocity

    # amplitude of velocity & convert to displacement amplitude
    Vamp = np.abs(V)                      # mm/s
    w0 = 2.0 * np.pi * args.f0
    A_est_mm = Vamp / w0                  # mm

    # instantaneous frequency deviation from envelope phase:
    # f_inst = f0 + (1/2π) d/dt arg(V)
    phi = np.unwrap(np.angle(V))
    dphi_dt = np.gradient(phi, 1.0 / fs)
    df_hz = dphi_dt / (2.0 * np.pi)
    f_inst = args.f0 + df_hz

    # deviations relative to nominal
    dA_pct = 100.0 * (A_est_mm - args.A0) / args.A0
    df_mHz = 1000.0 * (f_inst - args.f0)

    # Optional: fit an exponential decay to A_est_mm to estimate a time constant tau
    # --- Fit amplitude decay and compute Q ---
    fit_tmin = 30.0      # or expose as --fit_tmin
    fit_tmax = None      # or expose as --fit_tmax

    m, b, tau, mask, good = robust_ln_fit(t, A_est_mm, tmin=fit_tmin, tmax=fit_tmax, mad_k=4.0)

    f0 = args.f0
    Q_val = np.pi * f0 * tau

    print(f"Decay fit window: t >= {fit_tmin:.1f} s" + ("" if fit_tmax is None else f", t <= {fit_tmax:.1f} s"))
    print(f"Fit slope m = {m:.6e} 1/s  =>  tau = {tau:.1f} s")
    print(f"Q = pi * f0 * tau = {Q_val:.0f}   (f0={f0:.5f} Hz)")



    # --- Plots ---
    plt.figure()
    plt.plot(t, df_mHz, linewidth=1.0)
    plt.title(f"Frequency deviation vs nominal f0={args.f0:.5f} Hz")
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
    tt = t[mask][good]
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
    plt.plot(t[mask], np.log(A_est_mm[mask]), ".", markersize=2, label="ln(A) used (pre-outlier)")
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
    