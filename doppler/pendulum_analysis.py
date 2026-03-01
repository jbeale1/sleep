"""
pendulum_analysis.py  —  61 GHz IQ radar pendulum analyser
Usage:  python pendulum_analysis.py <IQ_csv_file> [output_png]
If output_png is omitted the plot is displayed interactively.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

# ── configuration ────────────────────────────────────────────────────────────
F_RADAR   = 61e9          # Hz  (BGT60LTR11AIP nominal)
HP_FC     = 0.40          # Hz  high-pass cutoff to remove slow drift
BP_BW     = 0.05          # Hz  half-bandwidth of narrow bandpass for phase extraction
ENV_LP_FC = 0.30          # Hz  low-pass for envelope smoothing
TRIM_S    = 3.0           # seconds to trim from each end for envelope fit
PHASE_TRIM_BW_FACTOR = 2.5  # trim = factor / (π × BP_BW) — covers filter group delay
SEG_MIN_TOTAL_S  = 300       # minimum record length (s) to attempt segmented Q analysis
SEG_WIN_S        = 240       # sliding window width (s) for local τ fits
SEG_STEP_S       = 60        # step between windows (s)
SEG_MIN_AMP_MM   = 0.15      # reject windows with mean amplitude below this (mm)
SEG_MAX_Q_FACTOR = 3.0       # reject windows where Q > this × median(Q)
SEG_MAX_TAU_ERR  = 0.15      # reject windows where τ_err/τ exceeds this fraction
# ─────────────────────────────────────────────────────────────────────────────

def load_iq(path):
    import csv
    seqs, I_raw, Q_raw = [], [], []
    ms_raw = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)                        # skip header
        for row in reader:
            if len(row) < 4:
                continue
            try:
                seqs.append(int(row[0]))
                ms_raw.append(int(row[1]))
                I_raw.append(int(row[2]))
                Q_raw.append(int(row[3]))
            except ValueError:
                continue
    seqs   = np.asarray(seqs)
    ms_raw = np.asarray(ms_raw)
    I_raw  = np.asarray(I_raw, dtype=np.float64)
    Q_raw  = np.asarray(Q_raw, dtype=np.float64)
    n      = len(seqs)

    # Check for dropped samples via sequence gaps
    seq_diffs = np.diff(seqs)
    gaps = np.where(seq_diffs != 1)[0]
    if len(gaps):
        total_dropped = int(np.sum(seq_diffs[gaps] - 1))
        print(f"  WARNING: {total_dropped} dropped samples at {len(gaps)} gap(s)")

    # Reconstruct elapsed time from ms%1000 with rollover tracking
    elapsed_ms = np.zeros(n, dtype=np.float64)
    epoch = 0
    for i in range(1, n):
        if ms_raw[i] < ms_raw[i-1] - 500:
            epoch += 1000
        elapsed_ms[i] = epoch + ms_raw[i] - ms_raw[0]

    T_total = elapsed_ms[-1] / 1000.0
    fs      = (n - 1) / T_total
    t       = np.arange(n) / fs
    print(f"  {n} samples, {T_total:.1f} s, fs = {fs:.1f} Hz")
    return t, I_raw, Q_raw, fs, T_total

def iq_to_displacement_mm(I_raw, Q_raw):
    lam = 3e8 / F_RADAR
    I   = I_raw - I_raw.mean()
    Q   = Q_raw - Q_raw.mean()
    phase  = np.unwrap(np.arctan2(Q, I))
    return phase * lam / (4 * np.pi) * 1000, lam

def find_fundamental(disp_hp, fs, f_search=(0.5, 2.0)):
    N   = len(disp_hp)
    win = np.hanning(N)
    D   = np.fft.rfft(disp_hp * win)
    frq = np.fft.rfftfreq(N, 1.0/fs)
    amp = np.abs(D) * 2 / win.sum()
    msk = (frq >= f_search[0]) & (frq <= f_search[1])
    f0  = frq[msk][np.argmax(amp[msk])]
    return f0

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def segmented_Q(t_fit, env_fit, f0, fs):
    """
    Slide a window across the envelope and fit τ independently in each window.
    Returns arrays of window-centre time, mean amplitude, and local Q.
    Only windows with a plausible fit are kept.
    """
    win_n  = int(SEG_WIN_S * fs)
    step_n = int(SEG_STEP_S * fs)
    trim   = int(TRIM_S * fs)

    seg_t, seg_A, seg_Q = [], [], []

    i = 0
    while i + win_n <= len(env_fit):
        t_w   = t_fit[i : i + win_n]
        env_w = env_fit[i : i + win_n]
        # Require minimum amplitude (low-SNR windows give garbage τ fits)
        if env_w.mean() < SEG_MIN_AMP_MM:
            i += step_n
            continue
        try:
            # Fit only if there is a visible decay (max/min ratio > 1.02)
            if env_w.max() / (env_w.min() + 1e-9) < 1.02:
                i += step_n
                continue
            # Initial guess: tau from ratio over window
            ratio = env_w[0] / (env_w[-1] + 1e-9)
            tau0  = SEG_WIN_S / np.log(max(ratio, 1.01))
            popt, pcov = curve_fit(
                exp_decay, t_w - t_w[0], env_w,
                p0=[env_w[0], tau0],
                bounds=([0, 10], [np.inf, 1e5]),
                maxfev=2000
            )
            _, tau_w = popt
            tau_err_w = np.sqrt(np.diag(pcov))[1]
            # Reject implausible or poorly constrained fits
            if tau_err_w / tau_w > SEG_MAX_TAU_ERR:
                i += step_n
                continue
            Q_w = np.pi * f0 * tau_w
            seg_t.append(t_w.mean())
            seg_A.append(env_w.mean())
            seg_Q.append(Q_w)
        except (RuntimeError, ValueError):
            pass
        i += step_n

    seg_t = np.array(seg_t); seg_A = np.array(seg_A); seg_Q = np.array(seg_Q)

    # Reject outliers: Q > SEG_MAX_Q_FACTOR × median
    if len(seg_Q) >= 4:
        q_med = np.median(seg_Q)
        keep  = seg_Q <= SEG_MAX_Q_FACTOR * q_med
        seg_t, seg_A, seg_Q = seg_t[keep], seg_A[keep], seg_Q[keep]

    return seg_t, seg_A, seg_Q


def analyse(csv_path):
    t, I_raw, Q_raw, fs, T_total = load_iq(csv_path)
    disp_mm, lam = iq_to_displacement_mm(I_raw, Q_raw)

    # ── high-pass to remove frequency drift ──────────────────────────────────
    sos_hp   = signal.butter(4, HP_FC / (fs/2), btype='high', output='sos')
    disp_hp  = signal.sosfiltfilt(sos_hp, disp_mm)

    # ── find fundamental, narrow bandpass, envelope ───────────────────────────
    f0_coarse = find_fundamental(disp_hp, fs)
    sos_bp    = signal.butter(4,
                    [(f0_coarse - BP_BW*3) / (fs/2),
                     (f0_coarse + BP_BW*3) / (fs/2)],
                    btype='band', output='sos')
    disp_bp   = signal.sosfiltfilt(sos_bp, disp_mm)

    envelope  = np.abs(signal.hilbert(disp_bp))
    sos_lp    = signal.butter(2, ENV_LP_FC / (fs/2), btype='low', output='sos')
    env_smooth = signal.sosfiltfilt(sos_lp, envelope)

    # ── exponential fit to envelope ───────────────────────────────────────────
    trim     = int(TRIM_S * fs)
    t_fit    = t[trim:-trim]
    env_fit  = env_smooth[trim:-trim]
    p0       = [env_fit[0], T_total / 2]
    popt, pcov = curve_fit(exp_decay, t_fit, env_fit, p0=p0, maxfev=10000)
    A0, tau  = popt
    tau_err  = np.sqrt(np.diag(pcov))[1]

    # ── refine f0 from narrow bandpass analytic phase ─────────────────────────
    # Group delay of order-6 Butterworth bandpass ≈ N/(π·BW); trim accordingly.
    phase_trim_s = PHASE_TRIM_BW_FACTOR / (np.pi * BP_BW)
    phase_trim   = int(phase_trim_s * fs)
    sos_narrow = signal.butter(6,
                    [(f0_coarse - BP_BW) / (fs/2),
                     (f0_coarse + BP_BW) / (fs/2)],
                    btype='band', output='sos')
    disp_narrow = signal.sosfiltfilt(sos_narrow, disp_mm)
    analytic    = signal.hilbert(disp_narrow)
    osc_phase   = np.unwrap(np.angle(analytic))

    t_p  = t[phase_trim:-phase_trim]
    op   = osc_phase[phase_trim:-phase_trim]
    coeffs        = np.polyfit(t_p, op, 1)
    omega_fit, _  = coeffs
    f0            = omega_fit / (2 * np.pi)

    Q_val   = np.pi * f0 * tau
    Q_err   = np.pi * f0 * tau_err

    # ── phase residual ────────────────────────────────────────────────────────
    ideal_phase    = np.polyval(coeffs, t_p)
    phase_resid    = op - ideal_phase             # radians
    timing_jitter  = phase_resid / (2 * np.pi * f0)  # seconds

    # ── log-envelope residual for linearity panel ─────────────────────────────
    log_env   = np.log(env_fit[:-trim])
    log_model = np.log(exp_decay(t_fit[:-trim], A0, tau))

    print(f"  f0   = {f0:.5f} Hz")
    print(f"  τ    = {tau:.1f} ± {tau_err:.1f} s")
    print(f"  Q    = {Q_val:.0f} ± {Q_err:.0f}")
    print(f"  A₀   = {A0:.3f} mm  →  A_end = {exp_decay(t[-1], A0, tau):.3f} mm")
    print(f"  Phase residual σ = {phase_resid.std()*1000:.2f} mrad rms")
    print(f"  Timing jitter  σ = {timing_jitter.std()*1000:.3f} ms rms")

    # ── segmented Q (only for long records) ──────────────────────────────────
    seg_t, seg_A, seg_Q = (np.array([]),) * 3
    if T_total >= SEG_MIN_TOTAL_S:
        seg_t, seg_A, seg_Q = segmented_Q(t_fit, env_fit, f0, fs)
        if len(seg_Q):
            print(f"  Q range (segments): {seg_Q.min():.0f} – {seg_Q.max():.0f}"
                  f"  over A = {seg_A.max():.2f} – {seg_A.min():.2f} mm")

    return dict(
        t=t, t_fit=t_fit, t_p=t_p,
        t_log=t_fit[:-trim],
        disp_bp=disp_bp, env_smooth=env_smooth, env_fit=env_fit,
        A0=A0, tau=tau, tau_err=tau_err,
        f0=f0, Q_val=Q_val, Q_err=Q_err,
        log_env=log_env, log_model=log_model,
        phase_resid=phase_resid, timing_jitter=timing_jitter,
        T_total=T_total,
        seg_t=seg_t, seg_A=seg_A, seg_Q=seg_Q,
    )

def plot_results(r, title=''):
    t       = r['t']
    t_fit   = r['t_fit']
    t_p     = r['t_p']
    A0      = r['A0'];  tau = r['tau'];  f0 = r['f0']
    Q_val   = r['Q_val'];  Q_err = r['Q_err'];  tau_err = r['tau_err']

    t_model = np.linspace(t_fit[0], t_fit[-1], 2000)

    seg_t = r['seg_t'];  seg_A = r['seg_A'];  seg_Q = r['seg_Q']
    has_seg = len(seg_Q) >= 4

    nrows = 4 if has_seg else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4*nrows + 1),
                             layout='constrained')
    fig.suptitle(title, fontsize=11, fontweight='bold')

    # ── panel 1: amplitude decay ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, r['disp_bp'],   color='steelblue', lw=0.35, alpha=0.65,
            label='Bandpass displacement')
    ax.plot(t, r['env_smooth'],  color='r', lw=1.5,  label='Envelope')
    ax.plot(t, -r['env_smooth'], color='r', lw=1.5)
    ax.plot(t_model,  exp_decay(t_model, A0, tau), 'k--', lw=1.8,
            label=f'Fit: A₀={A0:.3f} mm, τ={tau:.0f}±{tau_err:.0f} s, Q={Q_val:.0f}±{Q_err:.0f}')
    ax.plot(t_model, -exp_decay(t_model, A0, tau), 'k--', lw=1.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Displacement (mm)')
    end_amp = exp_decay(t[-1], A0, tau)
    ax.set_title(
        f'Amplitude Decay  |  f₀ = {f0:.5f} Hz  |  τ = {tau:.0f} s  |  Q = {Q_val:.0f}\n'
        f'Half-life: {tau*np.log(2):.0f} s ({tau*np.log(2)*f0:.0f} cycles)  |  '
        f'A: {A0:.3f} → {end_amp:.3f} mm'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── panel 2: log envelope ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(r['t_log'], r['log_env'],   color='steelblue', lw=0.8, label='ln(envelope)')
    ax.plot(r['t_log'], r['log_model'], 'k--', lw=1.5,
            label=f'Linear fit  (slope = −1/τ = {-1/tau:.5f} s⁻¹)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ln(Amplitude)')
    ax.set_title('Log Envelope  (linearity confirms single-mode exponential decay)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── panel 3: phase residual ───────────────────────────────────────────────
    ax = axes[2]
    tj_ms = r['timing_jitter'] * 1000
    pr_mrad = r['phase_resid'] * 1000
    ax.plot(t_p, tj_ms, color='darkorange', lw=0.6)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Timing residual (ms)')
    ax.set_title(
        f'Oscillation Phase Residual\n'
        f'σ = {tj_ms.std():.2f} ms rms  ({pr_mrad.std():.1f} mrad rms)'
    )
    ax.grid(True, alpha=0.3)

    # ── panel 4: Q vs amplitude (segmented) ──────────────────────────────────
    if has_seg:
        ax = axes[3]
        sc = ax.scatter(seg_A, seg_Q, c=seg_t, cmap='plasma',
                        s=40, zorder=3, label=f'Local Q ({SEG_WIN_S:.0f} s window)')
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Window centre time (s)', fontsize=9)

        # Fit Q = k/A (quadratic drag model) if amplitude spans > 2×
        if seg_A.max() / seg_A.min() > 1.5:
            try:
                popt_q, _ = curve_fit(lambda A, k: k/A, seg_A, seg_Q,
                                      p0=[seg_Q.mean() * seg_A.mean()])
                k_fit = popt_q[0]
                A_line = np.linspace(seg_A.min()*0.9, seg_A.max()*1.05, 200)
                ax.plot(A_line, k_fit / A_line, 'k--', lw=1.5,
                        label=f'Q = {k_fit:.1f} / A  (quadratic drag)')
            except RuntimeError:
                pass

        ax.axhline(Q_val, color='r', lw=1.2, ls=':', alpha=0.7,
                   label=f'Global fit Q = {Q_val:.0f}')
        ax.set_xlabel('Mean amplitude in window (mm)')
        ax.set_ylabel('Q')
        ax.set_title('Local Q vs Amplitude  '
                     f'(range: {seg_Q.min():.0f} – {seg_Q.max():.0f}  '
                     f'over A = {seg_A.max():.2f} – {seg_A.min():.2f} mm)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    return fig

def main():
    if len(sys.argv) < 2:
        print("Usage: python pendulum_analysis.py <IQ_csv_file> [output_png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None

    import os
    print(f"Analysing: {csv_path}")
    results = analyse(csv_path)

    title = (f'61 GHz Radar — Pendulum  |  {os.path.basename(csv_path)}'
             f'  |  {results["T_total"]/60:.1f} min  |  Q = {results["Q_val"]:.0f}')
    fig = plot_results(results, title=title)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
