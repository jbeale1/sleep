"""
pendulum_analysis.py  —  61 GHz IQ radar pendulum analyser
Usage:  python pendulum_analysis.py <IQ_csv_file> [output_png]
If output_png is omitted the plot is displayed interactively.
"""


VERSION = "1.9.7-peek-fix"
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

# ── configuration ────────────────────────────────────────────────────────────
F_RADAR   = 61e9          # Hz  (BGT60LTR11AIP nominal)
HP_FC     = 0.10          # Hz  high-pass cutoff to remove slow drift
BP_BW     = 0.05          # Hz  half-bandwidth of narrow bandpass for phase extraction
ENV_LP_FC = 0.15          # Hz  low-pass for envelope smoothing
TRIM_S    = 3.0           # seconds to trim from each end for envelope fit
PHASE_TRIM_BW_FACTOR = 2.5  # trim = factor / (π × BP_BW) — covers filter group delay
SEG_MIN_TOTAL_S  = 300       # minimum record length (s) to attempt segmented Q analysis
SEG_WIN_FRAC     = 0.10      # window width as fraction of record length (min 240 s, max 3600 s)
SEG_STEP_FRAC    = 0.025     # step size as fraction of record length (min 60 s, max 900 s)
SEG_MIN_AMP_MM   = 0.15      # reject windows with mean amplitude below this (mm)
SEG_MAX_Q_FACTOR = 3.0       # reject windows where Q > this × median(Q)
SEG_MAX_TAU_ERR  = 0.15      # reject windows where τ_err/τ exceeds this fraction

# ── large-record streaming configuration ──────────────────────────────────────
LARGE_RECORD_HOURS   = 2.0        # hours — use analyse_large() above this
CHUNK_MINUTES        = 5.0        # minutes per streaming chunk
HOURLY_CHUNKS        = 12         # chunks per hour (= 60 / CHUNK_MINUTES)
CUTOFF_SNR           = 5.0        # stop when chunk amplitude < this × noise floor
NOISE_FLOOR_INIT_CHUNKS = 6       # first N chunks used to estimate global noise floor

# ── noise analysis configuration ─────────────────────────────────────────────
NOISE_SMOOTH_S   = 20.0      # seconds — smoothing window for envelope/phase
NOISE_N_HARMONICS = 10       # number of harmonics to model (fundamental + this many)
NOISE_BP_BW_MULT = 0.15      # half-bandwidth of each harmonic bandpass, as fraction of f0
PHASE_TRACK_SMOOTH_S = 5.0   # smoothing window (s) — ~1–2 oscillation periods;
                              # tracks amplitude-envelope frequency variation while
                              # averaging out EKF sample noise
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
    """First-pass displacement via naive unwrap. Used for f0/Q/tau fitting only."""
    lam = 3e8 / F_RADAR
    I   = I_raw - I_raw.mean()
    Q   = Q_raw - Q_raw.mean()
    phase  = np.unwrap(np.arctan2(Q, I))
    return phase * lam / (4 * np.pi) * 1000, lam


def correct_iq_imbalance(I_raw, Q_raw):
    """
    Estimate and correct IQ gain imbalance and quadrature phase error.

    Real homodyne receivers have unequal I/Q gains and a quadrature angle
    that is not exactly 90°.  These cause the IQ phasor to trace an ellipse
    instead of a circle, creating a systematic displacement error that repeats
    identically on every oscillation cycle (visible as a deterministic residual
    at the oscillation frequency and its harmonics).

    The correction is estimated from high-amplitude samples, where the phasor
    fully traverses the I/Q plane.  For a pendulum with swing >> λ/2 (true
    here: ~8 mm ≈ 3.3 × λ/2), the phasor makes multiple complete rotations
    per half-swing, so the I and Q channels are approximately uniformly sampled
    over the circle — giving an unbiased estimate of the imbalance.

    Correction model applied:
        I_cor = I                          (I channel is reference)
        Q_cor = (Q − cross × I) × (g_I/g_Q)   (remove I→Q leak, equalize gain)

    Parameters
    ----------
    I_raw, Q_raw : raw ADC arrays (with DC bias)

    Returns
    -------
    I_cor_raw, Q_cor_raw : corrected arrays (same DC offset as inputs so they
                           can be dropped into any function that re-centres)
    g_ratio              : g_I / g_Q (diagnostic)
    cross_mrad           : quadrature error in mrad (diagnostic)
    """
    I = I_raw - I_raw.mean()
    Q = Q_raw - Q_raw.mean()
    amp = np.abs(I + 1j * Q)

    # Use only samples with amplitude > 30% of median to avoid biasing the
    # estimate using near-zero samples at the turning points.
    mask = amp > 0.30 * np.median(amp)
    I_hi, Q_hi = I[mask], Q[mask]

    g_I    = np.sqrt(np.mean(I_hi ** 2))
    g_Q    = np.sqrt(np.mean(Q_hi ** 2))
    cross  = np.mean(I_hi * Q_hi) / np.mean(I_hi ** 2)   # I→Q leakage fraction

    # Apply correction (keep I as the amplitude reference)
    I_cor = I
    Q_cor = (Q - cross * I) * (g_I / g_Q)

    return (I_cor + I_raw.mean(),
            Q_cor + Q_raw.mean(),
            g_I / g_Q,
            float(cross * 1000))   # g_ratio, cross in mrad


def iq_displacement_ekf(I_raw, Q_raw, lam, f0, fs, tau=None,
                         process_noise_mm=1e-4, meas_noise_frac=0.005):
    """
    Extract displacement via an Extended Kalman Filter on the raw IQ phasor.

    State:  x = [d (mm), v (mm/s)]
    Process model: discrete exact harmonic oscillator.
    Measurement:   unit IQ phasor [Re(z/|z|), Im(z/|z|)] — 2-element real vector.
    Measurement noise covariance R = (σ_base / amp_norm)² × I₂.

    When the IQ amplitude collapses near zero at the pendulum turning points
    (the phasor physically passes through the origin because the large swing
    ~8 mm ≈ 3×λ/2 causes the IQ circle to wrap through it), R grows by orders
    of magnitude, driving the Kalman gain to near zero.  The harmonic-oscillator
    dynamics then carry the state through the turning point without any phase
    extraction — eliminating the large spikes that corrupt the naive unwrap.

    Parameters
    ----------
    I_raw, Q_raw       : raw ADC arrays (DC bias removed internally)
    lam                : radar wavelength (m)
    f0                 : pendulum fundamental frequency (Hz)
    fs                 : sample rate (Hz)
    tau                : exponential decay time constant (s); optional
    process_noise_mm   : position process-noise σ per sample (mm)
    meas_noise_frac    : IQ noise as fraction of median amplitude

    Returns
    -------
    d_out (mm), v_out (mm/s), amp (IQ amplitude array)
    """
    dt = 1.0 / fs
    omega0 = 2.0 * np.pi * f0
    lam_mm = lam * 1000.0

    I = I_raw - I_raw.mean()
    Q = Q_raw - Q_raw.mean()
    z   = I + 1j * Q
    amp = np.abs(z)
    amp_median = np.median(amp)
    n = len(z)

    c, s = np.cos(omega0 * dt), np.sin(omega0 * dt)
    F_mat = np.array([[c,  s / omega0],
                      [-omega0 * s, c]])
    if tau is not None:
        F_mat *= np.exp(-dt / tau)

    qp = process_noise_mm ** 2
    qv = (omega0 * process_noise_mm) ** 2
    Q_proc = np.diag([qp, qv])

    phi_init = np.arctan2(Q[0], I[0])
    x = np.array([phi_init * lam_mm / (4.0 * np.pi), 0.0])
    P = np.diag([(lam_mm / 4.0) ** 2, (omega0 * lam_mm / 4.0) ** 2])

    k4pi_lam = 4.0 * np.pi / lam_mm
    d_out = np.empty(n); v_out = np.empty(n)

    for k in range(n):
        x = F_mat @ x
        P = F_mat @ P @ F_mat.T + Q_proc

        phi_pred = k4pi_lam * x[0]
        cos_p = np.cos(phi_pred); sin_p = np.sin(phi_pred)
        H = np.array([[-sin_p * k4pi_lam, 0.0],
                      [ cos_p * k4pi_lam, 0.0]])

        amp_norm = amp[k] / amp_median
        r = (meas_noise_frac / max(amp_norm, 1e-4)) ** 2
        R_mat = r * np.eye(2)

        z_unit = z[k] / (amp[k] + 1e-12)
        y = np.array([z_unit.real - cos_p, z_unit.imag - sin_p])
        S = H @ P @ H.T + R_mat
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        d_out[k] = x[0]; v_out[k] = x[1]

    return d_out, v_out, amp


def synthesize_pendulum_parametric(t, disp_hp, f0, tau,
                                    n_harmonics=NOISE_N_HARMONICS,
                                    n_env=1,
                                    phi_tracked=None):
    """
    Multi-harmonic pendulum model fitted by least squares.

    Basis: exp(−n·t/τ) × cos(h·φ(t))  and  exp(−n·t/τ) × sin(h·φ(t))
    for harmonics h = 1…n_harmonics and envelope powers n = 1…n_env.

    When phi_tracked is supplied (the preferred path), φ(t) is the actual
    accumulated phase from ekf_tracked_phase().  The basis functions then
    track whatever slow frequency variations actually occurred — temperature
    drift of the pendulum length, amplitude-dependent frequency shift,
    nonlinear damping — without needing higher-order envelope powers.
    n_env=1 is sufficient in this case.

    When phi_tracked is None the basis falls back to fixed ω₀·t with
    n_env=2 envelope terms, which captures the leading-order A²
    amplitude-frequency coupling analytically and avoids the hourglass
    residual on long records.

    Returns
    -------
    model, harmonic_info
    """
    fs_approx = 1.0 / (t[1] - t[0])

    if phi_tracked is not None:
        phi = phi_tracked
        n_env_use = n_env
    else:
        phi = 2.0 * np.pi * f0 * t
        # n_env=2 is needed for full-record fits (captures amplitude-frequency
        # coupling).  For short chunks the two envelope columns are nearly
        # identical (exp(-t/τ) barely changes over 5 min) making the basis
        # near-singular — so honour the caller's n_env request exactly.
        n_env_use = n_env

    cols = []; active_harmonics = []
    for h in range(1, n_harmonics + 1):
        if f0 * h >= 0.45 * fs_approx:
            break
        for n in range(1, n_env_use + 1):
            env_n = np.exp(-n * t / tau)
            cols.append(env_n * np.cos(h * phi))
            cols.append(env_n * np.sin(h * phi))
        active_harmonics.append(h)

    A_mat = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, disp_hp, rcond=None)
    model = A_mat @ coeffs

    harmonic_info = []
    for i, h in enumerate(active_harmonics):
        a, b = coeffs[2 * n_env_use * i], coeffs[2 * n_env_use * i + 1]
        harmonic_info.append((h, f0 * h, np.sqrt((a ** 2 + b ** 2) / 2.0)))

    return model, harmonic_info


def ekf_tracked_phase(d_ekf, v_ekf, f0, fs,
                       smooth_s=PHASE_TRACK_SMOOTH_S):
    """
    Extract a smoothly-varying accumulated oscillation phase from the EKF state.

    The EKF gives smooth d(t) and v(t), so the angle atan2(−v/ω₀, d) is
    well-defined everywhere including at turning points (d≈A_max, v≈0),
    unlike the Hilbert transform which is singular there.

    Crucially, only the *deviation* from the global ω₀ is tracked here.
    The mean instantaneous frequency from the EKF state may carry a small
    systematic bias; using it directly would accumulate a large phase drift.
    Instead, the tracked phase is: φ(t) = ω₀·t + δφ(t), where δφ is the
    smoothed integral of (ω_inst − ω₀).  The correction is then only the
    slow, physically meaningful part of the frequency variation.

    Parameters
    ----------
    d_ekf, v_ekf : EKF position (mm) and velocity (mm/s)
    f0           : nominal pendulum frequency (Hz) — sets the phase reference
    fs           : sample rate (Hz)
    smooth_s     : moving-average window (s); default 5 s ≈ 1–2 periods

    Returns
    -------
    phi_tracked  : accumulated oscillation phase (rad), same length as d_ekf
    """
    omega0 = 2.0 * np.pi * f0
    phi_raw = np.unwrap(np.arctan2(-v_ekf / omega0, d_ekf))
    omega_inst = np.gradient(phi_raw) * fs

    win = max(int(smooth_s * fs), 3)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win) / win
    delta_omega = np.convolve(omega_inst - omega0, kernel, mode='same')
    half = win // 2
    delta_omega[:half]  = delta_omega[half]
    delta_omega[-half:] = delta_omega[-half - 1]

    # Integrate the slow frequency deviation and add to the reference ω₀·t
    t_arr = np.arange(len(d_ekf)) / fs
    delta_phi = np.cumsum(delta_omega) / fs
    delta_phi -= delta_phi[0]   # zero the initial offset
    return omega0 * t_arr + delta_phi


def find_fundamental(disp_hp, fs, f_search=(0.1, 2.0)):
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
    T_total_s = t_fit[-1] - t_fit[0]
    seg_win_s  = float(np.clip(T_total_s * SEG_WIN_FRAC,  240,  3600))
    seg_step_s = float(np.clip(T_total_s * SEG_STEP_FRAC,  60,   900))
    win_n  = int(seg_win_s  * fs)
    step_n = int(seg_step_s * fs)
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
            tau0  = seg_win_s / np.log(max(ratio, 1.01))
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

    return seg_t, seg_A, seg_Q, seg_win_s


def synthesize_pendulum(disp_hp, fs, f0, n_harmonics=NOISE_N_HARMONICS):
    """
    Build a clean pendulum model by extracting each harmonic's slowly-varying
    amplitude and phase via the analytic signal, smoothing both, and
    reconstructing.  Returns the model waveform and per-harmonic info.
    """
    n = len(disp_hp)
    model = np.zeros(n)
    smooth_win = int(NOISE_SMOOTH_S * fs)
    if smooth_win % 2 == 0:
        smooth_win += 1
    kernel = np.ones(smooth_win) / smooth_win

    harmonic_info = []

    for h in range(1, n_harmonics + 1):
        fc = f0 * h
        if fc >= fs / 2 * 0.9:      # skip if too close to Nyquist
            break
        bw = f0 * NOISE_BP_BW_MULT  # same absolute BW for each harmonic
        f_lo = max(fc - bw, 0.05)
        f_hi = min(fc + bw, fs / 2 * 0.95)
        if f_hi <= f_lo:
            break

        try:
            sos = signal.butter(4,
                    [f_lo / (fs/2), f_hi / (fs/2)],
                    btype='band', output='sos')
            bp = signal.sosfiltfilt(sos, disp_hp)
        except ValueError:
            break

        analytic = signal.hilbert(bp)
        inst_amp   = np.abs(analytic)
        inst_phase = np.unwrap(np.angle(analytic))

        # Smooth amplitude with moving average
        amp_smooth = np.convolve(inst_amp, kernel, mode='same')
        # Smooth phase: fit local polynomial is better than moving average
        # for a monotonically increasing signal.  Use moving-average on the
        # *residual* after removing the linear trend.
        phase_slope = (inst_phase[-1] - inst_phase[0]) / (n - 1)
        phase_linear = inst_phase[0] + phase_slope * np.arange(n)
        phase_resid  = inst_phase - phase_linear
        phase_resid_smooth = np.convolve(phase_resid, kernel, mode='same')
        phase_smooth = phase_linear + phase_resid_smooth

        # Reconstruct this harmonic
        h_model = amp_smooth * np.cos(phase_smooth)
        model += h_model

        rms = np.sqrt(np.mean(bp**2))
        harmonic_info.append((h, fc, rms))

    return model, harmonic_info


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
    seg_t, seg_A, seg_Q, seg_win_s = (np.array([]),) * 3 + (240.0,)
    if T_total >= SEG_MIN_TOTAL_S:
        seg_t, seg_A, seg_Q, seg_win_s = segmented_Q(t_fit, env_fit, f0, fs)
        if len(seg_Q):
            print(f"  Q range (segments): {seg_Q.min():.0f} – {seg_Q.max():.0f}"
                  f"  over A = {seg_A.max():.2f} – {seg_A.min():.2f} mm")

    # ── noise analysis ────────────────────────────────────────────────────────
    I_cor, Q_cor, iq_g_ratio, iq_cross_mrad = correct_iq_imbalance(I_raw, Q_raw)
    print(f"  IQ correction: gain ratio = {iq_g_ratio:.4f}, "
          f"quadrature error = {iq_cross_mrad:.2f} mrad")

    disp_ekf, v_ekf, amp_iq = iq_displacement_ekf(
        I_cor, Q_cor, lam, f0=f0, fs=fs, tau=tau,
        process_noise_mm=1e-4, meas_noise_frac=0.005)
    disp_hp_ekf = signal.sosfiltfilt(sos_hp, disp_ekf)

    noise_trim = max(trim, int(5 * fs))
    t_noise   = t[noise_trim:-noise_trim]
    dhp_noise = disp_hp_ekf[noise_trim:-noise_trim]

    # Parametric harmonic model with dual-envelope basis.
    # Basis: exp(−n·t/τ) × cos/sin(h·ω₀·t) for n=1,2 and h=1..N_harmonics.
    # The n=2 terms absorb the large-angle pendulum frequency shift (ω ∝ 1−A²/8)
    # whose integral over the exponentially-decaying amplitude grows as
    # exp(−2t/τ), correcting the hourglass residual on long records.
    model, harmonic_info = synthesize_pendulum_parametric(
        t_noise, dhp_noise, f0=f0, tau=tau,
        phi_tracked=None, n_env=2)
    noise = dhp_noise - model
    noise_rms_mm = np.std(noise)
    # Convert to phase noise
    noise_rms_rad = noise_rms_mm / 1000 * (4 * np.pi) / lam
    # Noise PSD
    nperseg_noise = min(int(30 * fs), len(noise) // 4)
    f_noise, psd_noise = signal.welch(noise, fs=fs, nperseg=nperseg_noise,
                                       noverlap=nperseg_noise // 2)
    # Signal PSD for comparison
    _, psd_signal = signal.welch(disp_hp, fs=fs, nperseg=nperseg_noise,
                                  noverlap=nperseg_noise // 2)
    # Noise floor in quiet band (above last modeled harmonic)
    last_harm_f = f0 * min(NOISE_N_HARMONICS, int(0.9 * fs/2 / f0))
    quiet_lo = last_harm_f + f0 * 0.3
    quiet_hi = fs / 2 * 0.8
    quiet_mask = (f_noise >= quiet_lo) & (f_noise <= quiet_hi)
    if quiet_mask.any():
        noise_floor_density = np.mean(psd_noise[quiet_mask])
    else:
        noise_floor_density = np.mean(psd_noise)

    # Time-varying noise RMS (10s windows)
    noise_win_s = 10
    noise_win_n = int(noise_win_s * fs)
    noise_t_local, noise_rms_local = [], []
    for i in range(0, len(noise) - noise_win_n, noise_win_n // 2):
        noise_t_local.append(t_noise[i + noise_win_n // 2])
        noise_rms_local.append(np.std(noise[i:i + noise_win_n]))
    noise_t_local = np.array(noise_t_local)
    noise_rms_local = np.array(noise_rms_local)

    n_harm_used = len(harmonic_info)
    print(f"  Noise: {noise_rms_mm*1000:.1f} µm rms  ({noise_rms_rad*1000:.2f} mrad)"
          f"  [{n_harm_used} harmonics subtracted]")
    print(f"  Noise floor: {noise_floor_density:.2e} mm²/Hz"
          f"  ({10*np.log10(noise_floor_density + 1e-20):.1f} dB)"
          f"  [{quiet_lo:.1f}–{quiet_hi:.1f} Hz]")

    return dict(
        t=t, t_fit=t_fit, t_p=t_p, fs=fs,
        t_log=t_fit[:-trim],
        disp_hp=disp_hp, disp_bp=disp_bp,
        env_smooth=env_smooth, env_fit=env_fit,
        A0=A0, tau=tau, tau_err=tau_err,
        f0=f0, Q_val=Q_val, Q_err=Q_err,
        log_env=log_env, log_model=log_model,
        phase_resid=phase_resid, timing_jitter=timing_jitter,
        T_total=T_total,
        seg_t=seg_t, seg_A=seg_A, seg_Q=seg_Q, seg_win_s=seg_win_s,
        # noise analysis
        model=model, noise=noise, t_noise=t_noise,
        noise_rms_mm=noise_rms_mm, noise_rms_rad=noise_rms_rad,
        f_noise=f_noise, psd_noise=psd_noise, psd_signal=psd_signal,
        noise_floor_density=noise_floor_density,
        noise_t_local=noise_t_local, noise_rms_local=noise_rms_local,
        harmonic_info=harmonic_info,
    )

def plot_results(r, title=''):
    """
    Plot results in two separate figures to improve readability:
      - Figure 1: main analysis panels
      - Figure 2: noise analysis panels (residual + PSD)
    """
    print(f"pendulum_analysis version: {VERSION}")

    t       = r['t']
    t_fit   = r['t_fit']
    t_p     = r['t_p']
    A0      = r['A0'];  tau = r['tau'];  f0 = r['f0']
    Q_val   = r['Q_val'];  Q_err = r['Q_err'];  tau_err = r['tau_err']

    t_model = np.linspace(t_fit[0], t_fit[-1], 2000)

    seg_t = r.get('seg_t', np.array([]))
    seg_A = r.get('seg_A', np.array([]))
    seg_Q = r.get('seg_Q', np.array([]))
    has_seg = len(seg_Q) >= 4

    # ================= FIGURE 1 — MAIN ANALYSIS =================
    nrows_main = 4 if has_seg else 3
    fig1, axes1 = plt.subplots(nrows_main, 1, figsize=(12, 4*nrows_main + 1),
                               layout='constrained')
    fig1.suptitle(title + "  (Main Analysis)", fontsize=11, fontweight='bold')

    # Panel 1: amplitude decay
    ax = axes1[0]
    ax.plot(t, r['disp_bp'], color='steelblue', lw=0.35, alpha=0.65,
            label='Bandpass displacement')
    ax.plot(t, r['env_smooth'], color='r', lw=1.5, label='Envelope')
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

    # Panel 2: log envelope
    ax = axes1[1]
    ax.plot(r['t_log'], r['log_env'], color='steelblue', lw=0.8, label='ln(envelope)')
    ax.plot(r['t_log'], r['log_model'], 'k--', lw=1.5,
            label=f'Linear fit  (slope = −1/τ = {-1/tau:.5f} s⁻¹)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('ln(Amplitude)')
    ax.set_title('Log Envelope  (linearity confirms single-mode exponential decay)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: phase residual
    ax = axes1[2]
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

    # Panel 4: segmented Q vs amplitude (optional)
    if has_seg:
        ax = axes1[3]
        T_seg = r.get('seg_win_s', 240)
        sc = ax.scatter(seg_A, seg_Q, c=seg_t, cmap='plasma',
                        s=40, zorder=3, label=f'Local Q ({T_seg:.0f} s window)')
        cb = fig1.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label('Window centre time (s)', fontsize=9)

        if seg_A.max() / max(seg_A.min(), 1e-12) > 1.5:
            # Fit Q = k/A weighted by A².
            # The model Q·A = k means minimising Σ(Q·A − k)² gives k = mean(Q·A),
            # which is equivalent to weighting each point by A² in Q-space.
            # This prevents noisy low-amplitude windows (where τ is poorly
            # constrained and Q is overestimated) from dominating the fit.
            k_fit = float(np.average(seg_Q * seg_A, weights=seg_A ** 2))
            A_line = np.linspace(seg_A.min()*0.9, seg_A.max()*1.05, 200)
            ax.plot(A_line, k_fit / A_line, 'k--', lw=1.5,
                    label=f'Q = {k_fit:.1f} / A  (quadratic drag)')

        ax.axhline(Q_val, color='r', lw=1.2, ls=':', alpha=0.7,
                   label=f'Global fit Q = {Q_val:.0f}')
        ax.set_xlabel('Mean amplitude in window (mm)')
        ax.set_ylabel('Q')
        ax.set_title('Local Q vs Amplitude')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ================= FIGURE 2 — NOISE ANALYSIS =================
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 9), layout='constrained')
    fig2.suptitle(title + "  (Noise Analysis)", fontsize=11, fontweight='bold')

    # Panel 1: noise residual time series
    ax = axes2[0]
    noise = r['noise']
    t_noise = r.get('t_noise', t)
    noise_um = noise * 1000  # mm → µm
    ax.plot(t_noise, noise_um, color='gray', lw=0.5, alpha=0.85)
    if len(r.get('noise_rms_local', [])) > 0:
        ax.plot(r['noise_t_local'], r['noise_rms_local'] * 1000,
                color='red', lw=1.5, label='Local RMS (10 s)')
        ax.plot(r['noise_t_local'], -r['noise_rms_local'] * 1000,
                color='red', lw=1.5)
        ax.legend(fontsize=9)
    n_harm = len(r.get('harmonic_info', []))
    rms_um = r['noise_rms_mm'] * 1000
    rms_mrad = r['noise_rms_rad'] * 1000
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Residual (µm)')
    ax.set_title(f'System Noise  (measured − {n_harm}-harmonic model)\n'
                 f'σ = {rms_um:.1f} µm  ({rms_mrad:.2f} mrad)')
    ax.grid(True, alpha=0.3)

    # Panel 2: noise PSD
    ax = axes2[1]
    f_n = r['f_noise']
    psd_n = r['psd_noise']
    psd_s = r['psd_signal']
    ax.semilogy(f_n, psd_s, color='steelblue', lw=0.6, alpha=0.7,
                label='Signal (HP-filtered)')
    ax.semilogy(f_n, psd_n, color='red', lw=0.8,
                label='Noise (residual)')
    for h, fc, _ in r.get('harmonic_info', []):
        ax.axvline(fc, color='steelblue', ls=':', lw=0.5, alpha=0.4)
    nfd = r['noise_floor_density']
    ax.axhline(nfd, color='red', ls='--', lw=1, alpha=0.6,
               label=f'Noise floor: {10*np.log10(nfd+1e-20):.1f} dB(mm²/Hz)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (mm²/Hz)')
    ax.set_xlim(0, min(r['fs'] / 2, f0 * (n_harm + 3 if n_harm else 5)))
    ax.set_title('Power Spectral Density: Signal vs Noise Residual')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# LARGE-RECORD STREAMING PATH  (>LARGE_RECORD_HOURS hours)
# ═══════════════════════════════════════════════════════════════════════════════

def _peek_csv_header(path):
    """
    Quick pre-scan: read first and last rows to estimate total duration and fs.
    Returns (T_total_estimate_s, fs_estimate, n_rows_estimate).
    Reads only first 50 rows and last 50 rows — O(1) memory.
    """
    import csv, os
    rows_head = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= 50:
                break
            if len(row) >= 4:
                try:
                    rows_head.append((int(row[0]), int(row[1])))  # seq, ms_raw
                except ValueError:
                    pass

    # Seek near end of file to get last rows
    fsize = os.path.getsize(path)
    rows_tail = []
    with open(path, 'rb') as f:
        f.seek(max(0, fsize - 32768))
        tail_bytes = f.read()
    tail_text = tail_bytes.decode('utf-8', errors='replace')
    tail_lines = tail_text.splitlines()
    for line in reversed(tail_lines):
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                rows_tail.append((int(parts[0]), int(parts[1])))
                if len(rows_tail) >= 50:
                    break
            except ValueError:
                pass
    rows_tail.reverse()

    if not rows_head or not rows_tail:
        return None, None, None

    first_seq = rows_head[0][0]
    last_seq   = rows_tail[-1][0]
    n_rows_est = last_seq - first_seq + 1   # works regardless of starting seq number

    # Estimate fs from ms timestamps in first block
    ms_vals = [r[1] for r in rows_head]
    if len(ms_vals) > 2:
        diffs = [ms_vals[i+1] - ms_vals[i] for i in range(len(ms_vals)-1)
                 if 0 < ms_vals[i+1] - ms_vals[i] < 500]
        fs_est = 1000.0 / (sum(diffs)/len(diffs)) if diffs else None
    else:
        fs_est = None

    T_est = n_rows_est / (fs_est if fs_est else 200.0)
    return T_est, fs_est, n_rows_est


def _stream_chunks(path, chunk_samples):
    """
    Generator: yields (seq_arr, ms_arr, I_arr, Q_arr) numpy arrays one chunk
    at a time. Each chunk is exactly chunk_samples rows (last chunk may be shorter).
    Memory footprint: 4 × chunk_samples × 8 bytes ≈ a few MB per chunk.
    """
    import csv
    seqs = []; ms_list = []; I_list = []; Q_list = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                seqs.append(int(row[0]))
                ms_list.append(int(row[1]))
                I_list.append(int(row[2]))
                Q_list.append(int(row[3]))
            except ValueError:
                continue
            if len(seqs) == chunk_samples:
                yield (np.asarray(seqs),
                       np.asarray(ms_list),
                       np.asarray(I_list, dtype=np.float64),
                       np.asarray(Q_list, dtype=np.float64))
                seqs = []; ms_list = []; I_list = []; Q_list = []
    if seqs:
        yield (np.asarray(seqs),
               np.asarray(ms_list),
               np.asarray(I_list, dtype=np.float64),
               np.asarray(Q_list, dtype=np.float64))


def _chunk_ms_to_t(ms_raw):
    """Reconstruct elapsed time within a chunk from ms%1000 with rollover."""
    n = len(ms_raw)
    elapsed_ms = np.zeros(n, dtype=np.float64)
    epoch = 0
    for i in range(1, n):
        if ms_raw[i] < ms_raw[i-1] - 500:
            epoch += 1000
        elapsed_ms[i] = epoch + ms_raw[i] - ms_raw[0]
    return elapsed_ms / 1000.0


def _ekf_chunk(I_raw, Q_raw, lam, f0, fs, tau, x_init, P_init,
               process_noise_mm=1e-4, meas_noise_frac=0.005):
    """
    Run one chunk through the EKF with pre-loaded state (x_init, P_init).
    Returns (d_out, v_out, amp, x_final, P_final).
    Identical maths to iq_displacement_ekf(); factored out for chunk streaming.
    """
    dt = 1.0 / fs
    omega0 = 2.0 * np.pi * f0
    lam_mm = lam * 1000.0

    I = I_raw - I_raw.mean()
    Q = Q_raw - Q_raw.mean()
    z   = I + 1j * Q
    amp = np.abs(z)
    amp_median = np.median(amp)
    n = len(z)

    c, s = np.cos(omega0 * dt), np.sin(omega0 * dt)
    F_mat = np.array([[c,  s / omega0],
                      [-omega0 * s, c]]) * np.exp(-dt / tau)

    qp = process_noise_mm ** 2
    qv = (omega0 * process_noise_mm) ** 2
    Q_proc = np.diag([qp, qv])

    k4pi_lam = 4.0 * np.pi / lam_mm
    d_out = np.empty(n); v_out = np.empty(n)

    x = x_init.copy(); P = P_init.copy()
    for k in range(n):
        x = F_mat @ x
        P = F_mat @ P @ F_mat.T + Q_proc

        phi_pred = k4pi_lam * x[0]
        cos_p = np.cos(phi_pred); sin_p = np.sin(phi_pred)
        H = np.array([[-sin_p * k4pi_lam, 0.0],
                      [ cos_p * k4pi_lam, 0.0]])

        amp_norm = amp[k] / amp_median
        r_val = (meas_noise_frac / max(amp_norm, 1e-4)) ** 2
        R_mat = r_val * np.eye(2)

        z_unit = z[k] / (amp[k] + 1e-12)
        y = np.array([z_unit.real - cos_p, z_unit.imag - sin_p])
        S = H @ P @ H.T + R_mat
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        d_out[k] = x[0]; v_out[k] = x[1]

    return d_out, v_out, amp, x, P


def analyse_large(csv_path):
    """
    Memory-efficient analysis for multi-hour IQ CSV files.

    Strategy
    --------
    1. Pre-scan CSV for duration / fs estimate.
    2. Read first ~60 s block to determine f0, τ (global), noise floor.
    3. Stream file in CHUNK_MINUTES chunks, running EKF with state hand-off.
    4. Each chunk → one summary row: (t_centre, amplitude, phase_offset, noise_σ).
    5. Hourly groups → exp-decay fit → τ_h → Q_h ± uncertainty.
    6. Auto-stop when chunk amplitude < CUTOFF_SNR × global noise floor.
    7. Identify best-SNR chunk for Figure 2 noise window.

    Returns a dict compatible with plot_results_large().
    """
    import os
    print(f"  [large-record mode]  scanning {os.path.basename(csv_path)} ...")

    T_est, fs_est, n_rows_est = _peek_csv_header(csv_path)
    if fs_est is None:
        fs_est = 200.0
    print(f"  Estimated: {n_rows_est} rows, {T_est/3600:.2f} h, fs≈{fs_est:.1f} Hz")

    chunk_samples = int(CHUNK_MINUTES * 60 * fs_est)
    lam = 3e8 / F_RADAR

    # ── Phase 1: load first NOISE_FLOOR_INIT_CHUNKS chunks to bootstrap ─────
    boot_I = []; boot_Q = []; boot_ms = []; boot_seq = []
    boot_chunks_needed = NOISE_FLOOR_INIT_CHUNKS
    first_seq = None   # global sequence number of the very first sample
    for seq_a, ms_a, I_a, Q_a in _stream_chunks(csv_path, chunk_samples):
        if first_seq is None:
            first_seq = int(seq_a[0])
        boot_I.append(I_a); boot_Q.append(Q_a); boot_ms.append(ms_a)
        if len(boot_I) >= boot_chunks_needed:
            break

    I_boot = np.concatenate(boot_I)
    Q_boot = np.concatenate(boot_Q)
    ms_boot = np.concatenate(boot_ms)

    # Reconstruct elapsed time for boot block
    n_b = len(ms_boot)
    elapsed_boot = np.zeros(n_b)
    epoch = 0
    for i in range(1, n_b):
        if ms_boot[i] < ms_boot[i-1] - 500:
            epoch += 1000
        elapsed_boot[i] = epoch + ms_boot[i] - ms_boot[0]
    elapsed_boot /= 1000.0
    fs = (n_b - 1) / elapsed_boot[-1]
    # Recompute chunk_samples with accurate fs
    chunk_samples = int(CHUNK_MINUTES * 60 * fs)

    print(f"  Bootstrap block: {n_b} samples, fs = {fs:.2f} Hz")

    disp_boot, _ = iq_to_displacement_mm(I_boot, Q_boot)
    sos_hp = signal.butter(4, HP_FC / (fs/2), btype='high', output='sos')
    disp_hp_boot = signal.sosfiltfilt(sos_hp, disp_boot)
    f0 = find_fundamental(disp_hp_boot, fs)

    # Coarse envelope fit on boot block for initial τ and noise floor
    sos_bp_c = signal.butter(4,
                    [(f0 - BP_BW*3) / (fs/2), (f0 + BP_BW*3) / (fs/2)],
                    btype='band', output='sos')
    disp_bp_boot = signal.sosfiltfilt(sos_bp_c, disp_boot)
    env_boot = np.abs(signal.hilbert(disp_bp_boot))
    sos_lp = signal.butter(2, ENV_LP_FC / (fs/2), btype='low', output='sos')
    env_smooth_boot = signal.sosfiltfilt(sos_lp, env_boot)

    trim_b = int(TRIM_S * fs)
    t_b = elapsed_boot[trim_b:-trim_b]
    e_b = env_smooth_boot[trim_b:-trim_b]
    try:
        popt_b, _ = curve_fit(exp_decay, t_b, e_b,
                               p0=[e_b[0], elapsed_boot[-1] / 2], maxfev=5000)
        A0_global, tau_global = popt_b
    except RuntimeError:
        A0_global = float(e_b[0])
        tau_global = float(elapsed_boot[-1])

    # Global noise floor estimate from boot block residual
    I_cor_b, Q_cor_b, _, _ = correct_iq_imbalance(I_boot, Q_boot)
    d_ekf_b, v_ekf_b, _ = iq_displacement_ekf(
        I_cor_b, Q_cor_b, lam, f0=f0, fs=fs, tau=tau_global)
    dhp_b = signal.sosfiltfilt(sos_hp, d_ekf_b)
    model_b, _ = synthesize_pendulum_parametric(
        elapsed_boot, dhp_b, f0=f0, tau=tau_global, n_env=2)
    noise_b = dhp_b - model_b
    global_noise_floor_mm = float(np.std(noise_b))
    cutoff_amp = CUTOFF_SNR * global_noise_floor_mm
    print(f"  f0 = {f0:.5f} Hz | τ_est = {tau_global:.0f} s | "
          f"noise floor = {global_noise_floor_mm*1000:.1f} µm | "
          f"cutoff amp = {cutoff_amp*1000:.1f} µm")

    # ── Phase 2: stream all chunks ───────────────────────────────────────────
    # Initialise EKF state from boot block end
    omega0 = 2.0 * np.pi * f0
    lam_mm = lam * 1000.0
    # Use last EKF displacement/velocity as seed
    x_ekf = np.array([d_ekf_b[-1], v_ekf_b[-1]])
    P_ekf = np.diag([(lam_mm / 4.0) ** 2, (omega0 * lam_mm / 4.0) ** 2])

    # Summary arrays (one entry per chunk)
    chunk_t = []       # absolute time of chunk centre (s)
    chunk_amp = []     # mean smoothed envelope amplitude (mm)
    chunk_phase = []   # phase offset relative to ideal (rad)
    chunk_noise = []   # noise σ within chunk (mm)

    # For noise-window selection (best SNR chunk)
    best_snr = -1.0
    best_chunk_data = None   # will store (t_rel, d_ekf, fs) for one chunk

    chunk_idx = 0
    stopped_early = False
    _zc_t_ref_global = None   # set from first chunk with valid crossings
    _zc_global_count = 0      # cumulative crossing count across all chunks

    for seq_a, ms_a, I_a, Q_a in _stream_chunks(csv_path, chunk_samples):
        n_c = len(ms_a)

        # Absolute time from sequence numbers — exact, handles gaps, no accumulation error.
        # Subtract first_seq so t=0 is the start of the record regardless of
        # where in the sequence space this file begins.
        t_abs = (seq_a.astype(np.float64) - first_seq) / fs
        t_rel = t_abs - t_abs[0]   # chunk-relative time (for filter calls that need it)

        t_centre = float(np.mean(t_abs))

        # IQ correction
        I_cor, Q_cor, _, _ = correct_iq_imbalance(I_a, Q_a)

        # EKF with handed-off state
        d_ekf, v_ekf, amp_iq, x_ekf, P_ekf = _ekf_chunk(
            I_cor, Q_cor, lam, f0, fs, tau_global, x_ekf, P_ekf)

        # High-pass to remove slow drift
        d_hp = signal.sosfiltfilt(sos_hp, d_ekf)

        # Bandpass for envelope
        disp_bp_c = signal.sosfiltfilt(sos_bp_c, d_ekf)
        env_c = np.abs(signal.hilbert(disp_bp_c))
        env_sm_c = signal.sosfiltfilt(sos_lp, env_c)
        amp_mean = float(np.mean(env_sm_c))

        # Auto-cutoff
        if amp_mean < cutoff_amp and chunk_idx >= NOISE_FLOOR_INIT_CHUNKS:
            print(f"  Auto-cutoff at t={t_centre/3600:.3f} h "
                  f"(amp={amp_mean*1000:.1f} µm < {cutoff_amp*1000:.1f} µm)")
            stopped_early = True
            break

        # ── Phase residual via zero-crossing line fit ────────────────────────
        # All upward zero crossings in the chunk get sub-sample interpolated
        # absolute times.  We assign each a global crossing index (no ambiguity:
        # crossings within a chunk are sequential), fit a line through
        # (global_index, crossing_time) across ALL chunks, and the intercept
        # residual gives the phase offset.  Using all crossings (not just the
        # first) makes this immune to cycle slips from timing noise.
        bp_c = signal.sosfiltfilt(sos_bp_c, d_ekf)
        zc_idx = np.where((bp_c[:-1] < 0) & (bp_c[1:] >= 0))[0]
        if len(zc_idx) >= 2:
            frac = -bp_c[zc_idx] / (bp_c[zc_idx + 1] - bp_c[zc_idx] + 1e-30)
            zc_t_abs = t_abs[zc_idx] + frac / fs

            if _zc_t_ref_global is None:
                # First valid chunk: assign global indices starting at 0.
                _zc_t_ref_global = float(zc_t_abs[0])
                _zc_global_count = 0   # global index of next first crossing
            # Assign global indices for this chunk's crossings
            # (consecutive from _zc_global_count)
            zc_global_n = _zc_global_count + np.arange(len(zc_t_abs))
            # Accumulate into global lists for a final line fit later;
            # for the per-chunk summary use the mean residual from a local
            # line fit (same data, no slip risk).
            zc_n_loc = np.arange(len(zc_t_abs))
            slope_loc, intercept_loc = np.polyfit(zc_n_loc, zc_t_abs, 1)
            # Local phase: difference between fitted intercept and ideal
            # time predicted from the global reference
            t_ideal_intercept = _zc_t_ref_global + _zc_global_count / f0
            timing_resid_s = intercept_loc - t_ideal_intercept
            phase_off = timing_resid_s * 2 * np.pi * f0   # radians

            # Advance global crossing counter
            _zc_global_count += len(zc_t_abs)

            # Within-chunk timing jitter from local fit residuals
            zc_t_fit = intercept_loc + slope_loc * zc_n_loc
            zc_resid_s = zc_t_abs - zc_t_fit
            chunk_timing_jitter = float(np.std(zc_resid_s) * 1000)  # ms
        else:
            phase_off = float('nan')
            chunk_timing_jitter = float('nan')

        # ── Chunk noise σ ────────────────────────────────────────────────────
        # Use t_abs so exp(-t/τ) evaluates at the correct elapsed time.
        # n_env=1 is sufficient for a 5-min chunk (amplitude barely changes).
        try:
            model_c, _ = synthesize_pendulum_parametric(
                t_abs, d_hp, f0=f0, tau=tau_global, n_env=1)
            noise_c = d_hp - model_c
            noise_sigma = float(np.std(noise_c))
        except Exception as _exc:
            if chunk_idx == 0:
                print(f"  WARNING: noise model failed: {_exc}")
            noise_c = np.zeros_like(d_hp)
            noise_sigma = global_noise_floor_mm

        # Track best-SNR chunk for Figure 2
        snr_c = amp_mean / (noise_sigma + 1e-12)
        if snr_c > best_snr:
            best_snr = snr_c
            best_chunk_data = {
                't_rel': t_rel.copy(),
                'd_hp': d_hp.copy(),
                'noise': noise_c.copy() if 'noise_c' in dir() else np.array([]),
                'amp': amp_mean,
                'noise_sigma': noise_sigma,
                'fs': fs,
                'f0': f0,
                'tau': tau_global,
            }

        chunk_t.append(t_centre)
        chunk_amp.append(amp_mean)
        chunk_phase.append(phase_off)
        chunk_noise.append(noise_sigma)

        chunk_idx += 1
        if chunk_idx % 12 == 0:
            elapsed_h = t_centre / 3600
            print(f"  chunk {chunk_idx:4d}  t={elapsed_h:.2f} h  "
                  f"amp={amp_mean*1000:.1f} µm  noise={noise_sigma*1e6:.0f} nm")

    chunk_t     = np.array(chunk_t)
    chunk_amp   = np.array(chunk_amp)
    chunk_phase = np.array(chunk_phase)
    chunk_noise = np.array(chunk_noise)
    n_chunks    = len(chunk_t)
    print(f"  Processed {n_chunks} chunks  ({n_chunks * CHUNK_MINUTES / 60:.2f} h)")

    # ── Phase 3: hourly Q fits ───────────────────────────────────────────────
    hourly_t     = []
    hourly_Q     = []
    hourly_Q_err = []
    hourly_tau   = []
    hourly_noise = []

    i = 0
    while i + HOURLY_CHUNKS <= n_chunks:
        sl = slice(i, i + HOURLY_CHUNKS)
        t_h   = chunk_t[sl]
        A_h   = chunk_amp[sl]
        n_h   = chunk_noise[sl]

        if A_h.mean() < cutoff_amp:
            i += HOURLY_CHUNKS
            continue
        try:
            t_h0 = t_h - t_h[0]
            popt_h, pcov_h = curve_fit(
                exp_decay, t_h0, A_h,
                p0=[A_h[0], tau_global],
                bounds=([0, 10], [np.inf, 1e6]),
                maxfev=2000)
            _, tau_h = popt_h
            tau_h_err = np.sqrt(np.diag(pcov_h))[1]
            Q_h     = np.pi * f0 * tau_h
            Q_h_err = np.pi * f0 * tau_h_err
            hourly_t.append(float(np.mean(t_h)))
            hourly_Q.append(Q_h)
            hourly_Q_err.append(Q_h_err)
            hourly_tau.append(tau_h)
            hourly_noise.append(float(np.mean(n_h)))
        except (RuntimeError, ValueError):
            pass
        i += HOURLY_CHUNKS

    hourly_t     = np.array(hourly_t)
    hourly_Q     = np.array(hourly_Q)
    hourly_Q_err = np.array(hourly_Q_err)
    hourly_noise = np.array(hourly_noise)

    # Global Q from full chunk sequence
    if n_chunks >= 6:
        t_all = chunk_t - chunk_t[0]
        try:
            popt_g, pcov_g = curve_fit(exp_decay, t_all, chunk_amp,
                                        p0=[chunk_amp[0], tau_global], maxfev=5000)
            A0_global, tau_global_fit = popt_g
            tau_global_err = np.sqrt(np.diag(pcov_g))[1]
        except RuntimeError:
            tau_global_fit = tau_global
            tau_global_err = 0.0
        Q_global     = np.pi * f0 * tau_global_fit
        Q_global_err = np.pi * f0 * tau_global_err
    else:
        tau_global_fit  = tau_global
        tau_global_err  = 0.0
        Q_global        = np.pi * f0 * tau_global
        Q_global_err    = 0.0
        A0_global       = float(chunk_amp[0]) if n_chunks else 0.0

    print(f"  Global: τ = {tau_global_fit:.0f} ± {tau_global_err:.0f} s  "
          f"Q = {Q_global:.0f} ± {Q_global_err:.0f}")

    # Report phase residual stats (detrended, same as plot panel)
    if len(chunk_t) >= 4:
        t_h_all = chunk_t / 3600.0
        ph_unwrapped = np.unwrap(chunk_phase)
        ph_detrend_all = ph_unwrapped - np.polyval(
            np.polyfit(t_h_all, ph_unwrapped, 1), t_h_all)
        ph_rms_mrad = float(np.std(ph_detrend_all) * 1000)
        tj_rms_ms   = float(np.std(ph_detrend_all) / (2 * np.pi * f0) * 1000)
        print(f"  Phase residual σ = {ph_rms_mrad:.2f} mrad rms  "
              f"({tj_rms_ms:.3f} ms timing jitter)")

    return dict(
        f0=f0, fs=fs, lam=lam,
        A0=A0_global, tau=tau_global_fit, tau_err=tau_global_err,
        Q_val=Q_global, Q_err=Q_global_err,
        T_total=float(chunk_t[-1]) if n_chunks else 0.0,
        chunk_t=chunk_t, chunk_amp=chunk_amp,
        chunk_phase=chunk_phase, chunk_noise=chunk_noise,
        hourly_t=hourly_t, hourly_Q=hourly_Q, hourly_Q_err=hourly_Q_err,
        hourly_noise=hourly_noise,
        global_noise_floor=global_noise_floor_mm,
        cutoff_amp=cutoff_amp,
        stopped_early=stopped_early,
        best_chunk=best_chunk_data,
    )


def plot_results_large(r, title=''):
    """
    Three sparse figures for large (multi-hour) records.

    Figure 1: Amplitude peaks (one dot per chunk) + log-envelope + phase residual.
    Figure 2: Noise analysis on best-SNR 5-min chunk only.
    Figure 3: Hourly Q ± uncertainty + hourly noise floor.
    """
    print(f"pendulum_analysis version: {VERSION}  [large-record plots]")

    f0    = r['f0']
    tau   = r['tau']
    A0    = r['A0']
    Q_val = r['Q_val']
    Q_err = r['Q_err']
    tau_err = r['tau_err']

    chunk_t   = r['chunk_t']
    chunk_amp = r['chunk_amp']
    chunk_ph  = r['chunk_phase']
    t_h       = chunk_t / 3600.0   # convert to hours for x-axis
    t0_s      = float(chunk_t[0])  # absolute time of first chunk centre (s)

    t_model_h = np.linspace(t_h[0], t_h[-1], 500)
    # Model time must be relative to first chunk, matching curve_fit which used
    # t_all = chunk_t - chunk_t[0].  So t_model_s=0 → first chunk centre.
    t_model_s = (t_model_h - t_h[0]) * 3600.0

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 1, figsize=(13, 12), layout='constrained')
    fig1.suptitle(title + '  (Large-Record Analysis)', fontsize=11, fontweight='bold')

    # Panel 1a: amplitude decay (sparse dots + log model)
    ax = axes1[0]
    ax.scatter(t_h, chunk_amp, s=8, color='steelblue', alpha=0.7,
               label=f'Chunk amplitude ({CHUNK_MINUTES:.0f}-min)', zorder=3)
    ax.plot(t_model_h, exp_decay(t_model_s, A0, tau), 'k--', lw=1.8,
            label=f'Fit: A₀={A0*1000:.0f} µm, τ={tau:.0f}±{tau_err:.0f} s, '
                  f'Q={Q_val:.0f}±{Q_err:.0f}')
    if r['stopped_early']:
        ax.axvline(t_h[-1], color='r', ls=':', lw=1.2, label='Auto-cutoff')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Amplitude (mm)')
    ax.set_title(f'Amplitude Decay  |  f₀ = {f0:.5f} Hz  |  τ = {tau:.0f} s  |  Q = {Q_val:.0f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 1b: log envelope
    ax = axes1[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_amp = np.log(np.where(chunk_amp > 0, chunk_amp, np.nan))
    valid = np.isfinite(log_amp)
    ax.scatter(t_h[valid], log_amp[valid], s=8, color='steelblue', alpha=0.7)
    ax.plot(t_model_h, np.log(exp_decay(t_model_s, A0, tau)), 'k--', lw=1.5,
            label=f'slope = −1/τ = {-1/tau:.5f} s⁻¹')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('ln(Amplitude)')
    ax.set_title('Log Envelope  (linearity confirms single-mode exponential decay)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 1c: phase residual (unwrap then detrend)
    ax = axes1[2]
    ph_valid = np.isfinite(chunk_ph)
    if ph_valid.sum() >= 4:
        ph_uw = np.unwrap(chunk_ph[ph_valid])
        ph_detrend = ph_uw - np.polyval(
            np.polyfit(t_h[ph_valid], ph_uw, 1), t_h[ph_valid])
        ph_mrad = ph_detrend * 1000
        tj_ms = ph_detrend / (2 * np.pi * f0) * 1000
        ax.scatter(t_h[ph_valid], ph_mrad, s=8, color='darkorange', alpha=0.8)
        ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
        ax.set_title(f'Phase Residual  σ = {ph_mrad.std():.1f} mrad rms'
                     f'  ({tj_ms.std():.1f} ms timing jitter)')
    else:
        ax.set_title('Phase Residual  (insufficient data)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Phase residual (mrad)')
    ax.grid(True, alpha=0.3)

    # ── Figure 2: noise analysis on best-SNR chunk ───────────────────────────
    bc = r.get('best_chunk')
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 9), layout='constrained')
    fig2.suptitle(title + f'  (Noise — Best {CHUNK_MINUTES:.0f}-min Window)',
                  fontsize=11, fontweight='bold')

    if bc is not None:
        t_bc  = bc['t_rel']
        d_bc  = bc['d_hp']
        noise_bc = bc['noise']
        fs_bc = bc['fs']
        f0_bc = bc['f0']
        tau_bc = bc['tau']

        ax = axes2[0]
        noise_um = noise_bc * 1000
        ax.plot(t_bc, noise_um, color='gray', lw=0.5, alpha=0.85)
        rms_um = float(np.std(noise_um))
        ax.set_xlabel('Time within window (s)')
        ax.set_ylabel('Residual (µm)')
        ax.set_title(f'System Noise (measured − parametric model)\n'
                     f'σ = {rms_um:.1f} µm  SNR = {bc["amp"]/bc["noise_sigma"]:.0f}')
        ax.grid(True, alpha=0.3)

        ax = axes2[1]
        nperseg_n = min(int(30 * fs_bc), len(noise_bc) // 4)
        if nperseg_n >= 16:
            f_n, psd_n = signal.welch(noise_bc, fs=fs_bc, nperseg=nperseg_n)
            _, psd_s = signal.welch(d_bc, fs=fs_bc, nperseg=nperseg_n)
            ax.semilogy(f_n, psd_s, color='steelblue', lw=0.6, alpha=0.7,
                        label='Signal (HP-filtered)')
            ax.semilogy(f_n, psd_n, color='red', lw=0.8, label='Noise (residual)')
            nfd = float(np.mean(psd_n[f_n > f0_bc * 12])) if any(f_n > f0_bc * 12) else float(np.mean(psd_n))
            ax.axhline(nfd, color='red', ls='--', lw=1, alpha=0.6,
                       label=f'Noise floor: {10*np.log10(nfd+1e-30):.1f} dB(mm²/Hz)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (mm²/Hz)')
        ax.set_title('Power Spectral Density: Signal vs Noise')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    else:
        for ax in axes2:
            ax.text(0.5, 0.5, 'No chunk data available', ha='center', va='center',
                    transform=ax.transAxes)

    # ── Figure 3: hourly Q and noise floor ───────────────────────────────────
    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    fig3.suptitle(title + '  (Hourly Q & Noise)', fontsize=11, fontweight='bold')

    h_t = r['hourly_t'] / 3600.0
    h_Q = r['hourly_Q']
    h_Qe = r['hourly_Q_err']
    h_n  = r['hourly_noise']

    ax = axes3[0]
    if len(h_t) >= 2:
        ax.errorbar(h_t, h_Q, yerr=h_Qe, fmt='o-', color='steelblue',
                    capsize=4, lw=1.5, ms=6, label='Hourly Q')
        ax.axhline(Q_val, color='r', lw=1.2, ls=':', alpha=0.7,
                   label=f'Global Q = {Q_val:.0f} ± {Q_err:.0f}')
        ax.fill_between([h_t[0], h_t[-1]],
                        [Q_val - Q_err]*2, [Q_val + Q_err]*2,
                        color='r', alpha=0.12)
    else:
        ax.text(0.5, 0.5, 'Insufficient hourly windows', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Q factor')
    ax.set_title('Hourly Q Factor  (exp-decay fit over each 1-h block of chunks)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    if len(h_t) >= 2:
        ax.plot(h_t, h_n * 1e6, 'o-', color='darkorange', lw=1.5, ms=6,
                label='Hourly mean noise σ')
        ax.axhline(r['global_noise_floor'] * 1e6, color='k', ls='--', lw=1.2,
                   label=f"Global floor = {r['global_noise_floor']*1e6:.0f} nm rms")
        ax.axhline(r['cutoff_amp'] * 1e6, color='r', ls=':', lw=1,
                   label=f"Cutoff threshold = {r['cutoff_amp']*1e6:.0f} nm")
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Noise σ (nm rms)')
    ax.set_title('Hourly Noise Floor')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig1, fig2, fig3


def main():
    if len(sys.argv) < 2:
        print("Usage: python pendulum_analysis.py <IQ_csv_file> [output_png]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else None

    import os
    print(f"Analysing: {csv_path}")

    # Quick size check to decide which path to use
    T_est, _, n_rows_est = _peek_csv_header(csv_path)
    use_large = (T_est is not None and T_est / 3600.0 >= LARGE_RECORD_HOURS)
    if use_large:
        print(f"  Record ≈ {T_est/3600:.2f} h — using large-record streaming path.")

    if use_large:
        results = analyse_large(csv_path)
        title = (f'61 GHz Radar — Pendulum  |  {os.path.basename(csv_path)}'
                 f'  |  {results["T_total"]/3600:.2f} h  |  Q = {results["Q_val"]:.0f}')
        fig1, fig2, fig3 = plot_results_large(results, title=title)

        if out_path:
            base, ext = os.path.splitext(out_path)
            if ext == "":
                ext = ".png"
            for suffix, fig in [("_amplitude", fig1), ("_noise", fig2), ("_hourlyQ", fig3)]:
                fname = base + suffix + ext
                fig.savefig(fname, dpi=200, bbox_inches='tight')
                print(f"Saved: {fname}")
        else:
            plt.show()
    else:
        results = analyse(csv_path)
        title = (f'61 GHz Radar — Pendulum  |  {os.path.basename(csv_path)}'
                 f'  |  {results["T_total"]/60:.1f} min  |  Q = {results["Q_val"]:.0f}')
        fig_main, fig_noise = plot_results(results, title=title)

        if out_path:
            base, ext = os.path.splitext(out_path)
            if ext == "":
                ext = ".png"
            out_main  = base + "_main" + ext
            out_noise = base + "_noise" + ext
            fig_main.savefig(out_main, dpi=200, bbox_inches='tight')
            fig_noise.savefig(out_noise, dpi=200, bbox_inches='tight')
            print(f"Saved: {out_main}")
            print(f"Saved: {out_noise}")
        else:
            plt.show()

if __name__ == '__main__':
    main()
