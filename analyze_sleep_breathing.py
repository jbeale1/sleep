#!/usr/bin/env python3
"""
Sleep Breathing Analysis Tool - 2026-2-5 J.Beale

Analyzes overnight breathing audio recordings to detect:
- Breath cycles (inhale + exhale pairs)
- Obstruction severity (mild, moderate, severe)
- Breathing gaps (apnea events > 10 seconds)

Input: Directory containing sequential 15-minute MP3 files
       Filename format: ch4_YYYYMMDD_HHMMSS.mp3

Output: CSV summary file + console report by wall-clock hour
"""

import os
import sys
import glob
import subprocess
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import uniform_filter1d


# =============================================================================
# Configuration / Thresholds (calibrated from sample data)
# =============================================================================

# Breath detection: multiplier of noise floor
BREATH_THRESH_MULTIPLIER = 3.0

# Minimum breath event duration (seconds) - shorter events are pops/clicks
MIN_BREATH_DURATION = 0.3

# No-breathing (apnea) detection
APNEA_RMS_THRESHOLD = 0.002  # Below this = no breathing
MIN_APNEA_DURATION = 10.0    # Minimum gap duration to report (seconds)

# Obstruction severity thresholds (absolute, calibrated from sample data)
# Based on peak RMS amplitude and spectral low-frequency ratio
OBSTRUCTION_MILD_PEAK_RMS = 0.06      # Peak RMS above this = mild
OBSTRUCTION_MILD_LOW_RATIO = 1.5      # Low-freq ratio above this = mild (snoring character)
OBSTRUCTION_MODERATE_PEAK_RMS = 0.15  # Peak RMS above this + criteria = moderate
OBSTRUCTION_MODERATE_MIN_DURATION = 4.0  # Duration threshold for moderate
OBSTRUCTION_SEVERE_PEAK_RMS = 0.20    # Peak RMS above this = severe

# Analysis window sizes
RMS_WINDOW_MS = 50   # Window for RMS envelope computation
RMS_HOP_MS = 25      # Hop size for RMS envelope

# Multi-band breathing detection (quiet breathing rejection)
MULTIBAND_FREQ_BANDS = [(50, 300), (300, 1000), (1000, 3000)]  # Hz
MULTIBAND_ENV_WINDOW_MS = 100    # RMS window for band envelopes
MULTIBAND_SMOOTH_SAMPLES = 20    # ~0.5s smoothing at 40Hz envelope rate
MULTIBAND_PEAK_SNR = 2.0         # Peak must exceed 2x noise floor
MULTIBAND_PEAK_MIN_DIST = 20     # Min samples between peaks (~0.5s)
MULTIBAND_PEAK_MIN_WIDTH = 22    # Min peak width in samples (~0.55s); rejects clicks/pops
MULTIBAND_MIN_BREATH_RATE = 3.0  # Peaks/min above this = breathing detected
MULTIBAND_CLICK_FILTER_MS = 20   # Median filter kernel for click/pop removal on raw audio


# =============================================================================
# Audio Processing Functions
# =============================================================================

def load_mp3_as_wav(mp3_path):
    """Convert MP3 to WAV in memory and return sample rate + audio array."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', mp3_path, '-ac', '1', '-ar', '24000', tmp_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        sr, audio = wavfile.read(tmp_path)
        audio = audio.astype(np.float32) / 32768.0
        return sr, audio
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def compute_rms_envelope(audio, sr, window_ms=RMS_WINDOW_MS, hop_ms=RMS_HOP_MS):
    """Compute RMS envelope of audio signal."""
    window_samples = int(sr * window_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    rms_values = []
    times = []
    
    for i in range(0, len(audio) - window_samples, hop_samples):
        rms = np.sqrt(np.mean(audio[i:i+window_samples]**2))
        rms_values.append(rms)
        times.append((i + window_samples/2) / sr)
    
    return np.array(times), np.array(rms_values)


def analyze_breath_spectrum(audio, sr, t_start, t_end):
    """Compute spectral features for obstruction scoring."""
    idx1 = int(t_start * sr)
    idx2 = int(t_end * sr)
    seg = audio[idx1:idx2]
    
    if len(seg) < 256:
        return None
    
    f, psd = signal.welch(seg, sr, nperseg=min(1024, len(seg)))
    
    # Low freq energy (snoring: 50-500 Hz)
    low_mask = (f >= 50) & (f <= 500)
    low_energy = np.sum(psd[low_mask])
    
    # Mid freq energy (500-2000 Hz)
    mid_mask = (f >= 500) & (f <= 2000)
    mid_energy = np.sum(psd[mid_mask])
    
    # High freq energy (hiss: 2000-6000 Hz)
    high_mask = (f >= 2000) & (f <= 6000)
    high_energy = np.sum(psd[high_mask])
    
    return {
        'low_energy': low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy,
        'low_ratio': low_energy / (mid_energy + high_energy + 1e-10)
    }


# =============================================================================
# Multi-band Breathing Detection (quiet breathing rejection)
# =============================================================================

def compute_multiband_envelope(audio, sr):
    """
    Compute a noise-normalized multi-band envelope optimized for detecting
    quiet breathing that falls below the broadband RMS threshold.
    
    First applies a median filter to the raw audio to remove impulsive
    clicks/pops (<20ms transients), then splits into frequency bands,
    computes RMS envelope for each, normalizes each to its own noise floor,
    and takes the geometric mean.
    
    Returns (times, envelope) where envelope values represent SNR relative
    to the noise floor (1.0 = at noise floor).
    """
    # Click/pop rejection is handled downstream by the RMS windowing (100ms),
    # peak minimum-width constraint, and peak distance requirements.
    # A median filter was previously applied here but it destroys quiet
    # broadband breathing (preserving only 2-11% of band energy at 24kHz)
    # because the breathing is spectrally white and the 20ms kernel acts as
    # an aggressive low-pass filter on the higher detection bands.
    
    band_envs = []
    t_out = None
    
    for f_low, f_high in MULTIBAND_FREQ_BANDS:
        # Bandpass filter
        sos = signal.butter(4, [f_low, f_high], btype='band', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # Compute RMS envelope
        window_samples = int(sr * MULTIBAND_ENV_WINDOW_MS / 1000)
        hop_samples = int(sr * RMS_HOP_MS / 1000)
        
        rms_vals = []
        t_vals = []
        for i in range(0, len(filtered) - window_samples, hop_samples):
            rms_vals.append(np.sqrt(np.mean(filtered[i:i+window_samples]**2)))
            t_vals.append((i + window_samples / 2) / sr)
        
        rms_arr = np.array(rms_vals)
        t_out = np.array(t_vals)
        
        # Normalize to noise floor (5th percentile)
        noise_floor = np.percentile(rms_arr, 5)
        if noise_floor > 0:
            band_envs.append(rms_arr / noise_floor)
        else:
            band_envs.append(np.ones_like(rms_arr))
    
    # Geometric mean across bands
    combined = np.ones_like(band_envs[0])
    for env in band_envs:
        combined *= env
    combined = combined ** (1.0 / len(band_envs))
    
    return t_out, combined


def check_multiband_breathing(multiband_env, t_start, t_end):
    """
    Check whether a candidate apnea region contains quiet breathing by
    counting peaks in the multi-band SNR envelope.
    
    Quiet breathing produces regular peaks at 2-5 second intervals (exhale
    events) that are clearly visible in the multi-band SNR even when they
    fall below the broadband RMS threshold.  True apnea produces at most
    isolated pops/clicks with no periodic structure.
    
    This approach works directly on the gap boundaries with no windowing,
    so it cannot be contaminated by adjacent breathing.
    
    Args:
        multiband_env: (times, values) tuple from compute_multiband_envelope
        t_start, t_end: time bounds of the candidate apnea region
        
    Returns:
        (has_breathing, peak_rate, num_peaks) where has_breathing is True
        if peak rate exceeds MULTIBAND_MIN_BREATH_RATE.
    """
    env_times, env_values = multiband_env
    
    # Extract only data within the gap boundaries
    mask = (env_times >= t_start) & (env_times <= t_end)
    region = env_values[mask]
    
    if len(region) < MULTIBAND_PEAK_MIN_DIST * 2:
        return False, 0.0, 0
    
    # Smooth the envelope
    smoothed = uniform_filter1d(region, MULTIBAND_SMOOTH_SAMPLES)
    
    # Find peaks above SNR threshold, with minimum width to reject clicks/pops
    peaks, _ = signal.find_peaks(
        smoothed,
        height=MULTIBAND_PEAK_SNR,
        distance=MULTIBAND_PEAK_MIN_DIST,
        width=MULTIBAND_PEAK_MIN_WIDTH
    )
    
    duration_min = (t_end - t_start) / 60.0
    peak_rate = len(peaks) / duration_min if duration_min > 0 else 0
    has_breathing = peak_rate >= MULTIBAND_MIN_BREATH_RATE
    
    return has_breathing, peak_rate, len(peaks)




def detect_breath_events(audio, sr):
    """
    Detect individual breath events (inhales and exhales).
    Returns list of event dicts with timing and amplitude info.
    """
    times, rms = compute_rms_envelope(audio, sr)
    smoothed = uniform_filter1d(rms, size=5)
    
    # Auto-calibrate threshold from noise floor
    noise_floor = np.percentile(rms, 5)
    breath_threshold = noise_floor * BREATH_THRESH_MULTIPLIER
    
    # Find connected regions above threshold
    above_thresh = smoothed > breath_threshold
    transitions = np.diff(above_thresh.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    # Handle edge cases
    if len(starts) == 0 and len(ends) == 0:
        if above_thresh[0]:
            # Entire signal above threshold
            starts = np.array([0])
            ends = np.array([len(above_thresh)-1])
        else:
            return [], noise_floor, breath_threshold
    
    if above_thresh[0] and (len(starts) == 0 or starts[0] > 0):
        starts = np.insert(starts, 0, 0)
    if above_thresh[-1] and (len(ends) == 0 or ends[-1] < len(above_thresh)-1):
        ends = np.append(ends, len(above_thresh)-1)
    
    # Ensure equal length
    min_len = min(len(starts), len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]
    
    breath_events = []
    for s, e in zip(starts, ends):
        if s >= len(times) or e >= len(times):
            continue
        t_start = times[s]
        t_end = times[e]
        duration = t_end - t_start
        
        if duration >= MIN_BREATH_DURATION:
            peak_rms = smoothed[s:e+1].max()
            mean_rms = smoothed[s:e+1].mean()
            
            breath_events.append({
                'start': t_start,
                'end': t_end,
                'duration': duration,
                'peak_rms': peak_rms,
                'mean_rms': mean_rms
            })
    
    return breath_events, noise_floor, breath_threshold


def identify_breath_cycles(breath_events):
    """
    Group breath events into cycles (inhale + exhale pairs).
    Uses the observation that exhale->inhale gaps are longer than inhale->exhale gaps.
    
    Returns list of cycles, each containing constituent events.
    """
    if len(breath_events) < 2:
        return breath_events  # Can't pair, return as-is
    
    # Calculate inter-event gaps
    gaps = []
    for i in range(1, len(breath_events)):
        gap = breath_events[i]['start'] - breath_events[i-1]['end']
        gaps.append(gap)
    
    gaps = np.array(gaps)
    
    # Find gap threshold to separate within-cycle from between-cycle
    # Typically the short gaps (inhale->exhale) are < 0.5s
    # and the longer gaps (exhale->inhale) are > 1s
    if len(gaps) > 0:
        gap_threshold = np.median(gaps)
    else:
        gap_threshold = 0.5
    
    # Group events into cycles
    cycles = []
    current_cycle = [breath_events[0]]
    
    for i in range(1, len(breath_events)):
        gap = breath_events[i]['start'] - breath_events[i-1]['end']
        
        if gap > gap_threshold:
            # New cycle starts
            cycles.append(current_cycle)
            current_cycle = [breath_events[i]]
        else:
            # Continue current cycle
            current_cycle.append(breath_events[i])
    
    if current_cycle:
        cycles.append(current_cycle)
    
    # Convert cycles to summary format
    cycle_summaries = []
    for cycle in cycles:
        cycle_start = cycle[0]['start']
        cycle_end = cycle[-1]['end']
        peak_rms = max(e['peak_rms'] for e in cycle)
        mean_rms = np.mean([e['mean_rms'] for e in cycle])
        
        cycle_summaries.append({
            'start': cycle_start,
            'end': cycle_end,
            'duration': cycle_end - cycle_start,
            'peak_rms': peak_rms,
            'mean_rms': mean_rms,
            'num_events': len(cycle)
        })
    
    return cycle_summaries


def compute_obstruction_scores(breath_cycles, audio, sr):
    """
    Compute obstruction severity for each breath cycle using absolute thresholds.
    
    Classification:
    - Severe:   peak_rms > 0.20
    - Moderate: peak_rms > 0.15 AND (low_ratio > 1.0 OR duration > 4s)
    - Mild:     peak_rms > 0.06 OR low_ratio > 1.5
    - Normal:   everything else
    """
    # Add spectral features
    for cycle in breath_cycles:
        spec = analyze_breath_spectrum(audio, sr, cycle['start'], cycle['end'])
        if spec:
            cycle.update(spec)
        else:
            cycle['low_ratio'] = 0
    
    # Classify using absolute thresholds
    for cycle in breath_cycles:
        peak_rms = cycle['peak_rms']
        low_ratio = cycle.get('low_ratio', 0)
        duration = cycle['duration']
        
        if peak_rms > OBSTRUCTION_SEVERE_PEAK_RMS:
            cycle['obstruction_level'] = 'severe'
        elif peak_rms > OBSTRUCTION_MODERATE_PEAK_RMS and (low_ratio > 1.0 or duration > OBSTRUCTION_MODERATE_MIN_DURATION):
            cycle['obstruction_level'] = 'moderate'
        elif peak_rms > OBSTRUCTION_MILD_PEAK_RMS or low_ratio > OBSTRUCTION_MILD_LOW_RATIO:
            cycle['obstruction_level'] = 'mild'
        else:
            cycle['obstruction_level'] = 'normal'
    
    return breath_cycles


def detect_apnea_gaps(audio, sr, breath_cycles):
    """
    Detect gaps with no detectable breathing longer than MIN_APNEA_DURATION.
    
    Strategy: 
    1. Find sustained low-energy periods (baseline below threshold)
    2. Verify these are TRUE apnea by checking for absence of breath peaks
       - Quiet breathing has periodic peaks above 0.003 even if baseline is low
       - True apnea has no such peaks
    3. Multi-band peak check: count peaks in noise-normalized frequency-band
       envelope to reject quiet breathing that falls below the broadband
       RMS threshold but produces regular breath-rate peaks
    """
    times, rms = compute_rms_envelope(audio, sr)
    
    # Threshold for "no breathing baseline" - signal mostly below this
    BASELINE_THRESHOLD = APNEA_RMS_THRESHOLD  # 0.002
    
    # Threshold for "breath peak" - any sustained peak above this indicates breathing
    BREATH_PEAK_THRESHOLD = 0.0025
    
    # Find regions where baseline is mostly low
    below_thresh = rms < BASELINE_THRESHOLD
    
    transitions = np.diff(below_thresh.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    if below_thresh[0]:
        starts = np.insert(starts, 0, 0)
    if below_thresh[-1]:
        ends = np.append(ends, len(below_thresh)-1)
    
    min_len = min(len(starts), len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]
    
    # Collect candidate gaps (must be at least 5s)
    MIN_GAP_FOR_MERGE = 5.0
    MERGE_THRESHOLD = 3.0
    
    candidate_gaps = []
    for s, e in zip(starts, ends):
        if s >= len(times) or e >= len(times):
            continue
        t_start = times[s]
        t_end = times[e]
        duration = t_end - t_start
        
        if duration >= MIN_GAP_FOR_MERGE:
            # Check if this region has any breath peaks
            region_rms = rms[s:e+1]
            max_peak = region_rms.max()
            num_breath_peaks = np.sum(region_rms > BREATH_PEAK_THRESHOLD)
            
            candidate_gaps.append({
                'start': t_start,
                'end': t_end,
                'duration': duration,
                'max_peak': max_peak,
                'num_breath_peaks': num_breath_peaks
            })
    
    # Merge nearby gaps, but only if they don't contain breath peaks
    merged_gaps = []
    
    if candidate_gaps:
        current = candidate_gaps[0].copy()
        
        for i in range(1, len(candidate_gaps)):
            gap_between = candidate_gaps[i]['start'] - current['end']
            
            if gap_between < MERGE_THRESHOLD:
                # If the gap is shorter than a full breath cycle could possibly be,
                # merge unconditionally (single inhale+exhale takes at least ~1s)
                MIN_BREATH_CYCLE_DURATION = 1.0
                if gap_between < MIN_BREATH_CYCLE_DURATION:
                    between_has_breath = False
                else:
                    # Check if the gap between has SUSTAINED breath peaks (not brief pops)
                    gap_start_idx = np.searchsorted(times, current['end'])
                    gap_end_idx = np.searchsorted(times, candidate_gaps[i]['start'])
                    between_has_breath = False
                
                    if gap_end_idx > gap_start_idx:
                        between_rms = rms[gap_start_idx:gap_end_idx]
                        # Check for sustained peaks (>= MIN_BREATH_DURATION)
                        above_thresh = between_rms > BREATH_PEAK_THRESHOLD
                        if np.any(above_thresh):
                            transitions = np.diff(above_thresh.astype(int))
                            peak_starts = np.where(transitions == 1)[0]
                            peak_ends = np.where(transitions == -1)[0]
                            if above_thresh[0]:
                                peak_starts = np.insert(peak_starts, 0, 0)
                            if above_thresh[-1]:
                                peak_ends = np.append(peak_ends, len(above_thresh)-1)
                            min_len = min(len(peak_starts), len(peak_ends))
                            # Check if any peak is sustained (>= MIN_BREATH_DURATION)
                            min_peak_samples = int(MIN_BREATH_DURATION / (RMS_HOP_MS / 1000))
                            for ps, pe in zip(peak_starts[:min_len], peak_ends[:min_len]):
                                if (pe - ps) >= min_peak_samples:
                                    between_has_breath = True
                                    break
                
                if not between_has_breath:
                    # Merge: extend current gap
                    current['end'] = candidate_gaps[i]['end']
                    current['duration'] = current['end'] - current['start']
                    current['max_peak'] = max(current['max_peak'], candidate_gaps[i]['max_peak'])
                    current['num_breath_peaks'] += candidate_gaps[i]['num_breath_peaks']
                else:
                    # Gap between has breathing, don't merge
                    if current['duration'] >= MIN_APNEA_DURATION:
                        merged_gaps.append(current)
                    current = candidate_gaps[i].copy()
            else:
                if current['duration'] >= MIN_APNEA_DURATION:
                    merged_gaps.append(current)
                current = candidate_gaps[i].copy()
        
        if current['duration'] >= MIN_APNEA_DURATION:
            merged_gaps.append(current)
    
    # Final filter: only report gaps that are TRUE apnea
    # Check 1: no significant sustained breath peaks (amplitude-based)
    # Check 2: no periodic breathing pattern (multi-band autocorrelation)
    true_apnea_gaps = []
    for gap in merged_gaps:
        # Get full region including any merged sections
        start_idx = np.searchsorted(times, gap['start'])
        end_idx = np.searchsorted(times, gap['end'])
        full_region_rms = rms[start_idx:end_idx]
        
        # Count SUSTAINED breath-like peaks (not brief pops)
        above_thresh = full_region_rms > BREATH_PEAK_THRESHOLD
        transitions = np.diff(above_thresh.astype(int))
        peak_starts = np.where(transitions == 1)[0]
        peak_ends = np.where(transitions == -1)[0]
        
        if above_thresh[0] and (len(peak_starts) == 0 or peak_starts[0] > 0):
            peak_starts = np.insert(peak_starts, 0, 0)
        if above_thresh[-1] and (len(peak_ends) == 0 or peak_ends[-1] < len(above_thresh)-1):
            peak_ends = np.append(peak_ends, len(above_thresh)-1)
        
        min_len = min(len(peak_starts), len(peak_ends))
        peak_starts, peak_ends = peak_starts[:min_len], peak_ends[:min_len]
        
        # Count sustained peaks (>= MIN_BREATH_DURATION)
        MIN_BREATH_SAMPLES = int(MIN_BREATH_DURATION / (RMS_HOP_MS / 1000))
        sustained_peaks = sum(1 for s, e in zip(peak_starts, peak_ends) if (e - s) >= MIN_BREATH_SAMPLES)
        
        duration = gap['duration']
        # If there are sustained breath peaks every <15 seconds, it's quiet breathing
        expected_breaths_if_breathing = duration / 15.0
        
        if sustained_peaks >= expected_breaths_if_breathing:
            continue  # Amplitude check: too many breath peaks
        
        # Multi-band peak check: extract the candidate audio segment and
        # check for quiet breathing peaks in the noise-normalized envelope.
        # Only computed for candidates that pass the amplitude check, so the
        # median filter + bandpass cost is limited to short segments.
        pad_sec = 2.0  # small pad for filter edge effects
        seg_start = max(0, int((gap['start'] - pad_sec) * sr))
        seg_end = min(len(audio), int((gap['end'] + pad_sec) * sr))
        seg_audio = audio[seg_start:seg_end]
        mb_envelope = compute_multiband_envelope(seg_audio, sr)
        
        # Adjust time bounds relative to segment start
        seg_t_start = gap['start'] - seg_start / sr
        seg_t_end = gap['end'] - seg_start / sr
        has_breathing, peak_rate, num_peaks = check_multiband_breathing(
            mb_envelope, seg_t_start, seg_t_end)
        
        if has_breathing:
            # Quiet breathing detected despite low broadband RMS
            continue
        
        true_apnea_gaps.append({
            'start': gap['start'],
            'end': gap['end'],
            'duration': gap['duration']
        })
    
    return true_apnea_gaps


# =============================================================================
# File Handling
# =============================================================================

def parse_filename_timestamp(filename):
    """
    Extract datetime from filename like 'ch4_20260203_220000.mp3'
    Returns datetime object or None if parsing fails.
    """
    basename = os.path.basename(filename)
    # Remove extension
    name = os.path.splitext(basename)[0]
    
    # Try to find YYYYMMDD_HHMMSS pattern
    parts = name.split('_')
    for i in range(len(parts) - 1):
        if len(parts[i]) == 8 and len(parts[i+1]) == 6:
            try:
                date_str = parts[i]
                time_str = parts[i+1]
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return dt
            except ValueError:
                continue
    
    return None


def get_wall_clock_hour(timestamp):
    """Return the wall clock hour (0-23) for a timestamp."""
    return timestamp.hour


def load_and_analyze_file(filepath):
    """
    Load a single MP3 file and perform all breath analysis.
    Returns dict with all results, or None if file couldn't be processed.
    """
    try:
        sr, audio = load_mp3_as_wav(filepath)
        duration = len(audio) / sr
        
        # Detect breath events
        breath_events, noise_floor, breath_thresh = detect_breath_events(audio, sr)
        
        # Group into cycles
        breath_cycles = identify_breath_cycles(breath_events)
        
        # Score obstruction
        breath_cycles = compute_obstruction_scores(breath_cycles, audio, sr)
        
        # Detect apnea gaps
        apnea_gaps = detect_apnea_gaps(audio, sr, breath_cycles)
        
        return {
            'filepath': filepath,
            'duration': duration,
            'sample_rate': sr,
            'noise_floor': noise_floor,
            'breath_threshold': breath_thresh,
            'breath_events': breath_events,
            'breath_cycles': breath_cycles,
            'apnea_gaps': apnea_gaps
        }
    
    except Exception as e:
        print(f"  WARNING: Could not process {filepath}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Hourly Aggregation
# =============================================================================

def aggregate_by_hour(file_results):
    """
    Aggregate results by wall-clock hour.
    Returns dict: hour -> aggregated stats
    """
    hourly_data = defaultdict(lambda: {
        'breath_cycles': [],
        'apnea_gaps': [],
        'total_duration': 0.0,
        'files': []
    })
    
    for result in file_results:
        if result is None:
            continue
        
        filepath = result['filepath']
        file_start = parse_filename_timestamp(filepath)
        if file_start is None:
            print(f"  WARNING: Could not parse timestamp from {filepath}", file=sys.stderr)
            continue
        
        file_duration = result['duration']
        
        # Assign breath cycles to hours based on their timestamps
        for cycle in result['breath_cycles']:
            cycle_time = file_start + timedelta(seconds=cycle['start'])
            hour = get_wall_clock_hour(cycle_time)
            
            cycle_with_time = cycle.copy()
            cycle_with_time['wall_clock_time'] = cycle_time
            hourly_data[hour]['breath_cycles'].append(cycle_with_time)
        
        # Assign apnea gaps to hours
        for gap in result['apnea_gaps']:
            gap_time = file_start + timedelta(seconds=gap['start'])
            hour = get_wall_clock_hour(gap_time)
            
            gap_with_time = gap.copy()
            gap_with_time['wall_clock_time'] = gap_time
            gap_with_time['file_start_sec'] = gap['start']  # Time within file
            gap_with_time['file_end_sec'] = gap['end']      # Time within file
            gap_with_time['source_file'] = os.path.basename(filepath)
            hourly_data[hour]['apnea_gaps'].append(gap_with_time)
        
        # Track duration per hour (approximate)
        # For simplicity, assign file's duration to its starting hour
        hour = get_wall_clock_hour(file_start)
        hourly_data[hour]['total_duration'] += file_duration
        hourly_data[hour]['files'].append(filepath)
    
    return dict(hourly_data)


def compute_hourly_stats(hourly_data):
    """
    Compute summary statistics for each hour.
    """
    hourly_stats = {}
    
    for hour, data in hourly_data.items():
        cycles = data['breath_cycles']
        gaps = data['apnea_gaps']
        
        # Breath cycle stats
        num_cycles = len(cycles)
        
        # Inter-cycle intervals
        if num_cycles >= 2:
            # Sort by wall clock time (not file-relative start time)
            sorted_cycles = sorted(cycles, key=lambda c: c['wall_clock_time'])
            intervals = []
            for i in range(1, len(sorted_cycles)):
                interval = (sorted_cycles[i]['wall_clock_time'] - 
                           sorted_cycles[i-1]['wall_clock_time']).total_seconds()
                if 0 < interval < 60:  # Ignore negative or very large intervals
                    intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                median_interval = np.median(intervals)
                min_interval = np.min(intervals)
                max_interval = np.max(intervals)
            else:
                avg_interval = median_interval = min_interval = max_interval = None
        else:
            avg_interval = median_interval = min_interval = max_interval = None
        
        # Obstruction counts
        mild_count = sum(1 for c in cycles if c.get('obstruction_level') == 'mild')
        moderate_count = sum(1 for c in cycles if c.get('obstruction_level') == 'moderate')
        severe_count = sum(1 for c in cycles if c.get('obstruction_level') == 'severe')
        
        # Apnea gap stats
        num_gaps = len(gaps)
        if num_gaps > 0:
            gap_durations = [g['duration'] for g in gaps]
            avg_gap = np.mean(gap_durations)
            median_gap = np.median(gap_durations)
            max_gap = np.max(gap_durations)
        else:
            avg_gap = median_gap = max_gap = None
        
        hourly_stats[hour] = {
            'hour': hour,
            'num_breath_cycles': num_cycles,
            'avg_interval_sec': avg_interval,
            'median_interval_sec': median_interval,
            'min_interval_sec': min_interval,
            'max_interval_sec': max_interval,
            'mild_obstructions': mild_count,
            'moderate_obstructions': moderate_count,
            'severe_obstructions': severe_count,
            'num_apnea_gaps': num_gaps,
            'avg_gap_duration_sec': avg_gap,
            'median_gap_duration_sec': median_gap,
            'max_gap_duration_sec': max_gap,
            'recording_duration_sec': data['total_duration'],
            'num_files': len(data['files'])
        }
    
    return hourly_stats


# =============================================================================
# Output Functions
# =============================================================================

def write_csv_report(hourly_stats, output_path):
    """Write hourly statistics to CSV file."""
    import csv
    
    # Sort by hour
    sorted_hours = sorted(hourly_stats.keys())
    
    fieldnames = [
        'hour', 'num_breath_cycles', 
        'avg_interval_sec', 'median_interval_sec', 'min_interval_sec', 'max_interval_sec',
        'mild_obstructions', 'moderate_obstructions', 'severe_obstructions',
        'num_apnea_gaps', 'avg_gap_duration_sec', 'median_gap_duration_sec', 'max_gap_duration_sec',
        'recording_duration_sec', 'num_files'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for hour in sorted_hours:
            row = hourly_stats[hour].copy()
            # Format hour as HH:00
            row['hour'] = f"{hour:02d}:00"
            # Round floats
            for key in ['avg_interval_sec', 'median_interval_sec', 'min_interval_sec', 'max_interval_sec',
                       'avg_gap_duration_sec', 'median_gap_duration_sec', 'max_gap_duration_sec',
                       'recording_duration_sec']:
                if row[key] is not None:
                    row[key] = round(row[key], 2)
            writer.writerow(row)
    
    print(f"CSV report written to: {output_path}")


def write_events_csv(hourly_data, output_path):
    """Write detailed event list with wall-clock timestamps for plotting."""
    import csv
    
    # Collect all apnea gaps
    all_gaps = []
    for hour, data in hourly_data.items():
        for gap in data['apnea_gaps']:
            all_gaps.append(gap)
    
    # Sort by wall clock time
    all_gaps.sort(key=lambda g: g['wall_clock_time'])
    
    # Write apnea events CSV
    apnea_path = output_path.replace('.csv', '_apnea_events.csv')
    with open(apnea_path, 'w', newline='') as f:
        fieldnames = ['wall_clock_time', 'duration_sec', 'source_file', 'file_start_sec', 'file_end_sec']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for gap in all_gaps:
            writer.writerow({
                'wall_clock_time': gap['wall_clock_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'duration_sec': round(gap['duration'], 1),
                'source_file': gap['source_file'],
                'file_start_sec': round(gap['file_start_sec'], 2),
                'file_end_sec': round(gap['file_end_sec'], 2)
            })
    
    print(f"Apnea events CSV written to: {apnea_path}")
    
    # Collect all obstructed breath cycles (moderate and severe)
    all_obstructions = []
    for hour, data in hourly_data.items():
        for cycle in data['breath_cycles']:
            level = cycle.get('obstruction_level', 'normal')
            if level in ('moderate', 'severe'):
                all_obstructions.append(cycle)
    
    # Sort by wall clock time
    all_obstructions.sort(key=lambda c: c['wall_clock_time'])
    
    # Write obstructions CSV
    obstruct_path = output_path.replace('.csv', '_obstructions.csv')
    with open(obstruct_path, 'w', newline='') as f:
        fieldnames = ['wall_clock_time', 'duration_sec', 'severity', 'peak_rms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for cycle in all_obstructions:
            writer.writerow({
                'wall_clock_time': cycle['wall_clock_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'duration_sec': round(cycle['duration'], 2),
                'severity': cycle.get('obstruction_level', 'unknown'),
                'peak_rms': round(cycle.get('peak_rms', 0), 4)
            })
    
    print(f"Obstructions CSV written to: {obstruct_path}")


def write_breathing_rate_csv(hourly_data, output_path):
    """Write respiratory rate (breaths/min) in 1-minute epochs with wall-clock timestamps."""
    import csv

    # Collect all breath cycles with wall-clock times
    all_cycles = []
    for hour, data in hourly_data.items():
        for cycle in data['breath_cycles']:
            all_cycles.append(cycle)

    if not all_cycles:
        print("No breath cycles to write for breathing rate CSV.")
        return

    # Sort by wall clock time
    all_cycles.sort(key=lambda c: c['wall_clock_time'])

    # Determine time range and create 1-minute bins
    t_first = all_cycles[0]['wall_clock_time']
    t_last = all_cycles[-1]['wall_clock_time']

    # Round down to start of minute
    bin_start = t_first.replace(second=0, microsecond=0)
    bin_end = t_last.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Count cycles per 1-minute bin
    bins = {}
    t = bin_start
    while t < bin_end:
        bins[t] = 0
        t += timedelta(minutes=1)

    for cycle in all_cycles:
        ct = cycle['wall_clock_time']
        bin_key = ct.replace(second=0, microsecond=0)
        if bin_key in bins:
            bins[bin_key] += 1

    # Write CSV
    rate_path = output_path.replace('.csv', '_respiratory_rate.csv')
    with open(rate_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'breaths_per_minute'])

        for t in sorted(bins.keys()):
            writer.writerow([t.strftime('%Y-%m-%d %H:%M:%S'), bins[t]])

    print(f"Respiratory rate CSV written to: {rate_path}")


def print_console_report(hourly_stats, hourly_data=None):
    """Print formatted report to console."""
    sorted_hours = sorted(hourly_stats.keys())
    
    print("\n" + "="*80)
    print("SLEEP BREATHING ANALYSIS REPORT")
    print("="*80)
    
    # Determine if this is overnight (spans midnight)
    if sorted_hours and max(sorted_hours) - min(sorted_hours) > 12:
        # Reorder for overnight display (22, 23, 0, 1, 2, ...)
        evening = [h for h in sorted_hours if h >= 18]
        morning = [h for h in sorted_hours if h < 18]
        sorted_hours = evening + morning
    
    print(f"\n{'Hour':<8} {'Cycles':>8} {'Interval (s)':>14} {'Obstructions':>20} {'Apnea Gaps':>30}")
    print(f"{'':8} {'':>8} {'avg/med':>14} {'mild/mod/sev':>20} {'count   avg   med   max':>30}")
    print("-"*80)
    
    total_cycles = 0
    total_mild = 0
    total_moderate = 0
    total_severe = 0
    total_gaps = 0
    all_gap_durations = []
    max_gap_overall = 0
    
    for hour in sorted_hours:
        stats = hourly_stats[hour]
        
        total_cycles += stats['num_breath_cycles']
        total_mild += stats['mild_obstructions']
        total_moderate += stats['moderate_obstructions']
        total_severe += stats['severe_obstructions']
        total_gaps += stats['num_apnea_gaps']
        
        hour_str = f"{hour:02d}:00"
        cycles_str = str(stats['num_breath_cycles'])
        
        if stats['avg_interval_sec'] is not None:
            interval_str = f"{stats['avg_interval_sec']:.1f}/{stats['median_interval_sec']:.1f}"
        else:
            interval_str = "-"
        
        obstruct_str = f"{stats['mild_obstructions']}/{stats['moderate_obstructions']}/{stats['severe_obstructions']}"
        
        if stats['num_apnea_gaps'] > 0:
            gap_str = f"{stats['num_apnea_gaps']:>3}  {stats['avg_gap_duration_sec']:>5.1f} {stats['median_gap_duration_sec']:>5.1f} {stats['max_gap_duration_sec']:>5.1f}"
            all_gap_durations.append(stats['avg_gap_duration_sec'])
            if stats['max_gap_duration_sec'] > max_gap_overall:
                max_gap_overall = stats['max_gap_duration_sec']
        else:
            gap_str = "  0      -     -     -"
        
        print(f"{hour_str:<8} {cycles_str:>8} {interval_str:>14} {obstruct_str:>20} {gap_str:>30}")
    
    print("-"*80)
    
    # Totals
    print(f"\n{'TOTALS':}")
    print(f"  Breath cycles:     {total_cycles}")
    print(f"  Mild obstructions: {total_mild}")
    print(f"  Moderate obst.:    {total_moderate}")
    print(f"  Severe obst.:      {total_severe}")
    print(f"  Apnea gaps (>10s): {total_gaps}")
    if all_gap_durations:
        print(f"  Gap duration avg:  {np.mean(all_gap_durations):.1f}s")
        print(f"  Gap duration max:  {max_gap_overall:.1f}s")
    
    # List long gaps (>30s) for manual review
    LONG_GAP_THRESHOLD = 30.0
    if hourly_data:
        long_gaps = []
        for hour, data in hourly_data.items():
            for gap in data['apnea_gaps']:
                if gap['duration'] >= LONG_GAP_THRESHOLD:
                    long_gaps.append(gap)
        
        if long_gaps:
            # Sort by wall clock time
            long_gaps.sort(key=lambda g: g['wall_clock_time'])
            
            print(f"\n  APNEA GAPS > {LONG_GAP_THRESHOLD:.0f}s FOR MANUAL REVIEW:")
            print(f"  {'File':<30} {'Start':<10} {'End':<10} {'Duration':<10}")
            print(f"  {'-'*60}")
            for gap in long_gaps:
                start_min = int(gap['file_start_sec'] // 60)
                start_sec = gap['file_start_sec'] % 60
                end_min = int(gap['file_end_sec'] // 60)
                end_sec = gap['file_end_sec'] % 60
                print(f"  {gap['source_file']:<30} {start_min}:{start_sec:05.2f}  {end_min}:{end_sec:05.2f}  {gap['duration']:.1f}s")
    
    print("="*80 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sleep_breathing.py <directory>")
        print("  Analyzes all .mp3 files in the specified directory")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory")
        sys.exit(1)
    
    # Find all MP3 files
    mp3_files = sorted(glob.glob(os.path.join(input_dir, "*.mp3")))
    
    if not mp3_files:
        print(f"No .mp3 files found in '{input_dir}'")
        sys.exit(1)
    
    print(f"Found {len(mp3_files)} MP3 files in '{input_dir}'")
    
    # Process each file
    file_results = []
    for i, filepath in enumerate(mp3_files):
        filename = os.path.basename(filepath)
        print(f"Processing [{i+1}/{len(mp3_files)}]: {filename}")
        
        result = load_and_analyze_file(filepath)
        file_results.append(result)
        
        if result:
            n_cycles = len(result['breath_cycles'])
            n_gaps = len(result['apnea_gaps'])
            print(f"  -> {n_cycles} breath cycles, {n_gaps} apnea gaps detected")
    
    # Count successful files
    successful = sum(1 for r in file_results if r is not None)
    print(f"\nSuccessfully processed {successful}/{len(mp3_files)} files")
    
    if successful == 0:
        print("No files could be processed. Exiting.")
        sys.exit(1)
    
    # Aggregate by hour
    hourly_data = aggregate_by_hour(file_results)
    hourly_stats = compute_hourly_stats(hourly_data)
    
    # Output
    csv_path = os.path.join(input_dir, "breathing_analysis_report.csv")
    write_csv_report(hourly_stats, csv_path)
    write_events_csv(hourly_data, csv_path)
    write_breathing_rate_csv(hourly_data, csv_path)
    print_console_report(hourly_stats, hourly_data)


if __name__ == "__main__":
    main()
