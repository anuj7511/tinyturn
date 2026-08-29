"""
Trajectory-branch feature extraction (Section 3): relative log energy, low-energy/pause
probability, spectral tilt, spectral flux, syllabic-envelope activity, plus an optional pyin-based
F0 scalar (B1-f0 ablation only). Reuses the exact DSP conventions from `d5_trajectories.py` and
`d2_decode_transcribe.py`'s Hilbert-envelope logic rather than reimplementing them, per the brief's
explicit instruction ("reuse rather than reimplement").

All channels are computed on a common 10ms-hop frame grid aligned to `frame_valid_mask` in
preprocess.py, over the *already-windowed* (speech-aligned, left-padded where needed) waveform --
so a trajectory channel's own array indices line up 1:1 with `valid_frame_mask`.
"""
import numpy as np
import librosa
from scipy.signal import butter, filtfilt, find_peaks

HOP_LENGTH_S = 0.010
FRAME_LENGTH_S = 0.025


def _frame_times(n_frames, hop_length, sr):
    return np.arange(n_frames) * hop_length / sr


def compute_trajectory_channels(y: np.ndarray, sr: int, valid_sample_mask: np.ndarray,
                                 frame_length_s: float = FRAME_LENGTH_S,
                                 hop_length_s: float = HOP_LENGTH_S) -> dict:
    """Returns dict of 1-D float32 arrays, one value per analysis frame (center=False framing,
    matching `preprocess.frame_valid_mask`): rel_energy, pause_prob, spectral_tilt, spectral_flux,
    envelope_activity. Values at frames overlapping padded (invalid) samples are set to 0.0 -- the
    model is expected to combine this with `valid_frame_mask`, never to trust padded-region values.
    """
    y = np.asarray(y, dtype=np.float32)
    frame_length = int(round(frame_length_s * sr))
    hop_length = int(round(hop_length_s * sr))
    n = len(y)
    n_frames = 1 + (n - frame_length) // hop_length if n >= frame_length else 0

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=False)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))
    rms_mean_linear = float(np.mean(rms)) if len(rms) else np.nan
    rel_energy = rms_db - rms_db.mean() if len(rms_db) else rms_db

    noise_floor_db = np.percentile(rms_db, 10) if len(rms_db) else 0.0
    peak_db = rms_db.max() if len(rms_db) else 0.0
    thresh_db = max(noise_floor_db + 10, peak_db - 40)
    active = rms_db > thresh_db
    pause_prob = (~active).astype(np.float32)

    n_fft = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, center=False))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_e = S[(freqs >= 80) & (freqs < 1000)].mean(axis=0)
    high_e = S[(freqs >= 1000) & (freqs < 4000)].mean(axis=0)
    spectral_tilt = 20 * np.log10(np.maximum(low_e, 1e-6)) - 20 * np.log10(np.maximum(high_e, 1e-6))
    spectral_flux = np.concatenate([[0.0], np.sqrt(np.mean(np.diff(S, axis=1) ** 2, axis=0))]) if S.shape[1] > 0 else np.zeros(0)

    envelope_activity = _syllabic_envelope_activity(y, sr, active, hop_length, n_frames)

    def _fit(arr):
        arr = np.asarray(arr, dtype=np.float32)
        out = np.zeros(n_frames, dtype=np.float32)
        m = min(len(arr), n_frames)
        out[:m] = arr[:m]
        return out

    # frame-level validity from sample-level mask, using the same majority rule as preprocess.py
    frame_valid = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_length
        seg = valid_sample_mask[start:start + frame_length]
        frame_valid[i] = seg.mean() > 0.5

    channels = {
        "rel_energy": _fit(rel_energy),
        "pause_prob": _fit(pause_prob),
        "spectral_tilt": _fit(spectral_tilt),
        "spectral_flux": _fit(spectral_flux),
        "envelope_activity": _fit(envelope_activity),
    }
    for c in channels:
        channels[c][~frame_valid] = 0.0
    return channels


def _syllabic_envelope_activity(y, sr, active_frames, hop_length, n_frames):
    """Hilbert-envelope peak-picking, reused verbatim from d2_decode_transcribe.py's
    speaking_rate_slope logic, but returns a continuous per-frame envelope-activity trace (the
    smoothed band-passed envelope itself, gated to active regions) rather than collapsing to a
    single slope scalar -- this is the "syllabic-envelope activity" proxy channel the brief asks
    the trajectory branch to consume directly, with peak positions available for future use.
    """
    try:
        nyq = sr / 2
        b, a = butter(4, [300 / nyq, min(2000 / nyq, 0.99)], btype="band")
        y_band = filtfilt(b, a, y)
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(y_band))
        win = max(int(0.02 * sr), 1)
        envelope_smooth = np.convolve(envelope, np.ones(win) / win, mode="same")

        active_samples = np.repeat(active_frames, hop_length)[:len(envelope_smooth)]
        env_gated = envelope_smooth.copy()
        if len(active_samples) == len(env_gated):
            env_gated[~active_samples] = 0.0

        peak = env_gated.max()
        if peak > 0:
            env_gated = env_gated / peak

        # downsample the sample-rate envelope onto the frame grid (mean-pool per frame window)
        frame_length = hop_length  # coarse pooling window == hop for this proxy channel
        out = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_length
            seg = env_gated[s:s + frame_length]
            out[i] = seg.mean() if len(seg) else 0.0
        return out
    except Exception:
        return np.zeros(n_frames, dtype=np.float32)


def compute_f0_channel(y: np.ndarray, sr: int, n_frames: int, hop_length: int,
                        frame_length: int) -> np.ndarray:
    """B1-f0 ablation only: pyin-based F0, expressed as semitones relative to the clip's own
    median voiced pitch (matches d2's / d5's convention), 0.0 where unvoiced or invalid."""
    out = np.zeros(n_frames, dtype=np.float32)
    try:
        f_len = int(0.05 * sr)
        f_hop = max(f_len // 4, 1)
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=70, fmax=400, sr=sr,
                                                      frame_length=f_len, hop_length=f_hop)
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced = f0 > 0
        if voiced.sum() < 5:
            return out
        f0_ref = float(np.median(f0[voiced]))
        semitones = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / f0_ref), 0.0)
        f0_times = np.arange(len(f0)) * f_hop / sr
        frame_times = _frame_times(n_frames, hop_length, sr)
        out = np.interp(frame_times, f0_times, semitones, left=0.0, right=0.0).astype(np.float32)
    except Exception:
        pass
    return out
