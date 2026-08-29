"""
Canonical speech-end (turn-boundary) estimator.

Extracted from `scripts_part3/d2_decode_transcribe.py::compute_signal_features_v3` (estimator "A" /
the fixed energy threshold, per `e5_prediction_deltas.py`'s naming) -- this is the one boundary
convention reused identically across D2, D6 and E5 throughout Part 3 of the EDA. It is kept
standalone here (it was inlined three times upstream) so Step 1's preprocessing pipeline has one
place that defines "canonical v0 boundary", per the brief's Step 1 instruction to pick one estimator
for reproducibility while keeping alternatives computable later for VAD-boundary augmentation
(Step 9).

Do not change the threshold constants below without re-running D2/D6/E5 -- cached `last_active_t`
values in data_cache/d2_stratified_signal_features.parquet were produced with exactly this formula,
and preprocess.py prefers the cached value (falling back to recomputation here only when a cached
value isn't available) specifically so results stay numerically identical to the rest of the EDA.
"""
from dataclasses import dataclass

import numpy as np
import librosa

FRAME_LENGTH_S = 0.032
HOP_LENGTH_S = 0.010
MIN_PAUSE_S = 0.1


@dataclass
class BoundaryEstimate:
    speech_end_s: float          # last_active_t: end of the last energy-active run
    duration_s: float
    trailing_silence_s: float
    internal_pause_spans_s: list  # [(start_s, end_s), ...] interior pauses >= MIN_PAUSE_S


def segment_runs(active: np.ndarray, times: np.ndarray):
    """Run-length-encode a boolean activity mask into (start_s, end_s, is_active) tuples."""
    runs = []
    if len(active) == 0:
        return runs
    run_start = 0
    cur = bool(active[0])
    for i in range(1, len(active)):
        if bool(active[i]) != cur:
            runs.append((float(times[run_start]), float(times[i - 1]), cur))
            run_start = i
            cur = bool(active[i])
    runs.append((float(times[run_start]), float(times[-1]), cur))
    return runs


def estimate_speech_end(y: np.ndarray, sr: int) -> BoundaryEstimate:
    """Canonical v0 boundary estimator (fixed energy threshold on frame RMS).

    thresh_db = max(noise_floor_db + 10, peak_db - 40), noise_floor_db = 10th percentile of
    frame RMS-dB. Same formula as d2/d6/e5. Returns the end of the final active run as the
    detected speech-end (turn boundary).
    """
    y = np.asarray(y, dtype=np.float32)
    duration = len(y) / sr
    frame_length = int(FRAME_LENGTH_S * sr)
    hop_length = int(HOP_LENGTH_S * sr)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(np.maximum(rms, 1e-6))

    if len(rms_db) == 0:
        return BoundaryEstimate(speech_end_s=duration, duration_s=duration,
                                 trailing_silence_s=0.0, internal_pause_spans_s=[])

    noise_floor_db = float(np.percentile(rms_db, 10))
    peak_db = float(rms_db.max())
    thresh_db = max(noise_floor_db + 10, peak_db - 40)
    active = rms_db > thresh_db

    runs = segment_runs(active, times)
    speech_end_s = duration
    for start, end, is_act in runs[::-1]:
        if is_act:
            speech_end_s = end
            break
    trailing_silence_s = max(duration - speech_end_s, 0.0)

    internal_pause_spans = []
    if len(runs) >= 3:
        for start, end, is_act in runs[1:-1]:
            if not is_act and (end - start) >= MIN_PAUSE_S:
                internal_pause_spans.append((start, end))

    return BoundaryEstimate(
        speech_end_s=float(speech_end_s),
        duration_s=float(duration),
        trailing_silence_s=float(trailing_silence_s),
        internal_pause_spans_s=internal_pause_spans,
    )
