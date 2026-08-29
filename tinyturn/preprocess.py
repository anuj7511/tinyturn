"""
Step 1 / Phase-2 8b -- shared preprocessing: build one model-ready example from one dataset row's
waveform.

Input definition (Phase-2 brief, Section 0/8b -- supersedes the original Step 1 contract): `N`
seconds ending exactly at the detected speech end. No post-roll is baked into model input. The
200ms silence wait that used to be appended here is now runtime VAD policy (decided at the point a
turn-taking system chooses to act on a prediction), not part of what the model is trained or
evaluated on. This was locked on structural grounds -- it removes the possibility of a
variable-valid-post-roll region altogether (every window's right edge is exactly the detected
boundary, so there is no "did this clip have enough trailing audio to fill the post-roll" case left
to confound results) and gives internal-pause events and final-turn events the same convention (both
are just "N seconds ending at a boundary", differing only in which boundary).

This module is deliberately the *only* place that implements the input-construction contract, so
every experiment that consumes examples through `build_example` gets identical framing, padding and
masking by construction.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Example:
    waveform: np.ndarray          # float32, shape (context_s * sample_rate,), fixed length
    sample_rate: int
    label: bool                   # endpoint_bool: True = turn complete
    valid_sample_mask: np.ndarray  # bool, same length as waveform; False where left-padded
    valid_frame_mask: np.ndarray   # bool, per analysis-frame (see frame_valid_mask below)
    speech_end_sample: int         # index, in the output waveform, of the detected speech end --
                                    # always == len(waveform): the window ends exactly there now.
    source: Optional[str]          # dataset source name
    language: Optional[str]
    synthetic: Optional[bool]
    endfiller: Optional[bool]      # None = unknown (real audio without ASR-derived label merged in)
    id: str
    context_s: float
    left_pad_s: float = 0.0
    right_pad_s: float = 0.0       # structurally always 0.0 under this contract (see build_example);
                                    # kept for API stability and as a defensive/diagnostic field.
    extra: dict = field(default_factory=dict)


def frame_valid_mask(valid_sample_mask: np.ndarray, frame_length: int, hop_length: int,
                      center: bool = False) -> np.ndarray:
    """A frame is valid if a majority of its samples are valid (non-padded).

    center=False (default): frame i covers samples [i*hop, i*hop+frame_length) -- matches this
    module's own log-mel branch (librosa, center=False).
    center=True: frame i is centered at sample i*hop, covering [i*hop-frame_length//2,
    i*hop+frame_length//2), with the trailing partial frame dropped (n_frames = n // hop_length) --
    matches WhisperFeatureExtractor's actual internal framing (torch.stft's default center=True,
    reflect-padded convention, which drops the same trailing frame). Required for any mask consumed
    alongside WhisperFeatureExtractor output: the two conventions place frame i's window at
    different sample offsets, so using the center=False mask there mislabels the valid/padding
    boundary frame by about one frame.
    """
    n = len(valid_sample_mask)
    if center:
        half = frame_length // 2
        n_frames = n // hop_length
        out = np.empty(n_frames, dtype=bool)
        for i in range(n_frames):
            c = i * hop_length
            seg = valid_sample_mask[max(c - half, 0):min(c + half, n)]
            out[i] = seg.size > 0 and seg.mean() > 0.5
        return out
    if n < frame_length:
        return np.zeros(0, dtype=bool)
    n_frames = 1 + (n - frame_length) // hop_length
    out = np.empty(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * hop_length
        seg = valid_sample_mask[start:start + frame_length]
        out[i] = seg.mean() > 0.5
    return out


def build_example(
    y: np.ndarray,
    sr: int,
    speech_end_s: float,
    context_s: float,
    *,
    frame_length_s: float = 0.025,
    hop_length_s: float = 0.010,
    label: Optional[bool] = None,
    row_id: str = "",
    source: Optional[str] = None,
    language: Optional[str] = None,
    synthetic: Optional[bool] = None,
    endfiller: Optional[bool] = None,
    training: bool = False,  # reserved for Step 9 boundary jitter; ignored here by design (Step 1
                              # has no augmentation yet) -- kept so callers/tests can assert that
                              # train-mode and eval-mode calls are identical while this is the case.
) -> Example:
    """
    1. decode waveform            -- caller's responsibility (y, sr already decoded)
    2. find/read canonical speech boundary -- caller passes speech_end_s (tinyturn.boundary); for an
       internal-pause event, caller passes the pause's own start time instead (tinyturn.pause_events)
    3. retain exactly N seconds ending at that boundary -- no post-roll appended
    4. pad on the left if there isn't N seconds of audio before the boundary
    5. produce valid-sample and valid-frame masks
    6. return waveform, label, metadata, and masks
    7. deterministic: pure function of (y, sr, speech_end_s, context_s) -- `training` does not
       change behavior in this implementation.
    """
    y = np.asarray(y, dtype=np.float32)
    n_total_in = len(y)

    context_samples = int(round(context_s * sr))
    out_len = context_samples

    speech_end_sample_global = int(round(speech_end_s * sr))
    speech_end_sample_global = min(max(speech_end_sample_global, 0), n_total_in)

    window_start_global = speech_end_sample_global - context_samples
    window_end_global = speech_end_sample_global

    left_pad = max(0, -window_start_global)
    content_start_global = max(window_start_global, 0)
    content_end_global = min(window_end_global, n_total_in)
    content = y[content_start_global:content_end_global]
    # Provably 0 given the clamp above (window_end_global == speech_end_sample_global <=
    # n_total_in always) -- kept as a defensive clamp, not a real code path, in case a caller passes
    # a speech_end_s that rounds past the waveform it was measured from.
    right_pad = max(out_len - left_pad - len(content), 0)

    waveform = np.zeros(out_len, dtype=np.float32)
    waveform[left_pad:left_pad + len(content)] = content

    valid_sample_mask = np.zeros(out_len, dtype=bool)
    valid_sample_mask[left_pad:left_pad + len(content)] = True

    # By construction the speech end always lands exactly at the output's final sample -- left
    # padding exists precisely to preserve this invariant when there isn't N seconds of audio before
    # the boundary.
    speech_end_sample_local = out_len

    frame_length = int(round(frame_length_s * sr))
    hop_length = int(round(hop_length_s * sr))
    vfm = frame_valid_mask(valid_sample_mask, frame_length, hop_length)

    return Example(
        waveform=waveform,
        sample_rate=sr,
        label=bool(label) if label is not None else None,
        valid_sample_mask=valid_sample_mask,
        valid_frame_mask=vfm,
        speech_end_sample=speech_end_sample_local,
        source=source,
        language=language,
        synthetic=synthetic,
        endfiller=endfiller,
        id=row_id,
        context_s=context_s,
        left_pad_s=left_pad / sr,
        right_pad_s=right_pad / sr,
    )
