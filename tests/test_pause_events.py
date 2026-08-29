"""Unit tests for internal-pause event extraction and windowing under Phase-2 8b: a pause event's
input window ends exactly at the pause's start (no post-roll), and only pauses long enough to
matter *and* followed by real speech qualify."""
import numpy as np
import pytest

from tinyturn.boundary import estimate_speech_end
from tinyturn.preprocess import build_example
from tinyturn.pause_events import extract_pause_events_for_clip, RUNTIME_TRIGGER_S

SR = 16000


def _segment(duration_s, active, rng, seed_offset=0):
    n = int(duration_s * SR)
    if not active:
        return np.zeros(n, dtype=np.float32)
    return rng.uniform(-0.5, 0.5, size=n).astype(np.float32)


def _clip_with_pauses(*segments, seed=0):
    """segments: list of (duration_s, is_active) tuples, concatenated in order."""
    rng = np.random.RandomState(seed)
    return np.concatenate([_segment(d, a, rng) for d, a in segments])


def test_short_internal_pause_is_ineligible_but_long_one_is():
    # active 1.0s, pause 0.05s (< RUNTIME_TRIGGER_S) -- too short to be a real event, active 1.0s,
    # pause 0.3s (>= RUNTIME_TRIGGER_S) -- eligible, active 1.0s, trailing silence 0.5s (not
    # "internal" -- it's the clip's own final run, excluded by construction).
    y = _clip_with_pauses((1.0, True), (0.05, False), (1.0, True), (0.3, False),
                           (1.0, True), (0.5, False))
    events = extract_pause_events_for_clip("clip_a", y, SR)
    assert len(events) == 1
    ev = events[0]
    assert ev["pause_duration_s"] >= RUNTIME_TRIGGER_S
    assert ev["pause_duration_s"] == pytest.approx(0.3, abs=0.05)
    # the pause must fall roughly after the first two active+short-pause segments (~2.05s in)
    assert 1.8 < ev["pause_start_s"] < 2.3


def test_extracted_pause_always_has_speech_resuming_after_it():
    """Phase-2 8b: an event only qualifies if speech demonstrably resumes after the pause -- this
    holds by construction (tinyturn.boundary never records a trailing pause as "internal"), verified
    here directly rather than just assumed."""
    y = _clip_with_pauses((0.8, True), (0.25, False), (0.8, True), (0.4, False), (0.8, True))
    est = estimate_speech_end(y, SR)
    events = extract_pause_events_for_clip("clip_b", y, SR)
    assert len(events) == 2
    for ev in events:
        # there must be real (active) audio strictly after this pause's end, before the clip's own
        # final speech_end_s -- i.e. this pause is not the trailing "end of the clip" silence.
        assert ev["pause_end_s"] < est.speech_end_s


def test_max_events_per_clip_keeps_longest_pauses_first():
    y = _clip_with_pauses((0.5, True), (0.25, False), (0.5, True), (0.6, False),
                           (0.5, True), (0.3, False), (0.5, True))
    events = extract_pause_events_for_clip("clip_c", y, SR, max_events=1)
    assert len(events) == 1
    assert events[0]["pause_duration_s"] == pytest.approx(0.6, abs=0.05)  # the longest one kept


def test_pause_event_window_ends_at_pause_start_with_no_postroll():
    """The defining Phase-2 8b behavior for pause events: build_example's window ends exactly at
    the pause's start, not 200ms into it -- so the output's final samples are the real audio
    immediately preceding the pause, and there is zero right-padding."""
    y = _clip_with_pauses((1.5, True), (0.4, False), (1.0, True))
    events = extract_pause_events_for_clip("clip_d", y, SR)
    assert len(events) == 1
    pause_start_s = events[0]["pause_start_s"]

    ex = build_example(y, SR, pause_start_s, context_s=1.0, label=False, row_id="pause0")
    assert ex.right_pad_s == 0.0
    assert ex.speech_end_sample == len(ex.waveform)
    # the window's own final samples should be real (non-padded) speech, since there's plenty of
    # active audio directly before the pause
    assert ex.valid_sample_mask[-10:].all()
