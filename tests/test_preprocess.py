"""Unit tests for the input-construction contract (tinyturn_fusion_implementation_brief.md Section
9, items 1-8; contract itself superseded by Phase-2 8b -- speech-aligned, no baked-in post-roll)."""
import numpy as np
import pytest

from tinyturn.boundary import estimate_speech_end, segment_runs
from tinyturn.preprocess import build_example, frame_valid_mask


SR = 16000


def _synthetic_clip(duration_s=3.0, speech_end_s=2.0, sr=SR, seed=0):
    """A clip that's active (noise-like) up to speech_end_s then silent -- a clean, unambiguous
    boundary for testing (real formula behavior is exercised separately against cached values)."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    active_n = int(speech_end_s * sr)
    y[:active_n] = rng.uniform(-0.5, 0.5, size=active_n).astype(np.float32)
    return y


def test_boundary_finds_last_active_run():
    y = _synthetic_clip(duration_s=3.0, speech_end_s=2.0)
    est = estimate_speech_end(y, SR)
    assert abs(est.speech_end_s - 2.0) < 0.05
    assert est.trailing_silence_s > 0.9


def test_segment_runs_basic():
    active = np.array([True, True, False, False, True])
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    runs = segment_runs(active, times)
    assert runs == [(0.0, 0.1, True), (0.2, 0.3, False), (0.4, 0.4, True)]


def test_build_example_no_padding_needed():
    """speech well past N seconds in => no left pad; window ends exactly at the boundary, so there
    is no post-roll region left to need trailing audio for."""
    y = _synthetic_clip(duration_s=5.0, speech_end_s=3.0)
    ex = build_example(y, SR, speech_end_s=3.0, context_s=1.0, label=True, row_id="x")
    expected_len = int(1.0 * SR)
    assert len(ex.waveform) == expected_len
    assert ex.left_pad_s == 0.0
    assert ex.right_pad_s == 0.0
    assert ex.valid_sample_mask.all()
    assert ex.speech_end_sample == expected_len


def test_build_example_left_padding_when_boundary_too_early():
    """speech_end_s < context_s => must left-pad, and the padded region must be masked invalid."""
    y = _synthetic_clip(duration_s=2.0, speech_end_s=0.4)
    ex = build_example(y, SR, speech_end_s=0.4, context_s=1.0, label=False, row_id="y")
    expected_len = int(1.0 * SR)
    assert len(ex.waveform) == expected_len
    assert ex.left_pad_s == pytest.approx(0.6, abs=1e-3)
    left_pad_samples = int(round(ex.left_pad_s * SR))
    assert not ex.valid_sample_mask[:left_pad_samples].any()
    assert ex.valid_sample_mask[left_pad_samples:left_pad_samples + 10].all()
    # speech end must still land exactly at the output's final sample, regardless of padding
    assert ex.speech_end_sample == expected_len
    assert ex.right_pad_s == 0.0


def test_right_padding_never_occurs_under_speech_aligned_contract():
    """Phase-2 8b: the window's right edge is always the detected speech end, which is always <=
    the clip's own duration -- so unlike the old N+200ms-post-roll contract (~30% of clips ended
    before the post-roll edge), there is structurally no case left where the output needs
    right-padding. Exercise the exact-boundary edge case: speech end == clip duration."""
    y = _synthetic_clip(duration_s=1.0, speech_end_s=1.0)  # speech runs all the way to the last sample
    ex = build_example(y, SR, speech_end_s=1.0, context_s=1.0, label=True, row_id="z")
    assert ex.right_pad_s == 0.0
    assert ex.valid_sample_mask.all()


def test_build_example_deterministic_train_vs_eval():
    """Item 8: identical output in training and inference mode."""
    y = _synthetic_clip(duration_s=4.0, speech_end_s=2.5, seed=7)
    kwargs = dict(y=y, sr=SR, speech_end_s=2.5, context_s=1.0, label=True, row_id="det")
    ex_train = build_example(**kwargs, training=True)
    ex_eval = build_example(**kwargs, training=False)
    np.testing.assert_array_equal(ex_train.waveform, ex_eval.waveform)
    np.testing.assert_array_equal(ex_train.valid_sample_mask, ex_eval.valid_sample_mask)
    np.testing.assert_array_equal(ex_train.valid_frame_mask, ex_eval.valid_frame_mask)
    assert ex_train.speech_end_sample == ex_eval.speech_end_sample


def test_build_example_configurable_context_length():
    y = _synthetic_clip(duration_s=6.0, speech_end_s=5.0)
    for n_s in [0.5, 1.0, 2.0, 4.0]:
        ex = build_example(y, SR, speech_end_s=5.0, context_s=n_s, label=True, row_id=f"ctx{n_s}")
        assert len(ex.waveform) == int(round(n_s * SR))


def test_valid_frame_mask_flags_padded_frames():
    valid = np.array([False] * 100 + [True] * 400)
    vfm = frame_valid_mask(valid, frame_length=200, hop_length=100)
    # frame 0: samples [0,200) all invalid -> False
    assert vfm[0] == False  # noqa: E712
    # a frame fully inside the valid region -> True
    assert vfm[-1] == True  # noqa: E712


def test_output_fields_present():
    y = _synthetic_clip(duration_s=3.0, speech_end_s=2.0)
    ex = build_example(y, SR, speech_end_s=2.0, context_s=1.0, label=True,
                        row_id="full", source="chirp3_1", language="eng", synthetic=True,
                        endfiller=False)
    for field in ["waveform", "sample_rate", "label", "valid_sample_mask", "valid_frame_mask",
                  "speech_end_sample", "source", "language", "synthetic", "endfiller", "id"]:
        assert hasattr(ex, field)
