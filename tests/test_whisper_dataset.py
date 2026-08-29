"""Verifies the Phase-2 8e finding and its fix: WhisperFeatureExtractor's own per-utterance log-mel
normalization (`log_spec.max() - 8.0`, computed over the *whole* waveform) is sensitive to whatever
raw content sits in a build_example waveform's padded region -- a vulnerability upstream of Section
8c's model-side mel canonicalization. `silence_invalid_samples` / `extract_whisper_features`
(tinyturn.whisper_dataset) close it at the actual source."""
import numpy as np
import pytest

from tinyturn.whisper_dataset import silence_invalid_samples, extract_whisper_features


def test_silence_invalid_samples_zeros_only_invalid_region():
    waveform = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    valid = np.array([False, False, True, True, True])
    out = silence_invalid_samples(waveform, valid)
    np.testing.assert_array_equal(out[:2], [0.0, 0.0])
    np.testing.assert_array_equal(out[2:], waveform[2:])


def test_extractor_output_is_invariant_to_padded_content_after_silencing():
    """Without silencing, two waveforms differing only in the padded region produce different
    log-mel values on the *valid* frames (WhisperFeatureExtractor's own global-max normalization
    leaking across the padding boundary). After silencing, they're identical."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    sr = 16000
    rng = np.random.RandomState(0)
    n_valid = 20000
    valid_content = (rng.uniform(-0.3, 0.3, n_valid)).astype(np.float32)
    n_pad = 12000
    valid_mask = np.array([False] * n_pad + [True] * n_valid)

    quiet = np.concatenate([np.zeros(n_pad, dtype=np.float32), valid_content])
    loud = np.concatenate([(rng.normal(0, 5.0, n_pad)).astype(np.float32), valid_content])

    feats_quiet_raw = fe(quiet, sampling_rate=sr, padding=False, return_tensors="np")["input_features"][0]
    feats_loud_raw = fe(loud, sampling_rate=sr, padding=False, return_tensors="np")["input_features"][0]
    n_frames = feats_quiet_raw.shape[1]
    frame_hop = n_pad / n_frames if n_frames else 0  # rough: valid samples start after n_pad
    valid_frame_start = int(n_pad / (sr * 0.010))  # matches HOP_LENGTH_S used elsewhere
    # Sanity: raw extraction on the two variants disagrees somewhere on the valid frames (the bug).
    assert not np.allclose(feats_quiet_raw[:, valid_frame_start:], feats_loud_raw[:, valid_frame_start:],
                            atol=1e-4)

    input_features_quiet, _ = extract_whisper_features(fe, quiet, valid_mask, sr)
    input_features_loud, _ = extract_whisper_features(fe, loud, valid_mask, sr)
    # After silencing, identical valid content -> byte-identical features regardless of what was in
    # the padded region beforehand.
    np.testing.assert_array_equal(input_features_quiet, input_features_loud)


def test_frame_mask_matches_extractor_frame_count_and_boundary():
    """extract_whisper_features must align its mask to WhisperFeatureExtractor's own frame
    convention (center=True, frame t covers samples centered at t*hop_length), not build_example's
    center=False convention -- the two disagree both on total frame count and on which frame index
    covers a given sample range. Ground truth for "which frame is genuinely untouched by real
    content" is derived directly from the extractor's own output: for an all-zero (padded) region
    embedded in an otherwise loud utterance, every frame whose receptive field is *entirely* within
    the padded region comes out as one uniform "floor" value (the per-utterance clamp value) on
    every mel bin; any frame with even partial real content deviates from it."""
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    sr = 16000
    n_pad, n_valid = 8000, 8000  # 0.5s padding, 0.5s loud content
    rng = np.random.RandomState(1)
    valid_content = rng.uniform(-0.9, 0.9, n_valid).astype(np.float32)
    waveform = np.concatenate([np.zeros(n_pad, dtype=np.float32), valid_content])
    valid_mask = np.array([False] * n_pad + [True] * n_valid)

    feats = fe(waveform, sampling_rate=sr, padding=False, return_tensors="np")["input_features"][0]
    n_frames = feats.shape[1]
    assert n_frames == len(waveform) // 160  # WhisperFeatureExtractor's actual (center=True) count

    floor = feats[0, 0]
    is_floor_frame = np.all(np.isclose(feats, floor, atol=1e-5), axis=0)
    last_pure_floor_idx = np.argmax(~is_floor_frame) - 1
    assert last_pure_floor_idx > 0  # sanity: the test setup actually has some untouched frames

    _, vfm = extract_whisper_features(fe, waveform, valid_mask, sr)
    assert len(vfm) == n_frames
    # every frame strictly before the last pure-floor frame must be marked invalid (majority padding)
    assert not vfm[:last_pure_floor_idx].any()
    # frames well past the boundary, deep in real content, must be marked valid
    assert vfm[last_pure_floor_idx + 5:].all()
