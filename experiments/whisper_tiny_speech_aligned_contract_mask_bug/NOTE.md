# Superseded -- Whisper frame-mask misalignment bug

This run of A0 (val_auc=0.9387, threshold=0.8792) was trained and evaluated with a `valid_frame_mask`
computed via `build_example`'s `center=False` framing, then truncated/padded directly onto
`WhisperFeatureExtractor`'s own output. WhisperFeatureExtractor's internal STFT actually uses
`center=True` framing (`transformers.audio_utils.spectrogram`'s default: frame `t` centered at
`t * hop_length`, `n_frames = n // hop_length`), which produces a different frame count and places
frame `i`'s receptive field at a different sample range than the `center=False` convention.

Concretely, for a typical example this mismatch caused the mask to be 2 frames shorter than the
model's actual input length, and the padding step at the end of `WhisperTurnDataset.__getitem__`
(`if len(valid_frame_mask) < n_frames: pad with False`) always padded those missing frames as
*invalid* -- meaning the model's endpoint-aware pooling (which picks the "last valid position") was
systematically prevented from ever seeing the true last ~2 frames of every window, regardless of
whether real padding was present. That's exactly the region closest to the detected speech
boundary -- the most diagnostically important part of the input for this task.

Fixed in `tinyturn.whisper_dataset.extract_whisper_features` (derives the mask from
`valid_sample_mask` via `tinyturn.preprocess.frame_valid_mask(..., center=True)`, matching
WhisperFeatureExtractor's real framing) and `tinyturn.preprocess.frame_valid_mask`'s new `center`
parameter. Verified against the real feature extractor's own frame count and boundary alignment in
`tests/test_whisper_dataset.py`.

This checkpoint and its metrics are kept only as a historical/negative record (this mask bug, not
just a training-data artifact) -- do not use it as A0's qualifying checkpoint. See
`experiments/whisper_tiny_speech_aligned_contract/` for the corrected retrain.
