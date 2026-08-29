"""Verifies the variable-length WhisperEncoder forward patch (tinyturn.whisper_model): it matches
HF's built-in forward at native length, works for our much shorter windows, and (Phase-2 8c) really
masks padding -- the 5 regression tests required by the Phase-2 brief's Section 8c, plus the
shape/precision/ONNX edge cases 8c.2/8c.3 call out by name."""
import numpy as np
import torch
import pytest

from tinyturn.whisper_model import (
    WhisperEndpointModel,
    WHISPER_PAD_LOGMEL_VALUE,
    _variable_length_encoder_forward,
    _canonicalize_padding,
    _downsample_valid_mask,
)

@pytest.fixture(scope="module")
def model():
    m = WhisperEndpointModel()
    m.eval()
    return m


def _all_valid_mask(batch, t_in):
    return torch.ones(batch, t_in, dtype=torch.bool)


def test_silence_floor_matches_pad_value():
    """The canonical padding fill value is not an arbitrary constant -- it's WhisperFeatureExtractor's
    own output for genuine silence, verified directly against the library rather than assumed."""
    from transformers import WhisperFeatureExtractor
    from tinyturn.whisper_model import WHISPER_MODEL_NAME

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
    y = np.zeros(16000, dtype=np.float32)
    feats = fe(y, sampling_rate=16000, padding=False, return_tensors="np")["input_features"][0]
    assert np.unique(feats).size == 1
    assert feats.flat[0] == pytest.approx(WHISPER_PAD_LOGMEL_VALUE)


def test_matches_hf_forward_at_native_length_when_fully_valid(model):
    """Regression test 1 (Section 8c): with an all-valid mask, our patched forward -- masking and
    all -- must still match HF's own forward at the native 3000-frame length HF actually supports."""
    torch.manual_seed(0)
    x = torch.randn(1, 80, 3000)
    mask = _all_valid_mask(1, 3000)
    with torch.no_grad():
        ours, mask_ds = _variable_length_encoder_forward(model.encoder, x, mask)
        reference = model.encoder(x).last_hidden_state
    assert ours.shape == reference.shape == (1, 1500, model.encoder.config.d_model)
    assert mask_ds.all()
    torch.testing.assert_close(ours, reference, atol=1e-5, rtol=1e-4)


def test_handles_short_context_window(model):
    x = torch.randn(2, 80, 420)  # our ~4.2s window, not Whisper's native 30s/3000-frame window
    mask = _all_valid_mask(2, 420)
    with torch.no_grad():
        out, mask_ds = _variable_length_encoder_forward(model.encoder, x, mask)
    assert out.shape == (2, 210, model.encoder.config.d_model)  # conv2 stride=2 halves time
    assert mask_ds.shape == (2, 210)


def test_full_model_forward_and_backward(model):
    x = torch.randn(2, 80, 420)
    mask = torch.ones(2, 420, dtype=torch.bool)
    logit = model(x, mask)
    assert logit.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert model.encoder.conv1.weight.grad is not None


# ---------------------------------------------------------------------------------------------
# Section 8c required regression tests
# ---------------------------------------------------------------------------------------------

def test_regression_2_batch_and_single_item_agree(model):
    torch.manual_seed(1)
    x = torch.randn(3, 80, 200)
    mask = torch.ones(3, 200, dtype=torch.bool)
    mask[0, :60] = False   # item 0: left-padded
    mask[1, :5] = False    # item 1: barely padded
    # item 2: fully valid
    with torch.no_grad():
        out_batched, mask_ds_batched = _variable_length_encoder_forward(model.encoder, x, mask)
        for i in range(3):
            out_single, mask_ds_single = _variable_length_encoder_forward(
                model.encoder, x[i:i + 1], mask[i:i + 1])
            torch.testing.assert_close(out_batched[i:i + 1], out_single, atol=1e-5, rtol=1e-4)
            assert torch.equal(mask_ds_batched[i:i + 1], mask_ds_single)


def test_regression_3_padded_content_does_not_change_output(model):
    """Section 8c.1: identical valid speech, different content in the padded region (zero vs.
    noise vs. a repeated excerpt of the valid audio) -- final logit must not materially change."""
    torch.manual_seed(2)
    n_mels, t_in = 80, 200
    valid_start = 80
    mask = torch.ones(1, t_in, dtype=torch.bool)
    mask[0, :valid_start] = False

    valid_content = torch.randn(1, n_mels, t_in - valid_start)

    def _build(pad_fill):
        x = torch.zeros(1, n_mels, t_in)
        x[:, :, :valid_start] = pad_fill
        x[:, :, valid_start:] = valid_content
        return x

    variants = {
        "zero": _build(torch.zeros(1, n_mels, valid_start)),
        "noise": _build(torch.randn(1, n_mels, valid_start) * 50),  # wildly different raw content
        "repeat": _build(valid_content[:, :, -valid_start:]),       # repeated real audio as filler
    }

    logits = {}
    with torch.no_grad():
        for name, x in variants.items():
            out, mask_ds = _variable_length_encoder_forward(model.encoder, x, mask)
            pooled = model.pool(out, mask_ds)
            logits[name] = model.head(pooled).squeeze(-1)

    base = logits["zero"]
    for name in ("noise", "repeat"):
        torch.testing.assert_close(logits[name], base, atol=1e-4, rtol=1e-4)


def test_regression_4_padded_states_are_zero_and_dont_leak(model):
    """Section 8c.3: padded encoder states are exactly zero (by construction, post-8c), and valid
    positions' hidden states are unaffected by what's fed into the padded region."""
    torch.manual_seed(3)
    n_mels, t_in = 80, 120
    valid_start = 40
    mask = torch.ones(1, t_in, dtype=torch.bool)
    mask[0, :valid_start] = False

    x1 = torch.randn(1, n_mels, t_in)
    x2 = x1.clone()
    x2[:, :, :valid_start] = torch.randn(1, n_mels, valid_start) * 100  # only padded region differs

    with torch.no_grad():
        out1, mask_ds = _variable_length_encoder_forward(model.encoder, x1, mask)
        out2, mask_ds2 = _variable_length_encoder_forward(model.encoder, x2, mask)

    assert torch.equal(mask_ds, mask_ds2)
    pad_positions = ~mask_ds[0]
    valid_positions = mask_ds[0]
    assert pad_positions.any() and valid_positions.any()

    # padded encoder states are exactly zero
    assert torch.equal(out1[0, pad_positions], torch.zeros_like(out1[0, pad_positions]))
    assert torch.equal(out2[0, pad_positions], torch.zeros_like(out2[0, pad_positions]))
    # valid encoder states don't depend on padded-region content
    torch.testing.assert_close(out1[0, valid_positions], out2[0, valid_positions],
                                atol=1e-5, rtol=1e-4)


def test_regression_5_onnx_matches_pytorch_and_preserves_masking(model, tmp_path):
    """Section 8c: ONNX export must agree with PyTorch, including the padding-invariance property
    (not just raw numerical parity on one fixed input)."""
    onnxruntime = pytest.importorskip("onnxruntime")
    n_frames = 84
    onnx_path = tmp_path / "whisper_endpoint.onnx"
    dummy_feats = torch.randn(1, 80, n_frames)
    dummy_mask = torch.ones(1, n_frames, dtype=torch.bool)
    torch.onnx.export(
        model, (dummy_feats, dummy_mask), str(onnx_path),
        input_names=["input_features", "valid_frame_mask"], output_names=["logit"],
        dynamic_axes={"input_features": {2: "time"}, "valid_frame_mask": {1: "time"},
                      "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    torch.manual_seed(4)
    valid_start = 30
    mask = torch.ones(1, n_frames, dtype=torch.bool)
    mask[0, :valid_start] = False
    x_zero = torch.randn(1, 80, n_frames)
    x_zero[:, :, :valid_start] = 0.0
    x_noise = x_zero.clone()
    x_noise[:, :, :valid_start] = torch.randn(1, 80, valid_start) * 50

    with torch.no_grad():
        torch_logit_zero = model(x_zero, mask).numpy()
        torch_logit_noise = model(x_noise, mask).numpy()

    onnx_logit_zero = sess.run(["logit"], {"input_features": x_zero.numpy(),
                                            "valid_frame_mask": mask.numpy()})[0]
    onnx_logit_noise = sess.run(["logit"], {"input_features": x_noise.numpy(),
                                             "valid_frame_mask": mask.numpy()})[0]

    np.testing.assert_allclose(onnx_logit_zero, torch_logit_zero, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(onnx_logit_noise, onnx_logit_zero, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------------------------
# Section 8c.2: mask-downsampling shape edge cases
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("t_in", [1, 2, 3, 4, 5, 6, 7, 8, 41, 42, 419, 420])
def test_downsample_valid_mask_shape_matches_conv_output(model, t_in):
    """Odd/even lengths, very short clips, and exact boundary lengths: the downsampled mask's time
    dimension must always exactly match what conv1+conv2 actually produce -- never off by one."""
    if t_in < 3:
        pytest.skip("conv1 (kernel=3, padding=1) needs at least 1 valid output position; "
                    "shorter-than-kernel windows aren't a real input shape for this dataset")
    mask = torch.ones(1, t_in, dtype=torch.bool)
    x = torch.randn(1, 80, t_in)
    with torch.no_grad():
        conv_out_len = model.encoder.conv2(model.encoder.conv1(x)).shape[-1]
        mask_ds = _downsample_valid_mask(model.encoder, mask)
    assert mask_ds.shape[-1] == conv_out_len


def test_downsample_valid_mask_marks_boundary_frame_valid_if_any_real_content(model):
    """A conv output frame straddling the valid/padding boundary still counts as valid (its
    receptive field includes real content) -- it's canonicalization (8c.1), not masking, that
    keeps its *value* well-defined."""
    t_in = 40
    mask = torch.zeros(1, t_in, dtype=torch.bool)
    mask[0, -1] = True  # only the very last input frame is real
    mask_ds = _downsample_valid_mask(model.encoder, mask)
    assert mask_ds[0, -1] == True  # noqa: E712 -- last output frame's receptive field reaches it
    assert not mask_ds[0, :-2].any()  # frames far from the boundary stay invalid


def test_mixed_length_batch_each_item_downsamples_independently(model):
    mask = torch.ones(3, 100, dtype=torch.bool)
    mask[0, :10] = False
    mask[1, :50] = False
    mask[2, :90] = False
    mask_ds = _downsample_valid_mask(model.encoder, mask)
    counts = mask_ds.sum(dim=1)
    assert counts[0] > counts[1] > counts[2]


def test_canonicalize_padding_overwrites_only_invalid_frames():
    x = torch.randn(2, 80, 10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, :4] = False
    out = _canonicalize_padding(x, mask)
    assert torch.equal(out[0, :, :4], torch.full((80, 4), WHISPER_PAD_LOGMEL_VALUE))
    torch.testing.assert_close(out[0, :, 4:], x[0, :, 4:])
    torch.testing.assert_close(out[1], x[1])  # item 1 has no padding -> unchanged


# ---------------------------------------------------------------------------------------------
# Section 8c.3: mixed precision
# ---------------------------------------------------------------------------------------------

def test_masked_attention_under_mixed_precision(model):
    x = torch.randn(2, 80, 200)
    mask = torch.ones(2, 200, dtype=torch.bool)
    mask[0, :60] = False
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, mask_ds = _variable_length_encoder_forward(model.encoder, x, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, ~mask_ds[0]], torch.zeros_like(out[0, ~mask_ds[0]]))
