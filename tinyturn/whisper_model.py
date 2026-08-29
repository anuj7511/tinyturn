"""
A0 -- corrected Whisper-Tiny baseline (Section 5). Same input-construction contract as B0/B1 (Step
1's build_example, Phase-2 8b: speech-aligned, no baked-in post-roll; masked padding; identical
splits/metrics), swapping only the encoder: Whisper-Tiny's pretrained conv+transformer encoder in
place of the DS-CNN+TCN mel branch, full fine-tuning, same endpoint-focused pooling module as B0/B1
(`tinyturn.pooling`) so pooling strategy isn't a confound in the B0/B1-vs-A0 comparison.

Phase-2 8a/8c: both HF's stock `WhisperEncoder.forward` and this module's own prior
`_variable_length_encoder_forward` called every `encoder_layer(hidden_states, attention_mask=None,
...)` unconditionally -- confirmed directly against the installed HF source (see module-level
`_variable_length_encoder_forward` docstring). Padding was masked nowhere in the encoder itself
(only in pooling, downstream). This is now fixed: padded mel frames are canonicalized to one fixed
value before the conv frontend (8c.1) and a real attention mask, downsampled through the same conv
stride/kernel as the model itself, is threaded into every encoder layer (8c.3).
"""
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel
from transformers.masking_utils import create_bidirectional_mask

from tinyturn.pooling import EndpointPooling

WHISPER_MODEL_NAME = "openai/whisper-tiny"

# create_bidirectional_mask's embeddings kwarg was renamed input_embeds -> inputs_embeds at some
# point in transformers' history -- confirmed to differ between this project's own local venv
# (inputs_embeds) and its Kaggle notebooks' unpinned `pip install transformers` (input_embeds),
# so this can flip depending on whichever version happens to resolve, not just once historically.
# Detected once here via introspection rather than hardcoding either spelling, so the same source
# works unmodified on both.
_CBM_EMBEDS_KWARG = ("inputs_embeds" if "inputs_embeds" in inspect.signature(create_bidirectional_mask).parameters
                     else "input_embeds")

# Canonical fill value for padded mel frames (Section 8c.1), so that whatever raw waveform padding
# scheme produced them (zero/noise/repeat -- Section 8e), the conv frontend sees identical content
# for the padded region every time. Matches WhisperFeatureExtractor's own floor for genuine silence:
# an all-zero waveform normalizes to a uniform -1.5 across every mel bin and frame (log10(1e-10)
# clamped, then `(x + 4) / 4`) -- verified empirically against the feature extractor, not just
# derived on paper (see test_whisper_model.py::test_silence_floor_matches_pad_value).
WHISPER_PAD_LOGMEL_VALUE = -1.5


def _canonicalize_padding(input_features: torch.Tensor, valid_frame_mask: torch.Tensor,
                           pad_value: float = WHISPER_PAD_LOGMEL_VALUE) -> torch.Tensor:
    """Overwrite padded mel frames with one fixed value before conv1 (Section 8c.1). Without this,
    conv2's stride-2 kernel (spanning 3 input frames per output step) blends whatever real padding
    content existed with real speech at the valid/padding boundary, differently depending on the
    padding scheme -- masking attention afterward doesn't undo that blend, since it already happened
    inside the convolution."""
    mask = valid_frame_mask.unsqueeze(1)  # (B, 1, T) broadcasts over the mel-channel dim
    return torch.where(mask, input_features, torch.full_like(input_features, pad_value))


def _downsample_valid_mask(encoder, valid_frame_mask: torch.Tensor) -> torch.Tensor:
    """Push the mel-frame validity mask through conv1 then conv2 using each conv's own actual
    kernel/stride/padding (Section 8c.2), via max-pooling: an output frame counts as valid iff its
    conv receptive field includes at least one real (non-padded) input frame. This mirrors exactly
    what the convolution itself mixes -- unlike a nearest-neighbor resize (the prior
    implementation), it can't drift out of sync with the model architecture, and it is correct for
    odd/even lengths, very short clips, and exact boundary cases because it uses the identical
    padding/stride arithmetic as `F.conv1d` itself."""
    m = valid_frame_mask.to(torch.float32).unsqueeze(1)  # (B, 1, T_in)
    for conv in (encoder.conv1, encoder.conv2):
        m = F.max_pool1d(m, kernel_size=conv.kernel_size[0], stride=conv.stride[0],
                          padding=conv.padding[0])
    return m.squeeze(1) > 0.5  # (B, T_out) bool


def _variable_length_encoder_forward(encoder, input_features: torch.Tensor,
                                      valid_frame_mask: torch.Tensor):
    """HF's WhisperEncoder.forward hardcodes `input_features.shape[-1] == 3000` (the 30s
    convention) and errors otherwise -- it also adds the FULL 1500-entry sinusoidal
    position-embedding table unconditionally rather than slicing it to the actual (shorter)
    sequence length, and (Section 8a) calls every encoder layer with `attention_mask=None`
    regardless of what its own `attention_mask` argument received (its docstring says as much:
    "Whisper does not support masking of the input_features ... argument is preserved for
    compatibility, but it is not used"). This reimplements the same conv-stem +
    position-embedding + transformer-layers forward pass, slicing `embed_positions` to the actual
    conv output length like the original OpenAI Whisper implementation does for variable-length
    audio, *and* (Section 8c) actually uses `valid_frame_mask`: canonicalizing padded input before
    the conv stem, downsampling the mask through the real conv stride, and passing a real attention
    mask into every layer so padded positions cannot influence valid ones.

    Returns (hidden_states, mask_ds): mask_ds is the mask downsampled to encoder-frame resolution,
    for callers (pooling) that need to know which output positions are real.
    """
    input_features = _canonicalize_padding(input_features, valid_frame_mask)

    inputs_embeds = nn.functional.gelu(encoder.conv1(input_features))
    inputs_embeds = nn.functional.gelu(encoder.conv2(inputs_embeds))
    inputs_embeds = inputs_embeds.permute(0, 2, 1)  # (B, T_out, d_model)

    mask_ds = _downsample_valid_mask(encoder, valid_frame_mask)  # (B, T_out) bool
    keep = mask_ds.unsqueeze(-1)  # (B, T_out, 1), broadcasts over d_model

    t_out = inputs_embeds.shape[1]
    positions = torch.arange(t_out, device=inputs_embeds.device)
    hidden_states = inputs_embeds + encoder.embed_positions(positions)
    hidden_states = nn.functional.dropout(hidden_states, p=encoder.dropout, training=encoder.training)
    hidden_states = torch.where(keep, hidden_states, torch.zeros_like(hidden_states))

    # create_bidirectional_mask (transformers.masking_utils) builds whatever mask representation
    # the encoder's actual attention implementation expects -- an additive float mask for "eager",
    # a boolean keep-mask for "sdpa", a BlockMask for "flex" -- so this stays correct regardless of
    # which backend is configured, unlike hand-building one fixed representation. It also correctly
    # returns None when nothing in the batch is padded, matching HF's own convention for "no mask
    # needed". (Supersedes `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask`, which
    # is deprecated in this transformers version and only ever produced the "eager" format.)
    attn_mask = create_bidirectional_mask(config=encoder.config, attention_mask=mask_ds,
                                           **{_CBM_EMBEDS_KWARG: hidden_states})
    for encoder_layer in encoder.layers:
        layer_out = encoder_layer(hidden_states, attn_mask)
        # HF has changed WhisperEncoderLayer.forward's return type across versions (plain tensor
        # vs. a (hidden_states, attn_weights) tuple) -- normalize rather than assume one, since a
        # bare `[0]` on a plain tensor silently slices off batch item 0 instead of erroring.
        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        # Re-zero after every layer (not just once at the end): a padded position's own residual
        # stream would otherwise pick up nonzero, content-dependent values from its own (unmasked)
        # query-side attention pass and from layer-norm bias terms, which would then feed forward
        # into the next layer. Attention masking alone already keeps this from ever reaching a
        # *valid* position (masking is query-independent -- no query can attend to a padded key),
        # but re-zeroing keeps padded-position outputs themselves well-defined and deterministic,
        # which is what regression tests 3/4 (Section 8c) check directly.
        hidden_states = torch.where(keep, hidden_states, torch.zeros_like(hidden_states))

    hidden_states = encoder.layer_norm(hidden_states)
    hidden_states = torch.where(keep, hidden_states, torch.zeros_like(hidden_states))
    return hidden_states, mask_ds


class WhisperEndpointModel(nn.Module):
    def __init__(self, fusion_hidden: int = 64, pool_window_frames: int = 50,
                 model_name: str = WHISPER_MODEL_NAME):
        super().__init__()
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder  # full fine-tuning: no params frozen
        d_model = self.encoder.config.d_model
        self.pool = EndpointPooling(d_model, window_frames=pool_window_frames)
        self.head = nn.Sequential(
            nn.Linear(d_model * self.pool.out_multiplier, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, input_features: torch.Tensor, valid_frame_mask: torch.Tensor):
        """input_features: (B, n_mels, T_in) Whisper-convention log-mel.
        valid_frame_mask: (B, T_in) bool, at the same 10ms-hop frame rate as input_features."""
        out, mask_ds = _variable_length_encoder_forward(self.encoder, input_features, valid_frame_mask)
        pooled = self.pool(out, mask_ds)
        logit = self.head(pooled).squeeze(-1)
        return logit

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
