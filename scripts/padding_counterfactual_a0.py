"""
Phase-2 8e -- padding counterfactual on the corrected A0 (feeds Section 8g's padding criterion).

On the retrained, speech-aligned-contract A0 checkpoint (`experiments/whisper_tiny_speech_aligned_contract`,
produced by retrain_a0_speech_aligned_contract.py): take val-split clips short enough that build_example needs
substantial left-padding, build variants that share identical VALID (real) audio and differ only in
what fills the padded region -- zero (production default), small noise, large noise, and a repeat
of the valid audio -- then compare A0's predicted probability across variants for the same
underlying speech. Identical valid speech should produce (near-)identical predictions regardless of
padding scheme; this is the end-to-end (real audio -> feature extraction -> model) confirmation of
what test_whisper_model.py's regression test 3 already verified at the tensor level.

Reports exactly the fields Section 8g's frozen padding criterion needs:
  - mean absolute probability change (criterion: <= 0.02)
  - fraction of examples changing by more than 0.10 (criterion: <= 1%)
  - decision-flip rate at A0's own calibrated threshold (criterion: <= 1%)

First run of this script (against noise/repeat padding-scheme variants fed straight into
WhisperFeatureExtractor) failed the padding criterion badly (mean abs change 0.046, 18.6% of
examples changing >0.10). Root cause traced directly to WhisperFeatureExtractor's own per-utterance
normalization (`log_spec = np.maximum(log_spec, log_spec.max() - 8.0)`), computed over the *whole*
waveform including the padded region -- so loud padding content shifts the clamp applied to the
*valid* frames too, upstream of Section 8c's model-side mel canonicalization. Fixed at the actual
source (`tinyturn.whisper_dataset.silence_invalid_samples`, now applied in every production feature-
extraction call site) rather than worked around here -- this script goes through that same shared
path (`extract_whisper_features`), so it measures the real deployed behavior.

Usage:
  python scripts/padding_counterfactual_a0.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.preprocess import build_example
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from transformers import WhisperFeatureExtractor

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
# Optional sys.argv[1]: run against a different checkpoint dir (e.g. the 8g-remediation boundary-
# robust retrain) without touching the canonical A0's own recorded result. Output is always written
# inside EXP_DIR, so a different EXP_DIR can never clobber the canonical file.
EXP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments") / "whisper_tiny_speech_aligned_contract"
RNG_SEED = 42
SUBSTANTIAL_PAD_FRAC = 0.8  # "short enough to require substantial padding" (8g): at least 20% of
                            # the window is left-padding, i.e. speech_end_s < context_s * this.
                            # Chosen over a stricter 0.5 cut (which leaves only 25 val clips at
                            # context_s=4.0) to give the 1%-tolerance criteria a large enough n
                            # (118) to be statistically meaningful, while still requiring real,
                            # non-trivial padding rather than a single padded frame.
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010


def _load_wav(row_id):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    return y.astype(np.float32), sr


def _variant_waveform(ex, scheme: str, rng: np.random.RandomState):
    """Same valid content, different fill for the padded (left) region."""
    valid = ex.valid_sample_mask
    invalid = ~valid
    if not invalid.any() or scheme == "zero":
        return ex.waveform.copy()
    out = ex.waveform.copy()
    valid_content = ex.waveform[valid]
    n_pad = int(invalid.sum())
    if scheme == "noise":
        level = float(np.std(valid_content)) * 0.5 + 1e-5
        out[invalid] = rng.normal(0, level, n_pad).astype(np.float32)
    elif scheme == "noise_large":
        level = float(np.std(valid_content)) * 5.0 + 1e-3
        out[invalid] = rng.normal(0, level, n_pad).astype(np.float32)
    elif scheme == "repeat":
        reps = int(np.ceil(n_pad / max(len(valid_content), 1)))
        tiled = np.tile(valid_content, reps)[-n_pad:] if len(valid_content) else np.zeros(n_pad, np.float32)
        out[invalid] = tiled
    else:
        raise ValueError(scheme)
    return out


def main():
    if not (EXP_DIR / "checkpoint.pt").exists():
        print(f"ERROR: {EXP_DIR / 'checkpoint.pt'} not found -- run retrain_a0_speech_aligned_contract.py first.")
        sys.exit(1)

    cfg = json.load(open(EXP_DIR / "config.json"))
    metrics = json.load(open(EXP_DIR / "metrics.json"))
    threshold = float(metrics["threshold"])
    context_s = float(cfg["context_s"])

    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[
        ["id", "last_active_t", "endpoint_bool", "dataset", "language", "synthetic"]]
    df = splits[splits["split"] == "val"][["id"]].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_t"].notna()]
    df = df[df["last_active_t"] < context_s * SUBSTANTIAL_PAD_FRAC].reset_index(drop=True)
    print(f"8e: {len(df)} val clips with speech_end_s < {context_s * SUBSTANTIAL_PAD_FRAC:.2f}s "
          f"(substantial left-padding at context_s={context_s})", flush=True)
    if len(df) == 0:
        print("No qualifying clips -- nothing to test. Check context_s / substantial-pad threshold.")
        sys.exit(1)

    model = WhisperEndpointModel(model_name=cfg.get("model_name", WHISPER_MODEL_NAME))
    model.load_state_dict(torch.load(EXP_DIR / "checkpoint.pt", map_location="cpu"))
    model.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.get("model_name", WHISPER_MODEL_NAME))

    schemes = ["zero", "noise", "noise_large", "repeat"]
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    with torch.no_grad():
        for i, r in df.iterrows():
            y, sr = _load_wav(r["id"])
            ex = build_example(y, sr, float(r["last_active_t"]), context_s,
                                frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                                label=bool(r["endpoint_bool"]), row_id=r["id"])
            probs = {}
            for scheme in schemes:
                wav_variant = _variant_waveform(ex, scheme, rng)
                input_features, vfm = extract_whisper_features(
                    feature_extractor, wav_variant, ex.valid_sample_mask, sr)
                x = torch.from_numpy(input_features).unsqueeze(0)
                m = torch.from_numpy(vfm).unsqueeze(0)
                logit = model(x, m)
                probs[scheme] = float(torch.sigmoid(logit).item())
            vals = list(probs.values())
            rows.append({
                "id": r["id"], "real": not bool(r["synthetic"]), "dataset": r["dataset"],
                "endpoint_bool": bool(r["endpoint_bool"]), **{f"prob_{k}": v for k, v in probs.items()},
                "max_abs_diff_from_zero": max(abs(v - probs["zero"]) for v in vals),
                "max_abs_diff_any_pair": max(vals) - min(vals),
                "decision_flip": (max(vals) >= threshold) != (min(vals) >= threshold),
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)
    mean_abs_change = out["max_abs_diff_from_zero"].mean()
    frac_change_gt_010 = (out["max_abs_diff_from_zero"] > 0.10).mean()
    flip_rate = out["decision_flip"].mean()

    result = {
        "n": len(out),
        "context_s": context_s,
        "threshold": threshold,
        "schemes_tested": schemes,
        "mean_abs_prob_change": round(float(mean_abs_change), 5),
        "frac_change_gt_0.10": round(float(frac_change_gt_010), 5),
        "decision_flip_rate_at_threshold": round(float(flip_rate), 5),
        "criterion_mean_abs_change_le_0.02": bool(mean_abs_change <= 0.02),
        "criterion_frac_gt_010_le_0.01": bool(frac_change_gt_010 <= 0.01),
        "criterion_flip_rate_le_0.01": bool(flip_rate <= 0.01),
    }
    out_path = EXP_DIR / "padding_counterfactual.json"
    with open(out_path, "w") as f:
        json.dump({"summary": result, "per_clip": rows}, f, indent=2, default=str)

    print(json.dumps(result, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
