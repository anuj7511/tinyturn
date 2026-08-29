"""
Phase-3 8e-extended -- prefix-context reliance diagnostic on the corrected A0.

Not a padding-invariance test (that's 8e proper, already passing cleanly -- see
PHASE2_RESULTS_8a-9.md). 8e only ever changes what fills an already-invalid region, which by
construction carries no information. This diagnostic removes real information instead: starting
from val-split clips that already have close to a full window of genuine speech (>=90% of A0's
4s context, i.e. minimal/no structural left-padding to begin with), it progressively re-masks the
*earliest* part of that valid audio as invalid -- last 75% valid, last 50%, last 25% -- always
preserving the terminal region exactly as build_example does, and measures how A0's prediction
moves as long-range context is taken away.

A shift here is ambiguous by itself (brief Section 2, 8e-extended): it could mean the model
shortcuts on how much valid audio is present regardless of content, or that it's making legitimate
use of longer context when available. This script reports the raw shift AND two correlational
checks meant to pull those apart -- correlation of |shift| against (a) the absolute duration
removed (amount-dependence) and (b) the relative energy of the removed prefix vs. the retained
tail (content-dependence). Neither is causal proof; both are read together, not as a pass/fail
gate (this is an exploratory diagnostic, not a teacher-qualification criterion).

Usage:
  python scripts/prefix_context_reliance_diagnostic.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.preprocess import build_example, frame_valid_mask
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from transformers import WhisperFeatureExtractor

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
EXP_DIR = Path("experiments") / "whisper_tiny_speech_aligned_contract"
FULL_CONTEXT_FRAC = 0.9  # qualifying clip: valid (real) audio covers >= 90% of the context window,
                          # i.e. minimal structural left-padding to begin with -- so "last X% valid"
                          # below is actually removing real speech, not already-padded silence.
FRACTIONS = [1.0, 0.75, 0.5, 0.25]  # of the *valid* span, measured from the terminal region back
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010


def _load_wav(row_id):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    return y.astype(np.float32), sr


def _masked_variant(ex, frac: float):
    """New valid_sample_mask keeping only the last `frac` of the originally-valid span, terminal
    region always retained. Waveform itself is untouched -- extract_whisper_features silences
    whatever the mask marks invalid, exactly the same path production inference goes through."""
    valid = ex.valid_sample_mask
    valid_idx = np.flatnonzero(valid)
    valid_end = int(valid_idx[-1]) + 1  # == len(ex.waveform) by construction (speech end contract)
    valid_start = int(valid_idx[0])
    v = valid_end - valid_start
    keep_len = int(round(frac * v))
    new_valid_start = valid_end - keep_len
    new_mask = np.zeros_like(valid)
    new_mask[new_valid_start:valid_end] = True
    return new_mask, valid_start, new_valid_start, valid_end


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
    df = df[df["last_active_t"] >= context_s * FULL_CONTEXT_FRAC].reset_index(drop=True)
    print(f"8e-extended: {len(df)} val clips with >= {FULL_CONTEXT_FRAC:.0%} of the {context_s}s "
          f"context window occupied by real speech", flush=True)
    if len(df) == 0:
        print("No qualifying clips -- nothing to test.")
        sys.exit(1)

    model = WhisperEndpointModel(model_name=cfg.get("model_name", WHISPER_MODEL_NAME))
    model.load_state_dict(torch.load(EXP_DIR / "checkpoint.pt", map_location="cpu"))
    model.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.get("model_name", WHISPER_MODEL_NAME))

    rows = []
    with torch.no_grad():
        for i, r in df.iterrows():
            y, sr = _load_wav(r["id"])
            ex = build_example(y, sr, float(r["last_active_t"]), context_s,
                                frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                                label=bool(r["endpoint_bool"]), row_id=r["id"])
            probs = {}
            removed_bounds = {}
            for frac in FRACTIONS:
                mask, valid_start, new_valid_start, valid_end = _masked_variant(ex, frac)
                input_features, vfm = extract_whisper_features(feature_extractor, ex.waveform, mask, sr)
                x = torch.from_numpy(input_features).unsqueeze(0)
                m = torch.from_numpy(vfm).unsqueeze(0)
                logit = model(x, m)
                probs[frac] = float(torch.sigmoid(logit).item())
                removed_bounds[frac] = (valid_start, new_valid_start, valid_end)

            base = probs[1.0]
            # Content-vs-amount signal at the most aggressive fraction tested (0.25): energy of the
            # removed prefix relative to the retained tail. A removed prefix that's mostly silence
            # (low ratio) removing real *content* is a different signal than a removed prefix that
            # was itself high-energy speech.
            v_start, new_start_25, v_end = removed_bounds[0.25]
            removed_seg = ex.waveform[v_start:new_start_25]
            kept_seg = ex.waveform[new_start_25:v_end]
            removed_rms = float(np.sqrt(np.mean(removed_seg ** 2))) if len(removed_seg) else 0.0
            kept_rms = float(np.sqrt(np.mean(kept_seg ** 2))) if len(kept_seg) else 1e-8
            removed_duration_s = len(removed_seg) / sr

            row = {
                "id": r["id"], "real": not bool(r["synthetic"]), "dataset": r["dataset"],
                "language": r["language"], "endpoint_bool": bool(r["endpoint_bool"]),
                **{f"prob_frac_{frac}": p for frac, p in probs.items()},
                **{f"dprob_frac_{frac}": (p - base) for frac, p in probs.items()},
                "abs_dprob_at_025": abs(probs[0.25] - base),
                "flip_at_025": (probs[0.25] >= threshold) != (base >= threshold),
                "removed_duration_s_at_025": removed_duration_s,
                "removed_kept_rms_ratio_at_025": removed_rms / (kept_rms + 1e-8),
            }
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(df)}", flush=True)

    out = pd.DataFrame(rows)

    summary = {"n": len(out), "context_s": context_s, "threshold": threshold,
               "fractions_tested": FRACTIONS, "full_context_frac_cutoff": FULL_CONTEXT_FRAC}
    for frac in FRACTIONS:
        if frac == 1.0:
            continue
        d = out[f"dprob_frac_{frac}"]
        summary[f"frac_{frac}"] = {
            "mean_dprob": round(float(d.mean()), 5),
            "mean_abs_dprob": round(float(d.abs().mean()), 5),
            "median_abs_dprob": round(float(d.abs().median()), 5),
            "std_dprob": round(float(d.std()), 5),
            "frac_abs_dprob_gt_0.10": round(float((d.abs() > 0.10).mean()), 5),
            "flip_rate_vs_full": round(float(((out[f"prob_frac_{frac}"] >= threshold) !=
                                               (out["prob_frac_1.0"] >= threshold)).mean()), 5),
        }

    # Amount-dependence vs content-dependence at the 0.25 (most aggressive) fraction.
    valid_corr = out[out["removed_duration_s_at_025"] > 0]
    if len(valid_corr) >= 10:
        r_amount_p, p_amount_p = pearsonr(valid_corr["removed_duration_s_at_025"], valid_corr["abs_dprob_at_025"])
        r_amount_s, p_amount_s = spearmanr(valid_corr["removed_duration_s_at_025"], valid_corr["abs_dprob_at_025"])
        r_content_p, p_content_p = pearsonr(valid_corr["removed_kept_rms_ratio_at_025"], valid_corr["abs_dprob_at_025"])
        r_content_s, p_content_s = spearmanr(valid_corr["removed_kept_rms_ratio_at_025"], valid_corr["abs_dprob_at_025"])
        summary["amount_vs_content_at_025"] = {
            "n": int(len(valid_corr)),
            "note": "Neither correlation is causal proof by itself -- read together, not as a "
                    "pass/fail signal (brief: 'don't over-weight its priority regardless of what "
                    "it shows').",
            "abs_dprob_vs_removed_duration_s": {"pearson_r": round(r_amount_p, 4), "pearson_p": round(p_amount_p, 4),
                                                 "spearman_r": round(r_amount_s, 4), "spearman_p": round(p_amount_s, 4)},
            "abs_dprob_vs_removed_kept_rms_ratio": {"pearson_r": round(r_content_p, 4), "pearson_p": round(p_content_p, 4),
                                                     "spearman_r": round(r_content_s, 4), "spearman_p": round(p_content_s, 4)},
        }
    else:
        summary["amount_vs_content_at_025"] = {"note": f"skipped, n={len(valid_corr)} too small"}

    # Real vs synthetic breakdown at the most aggressive fraction, since duration's inert
    # coefficient (Part 1) was itself only checked in aggregate.
    for tag, sub in [("real", out[out["real"]]), ("synthetic", out[~out["real"]])]:
        if len(sub) == 0:
            continue
        summary[f"frac_0.25_{tag}"] = {
            "n": int(len(sub)),
            "mean_abs_dprob": round(float(sub["abs_dprob_at_025"].abs().mean()), 5),
            "flip_rate_vs_full": round(float(sub["flip_at_025"].mean()), 5),
        }

    out_path = EXP_DIR / "8e_extended_prefix_context.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_clip": rows}, f, indent=2, default=str)

    print(json.dumps(summary, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
