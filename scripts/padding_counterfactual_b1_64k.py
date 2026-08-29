"""
Step 10 planning -- padding counterfactual (8e-style) for the 64k B1 checkpoints, never run for any
B1 model before (the existing `experiments/*/8e_padding_counterfactual.json` files are all A0/
Whisper). Same method as `padding_counterfactual_a0.py`: take val clips short enough to need
substantial left-padding, build waveform variants that share identical valid (real) audio and differ
only in what fills the padded region (zero / small noise / large noise / repeat), and compare B1's
predicted probability across variants for the same underlying speech.

IMPORTANT data-characteristic finding, not a script bug: at context_s=1.0 (B1@1s), padding is a much
rarer scenario than at A0's context_s=4.0. `last_active_t` in the val split has median 6.73s (only
the 1.0th percentile is below 1.8s) -- almost every val clip already has >=1s of speech before the
endpoint, so it needs *no* left-padding at a 1-second window. Checked directly before writing this:
at the A0-precedent's SUBSTANTIAL_PAD_FRAC=0.8 cut (speech_end_s < context_s*0.8 = 0.8s), only 4 of
1,600 val clips qualify; even the loosest possible cut -- ANY left-padding at all, frac=1.0
(speech_end_s < context_s = 1.0s) -- only reaches 8. This script uses frac=1.0 to get the largest
usable n, but n=8 is still far too small for the criteria (mean abs change <=0.02, frac>0.10 <=1%,
flip rate <=1%) to be statistically meaningful the way they were at A0's n=118. Report the numbers,
but do not treat a pass/fail here as a confident verdict -- flag it as directional/exploratory only.

Threshold discipline: each checkpoint's own calib-calibrated threshold (metrics.json["threshold"]).

Usage:
  python scripts/padding_counterfactual_b1_64k.py
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
from tinyturn.models import TinyTurnModel
from scripts.vad_boundary_diagnostic_b1_64k import (
    N_MELS, TRAJECTORY_NAMES, FRAME_LENGTH_S, HOP_LENGTH_S, _b1_prob, CHECKPOINTS,
)

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
OUT_PATH = Path("experiments") / "8e_padding_counterfactual_b1_64k.json"
RNG_SEED = 42
# See module docstring: 1.0 (any left-padding at all), not the A0 precedent's 0.8 -- context_s=1.0
# makes even that too rare (n=4) to be usable; frac=1.0 gets to n=8, still small.
SUBSTANTIAL_PAD_FRAC = 1.0
CONTEXT_S = 1.0


def _load_wav(row_id):
    data, sr = sf.read(WAV_DIR / f"{row_id}.wav")
    y = data if data.ndim == 1 else data.mean(axis=1)
    return y.astype(np.float32), sr


def _variant_waveform(ex, scheme: str, rng: np.random.RandomState):
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


def run_checkpoint(ckpt_dir: Path, df: pd.DataFrame) -> dict:
    cfg = json.load(open(ckpt_dir / "config.json"))
    metrics = json.load(open(ckpt_dir / "metrics.json"))
    threshold = float(metrics["threshold"])
    context_s = float(cfg["context_s"])
    model = TinyTurnModel(n_mels=N_MELS, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()

    schemes = ["zero", "noise", "noise_large", "repeat"]
    rng = np.random.RandomState(RNG_SEED)
    rows = []
    for _, r in df.iterrows():
        y, sr = _load_wav(r["id"])
        ex = build_example(y, sr, float(r["last_active_t"]), context_s,
                            frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                            label=bool(r["endpoint_bool"]), row_id=r["id"])
        probs = {}
        for scheme in schemes:
            wav_variant = _variant_waveform(ex, scheme, rng)
            probs[scheme] = _b1_prob(model, wav_variant, sr, float(r["last_active_t"]), context_s)
        vals = list(probs.values())
        rows.append({
            "id": r["id"], "real": not bool(r["synthetic"]), "dataset": r["dataset"],
            "endpoint_bool": bool(r["endpoint_bool"]), **{f"prob_{k}": v for k, v in probs.items()},
            "max_abs_diff_from_zero": max(abs(v - probs["zero"]) for v in vals),
            "max_abs_diff_any_pair": max(vals) - min(vals),
            "decision_flip": (max(vals) >= threshold) != (min(vals) >= threshold),
        })

    out = pd.DataFrame(rows)
    mean_abs_change = out["max_abs_diff_from_zero"].mean()
    frac_change_gt_010 = (out["max_abs_diff_from_zero"] > 0.10).mean()
    flip_rate = out["decision_flip"].mean()

    return {
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
        "note": "n too small to treat criteria as statistically meaningful -- see module docstring",
        "per_clip": rows,
    }


def main():
    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")
    sf_feat = pd.read_parquet(CACHE_DIR / "d2_stratified_signal_features.parquet")[
        ["id", "last_active_t", "endpoint_bool", "dataset", "language", "synthetic"]]
    df = splits[splits["split"] == "val"][["id"]].merge(sf_feat, on="id", how="left")
    df = df[df["last_active_t"].notna()]
    df = df[df["last_active_t"] < CONTEXT_S * SUBSTANTIAL_PAD_FRAC].reset_index(drop=True)
    print(f"8e (B1@64k): {len(df)} val clips with speech_end_s < "
          f"{CONTEXT_S * SUBSTANTIAL_PAD_FRAC:.2f}s (any left-padding at context_s={CONTEXT_S})", flush=True)
    if len(df) == 0:
        print("No qualifying clips -- nothing to test.")
        sys.exit(1)

    all_results = {}
    for arm, seeds in CHECKPOINTS.items():
        all_results[arm] = {}
        for seed, path in seeds.items():
            d = Path(path)
            if not (d / "checkpoint.pt").exists():
                print(f"SKIP {arm} seed={seed}: {d} missing checkpoint.pt")
                continue
            print(f"running {arm} seed={seed} ({d})...", flush=True)
            all_results[arm][seed] = run_checkpoint(d, df)

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nsaved {OUT_PATH}")

    print("\n=== padding counterfactual summary (n={} -- exploratory only) ===".format(len(df)))
    for arm, seeds in all_results.items():
        for seed, r in seeds.items():
            print(f"{arm} seed={seed}: mean_abs_change={r['mean_abs_prob_change']} "
                  f"frac_gt_0.10={r['frac_change_gt_0.10']} flip_rate={r['decision_flip_rate_at_threshold']}")


if __name__ == "__main__":
    main()
