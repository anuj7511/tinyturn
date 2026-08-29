"""
Phase-3 8h-A0 -- confirm A0 has actually converged, then compare A0@2s vs. A0@4s.

Designed to run on either CPU (slow, ~25-28 min/epoch observed locally) or CUDA (e.g. a Colab GPU
runtime -- tinyturn.train_whisper now auto-detects and uses it). Mirrors what 8h already did for
B1: reports best-epoch vs. final-epoch for the existing fixed-epoch A0 run, then runs early-stopped
retrains at both context lengths under the identical `pv2-speechend` contract + corrected masking.

1. Confirms directly (not assumed) that A0_whisper_tiny_pv2speechend used a FIXED epoch budget (2
   epochs, no early stopping, no LR schedule) and whether val AUC was still rising at the end.
2. Trains ONE A0@4s with early stopping (max epochs / patience configurable below) to a NEW
   directory, so the existing qualifying checkpoint stays untouched.
3. Trains ONE A0@2s under the identical protocol, latency-relevant per the brief ("the 2-second
   model may be the better teacher if its operating-point performance is close").
4. Compares both on: FCR at fixed complete recall, implicit_incomplete FCR, real-audio FCR,
   internal-hold FCR (pause events, evaluated fresh here -- A0 has never been evaluated against
   these before per the brief), latency, and calibration -- not AUC alone.

Usage:
  python scripts/run_8h_a0_convergence_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.train_whisper import WhisperExperimentConfig, train_whisper_experiment
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.whisper_dataset import extract_whisper_features
from tinyturn.preprocess import build_example
from transformers import WhisperFeatureExtractor

CACHE_DIR = Path("data_cache")
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
ORIGINAL_4S_DIR = Path("experiments") / "A0_whisper_tiny_pv2speechend"
LONGRUN_4S_DIR = Path("experiments") / "A0_4s_8hA0_longrun"
LONGRUN_2S_DIR = Path("experiments") / "A0_2s_8hA0_longrun"

MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
FRAME_LENGTH_S, HOP_LENGTH_S = 0.025, 0.010


def analyze_existing_run():
    metrics = json.load(open(ORIGINAL_4S_DIR / "metrics.json"))
    history = metrics["history"]
    best_h = max(history, key=lambda h: h["val_auc"])
    best_epoch, best_val_auc = best_h["epoch"], best_h["val_auc"]
    final_epoch = history[-1]["epoch"]
    print("=== Existing A0_whisper_tiny_pv2speechend run (fixed 2-epoch protocol) ===")
    for h in history:
        marker = "  <- best" if h["epoch"] == best_epoch else ""
        print(f"  epoch {h['epoch']}: val_auc={h['val_auc']:.4f} loss={h['train_loss']:.4f}{marker}")
    print(f"best_epoch={best_epoch} (val_auc={best_val_auc:.4f}), final_epoch={final_epoch} "
          f"(val_auc={history[-1]['val_auc']:.4f})")
    print(f"val AUC was {'still rising' if best_epoch == final_epoch else 'NOT still rising'} "
          f"at the final epoch.")
    return {"best_epoch": best_epoch, "best_val_auc": best_val_auc, "final_epoch": final_epoch,
            "history": history}


def run_longer_a0(context_s: float, out_dir: Path, seed: int = 42):
    cfg = WhisperExperimentConfig(
        exp_id=f"A0_{context_s}s_8hA0_longrun", context_s=context_s,
        epochs=MAX_EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE, lr_schedule="plateau",
        batch_size=8, lr=1e-5, num_workers=0, seed=seed,
    )
    return train_whisper_experiment(cfg, out_dir)


def evaluate_internal_hold_fcr(ckpt_dir: Path, context_s: float, device):
    """A0 has never been separately validated on internal-pause events (brief: "that whole
    evaluation apparatus was built for B1 in Step 9, not for A0"). Evaluates fresh here: for every
    val-split pause event, build the A0 example anchored to the pause's own start (not the
    canonical final boundary), run the model, and report the false-complete rate (should predict
    incomplete; FCR = fraction predicted complete)."""
    cfg = json.load(open(ckpt_dir / "config.json"))
    metrics = json.load(open(ckpt_dir / "metrics.json"))
    threshold = float(metrics["threshold"])
    model = WhisperEndpointModel(model_name=cfg.get("model_name", WHISPER_MODEL_NAME))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.to(device).eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.get("model_name", WHISPER_MODEL_NAME))

    events = pd.read_parquet(CACHE_DIR / "tinyturn_pause_events.parquet")
    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")[["id", "split", "synthetic"]] \
        .rename(columns={"id": "clip_id"})
    df = events.merge(splits, on="clip_id", how="left")
    df = df[df["split"] == "val"].reset_index(drop=True)

    import soundfile as sf

    def _load_wav(clip_id):
        data, sr = sf.read(WAV_DIR / f"{clip_id}.wav")
        y = data if data.ndim == 1 else data.mean(axis=1)
        return y.astype(np.float32), sr

    probs, synth = [], []
    with torch.no_grad():
        for _, r in df.iterrows():
            y, sr = _load_wav(r["clip_id"])
            ex = build_example(y, sr, float(r["pause_start_s"]), context_s,
                                frame_length_s=FRAME_LENGTH_S, hop_length_s=HOP_LENGTH_S,
                                label=False, row_id=f"{r['clip_id']}__pause{r['event_idx']}")
            input_features, vfm = extract_whisper_features(feature_extractor, ex.waveform, ex.valid_sample_mask, sr)
            x = torch.from_numpy(input_features).unsqueeze(0).to(device)
            m = torch.from_numpy(vfm).unsqueeze(0).to(device)
            logit = model(x, m)
            probs.append(float(torch.sigmoid(logit).item()))
            synth.append(bool(r["synthetic"]))
    probs, synth = np.array(probs), np.array(synth, dtype=bool)
    pred_complete = probs >= threshold

    def _report(mask):
        p = pred_complete[mask]
        return {"n": int(mask.sum()), "fcr": round(float(p.mean()), 4) if mask.sum() else None}

    return {"threshold": threshold, "all": _report(np.ones_like(synth)),
            "real": _report(~synth), "synthetic": _report(synth)}


def summarize(name: str, ckpt_dir: Path, device) -> dict:
    metrics = json.load(open(ckpt_dir / "metrics.json"))
    hold_fcr = evaluate_internal_hold_fcr(ckpt_dir, float(json.load(open(ckpt_dir / "config.json"))["context_s"]), device)
    return {
        "name": name, "best_epoch": metrics.get("best_epoch"), "final_epoch": metrics.get("final_epoch"),
        "stopped_early": metrics.get("stopped_early"), "best_val_auc": metrics.get("best_val_auc"),
        "overall_auc": metrics["overall"]["auc"], "overall_fcr_at_recall95": metrics["overall"]["fcr_at_recall95"],
        "real_auc": metrics["real_all"]["auc"], "real_fcr_at_recall95": metrics["real_all"]["fcr_at_recall95"],
        "implicit_incomplete_fcr": metrics["implicit_incomplete"]["fcr"],
        "calibration": metrics["calibration"], "threshold": metrics["threshold"],
        "internal_hold_fcr": hold_fcr,
        "latency_p50_ms": metrics.get("onnx", {}).get("p50_ms"), "latency_p95_ms": metrics.get("onnx", {}).get("p95_ms"),
        "n_parameters": metrics.get("n_parameters"), "macs": metrics.get("macs"),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    if not (ORIGINAL_4S_DIR / "metrics.json").exists():
        print(f"ERROR: {ORIGINAL_4S_DIR / 'metrics.json'} not found -- run run_8d_retrain_a0.py first.")
        sys.exit(1)

    original = analyze_existing_run()

    print(f"\n=== Training A0@4s (early stopping, max_epochs={MAX_EPOCHS}, "
          f"patience={EARLY_STOP_PATIENCE}, plateau LR) ===", flush=True)
    if not (LONGRUN_4S_DIR / "checkpoint.pt").exists():
        run_longer_a0(4.0, LONGRUN_4S_DIR)
    else:
        print(f"{LONGRUN_4S_DIR} already exists, skipping", flush=True)

    print(f"\n=== Training A0@2s (identical protocol) ===", flush=True)
    if not (LONGRUN_2S_DIR / "checkpoint.pt").exists():
        run_longer_a0(2.0, LONGRUN_2S_DIR)
    else:
        print(f"{LONGRUN_2S_DIR} already exists, skipping", flush=True)

    print("\n=== Comparison: A0@4s (longrun) vs. A0@2s (longrun) ===", flush=True)
    summary_4s = summarize("A0@4s_longrun", LONGRUN_4S_DIR, device)
    summary_2s = summarize("A0@2s_longrun", LONGRUN_2S_DIR, device)
    for s in (summary_4s, summary_2s):
        print(f"\n{s['name']}: best_epoch={s['best_epoch']} final_epoch={s['final_epoch']} "
              f"stopped_early={s['stopped_early']} best_val_auc={s['best_val_auc']}")
        print(f"  overall_auc={s['overall_auc']} overall_fcr_at_recall95={s['overall_fcr_at_recall95']} "
              f"real_auc={s['real_auc']} real_fcr_at_recall95={s['real_fcr_at_recall95']} "
              f"implicit_incomplete_fcr={s['implicit_incomplete_fcr']}")
        print(f"  internal_hold_fcr: {s['internal_hold_fcr']}")
        print(f"  latency p50/p95: {s['latency_p50_ms']}/{s['latency_p95_ms']} ms")

    out = {"device": str(device), "original_fixed_epoch_run": original,
           "a0_4s_longrun": summary_4s, "a0_2s_longrun": summary_2s}
    out_path = Path("experiments") / "8h_a0_convergence_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
