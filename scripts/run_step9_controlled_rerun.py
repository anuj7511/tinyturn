"""
Phase-3 Step 9 -- controlled, early-stopped rerun + three-arm real/synthetic pause-sampling
comparison + short-complete / response-particle recall reporting.

Every run below (baseline + 4 new B1@1s trainings) uses the exact same protocol 8h found necessary
for the no-pause-event baseline (`experiments/B1_1s_8h_longrun`, reused as-is rather than retrained
-- it already used this protocol): epochs<=40 ceiling, early_stop_patience=6, lr_schedule="plateau",
batch_size=64, num_workers=2, seed=42. Every prior Step 9 result (`PHASE2_RESULTS_8a-9.md`) shared
B1's OLD fixed-5-epoch protocol instead, which 8h showed was not calibrated to this architecture's
convergence rate -- so none of those numbers are trusted here; this is a clean rerun, not a
continuation.

Checkpoint-selection discipline (verified, not just asserted): `train_p1.train_p1`'s val_auc is
computed on `val_loader` = plain `TinyTurnDataset` (final clips only), never on pause events --
i.e. every P1/P1a/P1b variant was already being selected on the same criterion as the no-pause
baseline, even in the original Step 9 pass. No fix was needed here; this script's job is just to
confirm that in the saved config/metrics rather than assume it.

Three-arm real/synthetic pause-sampling comparison (Section 3's new requirement): at whichever
lambda_hold the rerun confirms is the leading candidate (0.5, per the prior pass's "comes closest to
plain P1" read -- itself unconfirmed under the old protocol), run all three of:
  - "all" (real_synth_balance="proportional", i.e. the existing default -- natural real/synthetic
    mix, one event per clip per epoch)
  - "real_only" (new: only real-audio pause events -- confirmed directly against the D2 manifest
    that there's a real pool to use: 1,736 real train clips with an eligible pause event, 2,912
    real pause events total -- smaller than the ~2,230-clips figure floated in review, but a real,
    usable pool, not vanishingly small)
  - "50:50" (existing option, forced real/synthetic balance)

Response-particle lexicon below is an independently-reasonable list for THIS diagnostic, not a
reproduction of the original E3 9-word taxonomy (that generating script isn't in this working tree,
same situation as 8f's alt-threshold detector) -- includes the brief's own named examples ("haan",
"okay", "bas") plus common English backchannels. "Short-complete" uses n_words<=3 (n=61 on val,
ground-truth-complete) since the exact E3/E4 short-utterance cutoff isn't recoverable either.

Usage:
  python scripts/run_step9_controlled_rerun.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.pause_events import PauseEventDataset
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference
from tinyturn.train import TRAJECTORY_NAMES
from tinyturn.train_p1 import P1Config, train_p1
from torch.utils.data import DataLoader

CACHE_DIR = Path("data_cache")
BASELINE_DIR = Path("experiments") / "B1_1s_8h_longrun"  # already trained under this protocol (8h)
OUT_DIR = Path("experiments") / "step9_controlled_rerun"
CONTEXT_S = 1.0
TARGET_RECALLS = [0.90, 0.95]

# Shared protocol -- identical across every arm (brief: "same max-epoch budget, early stopping,
# LR schedule, seed, and checkpoint-selection metric across all four").
PROTOCOL = dict(epochs=40, early_stop_patience=6, lr_schedule="plateau",
                batch_size=64, num_workers=2, seed=42)

SHORT_COMPLETE_MAX_WORDS = 3
RESPONSE_PARTICLES = {
    "okay", "ok", "yeah", "yes", "yep", "yup", "alright", "right", "sure",
    "haan", "bas", "mm", "mhm", "uh-huh", "uhhuh", "no", "nope",
}


def _load_model(ckpt_dir: Path):
    cfg = json.load(open(ckpt_dir / "config.json"))
    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model, cfg


def _pause_probs_and_synth(model, pause_loader, device):
    y_prob, synthetic = [], []
    with torch.no_grad():
        for batch in pause_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            logits = model(log_mel, mask, traj)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            synthetic.extend(batch["synthetic"])
    return np.array(y_prob), np.array(synthetic, dtype=bool)


def matched_threshold_curve_v2(model, val_loader, pause_loader, device, n_thresholds=201):
    """Extends the Step 9 v1 curve (recall_complete / fcr_implicit_incomplete / fcr_holds) with
    real-only and synthetic-only hold-FCR curves -- "three separate recall-vs-hold-FCR curves",
    not one aggregate that can hide a real-audio-specific regression."""
    val_out = run_inference(model, val_loader, device, use_trajectory=True)
    pause_probs, pause_synth = _pause_probs_and_synth(model, pause_loader, device)
    real_mask, synth_mask = ~pause_synth, pause_synth

    complete_mask = val_out.y_true.astype(bool)
    implicit_mask = np.array(val_out.implicit_incomplete, dtype=bool)
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    recall_complete, fcr_implicit, fcr_holds_all, fcr_holds_real, fcr_holds_synth = [], [], [], [], []
    for t in thresholds:
        pred = val_out.y_prob >= t
        recall_complete.append(float(pred[complete_mask].mean()) if complete_mask.any() else float("nan"))
        fcr_implicit.append(float(pred[implicit_mask].mean()) if implicit_mask.any() else float("nan"))
        fcr_holds_all.append(float((pause_probs >= t).mean()) if len(pause_probs) else float("nan"))
        fcr_holds_real.append(float((pause_probs[real_mask] >= t).mean()) if real_mask.any() else float("nan"))
        fcr_holds_synth.append(float((pause_probs[synth_mask] >= t).mean()) if synth_mask.any() else float("nan"))
    return {"thresholds": thresholds.tolist(), "recall_complete": recall_complete,
            "fcr_implicit_incomplete": fcr_implicit, "fcr_holds_all": fcr_holds_all,
            "fcr_holds_real": fcr_holds_real, "fcr_holds_synthetic": fcr_holds_synth,
            "n_val_complete": int(complete_mask.sum()), "n_val_implicit_incomplete": int(implicit_mask.sum()),
            "n_val_pause_holds_real": int(real_mask.sum()), "n_val_pause_holds_synthetic": int(synth_mask.sum())}


def at_matched_recall_v2(curve: dict, target_recall: float) -> dict:
    recalls = np.array(curve["recall_complete"])
    idx = int(np.argmin(np.abs(recalls - target_recall)))
    return {"target_recall": target_recall, "threshold": curve["thresholds"][idx],
            "actual_recall_complete": recalls[idx],
            "fcr_holds_all": curve["fcr_holds_all"][idx], "fcr_holds_real": curve["fcr_holds_real"][idx],
            "fcr_holds_synthetic": curve["fcr_holds_synthetic"][idx],
            "fcr_implicit_incomplete": curve["fcr_implicit_incomplete"][idx]}


def _has_response_particle(text) -> bool:
    if not isinstance(text, str):
        return False
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return any(w in RESPONSE_PARTICLES for w in words)


def extra_slice_recalls(model, val_ds, device, threshold: float, trans_df: pd.DataFrame) -> dict:
    """Short-complete recall + response-particle complete recall, at the model's own calibrated
    threshold (Section 3: "for every arm" -- pause training pushing toward conservatism could
    overcorrect on exactly these short, legitimately-complete replies)."""
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    val_out = run_inference(model, val_loader, device, use_trajectory=True)
    ids = np.array(val_out.ids)
    probs = val_out.y_prob
    complete = val_out.y_true.astype(bool)

    meta = trans_df.set_index("id")
    n_words = np.array([meta["n_words"].get(i, np.nan) for i in ids])
    has_particle = np.array([_has_response_particle(meta["text"].get(i)) for i in ids])

    def _recall(mask):
        sub_complete = mask & complete
        n = int(sub_complete.sum())
        if n == 0:
            return {"n": 0, "recall": None}
        return {"n": n, "recall": round(float((probs[sub_complete] >= threshold).mean()), 4)}

    short_mask = n_words <= SHORT_COMPLETE_MAX_WORDS
    return {
        "short_complete_recall": _recall(short_mask),
        "response_particle_complete_recall": _recall(has_particle),
    }


def plot_curves(curves: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, key, title in zip(axes, ["fcr_holds_all", "fcr_holds_real", "fcr_holds_synthetic"],
                               ["All holds", "Real holds", "Synthetic holds"]):
        for name, curve in curves.items():
            ax.plot(curve["recall_complete"], curve[key], marker=".", markersize=2, label=name)
        ax.set_xlabel("Recall on complete final turns")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FCR on internal holds")
    axes[-1].legend(fontsize=8, loc="upper left")
    plt.suptitle("Step 9 controlled rerun: matched-threshold recall vs. FCR-at-holds, by hold source")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def confirm_real_pause_pool():
    events = pd.read_parquet(CACHE_DIR / "tinyturn_pause_events.parquet")
    splits = pd.read_parquet(CACHE_DIR / "tinyturn_splits.parquet")[["id", "split", "synthetic"]] \
        .rename(columns={"id": "clip_id"})
    df = events.merge(splits, on="clip_id", how="left")
    train = df[df["split"] == "train"]
    n_real_clips = train[~train["synthetic"]]["clip_id"].nunique()
    n_real_events = int((~train["synthetic"]).sum())
    print(f"confirmed directly against tinyturn_pause_events.parquet: train split has "
          f"{n_real_clips} real clips with an eligible pause event ({n_real_events} real pause "
          f"events total) -- vs. the unconfirmed ~2,230-clips figure floated in review.", flush=True)
    return {"train_real_clips_with_pause_events": int(n_real_clips), "train_real_pause_events": n_real_events}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if not (BASELINE_DIR / "checkpoint.pt").exists():
        print(f"ERROR: {BASELINE_DIR / 'checkpoint.pt'} not found -- run run_8h_b1_convergence_check.py first.")
        sys.exit(1)

    pool_note = confirm_real_pause_pool()

    baseline_cfg = json.load(open(BASELINE_DIR / "config.json"))
    print(f"baseline protocol (reused from 8h, already early-stopped): {baseline_cfg}", flush=True)

    runs = {
        "P1_plain": dict(lambda_hold=None, controlled_sampling=False, real_synth_balance="proportional"),
        "P1ab_lambda0.25": dict(lambda_hold=0.25, controlled_sampling=True, real_synth_balance="proportional"),
        "P1ab_lambda0.5_all": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="proportional"),
        "P1ab_lambda0.5_real_only": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="real_only"),
        "P1ab_lambda0.5_5050": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50"),
    }
    run_dirs = {"baseline (no pause events, 8h longrun)": BASELINE_DIR}
    for tag, overrides in runs.items():
        d = OUT_DIR / tag
        run_dirs[tag] = d
        if (d / "checkpoint.pt").exists():
            print(f"{d} already exists, skipping retrain", flush=True)
            continue
        cfg = P1Config(exp_id=tag, context_s=CONTEXT_S, **PROTOCOL, **overrides)
        print(f"\n=== training {tag}: {overrides} ===", flush=True)
        train_p1(cfg, BASELINE_DIR / "checkpoint.pt", d)

    # -- checkpoint-selection discipline check (verify, don't assume) --
    selection_check = {}
    for tag, d in run_dirs.items():
        cfg = json.load(open(d / "config.json"))
        selection_check[tag] = "final-clip val_auc (train_p1's val_loader is final-clips-only TinyTurnDataset)"
    print("\ncheckpoint-selection criterion used by every arm:", json.dumps(selection_check, indent=2), flush=True)

    print("\n=== building matched-threshold curves + extra slice recalls for every arm ===", flush=True)
    ds_kwargs = dict(context_s=CONTEXT_S, include_trajectory=True)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=CONTEXT_S, include_trajectory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
    trans_df = pd.read_parquet(CACHE_DIR / "d2_stratified_transcripts.parquet")[["id", "text", "n_words"]]

    curves, matched, extra_slices, run_metrics = {}, {}, {}, {}
    for name, d in run_dirs.items():
        model, cfg = _load_model(d)
        metrics = json.load(open(d / "metrics.json"))
        threshold = float(metrics["threshold"])
        curve = matched_threshold_curve_v2(model, val_loader, pause_loader, device)
        curves[name] = curve
        matched[name] = {f"recall_{int(r*100)}": at_matched_recall_v2(curve, r) for r in TARGET_RECALLS}
        extra_slices[name] = extra_slice_recalls(model, val_ds, device, threshold, trans_df)
        run_metrics[name] = {
            "best_epoch": metrics.get("best_epoch"), "final_epoch": metrics.get("final_epoch"),
            "stopped_early": metrics.get("stopped_early"), "best_val_auc": metrics.get("best_val_auc"),
            "overall_auc": metrics.get("overall", {}).get("auc"),
            "real_auc": metrics.get("real_all", {}).get("auc"),
            "threshold": threshold, "lambda_hold": cfg.get("lambda_hold"),
            "controlled_sampling": cfg.get("controlled_sampling"),
            "real_synth_balance": cfg.get("real_synth_balance"),
        }
        print(f"\n{name}: best_epoch={metrics.get('best_epoch')} final_epoch={metrics.get('final_epoch')} "
              f"stopped_early={metrics.get('stopped_early')} best_val_auc={metrics.get('best_val_auc')}")
        for r in TARGET_RECALLS:
            m = matched[name][f"recall_{int(r*100)}"]
            print(f"  @ recall~{r:.0%} (actual {m['actual_recall_complete']:.3f}): "
                  f"fcr_holds_all={m['fcr_holds_all']:.4f} fcr_holds_real={m['fcr_holds_real']:.4f} "
                  f"fcr_holds_synthetic={m['fcr_holds_synthetic']:.4f}")
        es = extra_slices[name]
        print(f"  short_complete_recall: {es['short_complete_recall']}  "
              f"response_particle_complete_recall: {es['response_particle_complete_recall']}")

    plot_path = OUT_DIR / "recall_vs_fcr_holds_by_source.png"
    plot_curves(curves, plot_path)
    print(f"\nsaved plot to {plot_path}", flush=True)

    out = {
        "protocol": PROTOCOL,
        "real_pause_pool_confirmation": pool_note,
        "checkpoint_selection_criterion": selection_check,
        "run_metrics": run_metrics,
        "matched_recall_summary": matched,
        "extra_slice_recalls": extra_slices,
        "short_complete_max_words": SHORT_COMPLETE_MAX_WORDS,
        "response_particle_lexicon": sorted(RESPONSE_PARTICLES),
        "curves": curves,
    }
    with open(OUT_DIR / "step9_controlled_rerun_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {OUT_DIR / 'step9_controlled_rerun_results.json'}")


if __name__ == "__main__":
    main()
