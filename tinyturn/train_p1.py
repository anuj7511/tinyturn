"""
Step 7 / Phase-2 Step 9 -- P1: final clips + internal-pause continuation events, vs. the
context_ablation_mel_trajectory_1s(_pv2speechend) baseline (final clips only, same architecture/context/epochs/batch/lr).

`lambda_hold=None` (default) reproduces Step 7's original recipe exactly: a straight, unweighted
blend of final clips and pause events in one combined loss mean, all pause events used every epoch.
This is what "before anything else, re-run the P1-vs-baseline comparison" (Phase-2 Step 9) means --
a fresh, contract-correct version of Step 7's own P1, to serve as the reference point before trying
P1a/P1b's refinements.

`lambda_hold=<float>` switches on P1a (Phase-2 Step 9): mean-normalized per component,
`L = mean(L_final) + lambda_hold * mean(L_hold)`, so the effective hold-loss weight doesn't drift
with how many pause events happen to land in a given batch (Step 7's plain blend effectively
weights every *example* equally regardless of group, which is not expressible as any single
lambda_hold under this normalization -- the two are genuinely different training objectives, not
compatible modes of the same knob).

`controlled_sampling=True` switches on P1b (Phase-2 Step 9): at most one internal-pause event per
parent clip per epoch (resampled fresh each epoch, so a clip with 2 eligible pauses still
contributes across training, just never both in the same epoch), preventing clips with more
eligible pauses from silently outweighing single-pause clips. Real/synthetic balance policy is
`real_synth_balance` ("proportional", the default -- real-audio pause events are already scarce, so
capping to a forced 50:50 split would throw most of that scarce signal away without a stated reason
to prefer forcing balance over the natural composition; "50:50" is available too), logged every
epoch either way.

Compares (unchanged from Step 7):
  (a) standard Section-8 report on val FINAL CLIPS ONLY, against the baseline's own report -- did
      augmenting training with pause events help or hurt the main endpoint task?
  (b) false-complete rate at internal holds: both the baseline (loaded from its own checkpoint,
      never trained on pause events) and P1 evaluated on val PAUSE EVENTS (all label=incomplete
      by construction, so only FCR is meaningful there), split real vs. synthetic.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import roc_auc_score

from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.pause_events import PauseEventDataset
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference, full_report, calibrate_threshold, count_macs
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes
from tinyturn.train import TRAJECTORY_NAMES


@dataclass
class P1Config:
    exp_id: str
    context_s: float = 1.0
    epochs: int = 5                             # hard ceiling on epochs run
    early_stop_patience: Optional[int] = None   # Phase-3 Step 9: stop if val_auc (final clips)
                                                 # hasn't beaten its best in this many epochs. None =
                                                 # original fixed-epoch-count behavior (Step 9 v1).
    lr_schedule: Optional[str] = None            # Phase-3 Step 9: None (fixed lr) | "plateau"
                                                 # (ReduceLROnPlateau on val_auc) -- matches the
                                                 # protocol 8h found necessary for the baseline.
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 2
    mel_channels: int = 112
    traj_channels: int = 24
    seed: int = 42
    lambda_hold: Optional[float] = None         # None = Step 7's original unweighted blend;
                                                 # float = Phase-2 Step 9 / P1a
    controlled_sampling: bool = False           # Phase-2 Step 9 / P1b
    real_synth_balance: str = "proportional"    # "proportional" | "50:50" | "real_only" -- only
                                                 # used when controlled_sampling=True; always logged.
                                                 # Phase-3 Step 9 adds "real_only" for the three-arm
                                                 # comparison (all / real-only / 50:50).


def _p1_loss(logits: torch.Tensor, labels: torch.Tensor, is_pause: torch.Tensor,
             lambda_hold: Optional[float]) -> torch.Tensor:
    if lambda_hold is None:
        return F.binary_cross_entropy_with_logits(logits, labels)
    final_mask = ~is_pause
    hold_mask = is_pause
    loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if final_mask.any():
        loss = loss + F.binary_cross_entropy_with_logits(logits[final_mask], labels[final_mask])
    if hold_mask.any():
        loss = loss + lambda_hold * F.binary_cross_entropy_with_logits(logits[hold_mask], labels[hold_mask])
    return loss


def _select_epoch_pause_events(pause_df: pd.DataFrame, balance: str, seed: int) -> pd.DataFrame:
    """P1b: at most one event per parent clip this epoch."""
    one_per_clip = pause_df.groupby("clip_id", group_keys=False).sample(n=1, random_state=seed)
    # Defensive: force real bool dtype rather than trusting it survived upstream merges/filters --
    # a left-merge against a `splits_path` that doesn't cover every event's clip_id (e.g. a partial
    # splits file) introduces NaN into "synthetic", silently upcasting the whole column to `object`.
    # `~` on an object-dtype column of Python bools does elementwise bitwise-NOT (~True=-2, ~False=
    # -1) instead of boolean negation, and indexing a DataFrame with that result is read as
    # column-label selection, not a row mask -- surfaces as a confusing KeyError far from the cause.
    one_per_clip = one_per_clip.assign(synthetic=one_per_clip["synthetic"].astype(bool))
    if balance == "proportional":  # Phase-3 Step 9's "all pause events" arm
        return one_per_clip
    if balance == "50:50":
        real = one_per_clip[~one_per_clip["synthetic"]]
        synth = one_per_clip[one_per_clip["synthetic"]]
        n = min(len(real), len(synth))
        if n == 0:  # can't balance (one side empty) -- fall back to the natural pool
            return one_per_clip
        return pd.concat([real.sample(n=n, random_state=seed), synth.sample(n=n, random_state=seed)])
    if balance == "real_only":  # Phase-3 Step 9's third arm -- tests whether synthetic pause
        # events are the ones dominating the learned hold representation (Section 3's monotonic
        # real-AUC-cost-with-lambda observation), not yet confirmed as such.
        real = one_per_clip[~one_per_clip["synthetic"]]
        if len(real) == 0:
            return one_per_clip
        return real
    raise ValueError(f"unknown real_synth_balance: {balance!r}")


def evaluate_fcr_at_holds(model, pause_loader, device, threshold: float) -> dict:
    model.eval()
    y_prob, synthetic = [], []
    with torch.no_grad():
        for batch in pause_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            logits = model(log_mel, mask, traj)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            synthetic.extend(batch["synthetic"])
    y_prob = np.array(y_prob)
    synthetic = np.array(synthetic, dtype=bool)

    def _report(mask):
        p = y_prob[mask]
        if len(p) == 0:
            return {"n": 0, "fcr": None}
        return {"n": int(len(p)), "fcr": round(float((p >= threshold).mean()), 4)}

    return {
        "threshold": round(float(threshold), 4),
        "all": _report(np.ones_like(synthetic, dtype=bool)),
        "real": _report(~synthetic),
        "synthetic": _report(synthetic),
    }


def train_p1(cfg: P1Config, baseline_checkpoint: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Auto-detect CUDA (e.g. Kaggle/Colab) -- every prior run in this project was CPU-only.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg.exp_id}] device: {device}", flush=True)

    ds_kwargs = dict(context_s=cfg.context_s, include_trajectory=True)
    train_final = TinyTurnDataset(split="train", **ds_kwargs)
    train_pause_full = PauseEventDataset(split="train", context_s=cfg.context_s, include_trajectory=True)
    n_clips_with_events = train_pause_full.df["clip_id"].nunique()
    print(f"[{cfg.exp_id}] train: {len(train_final)} final clips, {n_clips_with_events} clips with "
          f"eligible pause events ({len(train_pause_full)} events total)", flush=True)
    print(f"[{cfg.exp_id}] lambda_hold={cfg.lambda_hold} controlled_sampling={cfg.controlled_sampling} "
          f"real_synth_balance={cfg.real_synth_balance}", flush=True)

    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=cfg.context_s, include_trajectory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate,
                             persistent_workers=cfg.num_workers > 0)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               persistent_workers=cfg.num_workers > 0)
    val_pause_loader = DataLoader(val_pause_ds, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, collate_fn=collate)

    if not cfg.controlled_sampling:
        # Step 7's original behavior: one static combined dataset, every pause event, every epoch.
        train_combined = ConcatDataset([train_final, train_pause_full])
        static_train_loader = DataLoader(train_combined, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, collate_fn=collate,
                                          persistent_workers=cfg.num_workers > 0)

    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.mel_channels, traj_channels=cfg.traj_channels).to(device)
    n_params = model.num_parameters()
    print(f"[{cfg.exp_id}] model params: {n_params}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = None
    if cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    elif cfg.lr_schedule is not None:
        raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule!r}")

    best_val_auc, best_state, best_epoch, history = -1.0, None, None, []
    epochs_since_improvement = 0
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()

        if cfg.controlled_sampling:
            epoch_pause_df = _select_epoch_pause_events(
                train_pause_full.df, cfg.real_synth_balance, seed=cfg.seed * 1000 + epoch)
            n_real = int((~epoch_pause_df["synthetic"]).sum())
            n_synth = int(epoch_pause_df["synthetic"].sum())
            print(f"[{cfg.exp_id}] epoch {epoch+1} pause-event pool: {len(epoch_pause_df)} "
                  f"(real={n_real}, synthetic={n_synth})", flush=True)
            train_pause_epoch = Subset(train_pause_full, epoch_pause_df.index.tolist())
            train_combined = ConcatDataset([train_final, train_pause_epoch])
            train_loader = DataLoader(train_combined, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers, collate_fn=collate)
        else:
            train_loader = static_train_loader

        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            labels = batch["label"].to(device)
            is_pause = torch.tensor(batch["is_pause_event"], dtype=torch.bool, device=device)

            opt.zero_grad()
            logits = model(log_mel, mask, traj)
            loss = _p1_loss(logits, labels, is_pause, cfg.lambda_hold)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        # Checkpoint selection metric: val_auc on FINAL CLIPS ONLY (val_loader = TinyTurnDataset,
        # never pause events) -- same criterion the no-pause baseline uses, by construction, not
        # internal-hold performance. Verified directly (Phase-3 Step 9's checkpoint-selection-
        # discipline requirement) rather than assumed: this was already true in the original Step 9
        # pass, since val_loader here has always been final-clips-only.
        val_out = run_inference(model, val_loader, device, True)
        val_auc = roc_auc_score(val_out.y_true, val_out.y_prob) if len(set(val_out.y_true)) > 1 else float("nan")
        val_auc = max(val_auc, 1 - val_auc) if val_auc == val_auc else val_auc
        epoch_time = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"[{cfg.exp_id}] epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f} "
              f"val_auc(final-clips)={val_auc:.4f} lr={lr_now:.2e} ({epoch_time:.1f}s)", flush=True)
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_auc": val_auc,
                         "lr": lr_now, "epoch_time_s": round(epoch_time, 1)})
        if scheduler is not None and val_auc == val_auc:
            scheduler.step(val_auc)
        if val_auc == val_auc and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if (cfg.early_stop_patience is not None
                and epochs_since_improvement >= cfg.early_stop_patience):
            print(f"[{cfg.exp_id}] early stopping after epoch {epoch+1}: no val_auc improvement "
                  f"in {epochs_since_improvement} epochs (best={best_val_auc:.4f} @ epoch {best_epoch})",
                  flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "checkpoint.pt")

    calib_out = run_inference(model, calib_loader, device, True)
    threshold = calibrate_threshold(calib_out.y_true, calib_out.y_prob, target_fcr=0.05)
    print(f"[{cfg.exp_id}] calibrated threshold: {threshold:.4f}", flush=True)

    val_out = run_inference(model, val_loader, device, True)
    report = full_report(val_out, threshold)
    report["n_parameters"] = n_params
    report["history"] = history
    report["best_epoch"] = best_epoch
    report["final_epoch"] = history[-1]["epoch"] if history else None
    report["best_val_auc"] = best_val_auc
    report["stopped_early"] = history[-1]["epoch"] < cfg.epochs if history else False
    report["train_set_size"] = {"final_clips": len(train_final),
                                 "clips_with_eligible_pause_events": n_clips_with_events,
                                 "pause_events_total": len(train_pause_full)}

    # Everything below (MACs, ONNX export/latency, FCR-at-holds) is cheap regardless of device (B1
    # is tiny) and MACs/ONNX use CPU-default dummy tensors matching this project's deployment
    # target -- move off CUDA once, here, rather than risk a device mismatch further down.
    model = model.to("cpu")
    device = torch.device("cpu")

    n_frames = train_final[0]["log_mel"].shape[0]
    onnx_path = out_dir / "model.onnx"
    mac_inputs = (torch.randn(1, n_frames, 40), torch.ones(1, n_frames, dtype=torch.bool),
                  torch.randn(1, n_frames, len(TRAJECTORY_NAMES)))
    report["macs"] = int(count_macs(model, *mac_inputs))
    # ONNX export/latency are non-essential to Step 9's actual comparison (AUC / FCR-at-holds) and
    # this environment's `onnx` package is currently missing (present when mel_trajectory_1s_earlystopped_longrun was
    # trained -- environment drift, not a real code bug) -- caught rather than left to crash the
    # run after the expensive training loop already finished, matching train_whisper.py's identical
    # handling of the same gap.
    try:
        export_onnx(model, onnx_path, n_frames=n_frames, trajectory_dim=len(TRAJECTORY_NAMES))
        report["onnx_size_bytes"] = onnx_file_size_bytes(onnx_path)
        sample_row = val_ds.df.iloc[0]
        y_raw, sr = val_ds._load_wav(sample_row["id"])
        from tinyturn.preprocess import build_example
        ex = build_example(y_raw, sr, float(sample_row["last_active_t"]), cfg.context_s,
                            label=bool(sample_row["endpoint_bool"]), row_id=sample_row["id"])
        report["latency"] = measure_latency_ms(onnx_path, ex.waveform, sr, ex.valid_sample_mask,
                                                True, TRAJECTORY_NAMES)
    except Exception as e:
        report["onnx"] = {"error": str(e)}
        print(f"[{cfg.exp_id}] ONNX export/latency measurement failed: {e}", flush=True)

    # -- the headline P1 comparison: FCR at internal holds, baseline (never saw pause events)
    # vs. P1 (trained with them), on the same val pause-event set, real vs. synthetic.
    report["fcr_at_holds_p1"] = evaluate_fcr_at_holds(model, val_pause_loader, device, threshold)

    if baseline_checkpoint is not None:
        baseline_model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                                        mel_channels=cfg.mel_channels, traj_channels=cfg.traj_channels)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
        baseline_threshold = threshold  # report both at P1's own calibrated threshold for comparability
        report["fcr_at_holds_baseline"] = evaluate_fcr_at_holds(baseline_model, val_pause_loader, device,
                                                                 baseline_threshold)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{cfg.exp_id}] saved results to {out_dir}", flush=True)
    return report
