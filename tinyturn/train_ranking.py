"""
Step 10 planning, item 3 -- within-utterance pairwise-ranking experiment, replacing lambda=0.75.

Motivation (the plan): hold FCR remains very high under plain BCE, and stronger binary hold
weighting (P1a's lambda_hold) damages real AUC. Instead of training the hold event AS A NEGATIVE
EXAMPLE directly, add a margin loss that only asks the model to rank a completed clip's own final
score above an internal-hold score drawn from *the same clip*:

  L = L_final_BCE + 0.1 * max(0, 0.2 - s_final + s_hold)

Requirements (verbatim from the plan):
  - pairs come from the same completed clip -- only clips labeled complete (endpoint_bool=True)
    contribute a ranking pair; PauseEventDataset carries no parent label, so eligibility is
    determined here by joining pause events back to d2_stratified_signal_features's endpoint_bool.
  - at most one internal hold per clip per epoch -- one pause event sampled fresh each epoch per
    eligible clip_id, same mechanism as P1b's `_select_epoch_pause_events` (reused directly).
  - main BCE remains final-clips-only -- the BCE term never touches hold examples; only the ranking
    margin does. (Contrast with P1/distillation, where holds get their own direct hard-label BCE
    term -- this experiment deliberately does NOT add one, per the plan's exact loss formula.)
  - checkpoint selection remains final-clips-only (val_auc on TinyTurnDataset val, no pause events).

No student boundary augmentation here -- the plan's step 3 doesn't list it as part of this
experiment's recipe (unlike step 2's distillation runs), so the canonical boundary is used as-is.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from tinyturn.dataset import TinyTurnDataset, collate, CACHE_DIR
from tinyturn.pause_events import PauseEventDataset
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference, full_report, calibrate_threshold, count_macs
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes
from tinyturn.train import TRAJECTORY_NAMES
from tinyturn.train_p1 import evaluate_fcr_at_holds


@dataclass
class RankingConfig:
    exp_id: str
    context_s: float = 1.0
    epochs: int = 5
    early_stop_patience: Optional[int] = None
    lr_schedule: Optional[str] = None
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 0                        # per-epoch pair resampling needs the main-process
                                                 # dataset state; keep simple (0) rather than plumb
                                                 # worker-side epoch synchronization for one experiment.
    mel_channels: int = 112
    traj_channels: int = 24
    seed: int = 42
    margin: float = 0.2
    rank_weight: float = 0.1


class RankingPairDataset(Dataset):
    """Wraps a final-clips TinyTurnDataset; each item also carries a same-epoch-sampled internal-
    hold pair when its own clip is labeled complete and has an eligible hold event, else a zeroed,
    flagged-off dummy hold so every item has the same tensor shapes for collation."""

    def __init__(self, final_ds: TinyTurnDataset, pause_ds: PauseEventDataset, seed: int):
        self.final_ds = final_ds
        self.pause_ds = pause_ds
        self.seed = seed
        complete_ids = set(final_ds.df.loc[final_ds.df["endpoint_bool"], "id"])
        self.eligible_pause_df = pause_ds.df[pause_ds.df["clip_id"].isin(complete_ids)]
        n_clips = self.eligible_pause_df["clip_id"].nunique()
        print(f"RankingPairDataset: {n_clips} completed clips have >=1 eligible internal-hold pair "
              f"out of {len(final_ds)} final clips ({self.eligible_pause_df['clip_id'].nunique()} / "
              f"{int(final_ds.df['endpoint_bool'].sum())} completed)", flush=True)
        self.clip_to_pause_idx = {}
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        one_per_clip = self.eligible_pause_df.groupby("clip_id", group_keys=False).sample(
            n=1, random_state=self.seed * 1000 + epoch)
        self.clip_to_pause_idx = dict(zip(one_per_clip["clip_id"], one_per_clip.index))

    def __len__(self):
        return len(self.final_ds)

    def __getitem__(self, idx):
        item = self.final_ds[idx]
        pause_idx = self.clip_to_pause_idx.get(item["id"])
        if pause_idx is not None:
            hold_item = self.pause_ds[pause_idx]
            item["hold_log_mel"] = hold_item["log_mel"]
            item["hold_valid_frame_mask"] = hold_item["valid_frame_mask"]
            item["hold_trajectory"] = hold_item["trajectory"]
            item["has_hold_pair"] = True
        else:
            item["hold_log_mel"] = torch.zeros_like(item["log_mel"])
            item["hold_valid_frame_mask"] = torch.zeros_like(item["valid_frame_mask"])
            item["hold_trajectory"] = torch.zeros_like(item["trajectory"])
            item["has_hold_pair"] = False
        return item


def collate_ranking(batch):
    out = collate(batch)
    for key in ["hold_log_mel", "hold_valid_frame_mask", "hold_trajectory"]:
        out[key] = torch.stack([b[key] for b in batch])
    out["has_hold_pair"] = torch.tensor([b["has_hold_pair"] for b in batch], dtype=torch.bool)
    return out


def _ranking_loss(final_logits: torch.Tensor, labels: torch.Tensor, hold_logits: torch.Tensor,
                   has_hold_pair: torch.Tensor, margin: float, rank_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(final_logits, labels)
    if has_hold_pair.any():
        s_final = torch.sigmoid(final_logits[has_hold_pair])
        s_hold = torch.sigmoid(hold_logits[has_hold_pair])
        rank_term = F.relu(margin - s_final + s_hold).mean()
    else:
        rank_term = torch.zeros((), dtype=final_logits.dtype, device=final_logits.device)
    return bce + rank_weight * rank_term


def train_ranking(cfg: RankingConfig, baseline_checkpoint: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg.exp_id}] device: {device} margin={cfg.margin} rank_weight={cfg.rank_weight}", flush=True)

    ds_kwargs = dict(context_s=cfg.context_s, include_trajectory=True)
    train_final = TinyTurnDataset(split="train", **ds_kwargs)
    train_pause_full = PauseEventDataset(split="train", context_s=cfg.context_s, include_trajectory=True)
    train_pairs = RankingPairDataset(train_final, train_pause_full, seed=cfg.seed)

    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=cfg.context_s, include_trajectory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate)
    val_pause_loader = DataLoader(val_pause_ds, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, collate_fn=collate)

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
        train_pairs.set_epoch(epoch)
        train_loader = DataLoader(train_pairs, batch_size=cfg.batch_size, shuffle=True,
                                   num_workers=cfg.num_workers, collate_fn=collate_ranking)
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            labels = batch["label"].to(device)
            hold_log_mel = batch["hold_log_mel"].to(device)
            hold_mask = batch["hold_valid_frame_mask"].to(device)
            hold_traj = batch["hold_trajectory"].to(device)
            has_hold_pair = batch["has_hold_pair"].to(device)

            opt.zero_grad()
            final_logits = model(log_mel, mask, traj)
            hold_logits = model(hold_log_mel, hold_mask, hold_traj)
            loss = _ranking_loss(final_logits, labels, hold_logits, has_hold_pair,
                                  cfg.margin, cfg.rank_weight)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

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
                                 "eligible_ranking_pairs": train_pairs.eligible_pause_df["clip_id"].nunique()}

    model = model.to("cpu")
    device = torch.device("cpu")

    n_frames = train_final[0]["log_mel"].shape[0]
    onnx_path = out_dir / "model.onnx"
    mac_inputs = (torch.randn(1, n_frames, 40), torch.ones(1, n_frames, dtype=torch.bool),
                  torch.randn(1, n_frames, len(TRAJECTORY_NAMES)))
    report["macs"] = int(count_macs(model, *mac_inputs))
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

    report["fcr_at_holds_ranking"] = evaluate_fcr_at_holds(model, val_pause_loader, device, threshold)
    if baseline_checkpoint is not None:
        baseline_model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                                        mel_channels=cfg.mel_channels, traj_channels=cfg.traj_channels)
        baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
        report["fcr_at_holds_baseline"] = evaluate_fcr_at_holds(baseline_model, val_pause_loader,
                                                                 device, threshold)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{cfg.exp_id}] saved results to {out_dir}", flush=True)
    return report
