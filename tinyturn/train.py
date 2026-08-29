"""
Generic training loop shared by B0 / B1 / B1-f0 / (A0 uses train_whisper.py instead, since its
input is raw waveform + a different encoder, not log-mel + TinyTurnModel). Which experiment runs is
entirely a function of the config passed in -- no experiment-specific code lives here, per the
brief's ground rule against implementing multiple ideas at once.
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference, full_report, calibrate_threshold, count_macs
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes


TRAJECTORY_NAMES = ["rel_energy", "pause_prob", "spectral_tilt", "spectral_flux", "envelope_activity"]
TRAJECTORY_NAMES_F0 = TRAJECTORY_NAMES + ["f0_semitone"]


@dataclass
class ExperimentConfig:
    exp_id: str
    context_s: float = 4.0
    use_trajectory: bool = False
    use_f0: bool = False
    epochs: int = 3                              # hard ceiling on epochs run
    early_stop_patience: Optional[int] = None     # Phase-2 8h: stop if val_auc hasn't beaten its
                                                   # best in this many epochs. None = original
                                                   # fixed-epoch-count behavior (B0/B1/B1-f0 as run).
    lr_schedule: Optional[str] = None             # Phase-2 8h: None (fixed lr, as run) |
                                                   # "plateau" (ReduceLROnPlateau on val_auc) --
                                                   # its "own LR schedule", not A0's inherited one.
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 8
    mel_channels: int = 112
    traj_channels: int = 24
    seed: int = 42


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_experiment(cfg: ExperimentConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    device = torch.device("cpu")

    traj_names = TRAJECTORY_NAMES_F0 if cfg.use_f0 else TRAJECTORY_NAMES
    trajectory_dim = len(traj_names) if cfg.use_trajectory else 0

    ds_kwargs = dict(context_s=cfg.context_s,
                      include_trajectory=cfg.use_trajectory, include_f0=cfg.use_f0)
    train_ds = TinyTurnDataset(split="train", **ds_kwargs)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate,
                             persistent_workers=cfg.num_workers > 0)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               persistent_workers=cfg.num_workers > 0)

    model = TinyTurnModel(n_mels=40, trajectory_dim=trajectory_dim,
                           mel_channels=cfg.mel_channels, traj_channels=cfg.traj_channels).to(device)
    n_params = model.num_parameters()
    print(f"[{cfg.exp_id}] model params: {n_params}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = None
    if cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    elif cfg.lr_schedule is not None:
        raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule!r}")

    best_val_auc = -1.0
    best_state = None
    best_epoch = None
    epochs_since_improvement = 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device) if cfg.use_trajectory else None
            labels = batch["label"].to(device)

            opt.zero_grad()
            logits = model(log_mel, mask, traj)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        val_out = run_inference(model, val_loader, device, cfg.use_trajectory)
        val_auc = roc_auc_score(val_out.y_true, val_out.y_prob) if len(set(val_out.y_true)) > 1 else float("nan")
        val_auc = max(val_auc, 1 - val_auc) if val_auc == val_auc else val_auc
        epoch_time = time.time() - t0
        avg_loss = running_loss / max(n_batches, 1)
        lr_now = opt.param_groups[0]["lr"]
        print(f"[{cfg.exp_id}] epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f} "
              f"val_auc={val_auc:.4f} lr={lr_now:.2e} ({epoch_time:.1f}s)", flush=True)
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

    calib_out = run_inference(model, calib_loader, device, cfg.use_trajectory)
    threshold = calibrate_threshold(calib_out.y_true, calib_out.y_prob, target_fcr=0.05)
    print(f"[{cfg.exp_id}] calibrated threshold (target FCR<=0.05 on calib): {threshold:.4f}", flush=True)

    val_out = run_inference(model, val_loader, device, cfg.use_trajectory)
    report = full_report(val_out, threshold)
    report["n_parameters"] = n_params
    report["history"] = history
    report["best_epoch"] = best_epoch
    report["final_epoch"] = history[-1]["epoch"] if history else None
    report["best_val_auc"] = best_val_auc
    report["stopped_early"] = history[-1]["epoch"] < cfg.epochs if history else False

    # MACs/ONNX use CPU-default dummy tensors (matching this project's deployment target) -- move
    # off CUDA once, here, rather than risk a device mismatch (mirrors the identical fix in
    # train_whisper.py / train_p1.py).
    model = model.to("cpu")

    n_frames = train_ds[0]["log_mel"].shape[0]
    mac_inputs = (torch.randn(1, n_frames, 40), torch.ones(1, n_frames, dtype=torch.bool))
    if cfg.use_trajectory:
        mac_inputs = mac_inputs + (torch.randn(1, n_frames, trajectory_dim),)
    else:
        mac_inputs = mac_inputs + (None,)
    report["macs"] = int(count_macs(model, *mac_inputs))

    try:
        onnx_path = out_dir / "model.onnx"
        export_onnx(model, onnx_path, n_frames=n_frames, trajectory_dim=trajectory_dim)
        report["onnx_size_bytes"] = onnx_file_size_bytes(onnx_path)

        sample = val_ds[0]
        sample_row = val_ds.df.iloc[0]
        from tinyturn.boundary import estimate_speech_end
        y_raw, sr = val_ds._load_wav(sample_row["id"])
        from tinyturn.preprocess import build_example
        ex = build_example(y_raw, sr, float(sample_row["last_active_t"]), cfg.context_s,
                            label=bool(sample_row["endpoint_bool"]), row_id=sample_row["id"])
        latency = measure_latency_ms(onnx_path, ex.waveform, sr, ex.valid_sample_mask,
                                      cfg.use_trajectory, traj_names)
        report["latency"] = latency
        print(f"[{cfg.exp_id}] latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms "
              f"(batch-1, incl. feature extraction)", flush=True)
    except Exception as e:
        report["onnx"] = {"error": str(e)}
        print(f"[{cfg.exp_id}] ONNX export/latency measurement failed: {e}", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[{cfg.exp_id}] saved results to {out_dir}", flush=True)
    return report
