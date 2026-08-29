"""
Step 10 planning, item 2 -- minimal distillation ablation: B1@1s student, whisper_tiny_boundary_robust_retrain as an
offline teacher (despite its own 8g deployment-robustness FAIL -- the plan uses it anyway, on the
logic that "not qualified as a deployed teacher under boundary perturbation" and "produces a useful
soft target for a smaller student" are different questions).

Two runs, selected via `teacher_target`:
  D1: canonical-boundary A0 logit only            (teacher_logits_a0_boundary_robust_train.parquet's
  D2: mean of canonical/alt-threshold/Silero logits  "teacher_logit_d1"/"teacher_logit_d2" columns)

Fixed recipe (the plan's "use only", no grid):
  T=2, alpha=0.5, B1@1s student, teacher loss on final clips only, hard labels on internal holds,
  student boundary augmentation enabled.

Per-example loss:
  final clip:  alpha * BCE(hard_label, student) + (1-alpha) * T^2 * BCE(sigmoid(teacher/T), sigmoid(student/T))
  internal hold: BCE(hard_label=0, student)                      -- no teacher term, per the plan.
Batch loss is the plain mean over every example (final + hold combined), matching Step 7/P1's
original unweighted-blend default -- the plan doesn't ask for a lambda_hold-style reweighting here,
only P1a/P1b introduced that as a separate, optional knob.

Reuses train_p1.py's evaluation scaffolding (val AUC on final clips only for checkpoint selection,
FCR-at-holds on val pause events, baseline comparison) since the underlying question -- did this
change to *how B1 is trained* help or hurt -- is the same shape as P1's.
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
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

from tinyturn.dataset import TinyTurnDataset, collate, CACHE_DIR
from tinyturn.pause_events import PauseEventDataset
from tinyturn.whisper_dataset import worker_init_fn
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference, full_report, calibrate_threshold, count_macs
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes
from tinyturn.train import TRAJECTORY_NAMES
from tinyturn.train_p1 import evaluate_fcr_at_holds

TEACHER_LOGITS_PATH = CACHE_DIR / "teacher_logits_a0_boundary_robust_train.parquet"


@dataclass
class DistillConfig:
    exp_id: str
    teacher_target: str                          # "d1" | "d2"
    context_s: float = 1.0
    epochs: int = 5
    early_stop_patience: Optional[int] = None
    lr_schedule: Optional[str] = None
    batch_size: int = 64
    lr: float = 1e-3
    num_workers: int = 2
    mel_channels: int = 112
    traj_channels: int = 24
    seed: int = 42
    T: float = 2.0
    alpha: float = 0.5
    teacher_logits_path: Path = TEACHER_LOGITS_PATH


def _teacher_logit_column_path(cfg: DistillConfig) -> Path:
    """TinyTurnDataset expects a parquet with an `id`/`teacher_logit` pair -- derive one from the
    shared precompute file's `teacher_logit_{d1,d2}` column rather than making the dataset class
    aware of D1-vs-D2 selection."""
    assert cfg.teacher_target in ("d1", "d2"), cfg.teacher_target
    src_col = f"teacher_logit_{cfg.teacher_target}"
    df = pd.read_parquet(cfg.teacher_logits_path)[["id", src_col]].rename(columns={src_col: "teacher_logit"})
    out_path = cfg.teacher_logits_path.parent / f"_derived_teacher_logit_{cfg.teacher_target}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _distill_loss(student_logits: torch.Tensor, hard_labels: torch.Tensor,
                   teacher_logits: torch.Tensor, is_pause: torch.Tensor,
                   T: float, alpha: float) -> torch.Tensor:
    hard = F.binary_cross_entropy_with_logits(student_logits, hard_labels, reduction="none")
    is_final = ~is_pause
    per_example = hard.clone()
    if is_final.any():
        assert torch.isfinite(teacher_logits[is_final]).all(), \
            "NaN/inf teacher_logit on a final clip -- teacher-logit merge is incomplete for this split"
        s = torch.sigmoid(student_logits[is_final] / T)
        t = torch.sigmoid(teacher_logits[is_final] / T)
        soft = F.binary_cross_entropy(s, t, reduction="none") * (T ** 2)
        per_example = per_example.clone()
        per_example[is_final] = alpha * hard[is_final] + (1 - alpha) * soft
    return per_example.mean()


def train_distill(cfg: DistillConfig, baseline_checkpoint: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg.exp_id}] device: {device} teacher_target={cfg.teacher_target} "
          f"T={cfg.T} alpha={cfg.alpha}", flush=True)

    teacher_col_path = _teacher_logit_column_path(cfg)

    ds_kwargs = dict(context_s=cfg.context_s, include_trajectory=True)
    train_final = TinyTurnDataset(split="train", augment_boundaries=True,
                                   teacher_logit_path=teacher_col_path, **ds_kwargs)
    n_missing_teacher = int(train_final.df["teacher_logit"].isna().sum())
    if n_missing_teacher:
        raise RuntimeError(f"{n_missing_teacher}/{len(train_final)} train final clips have no "
                            f"precomputed teacher logit -- rerun the teacher-logit precompute script")
    train_pause = PauseEventDataset(split="train", context_s=cfg.context_s, include_trajectory=True)
    print(f"[{cfg.exp_id}] train: {len(train_final)} final clips (boundary-augmented), "
          f"{len(train_pause)} internal-hold events (hard label only)", flush=True)

    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)
    val_pause_ds = PauseEventDataset(split="val", context_s=cfg.context_s, include_trajectory=True)

    train_combined = ConcatDataset([train_final, train_pause])
    train_loader = DataLoader(train_combined, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
                               persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate,
                             persistent_workers=cfg.num_workers > 0)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               persistent_workers=cfg.num_workers > 0)
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
        model.train()
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for batch in train_loader:
            log_mel = batch["log_mel"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            traj = batch["trajectory"].to(device)
            labels = batch["label"].to(device)
            teacher_logits = batch["teacher_logit"].to(device)
            is_pause = torch.tensor(batch["is_pause_event"], dtype=torch.bool, device=device)

            opt.zero_grad()
            logits = model(log_mel, mask, traj)
            loss = _distill_loss(logits, labels, teacher_logits, is_pause, cfg.T, cfg.alpha)
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
    report["train_set_size"] = {"final_clips": len(train_final), "hold_events": len(train_pause)}

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

    report["fcr_at_holds_distill"] = evaluate_fcr_at_holds(model, val_pause_loader, device, threshold)
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
