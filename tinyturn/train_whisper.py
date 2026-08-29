"""
Step 5 -- A0: corrected Whisper-Tiny baseline, full fine-tuning. Same context/masks/splits/metrics
as B0/B1 (Section 5; Phase-2 8b: speech-aligned, no baked-in post-roll); only the encoder +
necessarily-Whisper-shaped mel input differ. Kept as its own training loop (not
tinyturn.train.train_experiment) because its dataset
(WhisperTurnDataset), model signature, and mel convention (80-bin Whisper filterbank vs. our 40-bin
branch) are genuinely different -- but every other design choice (pooling module, calibration
protocol, evaluation slices, ONNX export, latency harness) is intentionally identical.
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

from tinyturn.whisper_dataset import WhisperTurnDataset, collate, worker_init_fn
from tinyturn.whisper_model import WhisperEndpointModel, WHISPER_MODEL_NAME
from tinyturn.evaluate import EvalOutputs, full_report, calibrate_threshold, count_macs


@dataclass
class WhisperExperimentConfig:
    exp_id: str = "A0"
    context_s: float = 4.0
    epochs: int = 2                               # hard ceiling on epochs run
    early_stop_patience: Optional[int] = None      # 8h-A0 (mirrors tinyturn.train.ExperimentConfig's
                                                    # 8h field for B1): stop if val_auc hasn't beaten
                                                    # its best in this many epochs. None = original
                                                    # fixed-epoch-count behavior (A0 as run in 8d).
    lr_schedule: Optional[str] = None              # 8h-A0: None (fixed lr, as run) | "plateau"
                                                    # (ReduceLROnPlateau on val_auc) -- a fine-tuning-
                                                    # appropriate schedule, not the inherited fixed lr.
    batch_size: int = 16
    lr: float = 1e-5
    num_workers: int = 8
    seed: int = 42
    model_name: str = WHISPER_MODEL_NAME
    augment_boundaries: bool = False   # 8g remediation: train-time boundary augmentation (canonical/
                                        # alt-threshold/Silero, label-independent) via
                                        # tinyturn.whisper_dataset's precomputed boundary cache.
                                        # Train split only -- calib/val below never set this, so
                                        # calibration and model-selection metrics stay on the
                                        # canonical boundary regardless.


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def run_inference_whisper(model, loader, device) -> EvalOutputs:
    model.eval()
    ids, y_true, y_prob = [], [], []
    language, dataset, synthetic, implicit_incomplete = [], [], [], []
    for batch in loader:
        feats = batch["input_features"].to(device)
        mask = batch["valid_frame_mask"].to(device)
        logits = model(feats, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(batch["label"].numpy().tolist())
        ids.extend(batch["id"])
        language.extend(batch["language"])
        dataset.extend(batch["dataset"])
        synthetic.extend(batch["synthetic"])
        implicit_incomplete.extend(batch["implicit_incomplete"])
    return EvalOutputs(ids=ids, y_true=np.array(y_true), y_prob=np.array(y_prob),
                        language=language, dataset=dataset, synthetic=synthetic,
                        implicit_incomplete=implicit_incomplete)


def train_whisper_experiment(cfg: WhisperExperimentConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)
    # Every prior run in this project was CPU-only (no CUDA available in that environment) --
    # auto-detect here so the exact same training code runs unmodified on a CUDA box (e.g. Colab)
    # without silently staying on CPU there.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg.exp_id}] device: {device}", flush=True)

    ds_kwargs = dict(context_s=cfg.context_s, model_name=cfg.model_name)
    train_ds = WhisperTurnDataset(split="train", augment_boundaries=cfg.augment_boundaries, **ds_kwargs)
    val_ds = WhisperTurnDataset(split="val", **ds_kwargs)
    calib_ds = WhisperTurnDataset(split="calib", **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
                               persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate,
                             persistent_workers=cfg.num_workers > 0)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate,
                               persistent_workers=cfg.num_workers > 0)

    model = WhisperEndpointModel(model_name=cfg.model_name).to(device)
    n_params = model.num_parameters()
    print(f"[{cfg.exp_id}] model params (full fine-tune): {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()
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
            feats = batch["input_features"].to(device)
            mask = batch["valid_frame_mask"].to(device)
            labels = batch["label"].to(device)

            opt.zero_grad()
            logits = model(feats, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"[{cfg.exp_id}]   epoch {epoch+1} batch {n_batches} "
                      f"running_loss={running_loss/n_batches:.4f} ({time.time()-t0:.1f}s)", flush=True)

        val_out = run_inference_whisper(model, val_loader, device)
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
        # Checkpoint selection by the promotion metric (val_auc), never by training loss.
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

    calib_out = run_inference_whisper(model, calib_loader, device)
    threshold = calibrate_threshold(calib_out.y_true, calib_out.y_prob, target_fcr=0.05)
    print(f"[{cfg.exp_id}] calibrated threshold (target FCR<=0.05 on calib): {threshold:.4f}", flush=True)

    val_out = run_inference_whisper(model, val_loader, device)
    report = full_report(val_out, threshold)
    report["n_parameters"] = n_params
    report["history"] = history
    report["best_epoch"] = best_epoch
    report["final_epoch"] = history[-1]["epoch"] if history else None
    report["best_val_auc"] = best_val_auc
    report["stopped_early"] = history[-1]["epoch"] < cfg.epochs if history else False

    # MACs counting and ONNX export/latency measurement below use CPU-default tensors (matching
    # this project's deployment target) -- move off CUDA first so a GPU-trained model doesn't hit a
    # device mismatch here (training/calibration/val above are already done with `device`).
    model = model.to("cpu")

    sample0 = train_ds[0]
    mac_inputs = (sample0["input_features"].unsqueeze(0), sample0["valid_frame_mask"].unsqueeze(0))
    report["macs"] = int(count_macs(model, *mac_inputs))

    try:
        report["onnx"] = _export_and_measure_latency(model, train_ds, out_dir, torch.device("cpu"))
    except Exception as e:
        report["onnx"] = {"error": str(e)}
        print(f"[{cfg.exp_id}] ONNX export/latency measurement failed: {e}", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{cfg.exp_id}] saved results to {out_dir}", flush=True)
    return report


def _export_and_measure_latency(model, train_ds, out_dir: Path, device):
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import onnxruntime as ort

    sample = train_ds[0]
    n_mels, n_frames = sample["input_features"].shape
    onnx_path = out_dir / "model.onnx"
    model.eval()
    dummy_feats = torch.randn(1, n_mels, n_frames)
    dummy_mask = torch.ones(1, n_frames, dtype=torch.bool)
    # dynamo=False: see tinyturn/onnx_export.py's export_onnx for why.
    torch.onnx.export(
        model, (dummy_feats, dummy_mask), str(onnx_path),
        input_names=["input_features", "valid_frame_mask"], output_names=["logit"],
        dynamic_axes={"input_features": {2: "time"}, "valid_frame_mask": {1: "time"},
                      "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx_size_bytes = Path(onnx_path).stat().st_size

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # batch-1 CPU latency INCLUDING feature extraction (raw windowed waveform -> Whisper log-mel
    # -> ONNXRuntime forward), matching B0/B1's harness -- not encoder-forward-only, so the
    # brief's explicit "don't repeat the external reference's ambiguous 8.71ms" instruction holds
    # for A0 too.
    row = train_ds.df.iloc[0]
    y, sr = train_ds._load_wav(row["id"])
    from tinyturn.preprocess import build_example
    from tinyturn.whisper_dataset import silence_invalid_samples
    ex = build_example(y, sr, float(row["last_active_t"]), train_ds.context_s,
                        label=bool(row["endpoint_bool"]), row_id=row["id"])
    safe_waveform = silence_invalid_samples(ex.waveform, ex.valid_sample_mask)
    valid_frame_mask_np = sample["valid_frame_mask"].numpy()[None, :]

    def _one_pass():
        t0 = time.perf_counter()
        feats = train_ds.feature_extractor(safe_waveform, sampling_rate=sr, padding=False,
                                            return_tensors="np")["input_features"]
        sess.run(["logit"], {"input_features": feats.astype(np.float32),
                              "valid_frame_mask": valid_frame_mask_np})
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(5):
        _one_pass()
    times = [_one_pass() for _ in range(50)]
    return {
        "onnx_size_bytes": onnx_size_bytes,
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "n_runs": 50,
    }
