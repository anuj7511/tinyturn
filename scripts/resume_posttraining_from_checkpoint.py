"""One-off recovery script: B0's training loop completed all 5 epochs (val_auc up to 0.8119) and
saved checkpoint.pt, but the run crashed during the post-training calibration/eval/export phase
because the codebase was edited (added `use_disk_cache` to TinyTurnDataset) while the background
training process was still alive -- Windows' spawn-based DataLoader workers re-import the module
fresh, so newly-spawned calib/val workers picked up the edited class shape while the pickled
dataset instance (built in the older, already-running main process) didn't have the new attribute.
No more concurrent edits during a live run going forward. This just re-runs the post-training half
using the already-trained checkpoint, on the now-stable code.

Kept import/call-signature-compatible with Phase-2 8b's dropped `postroll_s` parameter, but the
checkpoints this script loads (B0/B1) were trained under the old N+200ms-post-roll contract and are
stale per Phase-2 8d -- re-running this reproduces the historical (pre-8b) numbers, not a
post-8b-contract result. Don't cite its output as a current result.
"""
import sys
import json
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from tinyturn.dataset import TinyTurnDataset, collate
from tinyturn.models import TinyTurnModel
from tinyturn.evaluate import run_inference, full_report, calibrate_threshold, count_macs
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes
from tinyturn.train import ExperimentConfig, TRAJECTORY_NAMES, TRAJECTORY_NAMES_F0


def finish(cfg: ExperimentConfig, out_dir: Path):
    device = torch.device("cpu")
    traj_names = TRAJECTORY_NAMES_F0 if cfg.use_f0 else TRAJECTORY_NAMES
    trajectory_dim = len(traj_names) if cfg.use_trajectory else 0

    ds_kwargs = dict(context_s=cfg.context_s,
                      include_trajectory=cfg.use_trajectory, include_f0=cfg.use_f0)
    train_ds = TinyTurnDataset(split="train", **ds_kwargs)
    val_ds = TinyTurnDataset(split="val", **ds_kwargs)
    calib_ds = TinyTurnDataset(split="calib", **ds_kwargs)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=collate)
    calib_loader = DataLoader(calib_ds, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=collate)

    model = TinyTurnModel(n_mels=40, trajectory_dim=trajectory_dim,
                           mel_channels=cfg.mel_channels, traj_channels=cfg.traj_channels).to(device)
    state = torch.load(out_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(state)
    n_params = model.num_parameters()
    print(f"[{cfg.exp_id}] loaded checkpoint, params={n_params}", flush=True)

    calib_out = run_inference(model, calib_loader, device, cfg.use_trajectory)
    threshold = calibrate_threshold(calib_out.y_true, calib_out.y_prob, target_fcr=0.05)
    print(f"[{cfg.exp_id}] calibrated threshold (target FCR<=0.05 on calib): {threshold:.4f}", flush=True)

    val_out = run_inference(model, val_loader, device, cfg.use_trajectory)
    report = full_report(val_out, threshold)
    report["n_parameters"] = n_params

    n_frames = train_ds[0]["log_mel"].shape[0]
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, onnx_path, n_frames=n_frames, trajectory_dim=trajectory_dim)
    report["onnx_size_bytes"] = onnx_file_size_bytes(onnx_path)

    mac_inputs = (torch.randn(1, n_frames, 40), torch.ones(1, n_frames, dtype=torch.bool))
    mac_inputs = mac_inputs + ((torch.randn(1, n_frames, trajectory_dim),) if cfg.use_trajectory else (None,))
    report["macs"] = int(count_macs(model, *mac_inputs))

    sample_row = val_ds.df.iloc[0]
    y_raw, sr = val_ds._load_wav(sample_row["id"])
    from tinyturn.preprocess import build_example
    ex = build_example(y_raw, sr, float(sample_row["last_active_t"]), cfg.context_s,
                        label=bool(sample_row["endpoint_bool"]), row_id=sample_row["id"])
    latency = measure_latency_ms(onnx_path, ex.waveform, sr, ex.valid_sample_mask,
                                  cfg.use_trajectory, traj_names)
    report["latency"] = latency
    print(f"[{cfg.exp_id}] latency p50={latency['p50_ms']}ms p95={latency['p95_ms']}ms", flush=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[{cfg.exp_id}] saved results to {out_dir}", flush=True)
    return report


EXPERIMENTS = {
    "B0": dict(use_trajectory=False, use_f0=False, dir_name="mel_only_baseline"),
    "B1": dict(use_trajectory=True, use_f0=False, dir_name="mel_trajectory_baseline"),
    "B1-f0": dict(use_trajectory=True, use_f0=True, dir_name="mel_trajectory_with_f0"),
}

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Re-run calibration/eval/export/report from a saved "
                                             "checkpoint (e.g. to regenerate metrics.json with an "
                                             "updated evaluate.py, without retraining).")
    p.add_argument("experiment", choices=list(EXPERIMENTS.keys()))
    p.add_argument("--context-s", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    spec = EXPERIMENTS[args.experiment]
    cfg = ExperimentConfig(exp_id=args.experiment, context_s=args.context_s,
                            use_trajectory=spec["use_trajectory"], use_f0=spec["use_f0"],
                            epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)
    finish(cfg, Path("experiments") / spec["dir_name"])
