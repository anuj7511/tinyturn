"""
Step 10 planning, item 8 -- ONNX + INT8 export of the frozen checkpoint. Independent of the official
test evaluation (uses only the frozen checkpoint + one sample input for tracing/latency), so this can
run before, during, or after the test-set pass without affecting the "touched once" discipline there.

Frozen checkpoint: experiments/B1_1s_64k_lambda0.5_5050_seed43/checkpoint.pt
  (sha256 ddaf7a8ea95b6675022920b68b95e7a1f8202ab403c3e7e11e08dc5f0892694f)

Produces:
  experiments/B1_1s_64k_lambda0.5_5050_seed43/model.onnx        -- FP32 ONNX export
  experiments/B1_1s_64k_lambda0.5_5050_seed43/model_int8.onnx   -- INT8 dynamic-quantized (weights
                                                                    only; no calibration data needed
                                                                    for dynamic quantization, so this
                                                                    doesn't touch val/calib/test)
  experiments/B1_1s_64k_lambda0.5_5050_seed43/export_manifest.json -- sizes, latency (FP32 vs INT8),
                                                                       MACs, PyTorch-vs-ONNX parity
                                                                       check on a real val clip.

Usage:
  python scripts/export_frozen_checkpoint_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tinyturn.models import TinyTurnModel
from tinyturn.onnx_export import export_onnx, measure_latency_ms, onnx_file_size_bytes
from tinyturn.evaluate import count_macs
from tinyturn.dataset import TinyTurnDataset, N_MELS
from tinyturn.train import TRAJECTORY_NAMES
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

CKPT_DIR = Path("experiments") / "B1_1s_64k_lambda0.5_5050_seed43"
ONNX_PATH = CKPT_DIR / "model.onnx"
ONNX_INT8_PATH = CKPT_DIR / "model_int8.onnx"
MANIFEST_PATH = CKPT_DIR / "export_manifest.json"


def main():
    cfg = json.load(open(CKPT_DIR / "config.json"))
    model = TinyTurnModel(n_mels=N_MELS, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg["mel_channels"], traj_channels=cfg["traj_channels"])
    model.load_state_dict(torch.load(CKPT_DIR / "checkpoint.pt", map_location="cpu"))
    model.eval()

    val_ds = TinyTurnDataset(split="val", context_s=cfg["context_s"], include_trajectory=True)
    sample = val_ds[0]
    n_frames = sample["log_mel"].shape[0]
    print(f"n_frames={n_frames}, mel_channels={cfg['mel_channels']}, traj_channels={cfg['traj_channels']}", flush=True)

    mac_inputs = (torch.randn(1, n_frames, N_MELS), torch.ones(1, n_frames, dtype=torch.bool),
                  torch.randn(1, n_frames, len(TRAJECTORY_NAMES)))
    macs = int(count_macs(model, *mac_inputs))
    print(f"MACs: {macs:,}", flush=True)

    print("exporting FP32 ONNX...", flush=True)
    export_onnx(model, ONNX_PATH, n_frames=n_frames, trajectory_dim=len(TRAJECTORY_NAMES))
    fp32_size = onnx_file_size_bytes(ONNX_PATH)
    print(f"FP32 ONNX saved: {ONNX_PATH} ({fp32_size:,} bytes)", flush=True)

    print("quantizing to INT8 (dynamic, weights-only -- no calibration data needed)...", flush=True)
    quantize_dynamic(str(ONNX_PATH), str(ONNX_INT8_PATH), weight_type=QuantType.QInt8)
    int8_size = onnx_file_size_bytes(ONNX_INT8_PATH)
    print(f"INT8 ONNX saved: {ONNX_INT8_PATH} ({int8_size:,} bytes)", flush=True)

    # Parity check: FP32 PyTorch vs FP32 ONNX vs INT8 ONNX on the same real val clip.
    with torch.no_grad():
        torch_logit = model(sample["log_mel"].unsqueeze(0), sample["valid_frame_mask"].unsqueeze(0),
                             sample["trajectory"].unsqueeze(0)).item()

    def _onnx_logit(path):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {
            "log_mel": sample["log_mel"].unsqueeze(0).numpy(),
            "valid_frame_mask": sample["valid_frame_mask"].unsqueeze(0).numpy(),
            "trajectory": sample["trajectory"].unsqueeze(0).numpy(),
        }
        return float(sess.run(["logit"], feed)[0].squeeze())

    onnx_fp32_logit = _onnx_logit(ONNX_PATH)
    onnx_int8_logit = _onnx_logit(ONNX_INT8_PATH)

    print("measuring latency (FP32 vs INT8, includes feature extraction)...", flush=True)
    y_raw, sr = val_ds._load_wav(val_ds.df.iloc[0]["id"])
    from tinyturn.preprocess import build_example
    row0 = val_ds.df.iloc[0]
    ex = build_example(y_raw, sr, float(row0["last_active_t"]), cfg["context_s"],
                        label=bool(row0["endpoint_bool"]), row_id=row0["id"])
    latency_fp32 = measure_latency_ms(ONNX_PATH, ex.waveform, sr, ex.valid_sample_mask, True, TRAJECTORY_NAMES)
    latency_int8 = measure_latency_ms(ONNX_INT8_PATH, ex.waveform, sr, ex.valid_sample_mask, True, TRAJECTORY_NAMES)

    manifest = {
        "checkpoint": str(CKPT_DIR / "checkpoint.pt"),
        "n_parameters": model.num_parameters(),
        "macs": macs,
        "fp32_onnx_size_bytes": fp32_size,
        "int8_onnx_size_bytes": int8_size,
        "size_reduction_pct": round(100 * (1 - int8_size / fp32_size), 2),
        "parity_check_val_clip0": {
            "torch_fp32_logit": round(torch_logit, 6),
            "onnx_fp32_logit": round(onnx_fp32_logit, 6),
            "onnx_int8_logit": round(onnx_int8_logit, 6),
            "torch_vs_onnx_fp32_abs_diff": round(abs(torch_logit - onnx_fp32_logit), 6),
            "torch_vs_onnx_int8_abs_diff": round(abs(torch_logit - onnx_int8_logit), 6),
        },
        "latency_fp32_ms": latency_fp32,
        "latency_int8_ms": latency_int8,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nsaved {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
