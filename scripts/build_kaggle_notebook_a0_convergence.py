"""One-off builder for A0_8h_convergence_kaggle.ipynb -- assembles the notebook programmatically
(safer than hand-writing large embedded code as JSON text) by reading the current source of the
tinyturn modules that need to be inlined, then emitting a self-contained notebook with no external
script/module dependencies -- everything needed lives in the notebook's own cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def code_cell(source: str, metadata=None):
    lines = source.splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": metadata or {},
            "outputs": [], "source": lines}


def md_cell(source: str):
    lines = source.splitlines(keepends=True)
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def strip_internal_imports(src: str, drop_names) -> str:
    """Remove `from tinyturn.X import ...` lines for modules we're inlining in earlier cells of the
    SAME notebook (so we don't re-trigger a real import of a `tinyturn` package that doesn't exist
    on Kaggle) -- keeps every other import (torch, numpy, transformers, sklearn, ...) untouched."""
    out = []
    for line in src.splitlines(keepends=True):
        if any(f"from tinyturn.{name}" in line or f"import tinyturn.{name}" in line for name in drop_names):
            continue
        out.append(line)
    return "".join(out)


preprocess_src = read("tinyturn/preprocess.py")
pooling_src = read("tinyturn/pooling.py")
whisper_model_src = strip_internal_imports(read("tinyturn/whisper_model.py"), ["pooling"])
whisper_dataset_src = strip_internal_imports(
    read("tinyturn/whisper_dataset.py"), ["preprocess", "whisper_model"])
evaluate_src = read("tinyturn/evaluate.py")
train_whisper_src = strip_internal_imports(
    read("tinyturn/train_whisper.py"), ["whisper_dataset", "whisper_model", "evaluate", "preprocess"])

# whisper_dataset.py hardcodes its own CACHE_DIR/WAV_DIR/FRAME_LENGTH_S/HOP_LENGTH_S at module
# level -- drop that block since the "paths" cell earlier in the notebook already defines
# CACHE_DIR/WAV_DIR for the Kaggle layout, and FRAME_LENGTH_S/HOP_LENGTH_S are defined identically
# in the preprocess.py cell's neighborhood already (re-declared below instead).
whisper_dataset_src = whisper_dataset_src.replace(
    'CACHE_DIR = Path("data_cache")\nWAV_DIR = CACHE_DIR / "d2_stratified_wavs"\nFRAME_LENGTH_S = 0.025\nHOP_LENGTH_S = 0.010\n',
    "# CACHE_DIR / WAV_DIR / FRAME_LENGTH_S / HOP_LENGTH_S come from the earlier 'paths' cell.\n",
)
train_whisper_src = train_whisper_src.replace(
    "from typing import Optional\n", "from typing import Optional\n", 1)

cells = []

cells.append(md_cell(
"""# 8h-A0 on Kaggle GPU: A0@2s vs. A0@4s early-stopped retrain (standalone)

Self-contained -- every `tinyturn` module needed is inlined directly in this notebook's own cells
(no `import tinyturn`, no external script files). Reads data from your uploaded dataset
(`/kaggle/input/datasets/choudhary7511an1/wav-files`, the same tar package used for the Colab
version: `tinyturn/`, `scripts/`, `data_cache/` with the 3 metadata parquets + 15,998 wav
files, and an `experiments/` dir this notebook does NOT use), extracts it to local (fast) disk, then
trains A0@4s and A0@2s from scratch with early stopping (max 10 epochs, patience 3,
`ReduceLROnPlateau`) and compares them to each other.

**Standalone comparison, by design:** this does not compare against A0's original fixed-2-epoch
run (that comparison was already done from log data directly, outside this notebook) or report
internal-hold FCR (the pause-events parquet isn't in this dataset upload) -- just the two fresh
context-length variants against each other on the full Section-8 slice report, calibration, and
latency.

**Before running:** Settings (right panel) -> Accelerator -> GPU T4 x2 (or better). Internet must be
ON (Settings -> Internet) for the `pip install` cell and the Whisper-tiny pretrained weights download."""
))

cells.append(code_cell(
"""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected -- Settings -> Accelerator -> GPU, then re-run this cell.")
"""))

cells.append(code_cell(
'''import os, tarfile
from pathlib import Path

KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/choudhary7511an1/wav-files")
EXTRACT_DIR = Path("/kaggle/tmp/pipecat_data")  # outside /kaggle/working on purpose -- not part of
                                                 # the output snapshot when you Save Version, and
                                                 # local (fast) disk rather than the input mount.
OUT_DIR = Path("/kaggle/working/experiments")   # small (checkpoints + json), persisted as output.

assert KAGGLE_INPUT_DIR.exists(), f"not found: {KAGGLE_INPUT_DIR} -- check your dataset is attached"

if not (EXTRACT_DIR / "data_cache" / "d2_stratified_wavs").exists():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    tar_candidates = list(KAGGLE_INPUT_DIR.glob("*.tar")) + list(KAGGLE_INPUT_DIR.rglob("*.tar"))
    if tar_candidates:
        tar_path = tar_candidates[0]
        print(f"extracting {tar_path} -> {EXTRACT_DIR} (one-time, ~3.7GB, may take a few minutes)")
        with tarfile.open(tar_path) as tf:
            tf.extractall(EXTRACT_DIR)
    else:
        # Dataset was uploaded already-extracted (Kaggle sometimes auto-extracts zips on upload) --
        # use the input mount's contents directly instead of copying/extracting anything.
        EXTRACT_DIR = KAGGLE_INPUT_DIR
        print(f"no .tar found under {KAGGLE_INPUT_DIR} -- assuming already extracted, using it directly")
else:
    print(f"{EXTRACT_DIR} already populated, skipping extraction")

CACHE_DIR = EXTRACT_DIR / "data_cache"
WAV_DIR = CACHE_DIR / "d2_stratified_wavs"
print("CACHE_DIR:", CACHE_DIR)
print("wav files:", len(list(WAV_DIR.glob("*.wav"))))
OUT_DIR.mkdir(parents=True, exist_ok=True)
'''))

cells.append(code_cell(
"""!pip install -q transformers librosa soundfile scikit-learn pandas pyarrow onnx onnxruntime
"""))

cells.append(md_cell(
"""## Inlined `tinyturn` code

Every cell below reproduces one module's current source unmodified (only cross-module
`from tinyturn.X import ...` lines are stripped, since everything now lives in one notebook
namespace) -- run them in order, top to bottom, exactly once."""
))

cells.append(code_cell(
'''FRAME_LENGTH_S = 0.025
HOP_LENGTH_S = 0.010

''' + preprocess_src
))

cells.append(code_cell(pooling_src))

cells.append(code_cell(whisper_model_src))

cells.append(code_cell(whisper_dataset_src))

cells.append(code_cell(evaluate_src))

cells.append(code_cell(train_whisper_src))

cells.append(md_cell(
"""## Train A0@4s and A0@2s

Same protocol for both (only `context_s` differs) -- max 10 epochs, early-stop patience 3,
`ReduceLROnPlateau`, batch_size=32 (raised from the original CPU run's 8 -- a T4/P100 has plenty of
headroom for Whisper-tiny full fine-tuning; adjust if you hit an out-of-memory error), lr=1e-5,
num_workers=4 (Kaggle's Linux containers don't have the Windows multiprocessing-spawn constraints
the original CPU runs were written around, so this is safe here)."""
))

cells.append(code_cell(
'''LONGRUN_4S_DIR = OUT_DIR / "whisper_tiny_4s_earlystopped_longrun"
LONGRUN_2S_DIR = OUT_DIR / "A0_2s_kaggle_longrun"
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
BATCH_SIZE = 32
NUM_WORKERS = 4

cfg_4s = WhisperExperimentConfig(
    exp_id="whisper_tiny_4s_earlystopped_longrun", context_s=4.0, epochs=MAX_EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE, lr_schedule="plateau",
    batch_size=BATCH_SIZE, lr=1e-5, num_workers=NUM_WORKERS, seed=42,
)
report_4s = train_whisper_experiment(cfg_4s, LONGRUN_4S_DIR)
'''))

cells.append(code_cell(
'''cfg_2s = WhisperExperimentConfig(
    exp_id="A0_2s_kaggle_longrun", context_s=2.0, epochs=MAX_EPOCHS,
    early_stop_patience=EARLY_STOP_PATIENCE, lr_schedule="plateau",
    batch_size=BATCH_SIZE, lr=1e-5, num_workers=NUM_WORKERS, seed=42,
)
report_2s = train_whisper_experiment(cfg_2s, LONGRUN_2S_DIR)
'''))

cells.append(md_cell(
"""## Compare A0@4s vs. A0@2s

Standalone comparison -- FCR at fixed complete recall, `implicit_incomplete` FCR, real-audio FCR,
latency, and calibration, not AUC alone (per the brief: "the 2-second model may be the better
teacher if its operating-point performance is close, given its latency advantage"). No internal-hold
FCR here (pause-events data isn't in this upload) and no comparison against the old fixed-2-epoch
run (already established separately)."""
))

cells.append(code_cell(
'''import json

def summarize(name, report):
    return {
        "name": name, "best_epoch": report.get("best_epoch"), "final_epoch": report.get("final_epoch"),
        "stopped_early": report.get("stopped_early"), "best_val_auc": report.get("best_val_auc"),
        "overall_auc": report["overall"]["auc"], "overall_fcr_at_recall95": report["overall"].get("fcr_at_recall95"),
        "real_auc": report["real_all"]["auc"], "real_fcr_at_recall95": report["real_all"].get("fcr_at_recall95"),
        "implicit_incomplete_fcr": report["implicit_incomplete"]["fcr"],
        "calibration": report["calibration"], "threshold": report["threshold"],
        "latency_p50_ms": report.get("onnx", {}).get("p50_ms"),
        "latency_p95_ms": report.get("onnx", {}).get("p95_ms"),
        "n_parameters": report.get("n_parameters"), "macs": report.get("macs"),
    }

summary_4s = summarize("A0@4s", report_4s)
summary_2s = summarize("A0@2s", report_2s)
for s in (summary_4s, summary_2s):
    print(f"\\n{s['name']}: best_epoch={s['best_epoch']} final_epoch={s['final_epoch']} "
          f"stopped_early={s['stopped_early']} best_val_auc={s['best_val_auc']}")
    print(f"  overall_auc={s['overall_auc']} overall_fcr_at_recall95={s['overall_fcr_at_recall95']} "
          f"real_auc={s['real_auc']} real_fcr_at_recall95={s['real_fcr_at_recall95']} "
          f"implicit_incomplete_fcr={s['implicit_incomplete_fcr']}")
    print(f"  latency p50/p95: {s['latency_p50_ms']}/{s['latency_p95_ms']} ms  "
          f"n_parameters={s['n_parameters']} macs={s['macs']}")

out = {"a0_4s": summary_4s, "a0_2s": summary_2s}
with open(OUT_DIR / "8h_a0_kaggle_comparison.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\\nsaved", OUT_DIR / "8h_a0_kaggle_comparison.json")
print("\\nEverything under /kaggle/working/experiments persists when you Save Version.")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = ROOT / "A0_8h_convergence_kaggle.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out_path}, {len(cells)} cells")
