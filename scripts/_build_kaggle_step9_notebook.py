"""One-off builder for step9_kaggle.ipynb -- inlines every tinyturn module Step 9 needs (no
`import tinyturn` anywhere in the notebook), embeds the small pause-events parquet as base64 (not
present in the already-uploaded tar), and reproduces run_step9_controlled_rerun.py's orchestration
logic (matched-threshold curves, extra-slice recalls) as notebook cells."""
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def code_cell(source: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def md_cell(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def strip_tinyturn_imports(src: str) -> str:
    """Drop every `from tinyturn.X import ...` / `import tinyturn.X` line -- everything is inlined
    into one notebook namespace already, in dependency order, so these would just re-trigger a real
    (nonexistent) package import."""
    out = []
    for line in src.splitlines(keepends=True):
        if re.match(r"^\s*(from|import)\s+tinyturn(\.\w+)?\s+(import|$)", line) or \
           re.match(r"^\s*from tinyturn\.\w+ import", line):
            continue
        out.append(line)
    return "".join(out)


def strip_module_level_constants(src: str, *snippets: str) -> str:
    for s in snippets:
        assert s in src, f"expected snippet not found:\n{s}"
        src = src.replace(s, "", 1)
    return src


preprocess_src = strip_tinyturn_imports(read("tinyturn/preprocess.py"))
boundary_src = strip_tinyturn_imports(read("tinyturn/boundary.py"))
pooling_src = strip_tinyturn_imports(read("tinyturn/pooling.py"))
features_src = strip_tinyturn_imports(read("tinyturn/features.py"))

dataset_src = strip_tinyturn_imports(read("tinyturn/dataset.py"))
dataset_src = strip_module_level_constants(
    dataset_src,
    'CACHE_DIR = Path("data_cache")\nWAV_DIR = CACHE_DIR / "d2_stratified_wavs"\nFEATURE_CACHE_DIR = CACHE_DIR / "tinyturn_feature_cache"\n',
)

_pause_events_raw = read("tinyturn/pause_events.py")
# pause_events.py imports several dataset.py constants via a multi-line `from tinyturn.dataset
# import (...)` -- strip that BEFORE strip_tinyturn_imports, which only matches single-line import
# statements and would otherwise remove just the opening `(` line, leaving the continuation lines
# dangling with a stray indent (caught by the syntax-check step below).
_pause_events_raw = re.sub(
    r"from tinyturn\.dataset import \([^)]*\)\n", "", _pause_events_raw, flags=re.S)
pause_events_src = strip_tinyturn_imports(_pause_events_raw)

models_src = strip_tinyturn_imports(read("tinyturn/models.py"))
onnx_export_src = strip_tinyturn_imports(read("tinyturn/onnx_export.py"))
evaluate_src = strip_tinyturn_imports(read("tinyturn/evaluate.py"))

train_src = strip_tinyturn_imports(read("tinyturn/train.py"))
train_p1_src = strip_tinyturn_imports(read("tinyturn/train_p1.py"))

step9_orchestration_src = read("scripts/run_step9_controlled_rerun.py")
# Pull out SHORT_COMPLETE_MAX_WORDS/RESPONSE_PARTICLES + every reusable function (skip the module
# docstring, imports, and the CACHE_DIR/BASELINE_DIR/OUT_DIR/PROTOCOL path constants + main() --
# paths and the run plan are redefined in later cells for the Kaggle layout).
m = re.search(r"\nSHORT_COMPLETE_MAX_WORDS.*?\ndef main\(\)", step9_orchestration_src, re.S)
assert m, "couldn't locate the constants+function block in run_step9_controlled_rerun.py"
step9_block_src = m.group(0).rsplit("\ndef main(", 1)[0]
# _load_model isn't needed (checkpoints are held as in-memory objects in this notebook, never
# reloaded from disk mid-session) -- drop ONLY that one function, keep the constants
# (SHORT_COMPLETE_MAX_WORDS/RESPONSE_PARTICLES) that come before it plus every function from
# _pause_probs_and_synth onward.
constants_part, _, rest = step9_block_src.partition("def _load_model")
funcs_part = "def _pause_probs_and_synth" + rest.split("def _pause_probs_and_synth", 1)[1]
# The regex above only captured this slice, not run_step9_controlled_rerun.py's top-level imports
# -- `re` (_has_response_particle) and matplotlib (plot_curves) are used but were never brought in;
# np/pd/torch are already in the notebook namespace by this point (imported by earlier inlined-
# module cells), so only these two need adding back explicitly.
step9_funcs_src = (
    'import re\nimport matplotlib\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n\n\n'
    + constants_part + "\n\n" + funcs_part
)

pause_events_parquet_b64 = base64.b64encode(
    (ROOT / "data_cache" / "tinyturn_pause_events.parquet").read_bytes()
).decode("ascii")

cells = []

cells.append(md_cell(
"""# Step 9 (Phase-3) on Kaggle GPU: controlled early-stopped rerun + three-arm pause-sampling comparison

Self-contained -- every `tinyturn` module Step 9 needs is inlined directly in this notebook's own
cells (no `import tinyturn`). Reads the same uploaded dataset as the 8h-A0 notebook
(`/kaggle/input/datasets/choudhary7511an1/wav-files`) for code/wavs/metadata, and additionally
embeds `tinyturn_pause_events.parquet` (598KB, not in that upload) directly as base64 in this
notebook -- byte-identical to the file used everywhere else in this project, not recomputed.

Trains 6 B1@1s models under one identical protocol (max 40 epochs, early-stop patience 6,
`ReduceLROnPlateau`, seed 42 -- the exact protocol 8h validated for the no-pause baseline):
baseline (no pause events, retrained fresh here rather than requiring another upload), plain P1,
P1a+P1b @ lambda=0.25, and the three-arm comparison at lambda=0.5 (all / real-only / 50:50).

Also confirms the real-clips-with-pause-events count directly against the manifest, verifies
checkpoint selection uses final-clip val_auc (not hold performance), and reports short-complete
recall + response-particle complete recall for every arm (Section 3's requirement -- pause training
pushing toward conservatism could overcorrect on short, legitimately-complete replies).

**Not included** (deferred, per the brief's own sequencing): lambda=0.75 (only after the converged
lambda=0.5 result is in hand) and the 3-seed promotion of finalists (after this rerun determines
which 1-2 configs to promote).

**Before running:** Settings -> Accelerator -> GPU T4 x2 (P100 needs a torch-version workaround for
this Kaggle image's default PyTorch build -- T4x2 works out of the box). Internet ON."""
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
'''import os, tarfile, base64, io
from pathlib import Path

KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/choudhary7511an1/wav-files")
EXTRACT_DIR = Path("/kaggle/tmp/pipecat_data")
OUT_DIR = Path("/kaggle/working/experiments")
CACHE_DIR = EXTRACT_DIR / "data_cache"     # always created as a REAL, writable directory below --
WAV_DIR = CACHE_DIR / "d2_stratified_wavs" # Step 9 (unlike the 8h-A0 notebook) needs to WRITE into
FEATURE_CACHE_DIR = CACHE_DIR / "tinyturn_feature_cache"  # this tree (feature cache, pause-events
                                                           # parquet), so it can never just be the
                                                           # read-only /kaggle/input mount itself.

assert KAGGLE_INPUT_DIR.exists(), f"not found: {KAGGLE_INPUT_DIR} -- check your dataset is attached"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not WAV_DIR.exists():
    tar_candidates = list(KAGGLE_INPUT_DIR.glob("*.tar")) + list(KAGGLE_INPUT_DIR.rglob("*.tar"))
    if tar_candidates:
        tar_path = tar_candidates[0]
        print(f"extracting {tar_path} -> {EXTRACT_DIR} (one-time, ~3.7GB, may take a few minutes)")
        with tarfile.open(tar_path) as tf:
            tf.extractall(EXTRACT_DIR)
    else:
        # Kaggle auto-extracted the uploaded tar into the dataset itself (no .tar file visible) --
        # its `data_cache/` is on the READ-ONLY /kaggle/input mount, so symlink its large,
        # never-written-to pieces (wavs + existing parquets) into our own writable CACHE_DIR
        # instead of using that mount directly (which would fail the very next line: creating
        # FEATURE_CACHE_DIR, or later, writing the pause-events parquet).
        src_cache = KAGGLE_INPUT_DIR / "data_cache"
        print(f"no .tar found -- symlinking {src_cache}'s contents into writable {CACHE_DIR}")
        assert src_cache.exists(), f"not found either: {src_cache}"
        for entry in src_cache.iterdir():
            link = CACHE_DIR / entry.name
            if not link.exists():
                os.symlink(entry, link)
else:
    print(f"{WAV_DIR} already populated, skipping extraction")

FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("CACHE_DIR:", CACHE_DIR)
print("wav files:", len(list(WAV_DIR.glob("*.wav"))))
OUT_DIR.mkdir(parents=True, exist_ok=True)
'''))

cells.append(code_cell(
f'''# tinyturn_pause_events.parquet -- not in the uploaded tar (that package was built for the 8h-A0
# notebook only). Embedded here as base64, byte-identical to the file used everywhere else in this
# project (safer than recomputing pause detection fresh, which risks numeric drift).
_PAUSE_EVENTS_PARQUET_B64 = """{pause_events_parquet_b64}"""

pause_events_path = CACHE_DIR / "tinyturn_pause_events.parquet"
if not pause_events_path.exists():
    pause_events_path.write_bytes(base64.b64decode(_PAUSE_EVENTS_PARQUET_B64))
    print(f"wrote {{pause_events_path}} ({{pause_events_path.stat().st_size}} bytes)")
else:
    print(f"{{pause_events_path}} already present, skipping")
'''))

cells.append(code_cell(
"""!pip install -q librosa soundfile scikit-learn pandas pyarrow scipy onnx onnxruntime matplotlib
"""))

cells.append(md_cell(
"""## Inlined `tinyturn` code

Run these in order, top to bottom, exactly once -- each reproduces one module's current source
unmodified (only cross-module `from tinyturn.X import ...` lines are stripped)."""
))

cells.append(code_cell(preprocess_src))
cells.append(code_cell(boundary_src))
cells.append(code_cell(pooling_src))
cells.append(code_cell(features_src))
cells.append(code_cell(dataset_src))
cells.append(code_cell(pause_events_src))
cells.append(code_cell(models_src))
cells.append(code_cell(onnx_export_src))
cells.append(code_cell(evaluate_src))
cells.append(code_cell(train_src))
cells.append(code_cell(train_p1_src))

cells.append(md_cell(
"""## Step 9 orchestration (matched-threshold curves, extra-slice recalls)

Reproduces `scripts/run_step9_controlled_rerun.py`'s reporting logic unmodified."""
))
cells.append(code_cell(step9_funcs_src))

cells.append(md_cell(
"""## Confirm the real pause-event pool, then train all 6 arms

Same protocol for every arm (only pause-sampling differs) -- max 40 epochs, early-stop patience 6,
`ReduceLROnPlateau`, batch_size=64, num_workers=4 (Kaggle's Linux containers don't have the Windows
multiprocessing-spawn / antivirus-file-scan constraints the original CPU runs fought)."""
))

cells.append(code_cell(
'''pool_note = confirm_real_pause_pool()

PROTOCOL = dict(epochs=40, early_stop_patience=6, lr_schedule="plateau",
                batch_size=64, num_workers=4, seed=42)
CONTEXT_S = 1.0
TARGET_RECALLS = [0.90, 0.95]  # used by the comparison cell below (at_matched_recall_v2) -- was
                                # missing here (a real bug, found live on Kaggle): the regex that
                                # extracts reusable functions from run_step9_controlled_rerun.py
                                # starts capturing after this constant is defined in that script.
RUN_DIRS = {}

print("\\n=== training baseline (no pause events) ===")
baseline_dir = OUT_DIR / "baseline_kaggle"
RUN_DIRS["baseline (no pause events)"] = baseline_dir
if not (baseline_dir / "checkpoint.pt").exists():
    baseline_cfg = ExperimentConfig(exp_id="baseline_kaggle", context_s=CONTEXT_S, use_trajectory=True,
                                     **PROTOCOL)
    train_experiment(baseline_cfg, baseline_dir)
else:
    print(f"{baseline_dir} already exists, skipping")
'''))

cells.append(code_cell(
'''runs = {
    "P1_plain": dict(lambda_hold=None, controlled_sampling=False, real_synth_balance="proportional"),
    "P1ab_lambda0.25": dict(lambda_hold=0.25, controlled_sampling=True, real_synth_balance="proportional"),
    "P1ab_lambda0.5_all": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="proportional"),
    "P1ab_lambda0.5_real_only": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="real_only"),
    "P1ab_lambda0.5_5050": dict(lambda_hold=0.5, controlled_sampling=True, real_synth_balance="50:50"),
}
for tag, overrides in runs.items():
    d = OUT_DIR / tag
    RUN_DIRS[tag] = d
    if (d / "checkpoint.pt").exists():
        print(f"{d} already exists, skipping retrain")
        continue
    cfg = P1Config(exp_id=tag, context_s=CONTEXT_S, **PROTOCOL, **overrides)
    print(f"\\n=== training {tag}: {overrides} ===")
    train_p1(cfg, baseline_dir / "checkpoint.pt", d)
'''))

cells.append(md_cell(
"""## Build matched-threshold curves + extra-slice recalls for every arm"""
))

cells.append(code_cell(
'''ds_kwargs = dict(context_s=CONTEXT_S, include_trajectory=True)
val_ds = TinyTurnDataset(split="val", **ds_kwargs)
val_pause_ds = PauseEventDataset(split="val", context_s=CONTEXT_S, include_trajectory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
pause_loader = DataLoader(val_pause_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
trans_df = pd.read_parquet(CACHE_DIR / "d2_stratified_transcripts.parquet")[["id", "text", "n_words"]]

device_cpu = torch.device("cpu")
curves, matched, extra_slices, run_metrics = {}, {}, {}, {}
for name, d in RUN_DIRS.items():
    model, cfg = _load_model(d)
    metrics = json.load(open(d / "metrics.json"))
    threshold = float(metrics["threshold"])
    curve = matched_threshold_curve_v2(model, val_loader, pause_loader, device_cpu)
    curves[name] = curve
    matched[name] = {f"recall_{int(r*100)}": at_matched_recall_v2(curve, r) for r in TARGET_RECALLS}
    extra_slices[name] = extra_slice_recalls(model, val_ds, device_cpu, threshold, trans_df)
    run_metrics[name] = {
        "best_epoch": metrics.get("best_epoch"), "final_epoch": metrics.get("final_epoch"),
        "stopped_early": metrics.get("stopped_early"), "best_val_auc": metrics.get("best_val_auc"),
        "overall_auc": metrics.get("overall", {}).get("auc"),
        "real_auc": metrics.get("real_all", {}).get("auc"),
        "threshold": threshold, "lambda_hold": cfg.get("lambda_hold"),
        "controlled_sampling": cfg.get("controlled_sampling"),
        "real_synth_balance": cfg.get("real_synth_balance"),
    }
    print(f"\\n{name}: best_epoch={metrics.get('best_epoch')} final_epoch={metrics.get('final_epoch')} "
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
print(f"\\nsaved plot to {plot_path}")

out = {
    "protocol": PROTOCOL, "real_pause_pool_confirmation": pool_note,
    "run_metrics": run_metrics, "matched_recall_summary": matched,
    "extra_slice_recalls": extra_slices, "short_complete_max_words": SHORT_COMPLETE_MAX_WORDS,
    "response_particle_lexicon": sorted(RESPONSE_PARTICLES), "curves": curves,
}
with open(OUT_DIR / "step9_kaggle_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\\nsaved {OUT_DIR / 'step9_kaggle_results.json'}")
print("\\nEverything under /kaggle/working/experiments persists when you Save Version.")
'''))


def _load_model_src():
    return '''def _load_model(ckpt_dir):
    cfg = json.load(open(ckpt_dir / "config.json"))
    model = TinyTurnModel(n_mels=40, trajectory_dim=len(TRAJECTORY_NAMES),
                           mel_channels=cfg.get("mel_channels", 112), traj_channels=cfg.get("traj_channels", 24))
    model.load_state_dict(torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model, cfg
'''


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

# Insert _load_model back into the orchestration cell's source (it's needed by the comparison
# cell below, and dropping it earlier was a mistake corrected here rather than left in place).
# Matched by substring, not startswith -- the cell no longer starts with this text once the
# re/matplotlib imports were prepended above, which silently broke this exact insertion earlier.
_patched = False
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if cell["cell_type"] == "code" and "def _pause_probs_and_synth" in src and "def _load_model" not in src:
        cell["source"] = (_load_model_src() + "\n\n" + src).splitlines(keepends=True)
        _patched = True
        break
assert _patched, "failed to insert _load_model into the orchestration cell -- check the match condition"

out_path = ROOT / "step9_kaggle.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out_path}, {len(cells)} cells")
