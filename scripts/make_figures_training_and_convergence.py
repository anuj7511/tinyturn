"""Generates the figures for planning/PHASE2_RESULTS_8a-9.md: 8d training curves, 8h convergence
comparison, 8f CI forest plot, Step 9 training curves (undertraining check), Step 9 AUC comparison.
Reads only already-produced experiment metrics.json / CI-analysis JSON -- no retraining."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXP = Path("experiments")
FIG_DIR = Path("planning") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load(p):
    return json.load(open(p))


def fig_8d_training_curves():
    a0 = load(EXP / "whisper_tiny_speech_aligned_contract" / "metrics.json")["history"]
    b1 = load(EXP / "mel_trajectory_1s_speech_aligned_contract" / "metrics.json")["history"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row, (name, hist) in enumerate([("A0 (context_s=4.0)", a0), ("B1@1s (context_s=1.0)", b1)]):
        epochs = [h["epoch"] for h in hist]
        loss = [h["train_loss"] for h in hist]
        auc = [h["val_auc"] for h in hist]
        axes[row, 0].plot(epochs, loss, marker="o", color="tab:red")
        axes[row, 0].set_title(f"{name}: train loss")
        axes[row, 0].set_xlabel("epoch"); axes[row, 0].set_ylabel("loss")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].plot(epochs, auc, marker="o", color="tab:blue")
        best_idx = int(np.argmax(auc))
        axes[row, 1].scatter([epochs[best_idx]], [auc[best_idx]], color="black", zorder=5,
                              label=f"best (epoch {epochs[best_idx]})")
        axes[row, 1].set_title(f"{name}: val AUC")
        axes[row, 1].set_xlabel("epoch"); axes[row, 1].set_ylabel("val AUC")
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(alpha=0.3)
    fig.suptitle("8d: A0 / B1@1s retrain curves (fixed epoch budget)")
    fig.tight_layout()
    out = FIG_DIR / "8d_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_8h_convergence():
    orig = load(EXP / "mel_trajectory_1s_speech_aligned_contract" / "metrics.json")["history"]
    long = load(EXP / "mel_trajectory_1s_earlystopped_longrun" / "metrics.json")["history"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    e1 = [h["epoch"] for h in orig]; a1 = [h["val_auc"] for h in orig]
    e2 = [h["epoch"] for h in long]; a2 = [h["val_auc"] for h in long]
    ax.plot(e1, a1, marker="o", label="original: fixed 5 epochs (inherited from A0's protocol)")
    ax.plot(e2, a2, marker="o", label="longer: early-stop patience=6, ReduceLROnPlateau, cap 40")
    b1 = int(np.argmax(a1)); b2 = int(np.argmax(a2))
    ax.scatter([e1[b1]], [a1[b1]], color="black", zorder=5)
    ax.annotate(f"best {a1[b1]:.4f} @ epoch {e1[b1]}", (e1[b1], a1[b1]), textcoords="offset points",
                xytext=(5, -12))
    ax.scatter([e2[b2]], [a2[b2]], color="black", zorder=5)
    ax.annotate(f"best {a2[b2]:.4f} @ epoch {e2[b2]}", (e2[b2], a2[b2]), textcoords="offset points",
                xytext=(5, 8))
    ax.axvline(e2[-1], color="gray", linestyle="--", alpha=0.6)
    ax.text(e2[-1], min(a1 + a2) - 0.002, f" early-stopped\n @ epoch {e2[-1]}", fontsize=8, color="gray")
    ax.set_xlabel("epoch"); ax.set_ylabel("val AUC")
    ax.set_title("8h: B1@1s convergence -- fixed-epoch vs. early-stopped protocol")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "8h_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_8f_forest_plot():
    d = load(EXP / "whisper_tiny_speech_aligned_contract" / "8f_ci_analysis_pilot.json")
    a0 = d["A0"]

    criteria = [
        ("frac_gt_020", "frac_gt_020_ci", 0.10, "frac Δprob>0.20 (≤10%)"),
        ("flip_rate", "flip_rate_ci", 0.05, "decision-flip rate (≤5%)"),
    ]
    comparisons = [("silero_vad", "vs. Silero VAD"), ("energy_alt", "vs. alt-threshold")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (point_key, ci_key, threshold, title) in zip(axes, criteria):
        ys = list(range(len(comparisons)))
        for y, (comp_key, comp_label) in zip(ys, comparisons):
            entry = a0[comp_key]
            point = entry[point_key]
            lo, hi = entry[ci_key]
            color = "tab:red" if lo > threshold else ("tab:green" if hi <= threshold else "tab:orange")
            ax.plot([lo, hi], [y, y], color=color, linewidth=3, solid_capstyle="round")
            ax.scatter([point], [y], color="black", zorder=5, s=25)
        ax.axvline(threshold, color="gray", linestyle="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([c[1] for c in comparisons])
        ax.set_xlabel("proportion")
        ax.set_title(title)
        ax.set_xlim(-0.02, max(0.30, max(a0[c][ci_key][1] for c, _ in comparisons) + 0.02))
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("8f: A0 VAD-boundary criteria, Wilson 95% CI (n=43) -- green=pass, red=fail, orange=borderline")
    fig.tight_layout()
    out = FIG_DIR / "8f_forest_plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_step9_training_curves():
    runs = [
        ("baseline (no pause events)", EXP / "mel_trajectory_1s_speech_aligned_contract"),
        ("P1 plain (Step 7 recipe)", EXP / "pause_events_contract_fixed_plain"),
        ("P1a+P1b λ=0.1", EXP / "pause_events_holdloss0.1_5epoch"),
        ("P1a+P1b λ=0.25", EXP / "pause_events_holdloss0.25_5epoch"),
        ("P1a+P1b λ=0.5", EXP / "pause_events_holdloss0.5_5epoch"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, d in runs:
        hist = load(d / "metrics.json")["history"]
        epochs = [h["epoch"] for h in hist]
        auc = [h["val_auc"] for h in hist]
        line, = ax.plot(epochs, auc, marker="o", label=name)
        if epochs[-1] == epochs[int(np.argmax(auc))]:
            ax.scatter([epochs[-1]], [auc[-1]], color=line.get_color(), s=80,
                       facecolors="none", edgecolors=line.get_color(), linewidths=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("val AUC (final clips)")
    ax.set_title("Step 9: val AUC per epoch -- open circle = still rising at epoch 5 (undertrained)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "step9_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_step9_auc_comparison():
    runs = [
        ("baseline", EXP / "mel_trajectory_1s_speech_aligned_contract"),
        ("P1 plain", EXP / "pause_events_contract_fixed_plain"),
        ("P1a+P1b\nλ=0.1", EXP / "pause_events_holdloss0.1_5epoch"),
        ("P1a+P1b\nλ=0.25", EXP / "pause_events_holdloss0.25_5epoch"),
        ("P1a+P1b\nλ=0.5", EXP / "pause_events_holdloss0.5_5epoch"),
    ]
    names, main_auc, real_auc = [], [], []
    for name, d in runs:
        m = load(d / "metrics.json")
        names.append(name)
        main_auc.append(m["overall"]["auc"])
        real_auc.append(m["real_all"]["auc"])

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, main_auc, width, label="main (overall) AUC", color="tab:blue")
    ax.bar(x + width / 2, real_auc, width, label="real-audio AUC", color="tab:orange")
    ax.axhline(main_auc[0], color="tab:blue", linestyle=":", alpha=0.6)
    ax.axhline(real_auc[0], color="tab:orange", linestyle=":", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.6, 0.9)
    ax.set_title("Step 9: main-task vs. real-audio AUC (dotted = baseline level)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "step9_auc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_8d_training_curves()
    fig_8h_convergence()
    fig_8f_forest_plot()
    fig_step9_training_curves()
    fig_step9_auc_comparison()
