"""One-off figure generation for PHASE3_RESULTS_8e-8h.md. Reuses the verified numbers already
pulled from the actual result JSONs (not re-derived/estimated) -- see the report for exact sourcing."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("planning/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig_8e_extended():
    fractions = [100, 75, 50, 25]
    mean_abs = [0.0, 2.988, 5.608, 12.737]
    median_abs = [0.0, 0.703, 1.362, 4.400]
    flip_rate = [0.0, 4.228, 7.681, 14.940]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fractions, mean_abs, marker="o", label="mean |Δprob|")
    ax1.plot(fractions, median_abs, marker="s", label="median |Δprob|")
    ax1.plot(fractions, flip_rate, marker="^", label="flip rate vs. full")
    ax1.set_xlabel("% of valid context kept")
    ax1.set_ylabel("% (probability shift / flip rate)")
    ax1.set_title("8e-extended: A0 prediction shift vs. prefix context removed\n(n=1,419)")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    cats = ["real\n(n=231)", "synthetic\n(n=1,188)"]
    mean_abs_rs = [17.296, 11.85]
    flip_rs = [20.779, 13.805]
    x = np.arange(2)
    w = 0.35
    ax2.bar(x - w/2, mean_abs_rs, w, label="mean |Δprob|")
    ax2.bar(x + w/2, flip_rs, w, label="flip rate")
    ax2.set_xticks(x, cats)
    ax2.set_ylabel("%")
    ax2.set_title("At 25% context kept: real vs. synthetic")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT / "8e_extended_prefix_context.png", dpi=150)
    plt.close()


def fig_8f_forest():
    rows = [
        ("alt-threshold: frac Δprob>0.20", 8.313, 7.058, 9.767, 10.0, "pass"),
        ("alt-threshold: flip rate", 10.500, 9.091, 12.098, 5.0, "fail"),
        ("Silero: frac Δprob>0.20", 11.625, 10.146, 13.288, 10.0, "fail"),
        ("Silero: flip rate", 12.250, 10.733, 13.948, 5.0, "fail"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (label, point, lo, hi, thresh, verdict) in enumerate(rows):
        y = len(rows) - i
        color = "#2a9d3a" if verdict == "pass" else "#d62728"
        ax.plot([lo, hi], [y, y], color=color, lw=2)
        ax.plot(point, y, "o", color=color)
        ax.axvline(thresh, color="gray", ls="--", lw=0.8)
        ax.text(max(hi, thresh) + 0.5, y, f"{verdict}", va="center", fontsize=8, color=color)
    ax.set_yticks([len(rows) - i for i in range(len(rows))], [r[0] for r in rows])
    ax.set_xlabel("%")
    ax.set_title("8f: VAD-boundary diagnostic, full val split (n=1,600)\npoint + 95% Wilson CI vs. frozen threshold (dashed)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(OUT / "8f_forest_plot_v2.png", dpi=150)
    plt.close()


def fig_8g_direction_specific():
    groups = ["safety-critical\n(→complete)", "latency-critical\n(→incomplete)", "real-audio\nsafety-critical"]
    alt = [10.761, 10.081, 14.444]
    silero = [14.416, 8.780, 19.444]
    alt_err = [[10.761-8.976, 10.081-7.944, 14.444-10.052], [12.851-10.761, 12.714-10.081, 20.323-14.444]]
    silero_err = [[14.416-12.361, 8.780-6.792, 19.444-14.326], [16.748-14.416, 11.281-8.780, 25.839-19.444]]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w/2, alt, w, yerr=alt_err, capsize=3, label="alt-threshold")
    ax.bar(x + w/2, silero, w, yerr=silero_err, capsize=3, label="Silero VAD")
    ax.axhline(2.0, color="red", ls="--", lw=1, label="safety-critical bound (2%)")
    ax.axhline(5.0, color="orange", ls="--", lw=1, label="latency-critical / aggregate bound (5%)")
    ax.set_xticks(x, groups)
    ax.set_ylabel("flip rate (%)")
    ax.set_title("8g: direction-specific flip rates vs. frozen bounds (n=1,600)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT / "8g_direction_specific_flip_rates.png", dpi=150)
    plt.close()


def fig_8h_a0_comparison():
    labels = ["A0@4s\n(original,\nfixed 2 epochs)", "A0@4s\n(Kaggle longrun,\nearly-stopped)", "A0@2s\n(Kaggle longrun,\nearly-stopped)"]
    overall_auc = [93.78, 93.62, 92.05]
    real_auc = [91.22, 90.47, 88.92]
    latency = [None, 45.653, 17.362]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, overall_auc, w, label="overall AUC")
    ax1.bar(x + w/2, real_auc, w, label="real-audio AUC")
    ax1.set_xticks(x, labels, fontsize=8)
    ax1.set_ylim(85, 96)
    ax1.set_ylabel("AUC (%)")
    ax1.set_title("8h-A0: accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(["A0@4s\nlongrun", "A0@2s\nlongrun"], [45.653, 17.362], color=["#1f77b4", "#ff7f0e"])
    ax2.set_ylabel("p50 latency (ms)")
    ax2.set_title("8h-A0: latency (batch-1, CPU, incl.\nfeature extraction)")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT / "8h_a0_context_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_8e_extended()
    fig_8f_forest()
    fig_8g_direction_specific()
    fig_8h_a0_comparison()
    print("saved 4 figures to", OUT)
