"""Summary figure for Step 9's controlled rerun (Section 7 of PHASE3_RESULTS_8e-8h.md). Numbers
pulled directly from step9_results/step9_kaggle_results.json and each arm's own metrics.json."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("planning/figures")
OUT.mkdir(parents=True, exist_ok=True)

labels = ["baseline", "P1_plain", "λ=0.25", "λ=0.5\n(all)", "λ=0.5\n(real-only)", "λ=0.5\n(50:50)"]
overall_auc = [83.16, 80.75, 83.61, 83.36, 81.19, 82.41]
real_auc = [74.36, 64.77, 72.54, 70.61, 64.17, 71.07]
fcr_holds_all_r95 = [77.73, 71.66, 75.66, 71.62, 78.34, 75.28]
fcr_holds_real_r95 = [86.98, 77.29, 79.50, 76.18, 64.27, 68.14]
short_complete_recall = [22.95, 19.67, 26.23, 21.31, 32.79, 27.87]
particle_recall = [30.0, 25.0, 40.0, 30.0, 20.0, 35.0]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(labels))
w = 0.35

ax = axes[0]
ax.bar(x - w/2, overall_auc, w, label="overall AUC")
ax.bar(x + w/2, real_auc, w, label="real-audio AUC")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("AUC (%)")
ax.set_ylim(60, 86)
ax.set_title("Main-task accuracy")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.bar(x - w/2, fcr_holds_all_r95, w, label="FCR @ holds, all")
ax.bar(x + w/2, fcr_holds_real_r95, w, label="FCR @ holds, real")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("FCR (%) @ matched recall=95%")
ax.set_title("Hold suppression (lower = better)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.bar(x - w/2, short_complete_recall, w, label="short-complete recall (n=61)")
ax.bar(x + w/2, particle_recall, w, label="response-particle recall (n=20)")
ax.set_xticks(x, labels, fontsize=8)
ax.set_ylabel("recall (%)")
ax.set_title("Hard-case recall (higher = better;\nsmall n -- noisy)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

plt.suptitle("Step 9 controlled rerun (early-stopped, n=1,600 val split)")
plt.tight_layout()
plt.savefig(OUT / "step9_v2_summary.png", dpi=150)
plt.close()
print("saved", OUT / "step9_v2_summary.png")
