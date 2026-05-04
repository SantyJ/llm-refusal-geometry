"""Generate clean, slide-friendly bootstrap stability plots."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORS_DIR = os.path.join(_ROOT, "results", "vectors")
PLOTS_DIR = os.path.join(_ROOT, "results", "plots")

with open(os.path.join(VECTORS_DIR, "bootstrap_results.json")) as f:
    data = json.load(f)

domains = ["harmful", "medical", "legal"]
colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}
sims = {d: np.array(data[d]) for d in domains}

# ── Plot 1: Strip + mean bar (clean, slide-friendly) ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

positions = {"harmful": 1, "medical": 2, "legal": 3}
for domain in domains:
    x = positions[domain]
    vals = sims[domain]
    mean = vals.mean()
    color = colors[domain]

    # Horizontal mean line
    ax.hlines(mean, x - 0.25, x + 0.25, colors=color, linewidth=3, zorder=4)

    # Individual points (jittered)
    jitter = np.random.RandomState(42).uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(x + jitter, vals, color=color, s=80, alpha=0.8, zorder=3,
               edgecolors="white", linewidth=0.8)

    # Shade between min and max
    ax.fill_between([x - 0.25, x + 0.25], vals.min(), vals.max(),
                    color=color, alpha=0.12, zorder=1)

    # Annotate mean
    ax.text(x, mean + 0.012, f"μ={mean:.3f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=color)

# Threshold line
ax.axhline(0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.6,
           label="0.90 threshold")
ax.text(3.4, 0.901, "0.90", va="bottom", color="gray", fontsize=9)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Harmful", "Medical", "Legal"], fontsize=13)
ax.set_ylabel("Cosine Similarity to Original Direction", fontsize=12)
ax.set_ylim(0.35, 1.08)
ax.set_xlim(0.5, 3.7)
ax.set_title("Bootstrap Stability: Are Refusal Directions Robust?\n"
             "(10 × 80% resamples — higher = more stable)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend note for harmful outliers
ax.annotate("2 outliers: artifact\n(only 2 answered\nsamples in harmful)",
            xy=(1, 0.443), xytext=(1.5, 0.50),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
            fontsize=8.5, color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.8))

plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "bootstrap_strip.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")


# ── Plot 2: Bar chart with error bars ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

# For harmful, report "clean" mean (exclude the 2 outliers below 0.9)
# so slide reads correctly — annotate both
x = np.arange(3)
means = [sims[d].mean() for d in domains]
stds  = [sims[d].std()  for d in domains]
clean_mean_harmful = sims["harmful"][sims["harmful"] > 0.9].mean()

bars = ax.bar(x, means, color=[colors[d] for d in domains],
              alpha=0.85, width=0.5, zorder=3,
              error_kw=dict(elinewidth=2, capsize=6, ecolor="black"))
ax.errorbar(x, means, yerr=stds, fmt="none", elinewidth=2, capsize=6,
            ecolor="black", zorder=4)

# Value labels
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.008, f"{m:.3f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")

# Clean mean annotation for harmful
ax.text(0, clean_mean_harmful + 0.008, f"({clean_mean_harmful:.3f} excl. outliers)",
        ha="center", va="bottom", fontsize=8, color="#e74c3c", style="italic")

ax.axhline(0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
           label="0.90 threshold")
ax.set_xticks(x)
ax.set_xticklabels(["Harmful", "Medical", "Legal"], fontsize=13)
ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
ax.set_ylim(0.7, 1.08)
ax.set_title("Bootstrap Stability — Mean ± Std\n"
             "(10 × 80% resamples per domain)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "bootstrap_bars.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ── Plot 3: Per-sample lines (shows consistency across 10 runs) ───────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

for domain in domains:
    vals = sims[domain]
    x = np.arange(1, len(vals) + 1)
    ax.plot(x, vals, "o-", color=colors[domain], linewidth=2,
            markersize=7, label=domain.capitalize(), alpha=0.85)
    ax.axhline(vals.mean(), color=colors[domain], linestyle=":",
               linewidth=1.5, alpha=0.5)

ax.axhline(0.9, color="gray", linestyle="--", linewidth=1.5, alpha=0.6,
           label="0.90 threshold")
ax.set_xlabel("Bootstrap Sample #", fontsize=12)
ax.set_ylabel("Cosine Similarity to Original Direction", fontsize=12)
ax.set_title("Bootstrap Stability Across 10 Resamples\n"
             "(Each point = direction from 80% of prompts)",
             fontsize=12, fontweight="bold")
ax.set_xticks(np.arange(1, 11))
ax.set_ylim(0.35, 1.08)
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "bootstrap_lines.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")
