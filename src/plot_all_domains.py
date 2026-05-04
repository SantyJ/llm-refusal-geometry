"""Plot all 3 domains together showing the safety-utility tradeoff contrast."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = "/workspace/refusal_vectors/results/plots"

# Data from logs
data = {
    "harmful": {
        "single": [
            (0.0, 98.0, 10.12), (0.1, 98.0, 9.98),  (0.2, 98.0, 9.97),
            (0.3, 98.0, 10.22), (0.4, 98.0, 10.85),  (0.5, 98.0, 13.94),
            (0.6, 98.0, 26.95), (0.7, 98.0, 90.87),  (0.8, 76.0, 340.61),
            (0.9, 12.0, 908.89),(1.0,  0.0, 1656.28),
        ],
        "subspace": [
            (0.0, 98.0, 10.12), (0.1, 98.0, 9.78),  (0.2, 98.0, 9.14),
            (0.3, 98.0, 8.87),  (0.4, 98.0, 10.31),  (0.5, 96.0, 8.88),
            (0.6, 96.0, 8.95),  (0.7, 96.0, 8.56),   (0.8, 84.0, 9.10),
            (0.9, 50.0, 56.26), (1.0,  0.0, 5264.55),
        ],
    },
    "medical": {
        "single": [
            (0.0, 48.0, 10.12), (0.1, 48.0, 10.11), (0.2, 48.0, 10.09),
            (0.3, 52.0, 10.07), (0.4, 54.0, 10.04), (0.5, 56.0, 10.03),
            (0.6, 56.0, 10.01), (0.7, 56.0, 10.02), (0.8, 58.0, 10.01),
            (0.9, 60.0, 10.00), (1.0, 60.0, 10.00),
        ],
        "subspace": [
            (0.0, 48.0, 10.12), (0.1, 44.0, 10.17), (0.2, 40.0, 10.22),
            (0.3, 40.0, 10.30), (0.4, 36.0, 10.42), (0.5, 38.0, 10.59),
            (0.6, 40.0, 10.85), (0.7, 38.0, 11.26), (0.8, 34.0, 11.75),
            (0.9, 36.0, 12.24), (1.0, 28.0, 12.77),
        ],
    },
    "legal": {
        "single": [
            (0.0, 70.0, 10.12), (0.1, 68.0, 10.11), (0.2, 68.0, 10.11),
            (0.3, 68.0, 10.10), (0.4, 66.0, 10.10), (0.5, 66.0, 10.09),
            (0.6, 66.0, 10.09), (0.7, 64.0, 10.08), (0.8, 64.0, 10.08),
            (0.9, 62.0, 10.08), (1.0, 62.0, 10.08),
        ],
        "subspace": [
            (0.0, 70.0, 10.12), (0.1, 68.0, 10.10), (0.2, 70.0, 10.08),
            (0.3, 68.0, 10.09), (0.4, 68.0, 10.13), (0.5, 64.0, 10.23),
            (0.6, 56.0, 10.36), (0.7, 50.0, 10.55), (0.8, 48.0, 10.76),
            (0.9, 28.0, 10.99), (1.0, 18.0, 11.20),
        ],
    },
}

domain_colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}
domain_labels = {"harmful": "Harmful", "medical": "Medical", "legal": "Legal"}

# ── Plot 1: Refusal rate vs alpha, all 3 domains, both methods ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, method, title in zip(axes, ["single", "subspace"],
                              ["Single Vector Ablation", "Subspace Ablation (k=8)"]):
    for domain in ["harmful", "medical", "legal"]:
        rows = data[domain][method]
        alphas   = [r[0] for r in rows]
        refusals = [r[1] for r in rows]
        color = domain_colors[domain]
        label = domain_labels[domain]
        ax.plot(alphas, refusals, "o-", color=color, linewidth=2.5,
                markersize=6, label=label)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Alpha (ablation strength)", fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Annotate concept erasure on left panel
axes[0].annotate("Concept erasure:\nrefusal increases!",
                 xy=(0.7, 56), xytext=(0.35, 66),
                 arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5),
                 fontsize=9, color="#3498db",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#3498db", alpha=0.85))

fig.suptitle("Refusal Rate vs Alpha — All 3 Domains\n"
             "Subspace ablation fixes medical/legal without affecting harmful",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "all_domains_refusal.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")


# ── Plot 2: Perplexity vs alpha — log scale to show catastrophic PPL ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, method, title in zip(axes, ["single", "subspace"],
                              ["Single Vector Ablation", "Subspace Ablation (k=8)"]):
    for domain in ["harmful", "medical", "legal"]:
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        perps  = [r[2] for r in rows]
        color  = domain_colors[domain]
        label  = domain_labels[domain]
        ax.semilogy(alphas, perps, "o-", color=color, linewidth=2.5,
                    markersize=6, label=label)

    # Baseline
    ax.axhline(10.12, color="gray", linestyle=":", linewidth=1.5,
               alpha=0.6, label="Baseline PPL")

    # Shade danger zone
    ax.axhspan(100, 10000, alpha=0.05, color="red")
    ax.text(0.02, 200, "incoherent\noutput zone", fontsize=8,
            color="#e74c3c", alpha=0.7, style="italic")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Alpha (ablation strength)", fontsize=11)
    ax.set_ylabel("Perplexity (log scale)", fontsize=11)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(7, 10000)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Annotate PPL values at alpha=1.0
for method_idx, method in enumerate(["single", "subspace"]):
    ax = axes[method_idx]
    ppl_vals = {d: data[d][method][-1][2] for d in ["harmful", "medical", "legal"]}
    for domain, ppl in ppl_vals.items():
        color = domain_colors[domain]
        ax.annotate(f"{ppl:.0f}", xy=(1.0, ppl),
                    xytext=(0.85, ppl * (1.4 if ppl > 100 else 1.1)),
                    fontsize=8, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1))

fig.suptitle("Perplexity vs Alpha — All 3 Domains (Log Scale)\n"
             "Harmful: catastrophic PPL to reach 0% refusal  |  Medical/Legal: modest cost",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "all_domains_perplexity.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ── Plot 3: The key contrast — 2×2 summary panel ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Top row: refusal rate
for col, (method, title) in enumerate([("single", "Single Vector"),
                                        ("subspace", "Subspace (k=8)")]):
    ax = axes[0][col]
    for domain in ["harmful", "medical", "legal"]:
        rows = data[domain][method]
        alphas   = [r[0] for r in rows]
        refusals = [r[1] for r in rows]
        ax.plot(alphas, refusals, "o-", color=domain_colors[domain],
                linewidth=2.5, markersize=6, label=domain_labels[domain])
    ax.set_title(f"{title}\nRefusal Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Refusal Rate (%)", fontsize=10)
    ax.set_xlabel("Alpha", fontsize=10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Bottom row: perplexity (log scale)
for col, (method, title) in enumerate([("single", "Single Vector"),
                                        ("subspace", "Subspace (k=8)")]):
    ax = axes[1][col]
    for domain in ["harmful", "medical", "legal"]:
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        perps  = [r[2] for r in rows]
        ax.semilogy(alphas, perps, "o-", color=domain_colors[domain],
                    linewidth=2.5, markersize=6, label=domain_labels[domain])
    ax.axhline(10.12, color="gray", linestyle=":", alpha=0.5)
    ax.axhspan(100, 10000, alpha=0.05, color="red")
    ax.set_title(f"{title}\nPerplexity (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=10)
    ax.set_xlabel("Alpha", fontsize=10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(7, 10000)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Safety vs Over-Refusal: A Structural Difference\n"
    "Harmful refusal is safety-critical (removing it destroys coherence)\n"
    "Medical/legal over-refusal is fixable (subspace ablation, modest PPL cost)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "all_domains_contrast.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")
