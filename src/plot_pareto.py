"""Clean, slide-friendly Pareto frontier plots."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR   = os.path.join(_ROOT, "results", "plots")
VECTORS_DIR = os.path.join(_ROOT, "results", "vectors")

# Load data from JSON (all 3 domains)
with open(os.path.join(VECTORS_DIR, "subspace_ablation_results.json")) as f:
    _raw = json.load(f)

def _to_tuples(rows):
    return [(r["alpha"], r["refusal_rate"], r["perplexity"]) for r in rows]

data = {d: {m: _to_tuples(rows) for m, rows in methods.items()}
        for d, methods in _raw.items()}
data = {
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

# ── Plot 1: Side-by-side subplots, one per domain ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

domain_colors = {"medical": "#3498db", "legal": "#2ecc71"}
method_styles = {
    "single":   {"marker": "^", "color": "#e74c3c", "label": "Single vector", "ls": "--"},
    "subspace": {"marker": "o", "color": None,       "label": "Subspace (k=8)", "ls": "-"},
}

for ax, domain in zip(axes, ["medical", "legal"]):
    dc = domain_colors[domain]
    for method, style in method_styles.items():
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        refusals = [r[1] for r in rows]
        perps = [r[2] for r in rows]
        color = dc if method == "subspace" else style["color"]

        ax.plot(refusals, perps, style["ls"], color=color,
                linewidth=2, alpha=0.7)
        ax.scatter(refusals, perps, marker=style["marker"], color=color,
                   s=70, zorder=4, label=style["label"])

        # Annotate only α=0, 0.5, 1.0
        for a, rr, ppl in zip(alphas, refusals, perps):
            if a in [0.0, 0.5, 1.0]:
                ax.annotate(f"α={a}", (rr, ppl),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=8, color=color)

    # Ideal corner marker
    ax.annotate("← ideal\n  corner", xy=(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 10, 10),
                fontsize=8, color="gray", style="italic")

    ax.set_title(f"{domain.capitalize()} Domain", fontsize=13, fontweight="bold")
    ax.set_xlabel("Refusal Rate (%)", fontsize=11)
    ax.set_ylabel("Perplexity (lower = more coherent)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Pareto Frontier: Safety-Utility Tradeoff\n"
    "Lower-left = lower refusal AND better language quality",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "pareto_clean.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")


# ── Plot 2: Refusal rate vs alpha line chart (more intuitive) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, domain in zip(axes, ["medical", "legal"]):
    dc = domain_colors[domain]
    for method, style in method_styles.items():
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        refusals = [r[1] for r in rows]
        color = dc if method == "subspace" else style["color"]

        ax.plot(alphas, refusals, style["ls"], color=color,
                linewidth=2.5, marker=style["marker"], markersize=7,
                label=style["label"])

    # Baseline reference
    baseline = data[domain]["single"][0][1]
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.2,
               alpha=0.6, label=f"Baseline ({baseline}%)")

    ax.set_title(f"{domain.capitalize()} Domain", fontsize=13, fontweight="bold")
    ax.set_xlabel("Alpha (ablation strength)", fontsize=11)
    ax.set_ylabel("Refusal Rate (%)", fontsize=11)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade the "concept erasure" zone for single vector (medical)
    if domain == "medical":
        single_refusals = [r[1] for r in data[domain]["single"]]
        single_alphas = [r[0] for r in data[domain]["single"]]
        ax.fill_between(single_alphas, baseline, single_refusals,
                        where=[r >= baseline for r in single_refusals],
                        alpha=0.1, color="#e74c3c", label="_nolegend_")
        ax.text(0.55, 57, "concept\nerasure", color="#e74c3c",
                fontsize=8, style="italic", ha="center")

fig.suptitle(
    "Refusal Rate vs Ablation Strength\n"
    "Single vector causes concept erasure; subspace ablation reduces refusal",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "refusal_vs_alpha.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")


# ── Plot 3: Combined 4-panel (medical + legal, refusal + perplexity) ───────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, domain in enumerate(["medical", "legal"]):
    dc = domain_colors[domain]

    # Row 0: refusal rate vs alpha
    ax = axes[0][col]
    for method, style in method_styles.items():
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        refusals = [r[1] for r in rows]
        color = dc if method == "subspace" else style["color"]
        ax.plot(alphas, refusals, style["ls"], color=color, linewidth=2.5,
                marker=style["marker"], markersize=6, label=style["label"])

    baseline = data[domain]["single"][0][1]
    ax.axhline(baseline, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"{domain.capitalize()} — Refusal Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Refusal Rate (%)", fontsize=10)
    ax.set_xlabel("Alpha", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Row 1: perplexity vs alpha
    ax = axes[1][col]
    for method, style in method_styles.items():
        rows = data[domain][method]
        alphas = [r[0] for r in rows]
        perps = [r[2] for r in rows]
        color = dc if method == "subspace" else style["color"]
        ax.plot(alphas, perps, style["ls"], color=color, linewidth=2.5,
                marker=style["marker"], markersize=6, label=style["label"])

    ax.axhline(10.12, color="gray", linestyle=":", alpha=0.5, label="Baseline ppl")
    ax.set_title(f"{domain.capitalize()} — Perplexity", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=10)
    ax.set_xlabel("Alpha", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "Subspace vs Single Vector Ablation — Full Alpha Sweep\n"
    "Top row: refusal rate  |  Bottom row: perplexity (language quality)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "ablation_4panel.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out3}")
