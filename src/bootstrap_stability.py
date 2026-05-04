"""Bootstrap stability analysis.

Resamples 80% of prompts 10 times per domain, recomputes mean-difference direction
from each bootstrap sample, measures cosine similarity between each bootstrap
direction and the original full-data direction.

High stability (cosine sim ~0.95+) = direction is real, not an artifact of the
specific 100 prompts chosen.
Expected: harmful ~0.95+, medical/legal ~0.7+ (high-rank directions are less stable)

Outputs:
    results/plots/bootstrap_stability.png
    results/vectors/bootstrap_results.json

Run:
    python src/bootstrap_stability.py   (CPU only, ~2 min)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/src")
from utils import VECTORS_DIR, PLOTS_DIR

N_BOOTSTRAP = 10
RESAMPLE_FRAC = 0.8


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compute_direction(refused_h, answered_h):
    diff = refused_h.mean(axis=0) - answered_h.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-10:
        return None
    return diff / norm


def bootstrap_domain(domain, vectors_dir=VECTORS_DIR, n=N_BOOTSTRAP, frac=RESAMPLE_FRAC):
    # Load original direction
    v_orig = np.load(os.path.join(vectors_dir, f"v_{domain}.npy"))

    # Load raw hidden states
    path = os.path.join(vectors_dir, f"hiddens_{domain}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run extract_directions.py --save-hiddens first")
    data = np.load(path)
    refused = data["refused"].astype(np.float32)
    answered = data["answered"].astype(np.float32)

    n_refused = len(refused)
    n_answered = len(answered)
    k_refused = max(2, int(n_refused * frac))
    k_answered = max(2, int(n_answered * frac))

    sims = []
    for i in range(n):
        idx_r = np.random.choice(n_refused, size=k_refused, replace=False)
        idx_a = np.random.choice(n_answered, size=k_answered, replace=False)
        v_boot = compute_direction(refused[idx_r], answered[idx_a])
        if v_boot is not None:
            sim = cosine_sim(v_orig, v_boot)
            sims.append(sim)

    return np.array(sims)


def plot_bootstrap(results, out_path):
    domains = list(results.keys())
    colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}

    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), sharey=True)

    for ax, domain in zip(axes, domains):
        sims = results[domain]
        mean_sim = sims.mean()
        std_sim = sims.std()

        ax.boxplot(sims, patch_artist=True,
                   boxprops=dict(facecolor=colors.get(domain, "#888"), alpha=0.6),
                   medianprops=dict(color="black", linewidth=2))
        ax.scatter([1] * len(sims), sims, color=colors.get(domain, "#888"),
                   alpha=0.7, zorder=3, s=40)
        ax.set_title(f"{domain.capitalize()}\nμ={mean_sim:.3f} ± {std_sim:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Cosine Similarity to Original Direction", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([])
        ax.axhline(mean_sim, color=colors.get(domain, "#888"), linestyle="--",
                   alpha=0.5, linewidth=1.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Bootstrap Stability Analysis ({N_BOOTSTRAP}x {int(RESAMPLE_FRAC*100)}% resampling)\n"
        "Cosine similarity between bootstrap directions and original full-data direction",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    np.random.seed(42)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    vectors_dir = VECTORS_DIR
    if "--vectors-dir" in sys.argv:
        idx = sys.argv.index("--vectors-dir")
        vectors_dir = sys.argv[idx + 1]

    domains = ["harmful", "medical", "legal"]
    print("=" * 50)
    print(f"Bootstrap Stability  (n={N_BOOTSTRAP}, frac={RESAMPLE_FRAC})")
    print("=" * 50)

    results = {}
    for domain in domains:
        print(f"\n[{domain}]")
        sims = bootstrap_domain(domain, vectors_dir)
        results[domain] = sims
        print(f"  Similarities: {np.array2string(sims, precision=3)}")
        print(f"  Mean ± Std:   {sims.mean():.3f} ± {sims.std():.3f}")
        print(f"  Min / Max:    {sims.min():.3f} / {sims.max():.3f}")

    # Save JSON
    out_json = os.path.join(vectors_dir, "bootstrap_results.json")
    with open(out_json, "w") as f:
        json.dump({d: results[d].tolist() for d in domains}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    plot_bootstrap(results, os.path.join(PLOTS_DIR, "bootstrap_stability.png"))

    # Summary
    print("\n" + "=" * 50)
    print(f"  {'Domain':<10} {'Mean':>8} {'Std':>8}  Interpretation")
    print(f"  {'-'*48}")
    interp = {
        "harmful": "stable (rank-1 direction)",
        "medical": "moderate (high-rank, some variance)",
        "legal":   "moderate (high-rank, some variance)"
    }
    for d in domains:
        s = results[d]
        print(f"  {d:<10} {s.mean():>8.3f} {s.std():>8.3f}  {interp.get(d, '')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
