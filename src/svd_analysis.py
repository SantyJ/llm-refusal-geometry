"""SVD subspace analysis: effective rank per domain + principal angles between subspaces.

Requires hidden states saved first:
    python src/extract_directions.py --save-hiddens

Outputs:
    results/plots/scree_plots.png        -- explained variance per domain
    results/plots/principal_angles.png   -- Grassmann geometry between subspaces
    results/vectors/subspaces.npz        -- top-k right singular vectors per domain (for ablation)

Run:
    python src/svd_analysis.py
    python src/svd_analysis.py --top-k 4   # change subspace rank
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/src")
from utils import VECTORS_DIR, PLOTS_DIR

TOP_K = 8  # Number of singular vectors kept for subspace ablation


def load_hiddens(domain, vectors_dir=VECTORS_DIR):
    path = os.path.join(vectors_dir, f"hiddens_{domain}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}\n"
            f"Run first: python src/extract_directions.py --save-hiddens"
        )
    data = np.load(path)
    return data["refused"].astype(np.float32), data["answered"].astype(np.float32)


def effective_rank(singular_values):
    """Spectral entropy: exp(-Σ p_i log p_i).  Clean 1-D signal → ~1.0."""
    s2 = singular_values ** 2
    p = s2 / (s2.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_subspace(domain, vectors_dir=VECTORS_DIR, top_k=TOP_K):
    """SVD of the difference matrix H_refused - mean(H_answered).

    Returns:
        S        -- all singular values
        Vt_k     -- top-k right singular vectors, shape (k, 4096)
        eff_r    -- effective rank
    """
    refused, answered = load_hiddens(domain, vectors_dir)
    mean_ans = answered.mean(axis=0)
    D = refused - mean_ans[np.newaxis, :]        # (n_refused, 4096)

    # Full SVD on the (small) n_refused × 4096 matrix
    _, S, Vt = np.linalg.svd(D, full_matrices=False)  # Vt: (min_dim, 4096)

    k = min(top_k, Vt.shape[0])
    eff_r = effective_rank(S)
    return S, Vt[:k, :], eff_r


def principal_angles(Vt_a, Vt_b):
    """Principal angles (degrees) between two subspaces given basis rows."""
    M = Vt_a @ Vt_b.T              # (k_a, k_b)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.arccos(sv) * 180 / np.pi


def plot_scree(scree_data, out_path):
    domains = list(scree_data.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), sharey=False)

    colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}

    for ax, domain in zip(axes, domains):
        S, eff_r = scree_data[domain]
        explained = (S ** 2) / ((S ** 2).sum() + 1e-12)
        cumulative = np.cumsum(explained)
        x = np.arange(1, min(21, len(S) + 1))

        ax.bar(x, explained[:len(x)] * 100, color=colors.get(domain, "#888"),
               alpha=0.7, label="Per component")
        ax2 = ax.twinx()
        ax2.plot(x, cumulative[:len(x)] * 100, "k-o", markersize=4, linewidth=1.5,
                 label="Cumulative")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative (%)", fontsize=9)
        ax2.tick_params(axis='y', labelsize=8)

        ax.set_title(f"{domain.capitalize()}\nEff. rank = {eff_r:.2f}", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("Singular value index", fontsize=10)
        ax.set_ylabel("Explained variance (%)", fontsize=10)
        ax.set_xlim(0.5, len(x) + 0.5)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle(
        "SVD Scree Plots — Refusal Subspace Structure per Domain\n"
        "(Llama-2-7b-chat, Layer 16 hidden states)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_principal_angles(angle_data, out_path):
    pairs = list(angle_data.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    pair_colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, pair in enumerate(pairs):
        angles = angle_data[pair]
        x = np.arange(1, len(angles) + 1)
        ax.plot(x, angles, "o-", label=pair, color=pair_colors[i % 3],
                linewidth=2, markersize=6)

    ax.axhline(90, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.6, label="90° (fully orthogonal)")
    ax.set_xlabel("Principal angle index", fontsize=11)
    ax.set_ylabel("Angle (degrees)", fontsize=11)
    ax.set_title(
        "Principal Angles Between Domain Refusal Subspaces\n"
        "(Near 90° = geometrically independent — Grassmann geometry)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    vectors_dir = VECTORS_DIR
    top_k = TOP_K

    if "--vectors-dir" in sys.argv:
        idx = sys.argv.index("--vectors-dir")
        vectors_dir = sys.argv[idx + 1]
    if "--top-k" in sys.argv:
        idx = sys.argv.index("--top-k")
        top_k = int(sys.argv[idx + 1])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    domains = ["harmful", "medical", "legal"]

    print("=" * 60)
    print(f"SVD Subspace Analysis  (top_k={top_k})")
    print("=" * 60)

    subspaces = {}
    scree_data = {}

    for domain in domains:
        print(f"\n[{domain}]")
        S, Vt_k, eff_r = compute_subspace(domain, vectors_dir, top_k=top_k)
        subspaces[domain] = Vt_k       # (k, 4096)
        scree_data[domain] = (S, eff_r)

        explained = (S ** 2) / ((S ** 2).sum() + 1e-12)
        print(f"  Effective rank : {eff_r:.3f}")
        print(f"  Top-1 explains : {explained[0]*100:.1f}%")
        print(f"  Top-3 explains : {explained[:3].sum()*100:.1f}%")
        print(f"  Top-{top_k} explains: {explained[:top_k].sum()*100:.1f}%")

    # Save subspaces as (k, 4096) matrices — used directly by subspace_ablation.py
    save_path = os.path.join(vectors_dir, "subspaces.npz")
    np.savez(save_path, **{f"U_{d}": subspaces[d] for d in domains})
    print(f"\nSubspaces saved → {save_path}")

    # Plots
    plot_scree(scree_data, os.path.join(PLOTS_DIR, "scree_plots.png"))

    # Principal angles between all pairs
    print("\nPrincipal Angles (degrees):")
    angle_data = {}
    pairs = [("harmful", "medical"), ("harmful", "legal"), ("medical", "legal")]
    for a, b in pairs:
        angles = principal_angles(subspaces[a], subspaces[b])
        label = f"{a} vs {b}"
        angle_data[label] = angles
        angle_str = ", ".join(f"{x:.1f}" for x in angles)
        print(f"  {label:25s}: [{angle_str}]")

    plot_principal_angles(angle_data, os.path.join(PLOTS_DIR, "principal_angles.png"))

    # Summary table
    print("\n" + "=" * 40)
    print(f"  {'Domain':<10} {'Eff. Rank':>10}  {'Interpretation'}")
    print(f"  {'-'*38}")
    interp = {"harmful": "~1  (clean 1D direction)", "medical": "~4-8 (entangled subspace)",
              "legal": "~4-8 (entangled subspace)"}
    for d, (S, eff_r) in scree_data.items():
        print(f"  {d:<10} {eff_r:>10.2f}  {interp.get(d, '')}")
    print("=" * 40)


if __name__ == "__main__":
    main()
