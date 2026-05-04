"""Compare direction vectors and plot cosine similarity heatmap."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VECTORS_DIR = "/workspace/results/vectors"
PLOTS_DIR = "/workspace/results/plots"


def load_vectors():
    """Load all saved direction vectors."""
    vectors = {}
    for name in ["harmful", "medical", "legal"]:
        path = os.path.join(VECTORS_DIR, f"v_{name}.npy")
        if os.path.exists(path):
            vectors[name] = np.load(path)
            print(f"Loaded v_{name} (dim={vectors[name].shape[0]})")
    return vectors


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def plot_heatmap(vectors):
    names = list(vectors.keys())
    n = len(names)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine_sim(vectors[names[i]], vectors[names[j]])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    labels = [f"v_{n}" for n in names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(sim[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")

    ax.set_title("Cosine Similarity: Refusal Direction Vectors\n"
                 "(Llama-2-7b-chat, Layer 16)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # Print matrix
    print("\nSimilarity matrix:")
    for i, ni in enumerate(names):
        row = "  ".join(f"{sim[i,j]:+.3f}" for j in range(n))
        print(f"  v_{ni:8s}: {row}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    vectors = load_vectors()
    if len(vectors) < 2:
        print(f"Need at least 2 vectors, found {len(vectors)}. Cannot plot heatmap.")
    else:
        plot_heatmap(vectors)
