"""Merge 4 partial patching JSONs and generate combined heatmap."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VECTORS_DIR = "/workspace/refusal_vectors/results/vectors"
PLOTS_DIR   = "/workspace/refusal_vectors/results/plots"
N_HEADS = 32

all_layers, all_aie = [], []
for start, end in [(10,13),(13,16),(16,19),(19,22)]:
    with open(os.path.join(VECTORS_DIR, f"patching_layer_{start}_{end}.json")) as f:
        d = json.load(f)
    all_layers.extend(d["layers"])
    all_aie.extend(d["aie_matrix"])

aie = np.array(all_aie)   # (12, 32)

# Save merged
with open(os.path.join(VECTORS_DIR, "patching_merged.json"), "w") as f:
    json.dump({"layers": all_layers, "aie_matrix": aie.tolist()}, f, indent=2)
print(f"Merged: {aie.shape[0]} layers × {N_HEADS} heads")

# ── Heatmap ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

im = ax.imshow(aie, cmap="RdBu", aspect="auto", vmin=-0.4, vmax=0.4)
plt.colorbar(im, ax=ax, label="AIE  (refusal rate change when head is patched back)")

ax.set_xlabel("Attention Head", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_yticks(range(len(all_layers)))
ax.set_yticklabels([f"L{l}" for l in all_layers], fontsize=9)
ax.set_xticks(range(0, N_HEADS, 2))
ax.set_xticklabels(range(0, N_HEADS, 2), fontsize=9)

ax.set_title(
    "Activation Patching: Average Indirect Effect per Attention Head (Layers 10–21)\n"
    "Blue = head causally contributes to refusal  |  N=5 prompts per head",
    fontsize=12, fontweight="bold"
)

# Highlight top-5 non-zero heads
flat = [(aie[li, h], li, h) for li in range(len(all_layers)) for h in range(N_HEADS)]
flat.sort(reverse=True)
top5 = [(li, h) for val, li, h in flat[:5] if val > 0]
for row, col in top5:
    ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))
    ax.text(col, row, f"{aie[row,col]:.2f}", ha="center", va="center",
            fontsize=7, color="gold", fontweight="bold")

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "activation_patching_merged.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# Print summary
print(f"\nAIE stats: min={aie.min():.3f} max={aie.max():.3f} mean={aie.mean():.4f}")
print(f"Non-zero heads: {(aie != 0).sum()}")
print("\nTop-5 causal heads:")
for val, li, h in flat[:5]:
    if val > 0:
        print(f"  Layer {all_layers[li]}, Head {h}: AIE={val:+.3f}")
