"""Subspace ablation: compare single-vector vs full-subspace projection on medical/legal.

This is the core fix for the concept erasure problem found in the midterm:
  - Single vector ablation INCREASES medical/legal refusal (concept erasure)
  - Subspace ablation (project out top-k directions) should REDUCE it

Hook:
  Single vector:   x' = x - α * (x·v) * v
  Subspace:        x' = x - α * U @ U.T @ x    (U shape: 4096 × k)

Outputs:
  results/vectors/subspace_ablation_results.json
  results/plots/subspace_vs_single.png   -- refusal rate comparison
  results/plots/alpha_sweep.png          -- fine alpha sweep (0→1 in 0.1 steps)
  results/plots/pareto.png               -- perplexity vs refusal rate

Run:
  CUDA_VISIBLE_DEVICES=0 python src/subspace_ablation.py
  CUDA_VISIBLE_DEVICES=1 python src/subspace_ablation.py --domain medical
  CUDA_VISIBLE_DEVICES=2 python src/subspace_ablation.py --domain legal
"""

import json
import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/src")
from utils import (
    load_model, is_refusal, format_prompt,
    DATASETS_DIR, VECTORS_DIR, PLOTS_DIR, LAYER, MAX_NEW_TOKENS, DOMAIN_FILES,
)

sys.stdout.reconfigure(line_buffering=True)

ALPHAS = [round(a * 0.1, 1) for a in range(11)]   # 0.0, 0.1, ..., 1.0
NUM_PROMPTS = 50
BATCH_SIZE = 4

# Fixed sentences for perplexity measurement (neutral, unrelated to refusal)
PERPLEXITY_SENTENCES = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The mitochondria is the powerhouse of the cell.",
    "Shakespeare wrote Hamlet in approximately 1600.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The human genome contains approximately 3 billion base pairs.",
    "Newton's first law states that an object at rest stays at rest.",
    "The Amazon River is the largest river by discharge volume.",
    "DNA stands for deoxyribonucleic acid.",
]


def load_vectors_and_subspaces(vectors_dir=VECTORS_DIR):
    vectors, subspaces = {}, {}
    for name in ["harmful", "medical", "legal"]:
        v_path = os.path.join(vectors_dir, f"v_{name}.npy")
        if os.path.exists(v_path):
            vectors[name] = np.load(v_path)

    sub_path = os.path.join(vectors_dir, "subspaces.npz")
    if os.path.exists(sub_path):
        data = np.load(sub_path)
        for name in ["harmful", "medical", "legal"]:
            key = f"U_{name}"
            if key in data:
                subspaces[name] = data[key]   # shape: (k, 4096)
    else:
        print(f"WARNING: {sub_path} not found. Run svd_analysis.py first.")
    return vectors, subspaces


def load_prompts(domain, n=NUM_PROMPTS):
    with open(os.path.join(DATASETS_DIR, DOMAIN_FILES[domain])) as f:
        return [e["prompt"] for e in json.load(f)][:n]


def make_single_hook(direction, alpha, device):
    """Hook that projects out a single vector: x' = x - α*(x·v)*v"""
    dir_t = torch.tensor(direction, dtype=torch.float32, device=device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            proj = (h @ dir_t).unsqueeze(-1) * dir_t.unsqueeze(0).unsqueeze(0)
            h = h - alpha * proj
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            proj = (h @ dir_t).unsqueeze(-1) * dir_t.unsqueeze(0).unsqueeze(0)
            h = h - alpha * proj
            return h.to(output.dtype)

    return hook_fn


def make_subspace_hook(U_rows, alpha, device):
    """Hook that projects out a k-dim subspace: x' = x - α * U @ U.T @ x
    U_rows: numpy (k, 4096) — rows are orthonormal basis vectors
    """
    U = torch.tensor(U_rows.T, dtype=torch.float32, device=device)  # (4096, k)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            proj = h @ U @ U.T
            h = h - alpha * proj
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            proj = h @ U @ U.T
            h = h - alpha * proj
            return h.to(output.dtype)

    return hook_fn


def run_batch(model, tokenizer, prompts, hook_fn=None):
    """Generate responses, optionally with a forward hook."""
    texts = [format_prompt(p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(model.device)
    handle = None
    if hook_fn is not None:
        handle = model.model.layers[LAYER].register_forward_hook(hook_fn)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    if handle is not None:
        handle.remove()

    results = []
    for i, prompt in enumerate(prompts):
        input_len = inputs["attention_mask"][i].sum().item()
        response = tokenizer.decode(gen[i][input_len:], skip_special_tokens=True)
        results.append((response, is_refusal(response)))

    del gen
    torch.cuda.empty_cache()
    return results


def compute_perplexity(model, tokenizer, sentences, hook_fn=None):
    """Average token-level NLL loss over a fixed sentence set."""
    total_loss, total_tokens = 0.0, 0
    handle = None
    if hook_fn is not None:
        handle = model.model.layers[LAYER].register_forward_hook(hook_fn)

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        n_tok = inputs["input_ids"].shape[-1]
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok

    if handle is not None:
        handle.remove()

    return float(np.exp(total_loss / total_tokens))


def sweep_domain(model, tokenizer, domain, vectors, subspaces):
    """Run full alpha sweep for a single domain. Returns dict of results."""
    prompts = load_prompts(domain)
    device = model.device
    results = {}

    for method in ["none", "single", "subspace"]:
        for alpha in ALPHAS:
            if method == "none" and alpha > 0:
                continue   # baseline only once

            key = f"{method}_alpha_{alpha}"
            print(f"\n  [{domain}] method={method} alpha={alpha}")

            # Build hook
            if method == "none" or alpha == 0.0:
                hook = None
            elif method == "single":
                if domain not in vectors:
                    print(f"    SKIP: no single vector for {domain}")
                    continue
                hook = make_single_hook(vectors[domain], alpha, device)
            else:  # subspace
                if domain not in subspaces:
                    print(f"    SKIP: no subspace for {domain}")
                    continue
                hook = make_subspace_hook(subspaces[domain], alpha, device)

            # Refusal rate
            n_refused = 0
            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                batch_results = run_batch(model, tokenizer, batch, hook)
                n_refused += sum(r for _, r in batch_results)
                print(f"    {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} done", end="\r")

            refusal_rate = n_refused / len(prompts) * 100

            # Perplexity
            ppl = compute_perplexity(model, tokenizer, PERPLEXITY_SENTENCES, hook)

            results[key] = {"refusal_rate": refusal_rate, "perplexity": ppl}
            print(f"    refusal={refusal_rate:.1f}%  perplexity={ppl:.2f}")

    return results


def plot_comparison(all_results, domains, out_path):
    """Side-by-side: single vector vs subspace for each domain."""
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5), sharey=False)
    if len(domains) == 1:
        axes = [axes]

    colors = {"single": "#e74c3c", "subspace": "#3498db"}

    for ax, domain in zip(axes, domains):
        res = all_results.get(domain, {})
        for method in ["single", "subspace"]:
            rates = []
            for alpha in ALPHAS:
                key = f"{method}_alpha_{alpha}"
                if alpha == 0.0:
                    key = "none_alpha_0.0"
                entry = res.get(key, {})
                rates.append(entry.get("refusal_rate", None))

            valid = [(a, r) for a, r in zip(ALPHAS, rates) if r is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, "o-", color=colors[method], linewidth=2,
                        markersize=5, label=method)

        baseline = res.get("none_alpha_0.0", {}).get("refusal_rate")
        if baseline is not None:
            ax.axhline(baseline, color="gray", linestyle=":", alpha=0.6, label="baseline")

        ax.set_title(f"{domain.capitalize()} Domain", fontsize=12, fontweight="bold")
        ax.set_xlabel("Alpha", fontsize=11)
        ax.set_ylabel("Refusal Rate (%)", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Subspace Ablation vs Single-Vector Ablation\n"
        "(Llama-2-7b-chat, Layer 16 — fine alpha sweep)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_pareto(all_results, domains, out_path):
    """Perplexity vs refusal rate (Pareto frontier)."""
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5))
    if len(domains) == 1:
        axes = [axes]

    colors = {"single": "#e74c3c", "subspace": "#3498db"}
    markers = {"single": "^", "subspace": "o"}

    for ax, domain in zip(axes, domains):
        res = all_results.get(domain, {})
        for method in ["single", "subspace"]:
            xs, ys = [], []
            for alpha in ALPHAS:
                key = f"{method}_alpha_{alpha}"
                if alpha == 0.0:
                    key = "none_alpha_0.0"
                entry = res.get(key, {})
                rr = entry.get("refusal_rate")
                ppl = entry.get("perplexity")
                if rr is not None and ppl is not None:
                    xs.append(rr)
                    ys.append(ppl)
            if xs:
                ax.scatter(xs, ys, color=colors[method], marker=markers[method],
                           s=60, label=method, zorder=3)
                ax.plot(xs, ys, color=colors[method], alpha=0.4, linewidth=1)

                # Annotate alpha values
                for a, x, y in zip(ALPHAS, xs, ys):
                    if a in [0.0, 0.5, 1.0]:
                        ax.annotate(f"α={a}", (x, y), textcoords="offset points",
                                    xytext=(5, 5), fontsize=7)

        ax.set_title(f"{domain.capitalize()} — Pareto Frontier", fontsize=11,
                     fontweight="bold")
        ax.set_xlabel("Refusal Rate (%)", fontsize=10)
        ax.set_ylabel("Perplexity", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Pareto Frontier: Refusal Rate vs Language Quality\n"
        "(lower-left = better; subspace ablation should be closer to ideal corner)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def find_optimal_alpha(results):
    """alpha* = argmax(baseline_refusal_drop / (perplexity_increase + ε))."""
    baseline_rr = results.get("none_alpha_0.0", {}).get("refusal_rate", 100)
    baseline_ppl = results.get("none_alpha_0.0", {}).get("perplexity", 10)

    best_alpha, best_score = None, -1
    for alpha in ALPHAS[1:]:
        for method in ["subspace", "single"]:
            key = f"{method}_alpha_{alpha}"
            entry = results.get(key, {})
            rr = entry.get("refusal_rate")
            ppl = entry.get("perplexity")
            if rr is None or ppl is None:
                continue
            refusal_drop = max(0, baseline_rr - rr)
            ppl_increase = max(0, ppl - baseline_ppl)
            score = refusal_drop / (ppl_increase + 1e-3)
            if score > best_score:
                best_score = score
                best_alpha = (method, alpha)

    return best_alpha, best_score


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    vectors_dir = VECTORS_DIR
    target_domains = ["medical", "legal"]  # where concept erasure happened

    if "--vectors-dir" in sys.argv:
        idx = sys.argv.index("--vectors-dir")
        vectors_dir = sys.argv[idx + 1]
    if "--domain" in sys.argv:
        idx = sys.argv.index("--domain")
        target_domains = [sys.argv[idx + 1]]
    if "--all-domains" in sys.argv:
        target_domains = ["harmful", "medical", "legal"]

    vectors, subspaces = load_vectors_and_subspaces(vectors_dir)
    print(f"Loaded vectors: {list(vectors.keys())}")
    print(f"Loaded subspaces: {list(subspaces.keys())}")

    model, tokenizer = load_model()

    all_results = {}
    for domain in target_domains:
        print(f"\n{'='*60}\nDomain: {domain}\n{'='*60}")
        all_results[domain] = sweep_domain(model, tokenizer, domain, vectors, subspaces)

        # Save after each domain in case of crash
        out_path = os.path.join(vectors_dir, "subspace_ablation_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved intermediate results → {out_path}")

        opt = find_optimal_alpha(all_results[domain])
        print(f"Optimal alpha for {domain}: {opt[0]} (score={opt[1]:.2f})")

    # Plots
    plot_comparison(all_results, target_domains,
                    os.path.join(PLOTS_DIR, "subspace_vs_single.png"))
    plot_pareto(all_results, target_domains,
                os.path.join(PLOTS_DIR, "pareto.png"))

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print(f"Results → {os.path.join(vectors_dir, 'subspace_ablation_results.json')}")


if __name__ == "__main__":
    main()
