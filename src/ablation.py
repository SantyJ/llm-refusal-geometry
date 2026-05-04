"""Run ablation: subtract direction vectors at alpha 0, 0.5, 1.0 and measure refusal rates."""

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
    DATASETS_DIR, VECTORS_DIR, PLOTS_DIR, LAYER, MAX_NEW_TOKENS,
)

sys.stdout.reconfigure(line_buffering=True)
ALPHAS = [0.0, 0.5, 1.0]
BATCH_SIZE = 8
NUM_PROMPTS = 50


def load_vectors(vectors_dir=None):
    vdir = vectors_dir or VECTORS_DIR
    vectors = {}
    for name in ["harmful", "medical", "legal"]:
        path = os.path.join(vdir, f"v_{name}.npy")
        if os.path.exists(path):
            vectors[name] = np.load(path)
    return vectors


def load_harmful_prompts():
    with open(os.path.join(DATASETS_DIR, "harmful_real.json")) as f:
        return [e["prompt"] for e in json.load(f)][:NUM_PROMPTS]


def load_domain_prompts(domain):
    from utils import DOMAIN_FILES
    with open(os.path.join(DATASETS_DIR, DOMAIN_FILES[domain])) as f:
        return [e["prompt"] for e in json.load(f)][:NUM_PROMPTS]


def generate_batch(model, tokenizer, prompts, direction=None, alpha=0.0):
    """Generate responses for a batch of prompts, optionally with ablation hook."""
    texts = [format_prompt(p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(model.device)

    handle = None
    if direction is not None and alpha > 0:
        dir_t = torch.tensor(direction, dtype=torch.float32, device=model.device)

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

        handle = model.model.layers[LAYER].register_forward_hook(hook_fn)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    if handle is not None:
        handle.remove()

    results = []
    for i in range(len(prompts)):
        input_len = inputs["attention_mask"][i].sum().item()
        response = tokenizer.decode(gen[i][input_len:], skip_special_tokens=True)
        results.append((response, is_refusal(response)))

    del gen
    torch.cuda.empty_cache()
    return results


def run_ablation(model, tokenizer, vectors, prompts):
    """Run all (vector, alpha) combos on harmful prompts using batch processing."""
    results = {}

    for domain, vec in vectors.items():
        for alpha in ALPHAS:
            n_refused = 0
            print(f"\nv_{domain}, alpha={alpha}")

            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                if alpha == 0.0:
                    batch_results = generate_batch(model, tokenizer, batch)
                else:
                    batch_results = generate_batch(model, tokenizer, batch, vec, alpha)

                for j, (resp, refused) in enumerate(batch_results):
                    idx = i + j
                    if refused:
                        n_refused += 1
                    if idx < 2 or (idx + 1) % 25 == 0:
                        label = "REF" if refused else "ANS"
                        print(f"  [{idx+1:3d}/{len(prompts)}] {label} | {prompts[idx][:50]}...")

                print(f"  Batch done: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}")

            rate = n_refused / len(prompts) * 100
            results[(domain, alpha)] = rate
            print(f"  Refusal rate: {rate:.1f}%")

    return results


def plot_refusal_rates(results, vectors):
    domains = list(vectors.keys())
    x = np.arange(len(ALPHAS))
    width = 0.8 / len(domains)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, domain in enumerate(domains):
        rates = [results.get((domain, a), 0) for a in ALPHAS]
        offset = (i - len(domains) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=f"v_{domain}", color=colors[i % 3])
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("Refusal Rate on Harmful Prompts (%)", fontsize=12)
    ax.set_title("Refusal Rate vs Ablation Strength\n(Llama-2-7b-chat, Layer 16)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "refusal_rates.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {path}")


def run_by_domain(model, tokenizer, vectors):
    """Ablate each vector on its OWN domain's prompts."""
    results = {}

    for domain, vec in vectors.items():
        prompts = load_domain_prompts(domain)
        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        for alpha in ALPHAS:
            n_refused = 0
            print(f"\nv_{domain}, alpha={alpha}, tested on {domain} prompts")

            for i in range(0, len(prompts), BATCH_SIZE):
                batch = prompts[i:i + BATCH_SIZE]
                if alpha == 0.0:
                    batch_results = generate_batch(model, tokenizer, batch)
                else:
                    batch_results = generate_batch(model, tokenizer, batch, vec, alpha)

                for j, (resp, refused) in enumerate(batch_results):
                    idx = i + j
                    if refused:
                        n_refused += 1
                    if idx < 2 or (idx + 1) % 25 == 0:
                        label = "REF" if refused else "ANS"
                        print(f"  [{idx+1:3d}/{len(prompts)}] {label} | {prompts[idx][:50]}...")

                print(f"  Batch done: {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)}")

            rate = n_refused / len(prompts) * 100
            results[(domain, alpha)] = rate
            print(f"  Refusal rate: {rate:.1f}%")

    return results


def plot_by_domain(results, plot_name="refusal_rates_by_domain.png"):
    domains = ["harmful", "medical", "legal"]
    x = np.arange(len(ALPHAS))
    width = 0.8 / len(domains)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, domain in enumerate(domains):
        rates = [results.get((domain, a), 0) for a in ALPHAS]
        offset = (i - len(domains) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=f"v_{domain} on {domain}", color=colors[i])
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Alpha", fontsize=12)
    ax.set_ylabel("Refusal Rate (%)", fontsize=12)
    ax.set_title("Refusal Rate: Each Vector Ablated on Its Own Domain\n"
                 "(Llama-2-7b-chat, Layer 16)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, plot_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    vectors_dir = None
    plot_name = "refusal_rates_by_domain.png"
    if "--vectors-dir" in sys.argv:
        idx = sys.argv.index("--vectors-dir")
        vectors_dir = sys.argv[idx + 1]
    if "--plot-name" in sys.argv:
        idx = sys.argv.index("--plot-name")
        plot_name = sys.argv[idx + 1]

    vectors = load_vectors(vectors_dir)
    if not vectors:
        print("No vectors found. Run extract_directions.py first.")
        return
    print(f"Loaded vectors: {list(vectors.keys())} from {vectors_dir or VECTORS_DIR}")

    mode = "by-domain" if "--by-domain" in sys.argv else "harmful"

    if mode == "by-domain":
        print(f"Mode: by-domain (batch_size={BATCH_SIZE}, {NUM_PROMPTS} prompts per domain)")
        model, tokenizer = load_model()
        results = run_by_domain(model, tokenizer, vectors)

        out_dir = vectors_dir or VECTORS_DIR
        save = {f"v_{d}_on_{d}_alpha_{a}": r for (d, a), r in results.items()}
        with open(os.path.join(out_dir, "ablation_by_domain.json"), "w") as f:
            json.dump(save, f, indent=2)

        plot_by_domain(results, plot_name)
    else:
        prompts = load_harmful_prompts()
        print(f"Mode: harmful-only ({len(prompts)} prompts, batch_size={BATCH_SIZE})")
        model, tokenizer = load_model()
        results = run_ablation(model, tokenizer, vectors, prompts)

        save = {f"v_{d}_alpha_{a}": r for (d, a), r in results.items()}
        with open(os.path.join(VECTORS_DIR, "ablation_results.json"), "w") as f:
            json.dump(save, f, indent=2)

        plot_refusal_rates(results, vectors)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
