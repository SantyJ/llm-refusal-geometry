"""Cross-domain refusal transfer experiment.

Tests whether injecting v_harmful into medical/legal hidden states transfers
refusal intent across domains — and vice versa.

If transfer is near zero → directions are domain-locked (supports independence claim).
If transfer is significant → partial overlap in refusal mechanisms.

Transfers tested:
  v_harmful → medical prompts  (does model start treating medical as harmful?)
  v_harmful → legal prompts    (does model start treating legal as harmful?)
  v_medical → harmful prompts  (does model start treating harmful as medical?)
  v_legal   → harmful prompts  (does model start treating harmful as legal?)

Math:
  h_modified = h + alpha * v_source
  Transfer Effectiveness = refusal_rate(modified) - refusal_rate(baseline)

Outputs:
    results/plots/cross_domain_transfer.png
    results/vectors/cross_domain_transfer.json

Run:
    CUDA_VISIBLE_DEVICES=0 python src/cross_domain_transfer.py
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

TRANSFER_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
BATCH_SIZE = 4
N_PROMPTS = 20


def load_prompts(domain, n=N_PROMPTS):
    with open(os.path.join(DATASETS_DIR, DOMAIN_FILES[domain])) as f:
        return [e["prompt"] for e in json.load(f)][:n]


def make_injection_hook(direction, alpha, device):
    dir_t = torch.tensor(direction, dtype=torch.float32, device=device)
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].float()
            h = h + alpha * dir_t.unsqueeze(0).unsqueeze(0)
            return (h.to(output[0].dtype),) + output[1:]
        else:
            h = output.float()
            h = h + alpha * dir_t.unsqueeze(0).unsqueeze(0)
            return h.to(output.dtype)
    return hook_fn


def run_prompts(model, tokenizer, prompts, hook=None):
    n_refused = 0
    handle = None
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        texts = [format_prompt(p) for p in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        if hook is not None:
            handle = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        if handle is not None:
            handle.remove()
        for j in range(len(batch)):
            input_len = inputs["attention_mask"][j].sum().item()
            response = tokenizer.decode(gen[j][input_len:], skip_special_tokens=True)
            if is_refusal(response):
                n_refused += 1
        del gen
        torch.cuda.empty_cache()
    return n_refused / len(prompts)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load vectors
    v_harmful = np.load(os.path.join(VECTORS_DIR, "v_harmful.npy"))
    v_medical = np.load(os.path.join(VECTORS_DIR, "v_medical.npy"))
    v_legal   = np.load(os.path.join(VECTORS_DIR, "v_legal.npy"))

    # Load prompts
    harmful_prompts = load_prompts("harmful")
    medical_prompts = load_prompts("medical")
    legal_prompts   = load_prompts("legal")

    model, tokenizer = load_model()

    # Define transfer experiments:
    # (source_vector, source_name, target_prompts, target_name)
    transfers = [
        (v_harmful, "harmful", medical_prompts, "medical"),
        (v_harmful, "harmful", legal_prompts,   "legal"),
        (v_medical, "medical", harmful_prompts, "harmful"),
        (v_legal,   "legal",   harmful_prompts, "harmful"),
    ]

    results = {}

    for vec, src_name, prompts, tgt_name in transfers:
        key = f"v_{src_name}→{tgt_name}"
        print(f"\n{'='*60}\nTransfer: {key}\n{'='*60}")
        sweep = []

        for alpha in TRANSFER_ALPHAS:
            hook = make_injection_hook(vec, alpha, model.device) if alpha > 0 else None
            rate = run_prompts(model, tokenizer, prompts, hook)
            sweep.append({"alpha": alpha, "refusal_rate": round(rate * 100, 1)})
            print(f"  alpha={alpha:.1f}: refusal_rate={rate*100:.1f}%")

        baseline = sweep[0]["refusal_rate"]
        max_rate  = max(r["refusal_rate"] for r in sweep)
        te = max_rate - baseline
        print(f"  Transfer Effectiveness: {te:+.1f}% (baseline={baseline:.1f}%, max={max_rate:.1f}%)")
        results[key] = {"sweep": sweep, "baseline": baseline, "transfer_effectiveness": te}

    # Save
    out_json = os.path.join(VECTORS_DIR, "cross_domain_transfer.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    colors = {
        "v_harmful→medical": "#3498db",
        "v_harmful→legal":   "#2ecc71",
        "v_medical→harmful": "#e74c3c",
        "v_legal→harmful":   "#9b59b6",
    }

    for ax, (key, res) in zip(axes, results.items()):
        alphas  = [r["alpha"]       for r in res["sweep"]]
        rates   = [r["refusal_rate"] for r in res["sweep"]]
        baseline = res["baseline"]
        te = res["transfer_effectiveness"]
        color = colors.get(key, "#888")

        ax.plot(alphas, rates, "o-", color=color, linewidth=2.5, markersize=7)
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.5,
                   alpha=0.6, label=f"Baseline ({baseline:.0f}%)")
        ax.fill_between(alphas, baseline, rates,
                        where=[r >= baseline for r in rates],
                        alpha=0.15, color=color)

        parts = key.replace("v_", "").split("→")
        ax.set_title(f"Inject v_{parts[0]} → {parts[1]} prompts\n"
                     f"Transfer Effectiveness: {te:+.1f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Alpha (injection strength)", fontsize=10)
        ax.set_ylabel("Refusal Rate (%)", fontsize=10)
        ax.set_xticks(TRANSFER_ALPHAS)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Cross-Domain Refusal Transfer\n"
        "Near-zero transfer → directions are domain-locked and independent",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out_plot = os.path.join(PLOTS_DIR, "cross_domain_transfer.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_plot}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"  {'Transfer':<25} {'Baseline':>9}  {'Max':>6}  {'TE':>6}")
    print(f"  {'-'*53}")
    for key, res in results.items():
        print(f"  {key:<25} {res['baseline']:>8.1f}%  "
              f"{max(r['refusal_rate'] for r in res['sweep']):>5.1f}%  "
              f"{res['transfer_effectiveness']:>+5.1f}%")
    print("=" * 55)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
