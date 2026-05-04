"""Zero-shot safety probe.

Uses the dot product score h · v_harmful as a lightweight classifier to detect
harmful prompts at layer 16, without running a full safety model.

score = hidden_state_layer16 · v_harmful

If score > threshold tau → classify as harmful
If score <= tau → classify as benign

Generates ROC curve vs ground truth refusal labels.
Compares inference cost vs full forward pass (Llama Guard proxy).

Outputs:
    results/plots/zero_shot_roc.png
    results/vectors/zero_shot_results.json

Run:
    CUDA_VISIBLE_DEVICES=0 python src/zero_shot_probe.py
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
    DATASETS_DIR, VECTORS_DIR, PLOTS_DIR, LAYER, DOMAIN_FILES,
)

sys.stdout.reconfigure(line_buffering=True)
BATCH_SIZE = 8


def get_hidden_state(model, tokenizer, prompt):
    """Get layer-16 hidden state at last token position for a single prompt."""
    text = format_prompt(prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    mask = inputs["attention_mask"][0]
    last_idx = mask.sum().item() - 1
    h = out.hidden_states[LAYER][0, last_idx, :].float().cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h


def get_hidden_states_batch(model, tokenizer, prompts):
    """Batch version for efficiency."""
    texts = [format_prompt(p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hiddens = []
    for i in range(len(prompts)):
        mask = inputs["attention_mask"][i]
        last_idx = mask.sum().item() - 1
        h = out.hidden_states[LAYER][i, last_idx, :].float().cpu().numpy()
        hiddens.append(h)
    del out
    torch.cuda.empty_cache()
    return hiddens


def compute_roc(scores, labels):
    """Compute ROC curve points and AUC."""
    thresholds = np.linspace(scores.min() - 0.01, scores.max() + 0.01, 200)
    tprs, fprs = [], []
    for tau in thresholds:
        pred = (scores >= tau).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tpr = tp / (tp + fn + 1e-10)
        fpr = fp / (fp + tn + 1e-10)
        tprs.append(tpr)
        fprs.append(fpr)
    # AUC via trapezoidal rule
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    sorted_idx = np.argsort(fprs_arr)
    auc = float(np.trapezoid(tprs_arr[sorted_idx], fprs_arr[sorted_idx]))
    return fprs_arr, tprs_arr, thresholds, auc


def find_best_threshold(scores, labels, thresholds):
    """Find threshold that maximizes F1."""
    best_f1, best_tau = 0, thresholds[0]
    for tau in thresholds:
        pred = (scores >= tau).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return best_tau, best_f1


def plot_roc(roc_data, out_path):
    """Plot ROC curves for all domains."""
    fig, axes = plt.subplots(1, len(roc_data), figsize=(5 * len(roc_data), 5))
    if len(roc_data) == 1:
        axes = [axes]

    colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}

    for ax, (domain, res) in zip(axes, roc_data.items()):
        fprs = res["fprs"]
        tprs = res["tprs"]
        auc = res["auc"]
        best_tau = res["best_threshold"]
        best_f1 = res["best_f1"]

        ax.plot(fprs, tprs, color=colors.get(domain, "#888"),
                linewidth=2.5, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")

        # Mark best threshold point
        scores = np.array(res["scores"])
        labels = np.array(res["labels"])
        pred = (scores >= best_tau).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        best_fpr = fp / (fp + tn + 1e-10)
        best_tpr = tp / (tp + fn + 1e-10)
        ax.scatter([best_fpr], [best_tpr], color=colors.get(domain, "#888"),
                   s=100, zorder=5, label=f"Best F1={best_f1:.3f}")

        ax.set_title(f"{domain.capitalize()}\nAUC = {auc:.3f}", fontsize=12,
                     fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        "Zero-Shot Safety Probe: ROC Curves\n"
        "Classifier: score = h · v_harmful  (single dot product at layer 16)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_score_distributions(score_data, out_path):
    """Histogram of scores for refused vs answered prompts."""
    domains = list(score_data.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4))
    if len(domains) == 1:
        axes = [axes]

    colors = {"harmful": "#e74c3c", "medical": "#3498db", "legal": "#2ecc71"}

    for ax, domain in zip(axes, domains):
        res = score_data[domain]
        scores = np.array(res["scores"])
        labels = np.array(res["labels"])

        refused_scores = scores[labels == 1]
        answered_scores = scores[labels == 0]

        bins = np.linspace(scores.min(), scores.max(), 25)
        ax.hist(answered_scores, bins=bins, alpha=0.6, color="steelblue",
                label="Answered", density=True)
        ax.hist(refused_scores, bins=bins, alpha=0.6,
                color=colors.get(domain, "#e74c3c"),
                label="Refused", density=True)

        # Best threshold line
        ax.axvline(res["best_threshold"], color="black", linestyle="--",
                   linewidth=1.5, label=f"τ={res['best_threshold']:.2f}")

        ax.set_title(f"{domain.capitalize()}\nAUC={res['auc']:.3f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Projection Score  h · v_harmful", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Zero-Shot Probe: Score Distributions\n"
        "Separation between refused and answered prompts",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    vectors_dir = VECTORS_DIR
    if "--vectors-dir" in sys.argv:
        idx = sys.argv.index("--vectors-dir")
        vectors_dir = sys.argv[idx + 1]

    # Load v_harmful — the probe direction
    v_harmful = np.load(os.path.join(vectors_dir, "v_harmful.npy"))
    print(f"Loaded v_harmful (dim={v_harmful.shape[0]})")

    model, tokenizer = load_model()

    roc_data = {}
    domains = ["harmful", "medical", "legal"]

    for domain in domains:
        print(f"\n{'='*50}\nDomain: {domain}\n{'='*50}")

        # Load prompts
        with open(os.path.join(DATASETS_DIR, DOMAIN_FILES[domain])) as f:
            entries = json.load(f)
        prompts = [e["prompt"] for e in entries]

        # Extract hidden states + ground truth labels
        all_hiddens = []
        all_labels = []

        print(f"Extracting hidden states + generating responses...")
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            hiddens = get_hidden_states_batch(model, tokenizer, batch)
            all_hiddens.extend(hiddens)

            # Generate responses for labels
            for prompt in batch:
                text = format_prompt(prompt)
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=512).to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=80, do_sample=False)
                response = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
                all_labels.append(1 if is_refusal(response) else 0)
                del gen
            torch.cuda.empty_cache()
            print(f"  {min(i + BATCH_SIZE, len(prompts))}/{len(prompts)} done", end="\r")

        print(f"\n  Labels: {sum(all_labels)} refused / {len(all_labels)-sum(all_labels)} answered")

        # Compute projection scores: h · v_harmful
        scores = np.array([np.dot(h, v_harmful) for h in all_hiddens])
        labels = np.array(all_labels)

        print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"  Refused mean score:  {scores[labels==1].mean():.3f}")
        print(f"  Answered mean score: {scores[labels==0].mean():.3f}")

        # ROC + AUC
        fprs, tprs, thresholds, auc = compute_roc(scores, labels)
        best_tau, best_f1 = find_best_threshold(scores, labels, thresholds)

        print(f"  AUC: {auc:.3f}")
        print(f"  Best threshold tau={best_tau:.3f}, F1={best_f1:.3f}")

        roc_data[domain] = {
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "fprs": fprs.tolist(),
            "tprs": tprs.tolist(),
            "auc": auc,
            "best_threshold": float(best_tau),
            "best_f1": float(best_f1),
        }

    # Save results
    out_json = os.path.join(vectors_dir, "zero_shot_results.json")
    # Save without raw score arrays to keep file small
    save_data = {d: {k: v for k, v in r.items() if k not in ["fprs", "tprs"]}
                 for d, r in roc_data.items()}
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plots
    plot_roc(roc_data, os.path.join(PLOTS_DIR, "zero_shot_roc.png"))
    plot_score_distributions(roc_data, os.path.join(PLOTS_DIR, "zero_shot_distributions.png"))

    # Summary
    print("\n" + "=" * 50)
    print(f"  {'Domain':<10} {'AUC':>6}  {'Best F1':>8}  {'tau':>8}")
    print(f"  {'-'*36}")
    for d, r in roc_data.items():
        print(f"  {d:<10} {r['auc']:>6.3f}  {r['best_f1']:>8.3f}  {r['best_threshold']:>8.3f}")
    print("=" * 50)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
