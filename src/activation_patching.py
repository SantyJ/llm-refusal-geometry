"""Activation patching — causal head identification.

For each (layer, head) pair in layers 10-20, patches the clean run's attention
output into the corrupt run and measures the Average Indirect Effect (AIE):
    AIE = P(refuse|patched) - P(refuse|corrupt)

High AIE = this head is causally responsible for refusal.

Corrupt run: harmful prompt with v_harmful ablated (model answers instead of refusing)
Clean run:   same harmful prompt with no ablation (model refuses)

Generates a heatmap: layers × heads, colored by AIE.

Outputs:
    results/plots/activation_patching.png
    results/vectors/patching_results.json

Run on 4 GPUs:
    CUDA_VISIBLE_DEVICES=0 python src/activation_patching.py --layers 10 13 &
    CUDA_VISIBLE_DEVICES=1 python src/activation_patching.py --layers 13 16 &
    CUDA_VISIBLE_DEVICES=2 python src/activation_patching.py --layers 16 19 &
    CUDA_VISIBLE_DEVICES=3 python src/activation_patching.py --layers 19 22 &
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

N_PROMPTS = 5        # prompts per (layer, head) test
LAYERS_START = 10    # inclusive
LAYERS_END = 22      # exclusive  → layers 10-21
N_HEADS = 32
BATCH_SIZE = 4


def load_prompts(n=N_PROMPTS):
    with open(os.path.join(DATASETS_DIR, DOMAIN_FILES["harmful"])) as f:
        return [e["prompt"] for e in json.load(f)][:n]


def make_ablation_hook(direction, alpha, device):
    """Ablate v_harmful to create 'corrupt' run (model answers harmful)."""
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


def get_refusal_rate(model, tokenizer, prompts, hook=None):
    """Run prompts and return refusal rate, optionally with a hook."""
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


def collect_clean_head_outputs(model, tokenizer, prompts):
    """Run clean forward pass, collect output of each attention head per layer."""
    clean_outputs = {}  # (layer, head) -> list of tensors (one per prompt)

    for i, prompt in enumerate(prompts):
        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(model.device)

        handles = []
        layer_head_outputs = {}

        for layer_idx in range(LAYERS_START, LAYERS_END):
            def make_hook(l):
                def hook_fn(module, input, output):
                    # output[0]: (1, seq, n_heads*head_dim) — capture it
                    if isinstance(output, tuple):
                        layer_head_outputs[l] = output[0].detach().cpu()
                    else:
                        layer_head_outputs[l] = output.detach().cpu()
                return hook_fn
            h = model.model.layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx))
            handles.append(h)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for layer_idx in range(LAYERS_START, LAYERS_END):
            if layer_idx in layer_head_outputs:
                if layer_idx not in clean_outputs:
                    clean_outputs[layer_idx] = []
                clean_outputs[layer_idx].append(layer_head_outputs[layer_idx])

        del inputs
        torch.cuda.empty_cache()

        if (i + 1) % 5 == 0:
            print(f"  Collected clean outputs: {i+1}/{len(prompts)}")

    return clean_outputs


def patch_and_measure(model, tokenizer, prompts, clean_outputs,
                      patch_layer, patch_head, v_harmful, device):
    """
    For a single (layer, head):
    - Run corrupt (ablated) forward pass
    - Patch head output from clean run
    - Measure refusal rate change
    """
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    n_refused_patched = 0
    n_refused_corrupt = 0

    v_dir = torch.tensor(v_harmful, dtype=torch.float32, device=device)

    for i, prompt in enumerate(prompts):
        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(device)
        input_len = inputs["input_ids"].shape[-1]

        clean_attn_out = clean_outputs.get(patch_layer, [None] * len(prompts))[i]

        # --- Corrupt run (ablate v_harmful at LAYER) ---
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].float()
                proj = (h @ v_dir).unsqueeze(-1) * v_dir.unsqueeze(0).unsqueeze(0)
                h = h - 1.0 * proj
                return (h.to(output[0].dtype),) + output[1:]
            else:
                h = output.float()
                proj = (h @ v_dir).unsqueeze(-1) * v_dir.unsqueeze(0).unsqueeze(0)
                h = h - 1.0 * proj
                return h.to(output.dtype)

        h1 = model.model.layers[LAYER].register_forward_hook(ablation_hook)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        h1.remove()
        resp_corrupt = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
        n_refused_corrupt += int(is_refusal(resp_corrupt))
        del gen

        # --- Patched run (ablate v_harmful + patch head output) ---
        if clean_attn_out is not None:
            clean_out_device = clean_attn_out.to(device)

            def patch_hook(module, input, output):
                # Replace this head's output slice with clean version
                # output: (1, seq, hidden_size) — replace head slice
                if isinstance(output, tuple):
                    h = output[0].clone()
                    start = patch_head * head_dim
                    end = start + head_dim
                    seq_len = min(h.shape[1], clean_out_device.shape[1])
                    h[:, :seq_len, start:end] = clean_out_device[:, :seq_len, start:end].to(h.dtype)
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    start = patch_head * head_dim
                    end = start + head_dim
                    seq_len = min(h.shape[1], clean_out_device.shape[1])
                    h[:, :seq_len, start:end] = clean_out_device[:, :seq_len, start:end].to(h.dtype)
                    return h

            h1 = model.model.layers[LAYER].register_forward_hook(ablation_hook)
            h2 = model.model.layers[patch_layer].self_attn.register_forward_hook(patch_hook)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            h1.remove()
            h2.remove()
            resp_patched = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
            n_refused_patched += int(is_refusal(resp_patched))
            del gen

        torch.cuda.empty_cache()

    refusal_corrupt = n_refused_corrupt / len(prompts)
    refusal_patched = n_refused_patched / len(prompts)
    aie = refusal_patched - refusal_corrupt
    return aie, refusal_corrupt, refusal_patched


def plot_heatmap(aie_matrix, layers, out_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(aie_matrix, cmap="RdBu", aspect="auto",
                   vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="AIE (refusal rate change)")

    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xticks(range(0, N_HEADS, 2))
    ax.set_xticklabels(range(0, N_HEADS, 2), fontsize=9)

    ax.set_title(
        "Activation Patching: Average Indirect Effect per Attention Head\n"
        "(Blue = head causally contributes to refusal; Red = suppresses refusal)",
        fontsize=12, fontweight="bold"
    )

    # Highlight top-5 heads
    flat_idx = np.argsort(aie_matrix.flatten())[-5:]
    for idx in flat_idx:
        row = idx // N_HEADS
        col = idx % N_HEADS
        ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                   fill=False, edgecolor="gold", linewidth=2))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    t0 = time.time()

    # Parse layer range from args
    layers_start = LAYERS_START
    layers_end = LAYERS_END
    if "--layers" in sys.argv:
        idx = sys.argv.index("--layers")
        layers_start = int(sys.argv[idx + 1])
        layers_end = int(sys.argv[idx + 2])

    layers = list(range(layers_start, layers_end))
    print(f"Activation Patching: layers {layers_start}-{layers_end-1}, {N_HEADS} heads, {N_PROMPTS} prompts")

    v_harmful = np.load(os.path.join(VECTORS_DIR, "v_harmful.npy"))
    prompts = load_prompts()

    model, tokenizer = load_model()
    device = model.device

    # Collect clean head outputs first
    print("\nCollecting clean attention outputs...")
    clean_outputs = collect_clean_head_outputs(model, tokenizer, prompts)

    # AIE matrix: rows=layers, cols=heads
    aie_matrix = np.zeros((len(layers), N_HEADS))

    print(f"\nPatching {len(layers)} layers × {N_HEADS} heads...")
    for li, layer in enumerate(layers):
        print(f"\n  Layer {layer}:")
        for head in range(N_HEADS):
            aie, r_corrupt, r_patched = patch_and_measure(
                model, tokenizer, prompts, clean_outputs,
                layer, head, v_harmful, device
            )
            aie_matrix[li, head] = aie
            if head % 8 == 0:
                print(f"    head {head:2d}: corrupt={r_corrupt:.2f} patched={r_patched:.2f} AIE={aie:+.3f}")

        # Save after each layer in case of crash
        out_json = os.path.join(VECTORS_DIR, f"patching_layer_{layers_start}_{layers_end}.json")
        with open(out_json, "w") as f:
            json.dump({
                "layers": layers[:li+1],
                "aie_matrix": aie_matrix[:li+1].tolist(),
            }, f)

    # Save final
    out_json = os.path.join(VECTORS_DIR, f"patching_layer_{layers_start}_{layers_end}.json")
    with open(out_json, "w") as f:
        json.dump({"layers": layers, "aie_matrix": aie_matrix.tolist()}, f, indent=2)
    print(f"\nSaved {out_json}")

    # Plot heatmap
    plot_heatmap(aie_matrix, layers,
                 os.path.join(PLOTS_DIR, f"activation_patching_{layers_start}_{layers_end}.png"))

    # Top heads
    print("\nTop-5 causal heads (highest AIE):")
    flat = [(aie_matrix[li, h], layers[li], h)
            for li in range(len(layers)) for h in range(N_HEADS)]
    flat.sort(reverse=True)
    for aie, layer, head in flat[:5]:
        print(f"  Layer {layer}, Head {head}: AIE={aie:+.3f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
