"""Extract refusal direction vectors from layer 16 hidden states."""

import json
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_model, is_refusal, format_prompt,
    DATASETS_DIR, VECTORS_DIR, DOMAIN_FILES, LAYER, MAX_NEW_TOKENS,
)

sys.stdout.reconfigure(line_buffering=True)
BATCH_SIZE = 8

REFUSAL_MARKERS = [
    "I cannot", "I can't", "I'm sorry", "I'm unable", "I won't", "I apologize",
]


def classify_response(text):
    """Classify response as 'refused' or 'answered'.

    Refusal: contains at least 1 refusal marker.
    Answer: contains none.
    """
    text_lower = text.lower()
    has_marker = any(phrase.lower() in text_lower for phrase in REFUSAL_MARKERS)
    return "refused" if has_marker else "answered"


def get_hidden_states_batch(model, tokenizer, prompts):
    """Forward pass on a batch, return layer-16 hidden state per prompt."""
    texts = [format_prompt(p) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=512).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    # For each prompt, grab the last non-pad token's hidden state at LAYER
    hiddens = []
    for i in range(len(prompts)):
        mask = inputs["attention_mask"][i]
        last_idx = mask.sum().item() - 1
        h = out.hidden_states[LAYER][i, last_idx, :].float().cpu().numpy()
        hiddens.append(h)

    del out
    torch.cuda.empty_cache()
    return hiddens


def generate_responses(model, tokenizer, prompts):
    """Generate responses one at a time, return (response, refused) pairs."""
    results = []
    for prompt in prompts:
        text = format_prompt(prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        response = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
        results.append((response, is_refusal(response)))
        del gen
    torch.cuda.empty_cache()
    return results


def process_domain(model, tokenizer, domain, filepath, use_filter=False):
    """Run all prompts for one domain. Return refused/answered hidden states."""
    with open(filepath) as f:
        prompts = [e["prompt"] for e in json.load(f)]

    print(f"\n{'='*60}")
    print(f"{domain} ({len(prompts)} prompts, filter={'ON' if use_filter else 'OFF'})")
    print(f"{'='*60}")

    # Collect hidden states in batches
    all_hiddens = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        hiddens = get_hidden_states_batch(model, tokenizer, batch)
        all_hiddens.extend(hiddens)
        print(f"  Hidden states: {i+len(batch)}/{len(prompts)}")

    # Generate responses one by one
    all_responses = []
    for i, prompt in enumerate(prompts):
        resp, refused = generate_responses(model, tokenizer, [prompt])[0]
        all_responses.append((resp, refused))
        label = "REFUSED" if refused else "ANSWERED"
        if i < 3 or (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{len(prompts)}] {label:8s} | {prompt[:50]}...")
            print(f"           {resp[:70]}...")

    if use_filter:
        # Apply confidence filter
        refused_h, answered_h = [], []
        for i, (resp, _) in enumerate(all_responses):
            label = classify_response(resp)
            if label == "refused":
                refused_h.append(all_hiddens[i])
            else:
                answered_h.append(all_hiddens[i])
        print(f"  Filter: {len(refused_h)} refused / {len(answered_h)} answered (100% kept)")
    else:
        # Original behavior
        refused_h = [all_hiddens[i] for i, (_, r) in enumerate(all_responses) if r]
        answered_h = [all_hiddens[i] for i, (_, r) in enumerate(all_responses) if not r]
        print(f"  Result: {len(refused_h)} refused / {len(answered_h)} answered")

    return refused_h, answered_h


def compute_direction(refused_h, answered_h):
    """Normalized mean difference: refused - answered."""
    if not refused_h or not answered_h:
        return None
    diff = np.mean(refused_h, axis=0) - np.mean(answered_h, axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-10:
        return None
    return diff / norm


def main():
    use_filter = "--filtered" in sys.argv
    save_hiddens = "--save-hiddens" in sys.argv
    filtered_domains = None
    if "--domains" in sys.argv:
        idx = sys.argv.index("--domains")
        filtered_domains = sys.argv[idx + 1].split(",")

    if use_filter:
        import os; out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "vectors_filtered")
    else:
        out_dir = VECTORS_DIR

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model, tokenizer = load_model()

    vectors = {}
    stats = {}

    domains_to_run = {k: v for k, v in DOMAIN_FILES.items()
                      if filtered_domains is None or k in filtered_domains}

    for domain, filename in domains_to_run.items():
        filepath = os.path.join(DATASETS_DIR, filename)
        refused_h, answered_h = process_domain(model, tokenizer, domain, filepath,
                                               use_filter=use_filter)
        stats[domain] = {"refused": len(refused_h), "answered": len(answered_h)}

        vec = compute_direction(refused_h, answered_h)
        if vec is not None:
            vectors[domain] = vec
            np.save(os.path.join(out_dir, f"v_{domain}.npy"), vec)
            print(f"  Saved v_{domain}.npy (dim={vec.shape[0]})")
        else:
            print(f"  WARNING: no vector for {domain} (need both refused + answered)")

        if save_hiddens and refused_h and answered_h:
            np.savez(
                os.path.join(out_dir, f"hiddens_{domain}.npz"),
                refused=np.array(refused_h),
                answered=np.array(answered_h),
            )
            print(f"  Saved hiddens_{domain}.npz "
                  f"({len(refused_h)} refused, {len(answered_h)} answered)")

    # Save stats
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for d, s in stats.items():
        total = s["refused"] + s["answered"]
        pct = s["refused"] / total * 100 if total > 0 else 0
        print(f"  {d:10s}: {s['refused']:3d} refused / {s['answered']:3d} answered ({pct:.1f}%)")
    print(f"  Vectors saved to: {out_dir}")
    print(f"  Vectors: {list(vectors.keys())}")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
