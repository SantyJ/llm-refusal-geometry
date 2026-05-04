# LLM Refusal Geometry

**Mechanistic Interpretability of LLM Refusal Behavior — Domain-Specific Subspace Analysis**

> CS 16:198:671 — Interpretable & Explainable AI, Rutgers University  
> Santhosh Janakiraman (sj1230) & Harsha Rajendra (hr458)

---

## Overview

This project studies whether LLM refusal behavior decomposes into **domain-specific directions** in hidden state space, and whether over-refusal on medical and legal queries can be surgically reduced without compromising safety.

We find that:
- **Harmful refusal** is low-dimensional (effective rank ≈ 1) — a single direction captures 98.6% of variance
- **Medical/legal refusal** is high-dimensional (effective rank ≈ 26–31) — entangled with domain knowledge across many dimensions
- **Subspace ablation** (projecting out top-8 SVD directions) fixes over-refusal without concept erasure
- **Harmful refusal is structurally protected** — forcing 0% refusal causes perplexity to explode to 1,600–5,200 (model breakdown), while medical/legal can be fixed with PPL ≈ 12
- All three domain directions are **geometrically and causally independent** (cosine sim ≈ 0, cross-domain transfer = 0%)

---

## Key Results

| Experiment | Finding |
|-----------|---------|
| Effective rank (SVD) | Harmful=1.12, Medical=31.35, Legal=26.14 |
| Principal angles (Grassmann) | Mean angles 63–72° — subspaces substantially independent |
| Single vector ablation (medical) | Refusal **increases** 48%→60% at α=1.0 (concept erasure) |
| Subspace ablation (medical) | Refusal **decreases** 48%→28% at α=1.0, PPL=12.77 |
| Subspace ablation (harmful) | 0% refusal only at PPL=5,265 (model breakdown) |
| Bootstrap stability | Medical/legal mean cosine sim ≈ 0.92–0.93 (std ≈ 0.02) |
| Cross-domain transfer | Transfer Effectiveness = 0% across all domain pairs |
| Activation patching | Causal heads concentrated in layers 19–21 |

---

## Model & Infrastructure

- **Model**: `meta-llama/Llama-2-7b-chat-hf` (4-bit NF4 quantized via BitsAndBytes)
- **Layer**: Hidden states extracted at layer 16 (4096-dimensional)
- **Hardware**: 4× NVIDIA RTX 4090 (24GB each)
- **Tools**: PyTorch, HuggingFace Transformers, NumPy, Matplotlib

---

## Repository Structure

```
├── datasets/
│   ├── harmful_real.json       # 100 prompts from JailbreakBench
│   ├── medical_real.json       # 100 custom medical over-refusal prompts
│   └── legal_real.json         # 100 custom legal over-refusal prompts
│
├── src/
│   ├── utils.py                # Model loading, helpers, shared constants
│   ├── extract_directions.py   # Mean-difference direction extraction
│   ├── svd_analysis.py         # SVD, effective rank, principal angles
│   ├── subspace_ablation.py    # Subspace vs single vector ablation sweep
│   ├── bootstrap_stability.py  # Bootstrap resampling stability analysis
│   ├── activation_patching.py  # Causal head identification (AIE)
│   ├── injection_attack.py     # Refusal direction injection experiment
│   ├── zero_shot_probe.py      # Zero-shot safety classifier via dot product
│   ├── cross_domain_transfer.py# Cross-domain direction transfer test
│   ├── merge_patching.py       # Merge parallel patching results + heatmap
│   ├── plot_pareto.py          # Pareto frontier plots
│   ├── plot_bootstrap.py       # Bootstrap visualization
│   └── plot_all_domains.py     # All-domain contrast plots (PPL explosion)
│
├── results/
│   ├── plots/                  # All generated figures (PNG + all_plots.zip)
│   ├── vectors/                # Extracted direction vectors + experiment JSONs
│   └── presentation_results.txt# Full results writeup for slides
│
├── requirements.txt
└── run_all.sh
```

---

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_huggingface_token>
```

You need HuggingFace access to `meta-llama/Llama-2-7b-chat-hf`. Request access at [huggingface.co/meta-llama](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

---

## Running the Pipeline

### 1. Extract direction vectors
```bash
python src/extract_directions.py --domain harmful --save-hiddens
python src/extract_directions.py --domain medical --save-hiddens
python src/extract_directions.py --domain legal --save-hiddens
```

### 2. SVD analysis + subspace extraction
```bash
python src/svd_analysis.py
```

### 3. Subspace vs single vector ablation (full alpha sweep)
```bash
# Run all 3 domains in parallel
CUDA_VISIBLE_DEVICES=0 python src/subspace_ablation.py --domain harmful &
CUDA_VISIBLE_DEVICES=1 python src/subspace_ablation.py --domain medical &
CUDA_VISIBLE_DEVICES=2 python src/subspace_ablation.py --domain legal &
```

### 4. Bootstrap stability (CPU only, fast)
```bash
python src/bootstrap_stability.py
```

### 5. Activation patching (parallel across 4 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0 python src/activation_patching.py --layers 10 13 &
CUDA_VISIBLE_DEVICES=1 python src/activation_patching.py --layers 13 16 &
CUDA_VISIBLE_DEVICES=2 python src/activation_patching.py --layers 16 19 &
CUDA_VISIBLE_DEVICES=3 python src/activation_patching.py --layers 19 22 &
# After all finish:
python src/merge_patching.py
```

### 6. Additional experiments
```bash
CUDA_VISIBLE_DEVICES=0 python src/zero_shot_probe.py
CUDA_VISIBLE_DEVICES=0 python src/injection_attack.py
CUDA_VISIBLE_DEVICES=0 python src/cross_domain_transfer.py
```

### 7. Generate plots
```bash
python src/plot_pareto.py
python src/plot_bootstrap.py
python src/plot_all_domains.py
```

---

## Core Concepts

### Mean-Difference Direction Extraction
```
v_d = normalize( mean(H_refused) - mean(H_answered) )
```
Hidden states at layer 16 are averaged over refused and answered prompts separately. The normalized difference is the refusal direction.

### Single Vector Ablation
```
x' = x - α * (x · v_d) * v_d
```
Projects out the refusal component. Works perfectly for harmful (eff_rank ≈ 1) but causes **concept erasure** for medical (eff_rank ≈ 31) — refusal increases instead of decreasing.

### Subspace Ablation
```
x' = x - α * U_d @ U_d.T @ x
```
where `U_d` = top-8 left singular vectors from SVD of the difference matrix. Correctly separates refusal from domain knowledge in high-rank domains.

### Effective Rank
```
eff_rank = exp( -Σ p_i * log(p_i) )    where p_i = σ_i² / Σ σ_j²
```
Measures the dimensionality of the refusal subspace. Low rank → clean single direction. High rank → entangled multi-dimensional structure.

---

## Selected Plots

| Plot | Description |
|------|-------------|
| `scree_plots.png` | SVD explained variance — harmful vs medical/legal |
| `principal_angles.png` | Grassmann geometry between domain subspaces |
| `refusal_vs_alpha.png` | Main result: single vs subspace ablation comparison |
| `all_domains_contrast.png` | **Key synthesis**: harmful PPL explosion vs medical/legal fix |
| `bootstrap_bars.png` | Direction stability across 10× resamples |
| `activation_patching_merged.png` | 12×32 AIE heatmap — causal heads in layers 19–21 |

---

## References

- Arditi et al. ["Refusal in Language Models Is Mediated by a Single Direction"](https://arxiv.org/abs/2406.11717) (NeurIPS 2024)
- ["The Geometry of Refusal in Large Language Models"](https://arxiv.org/abs/2502.17420) (2025)
- Wang et al. ["Mitigating False Refusal via Single Vector Ablation"](https://arxiv.org/abs/2602.02132) (ICLR 2025)
- Zou et al. ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043) (2023)
- Chao et al. ["JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models"](https://arxiv.org/abs/2404.01318) (2024)
