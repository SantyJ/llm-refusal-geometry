# Refusal Direction Experiments

Extracting and comparing domain-specific refusal direction vectors from Llama-2-7b-chat-hf across harmful, medical, and legal prompts.

## Setup

```
pip install -r requirements.txt
export HF_TOKEN=<your_token>
```

## How to run

```
python src/extract_directions.py
python src/cosine_similarity.py
python src/ablation.py
python src/ablation.py --by-domain
```

For filtered vectors:
```
python src/extract_directions.py --filtered --domains medical,legal
python src/ablation.py --by-domain --vectors-dir results/vectors_filtered --plot-name refusal_rates_filtered.png
```

## Approach

### Datasets

We use three prompt categories:
- **harmful** -- 100 prompts from JailbreakBench (unchanged)
- **medical** -- 100 custom prompts requesting personal medical advice, prescriptions, lethal doses, self-treatment
- **legal** -- 100 custom prompts asking for help with asset hiding, tax evasion, avoiding charges, illegal eviction

The medical and legal prompts replace the original MedQuad and LegalBench datasets which were too safe (0% refusal) to extract meaningful direction vectors.

### Direction extraction

We run each prompt through Llama-2-7b-chat (4-bit quantized), extract hidden states at layer 16, generate a response, and classify it as refused or answered. The refusal direction is the normalized mean difference between refused and answered hidden states.

### Cosine similarity

We compare the three direction vectors to see if refusal mechanisms are shared or domain-specific.

### Ablation

We subtract alpha * projection onto a direction vector from layer 16 activations during generation using a forward hook. We test alpha = 0, 0.5, 1.0 and measure how refusal rates change.

## Results

Plots and vectors are saved to `results/`.

## Limitations

- Single layer (16) only; Arditi et al. (2024) ablate across all layers
- Medical and legal datasets are hand-written, not from benchmarks
- 4-bit quantization may affect extraction fidelity
- Keyword-based refusal detection

## References

- Arditi et al. "Refusal in Language Models Is Mediated by a Single Direction" (NeurIPS 2024)
- "The Geometry of Refusal in Large Language Models" (2025) arXiv:2502.17420
- Wang et al. "Mitigating False Refusal via Single Vector Ablation" (ICLR 2025)
