#!/bin/bash
# One-day execution plan: 4x RTX 4090 GPUs
# Llama-2-7B-Chat (4-bit) fits comfortably on 1 GPU (~5GB VRAM)
#
# STEP 1 (sequential, ~45 min): Re-extract directions + save raw hidden states
#   - Must run BEFORE svd_analysis.py and subspace_ablation.py
#   - Runs on GPU 0; other GPUs idle
#
# STEP 2 (sequential, ~2 min, CPU): SVD analysis
#
# STEP 3 (parallel, ~2-3 hrs): Core experiments across all 4 GPUs
#
# HOW TO RUN:
#   chmod +x run_all.sh
#   bash run_all.sh
# Or run each step manually (recommended so you can monitor progress).

set -e
cd /workspace/refusal_vectors

LOG_DIR="/workspace/refusal_vectors/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Step 1: Re-extract directions + save hidden states"
echo " (Runs on GPU 0, ~45 min)"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0 python src/extract_directions.py --save-hiddens \
    2>&1 | tee "$LOG_DIR/extract_directions.log"

echo ""
echo "============================================================"
echo " Step 2: SVD subspace analysis (CPU, ~2 min)"
echo "============================================================"
python src/svd_analysis.py 2>&1 | tee "$LOG_DIR/svd_analysis.log"

echo ""
echo "============================================================"
echo " Step 3: Parallel experiments on 4 GPUs"
echo "============================================================"

# GPU 0: Subspace ablation on MEDICAL domain
CUDA_VISIBLE_DEVICES=0 python src/subspace_ablation.py --domain medical \
    2>&1 | tee "$LOG_DIR/subspace_medical.log" &
PID_MEDICAL=$!
echo "Started medical subspace ablation (PID=$PID_MEDICAL) on GPU 0"

# GPU 1: Subspace ablation on LEGAL domain
CUDA_VISIBLE_DEVICES=1 python src/subspace_ablation.py --domain legal \
    2>&1 | tee "$LOG_DIR/subspace_legal.log" &
PID_LEGAL=$!
echo "Started legal subspace ablation  (PID=$PID_LEGAL) on GPU 1"

# GPU 2: Subspace ablation on HARMFUL domain (cross-check: should be unaffected)
CUDA_VISIBLE_DEVICES=2 python src/subspace_ablation.py --domain harmful \
    2>&1 | tee "$LOG_DIR/subspace_harmful.log" &
PID_HARMFUL=$!
echo "Started harmful subspace ablation (PID=$PID_HARMFUL) on GPU 2"

# GPU 3: Injection attack (novel experiment)
CUDA_VISIBLE_DEVICES=3 python src/injection_attack.py \
    2>&1 | tee "$LOG_DIR/injection_attack.log" &
PID_INJECT=$!
echo "Started injection attack          (PID=$PID_INJECT) on GPU 3"

echo ""
echo "Waiting for all parallel jobs to finish..."
wait $PID_MEDICAL && echo "  [DONE] medical subspace ablation"
wait $PID_LEGAL   && echo "  [DONE] legal subspace ablation"
wait $PID_HARMFUL && echo "  [DONE] harmful subspace ablation"
wait $PID_INJECT  && echo "  [DONE] injection attack"

echo ""
echo "============================================================"
echo " All experiments complete. Results in:"
echo "   results/vectors/subspace_ablation_results.json"
echo "   results/vectors/injection_results.json"
echo "   results/plots/scree_plots.png"
echo "   results/plots/principal_angles.png"
echo "   results/plots/subspace_vs_single.png"
echo "   results/plots/pareto.png"
echo "   results/plots/injection_attack.png"
echo "============================================================"
