#!/usr/bin/env bash
set -euo pipefail
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
/home/tangyuxin/miniconda3/envs/hgr/bin/python scripts/hgr_v2_experiments.py compare \
  --manifest cfg/manifests/goat_train_fast_3x2_seed77.json \
  --start-subtask 1 \
  --baseline "results/v2_memory_fast_3x2_20260912_r1_baseline_warm_r1_s1,results/v2_memory_fast_3x2_20260912_r1_baseline_warm_r1_s2" \
  --candidate "results/v2_memory_fast_3x2_20260912_r1_v2_warm_r1_s1,results/v2_memory_fast_3x2_20260912_r1_v2_warm_r1_s2" \
  --baseline-cold "results/v2_memory_fast_3x2_20260912_r1_baseline_cold_r1_s1,results/v2_memory_fast_3x2_20260912_r1_baseline_cold_r1_s2" \
  --candidate-cold "results/v2_memory_fast_3x2_20260912_r1_v2_cold_r1_s1,results/v2_memory_fast_3x2_20260912_r1_v2_cold_r1_s2" \
  --output results/v2_memory_fast_3x2_20260912_r1_comparison.json
