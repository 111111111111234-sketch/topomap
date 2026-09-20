#!/usr/bin/env bash
set -euo pipefail
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running}"
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement/cfg/generated/v2_memory_pilot_aczz_20260912_r2/v2_memory_pilot_aczz_20260912_r2_baseline_cold_r1_s1.yaml --split 1
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement/cfg/generated/v2_memory_pilot_aczz_20260912_r2/v2_memory_pilot_aczz_20260912_r2_baseline_warm_r1_s1.yaml --split 1
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement/cfg/generated/v2_memory_pilot_aczz_20260912_r2/v2_memory_pilot_aczz_20260912_r2_v2_cold_r1_s1.yaml --split 1
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement/cfg/generated/v2_memory_pilot_aczz_20260912_r2/v2_memory_pilot_aczz_20260912_r2_v2_warm_r1_s1.yaml --split 1
