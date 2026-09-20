#!/usr/bin/env bash
set -euo pipefail
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running}"
cd /home/hdd/tangyuxin/projects/Hypothesis_Graph_Refinement
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf cfg/eval_goatbench_hgr_dual_topo_v2_memory_export_fast_3x2.yaml --split 1
CUDA_VISIBLE_DEVICES=7 /home/tangyuxin/miniconda3/envs/hgr/bin/python run_goatbench_evaluation.py -cf cfg/eval_goatbench_hgr_dual_topo_v2_memory_export_fast_3x2.yaml --split 2
