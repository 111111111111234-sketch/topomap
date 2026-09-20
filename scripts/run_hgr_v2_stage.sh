#!/usr/bin/env bash
set -euo pipefail

# Run one HGR V2 validation stage, the whole sequence, or the stages after A.
# Individual modes keep each online acceptance result independently auditable.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_hgr_v2_stage.sh tests [GPU]
  bash scripts/run_hgr_v2_stage.sh all [GPU]
  bash scripts/run_hgr_v2_stage.sh after-a [GPU]
  bash scripts/run_hgr_v2_stage.sh fixes [GPU]
  bash scripts/run_hgr_v2_stage.sh verify-r3 [GPU]
  bash scripts/run_hgr_v2_stage.sh <baseline|a|b|c-shadow|c|c-persistent|d> [GPU]

Examples:
  bash scripts/run_hgr_v2_stage.sh tests 7
  bash scripts/run_hgr_v2_stage.sh all 7
  bash scripts/run_hgr_v2_stage.sh after-a 7
  bash scripts/run_hgr_v2_stage.sh a 7
  bash scripts/run_hgr_v2_stage.sh b 7

Environment overrides:
  HGR_PYTHON     Python executable (default: current conda python if active)
  HGR_RESULTS    result parent directory (default: results)
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

stage="$1"
gpu="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -n "${HGR_PYTHON:-}" ]]; then
  python_bin="$HGR_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  python_bin="${CONDA_PREFIX}/bin/python"
else
  python_bin="/home/tangyuxin/miniconda3/envs/hgr/bin/python"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  echo "Set HGR_PYTHON to the hgr environment's Python executable." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

if [[ "$stage" == "all" || "$stage" == "fixes" || "$stage" == "verify-r3" ]]; then
  stages=(tests a b c-shadow c c-persistent d)
  if [[ "$stage" == "fixes" ]]; then
    stages=(tests b c-shadow c c-persistent d)
  elif [[ "$stage" == "verify-r3" ]]; then
    stages=(tests c-persistent d)
  fi
  echo "[HGR V2] running all stages sequentially on GPU=$gpu"
  echo "[HGR V2] order: ${stages[*]}"
  for next_stage in "${stages[@]}"; do
    echo
    echo "[HGR V2] ===== starting $next_stage ====="
    bash "$repo_dir/scripts/run_hgr_v2_stage.sh" "$next_stage" "$gpu"
  done
  echo
  echo "[HGR V2] all requested stages completed"
  exit 0
fi

if [[ "$stage" == "after-a" ]]; then
  stages=(b c-shadow c c-persistent d)
  echo "[HGR V2] continuing after an already accepted A-R2 run on GPU=$gpu"
  echo "[HGR V2] order: ${stages[*]}"
  for next_stage in "${stages[@]}"; do
    bash "$repo_dir/scripts/run_hgr_v2_stage.sh" "$next_stage" "$gpu"
  done
  exit 0
fi

if [[ "$stage" == "tests" ]]; then
  echo "[HGR V2] GPU=$gpu: running focused tests"
  "$python_bin" -m unittest tests.test_hgr_dual_topo_v2 tests.test_v2_route_stability -v
  echo "[HGR V2] GPU=$gpu: running the complete regression suite"
  "$python_bin" -m unittest discover -s tests -v
  exit 0
fi

case "$stage" in
  baseline)
    base_config="eval_goatbench_hgr_dual_topo_v2_baseline_smoke.yaml"
    audit_kind="none"
    ;;
  a)
    base_config="eval_goatbench_hgr_dual_topo_v2_a_smoke_r2.yaml"
    audit_kind="replay"
    ;;
  b)
    base_config="eval_goatbench_hgr_dual_topo_v2_b_smoke.yaml"
    audit_kind="v2"
    ;;
  c-shadow)
    base_config="eval_goatbench_hgr_dual_topo_v2_c_shadow_smoke.yaml"
    audit_kind="v2"
    ;;
  c)
    base_config="eval_goatbench_hgr_dual_topo_v2_c_smoke.yaml"
    audit_kind="v2"
    ;;
  c-persistent)
    base_config="eval_goatbench_hgr_dual_topo_v2_c_persistent_smoke.yaml"
    audit_kind="v2"
    ;;
  d)
    base_config="eval_goatbench_hgr_dual_topo_v2_d_smoke.yaml"
    audit_kind="v2"
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    usage
    exit 2
    ;;
esac

safe_stage="${stage//-/_}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp_dev_goatbench_hgr_dual_topo_v2_${safe_stage}_smoke_${timestamp}"
generated_dir="cfg/generated"
generated_config="${generated_dir}/${run_name}.yaml"
result_parent="${HGR_RESULTS:-results}"
result_dir="${result_parent}/${run_name}"

mkdir -p "$generated_dir"
if [[ -e "$generated_config" || -e "$result_dir" ]]; then
  echo "Refusing to overwrite an existing run: $run_name" >&2
  exit 1
fi

{
  printf 'extends: ../%s\n' "$base_config"
  printf 'exp_name: %s\n' "$run_name"
  printf 'output_parent_dir: %s\n' "$result_parent"
  if [[ "$stage" == "a" ]]; then
    printf 'record_run_fingerprint: true\n'
    printf 'hgr_topology_fusion:\n'
    printf '  record_decisions: true\n'
  fi
} > "$generated_config"

echo "[HGR V2] stage=$stage GPU=$gpu"
echo "[HGR V2] config=$generated_config"
echo "[HGR V2] result=$result_dir"

"$python_bin" run_goatbench_evaluation.py -cf "$generated_config" --split 1

if [[ "$audit_kind" == "replay" ]]; then
  "$python_bin" scripts/hgr_v2_experiments.py replay "$result_dir/decision_records"
elif [[ "$audit_kind" == "v2" ]]; then
  "$python_bin" scripts/hgr_v2_experiments.py replay "$result_dir/decision_records"
  "$python_bin" scripts/hgr_v2_experiments.py audit \
    "$result_dir/active_topology_traces" \
    --output "$result_dir/v2_trace_audit.json"
fi

echo "[HGR V2] completed stage=$stage"
if [[ "$stage" == "d" ]]; then
  "$python_bin" scripts/hgr_v2_experiments.py cache-replay "$result_dir/decision_records" \
    --output "$result_dir/v2_cache_replay.json"
fi
echo "[HGR V2] send back: $result_dir"
