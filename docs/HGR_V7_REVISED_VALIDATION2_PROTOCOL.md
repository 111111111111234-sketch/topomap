# Revised V7 Validation-2 Protocol

> Frozen on 2026-09-02 before running either method on this manifest.

## Data lock

- Dataset: GOAT-Bench `train`
- Manifest: `cfg/manifests/goat_train_validation2_12x2_seed77.json`
- SHA-256: `aefa1a81d36879aedfc4123507268e517c524dba73339d1655bac64bcc1f95cc`
- Deterministic selection: seed 77, legacy scene interval `[0.18, 0.27)`
- Scale: 12 scenes × 2 episodes = 24 episodes, 178 subtasks
- Goal counts: Object 63, Description 60, Image 55
- Scene overlap with the 12-scene development set: 0
- Scene overlap with the first 12-scene internal validation set: 0

## Frozen methods

Baseline:

```text
repository: Hypothesis_Graph_Refinement_baseline
config: cfg/eval_goatbench_qwen3vl_dashscope_train_validation2.yaml
method: original HGR with provider-only Qwen image-shape normalization
```

Candidate:

```text
repository: Hypothesis_Graph_Refinement
config: cfg/eval_goatbench_goal_topomap_v7_revised_qwen3vl_dashscope_train_validation2.yaml
method: Persistent Topology + Goal-conditioned Overlay + strict revisit gate
        bound to the selected historical topology node
```

Both use Qwen3-VL 30B through DashScope and the same fixed episodes, sensors,
perception models, planner settings, success distance, and HGR prompt policy.

## Acceptance criteria

The revised V7 direction is accepted only if all conditions hold:

1. both methods contain the same 178 subtask IDs;
2. revised V7 Distance SR is at least 2 percentage points above baseline;
3. revised V7 Distance SPL is no more than 2 percentage points below baseline;
4. no goal type loses more than 3 percentage points in Distance SR;
5. all invalid metrics, VLM failures, and API failures are reported;
6. every executed Topo Route is authorized for the exact selected node.

Do not change code, thresholds, the manifest, or configs after either run is
opened. Run baseline splits first, then revised V7 splits, and compare only
after all four runs finish.
