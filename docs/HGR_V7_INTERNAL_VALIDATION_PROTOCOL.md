# HGR V7 Internal Validation Protocol

> Frozen on 2026-09-01, before viewing any internal-validation result.

## Data lock

- Dataset: GOAT-Bench `train`
- Manifest: `cfg/manifests/goat_train_internal_val_12x2_seed77.json`
- SHA-256: `a96fcce33f26df9876dd22cf58a1dd3d3ead7cef76f2797174f990e132d6696b`
- Selection: seed 77, legacy scene interval `[0.09, 0.18)`
- Scale: 12 scenes × 2 episodes = 24 episodes, 170 filtered subtasks
- Goal counts: Object 61, Description 52, Image 57
- Overlap with `goat_train_dev_12x2_seed77.json`: 0 scenes

Locked scenes:

```text
1UnKg1rAb8A  DqJKU7YU7dA  GtM3JtRvvvR  JNiWU5TZLtt
Jfyvj3xn2aJ  LcAd9dhvVwh  RTV2n6fXB2w  erXNfWVjqZ8
gjhYih4upQ9  qgZhhx1MpTi  w8GiikYuFRk  wPLokgvCnuk
```

The manifest must not be regenerated, reordered, or edited after either method
has started. Split 1 contains episode 0 and split 2 contains episode 1 for the
same 12 scenes.

## Frozen methods

Baseline:

```text
repository: Hypothesis_Graph_Refinement_baseline
config: cfg/eval_goatbench_qwen3vl_dashscope_train_internal_val.yaml
method: original HGR navigation/prompt policy
model: qwen3-vl-30b-a3b-instruct through DashScope
```

Final V7:

```text
repository: Hypothesis_Graph_Refinement
config: cfg/eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train_internal_val.yaml
method: Persistent Topology + Goal-conditioned Overlay + C-strict gate
model: qwen3-vl-30b-a3b-instruct through DashScope
```

The two resolved configurations have identical seed, sensors, detector, SAM,
CLIP, planner, success distance, HGR settings, prompt dimensions, and maximum
step settings. Baseline-only code changes are provider/path adapters and fixed
manifest selection; they do not alter candidate choice or navigation.

### Baseline transport audit

The first baseline run is quarantined and must not be used for method
comparison. It produced 210 DashScope HTTP 400 responses and 16 explicit
`query_vlm_for_response failed` subtasks because the original GPT-4o adapter
sent tiny/extreme-aspect detector crops that Qwen3-VL rejects. V7 had zero VLM
failures because its provider adapter already pads such crops without removing
or synthesizing image content.

The valid baseline rerun uses
`cfg/eval_goatbench_qwen3vl_dashscope_train_internal_val_r2.yaml` and only adds
the identical black-padding/image-shape normalization to the baseline API
adapter. It does not change candidates, prompt text, navigation, thresholds,
the manifest, or any V7 code. Its new output directory prevents reuse of the
invalid run. The baseline comparison remains unopened until R2 completes with
zero HTTP 400 failures.

## Predeclared evaluation

Primary metrics:

1. Distance SR;
2. Distance SPL.

Secondary diagnostics:

- Snapshot SR/SPL;
- Object/Description/Image Distance SR/SPL;
- paired fail-to-success and success-to-fail counts;
- frames, path length, VLM calls/failures, and V7 gate intervention rate.

The validation direction is accepted only if all of the following hold:

1. baseline and V7 contain exactly the same 170 subtask IDs;
2. V7 Distance SR is at least 2 percentage points above baseline;
3. V7 Distance SPL is not more than 2 percentage points below baseline;
4. no goal type loses more than 3 percentage points in Distance SR;
5. invalid metrics and API failures are reported, with no silent deletion.

These thresholds and the method must not be changed after validation results are
opened. If they fail, report the negative result and stop; do not tune on this
set. If they pass, freeze the same method for the official GOAT benchmark.

## Execution order

Run both baseline splits first, then both V7 splits. Do not compare partial
results. Aggregate and paired-compare only after all four runs finish.
