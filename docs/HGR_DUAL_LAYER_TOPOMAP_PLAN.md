# HGR Dual-Layer Topological Map Plan

Historical V7 branch plan. For the current 2026-09-10 architecture and implementation priorities,
use [the general specification](HGR_DUAL_TOPO_GENERAL_DEVELOPMENT_SPEC.md) and
[topology-guided continuous execution design](HGR_TOPO_GUIDED_CONTINUOUS_DESIGN.md).
The V7 allowlist/Region mechanisms below are not the current dual-topology default.

## Implementation status

Implemented as an isolated V7 dual-layer branch:

- high-level policy: `src/dual_layer_topology.py`;
- optional Snapshot/Frontier prompt allowlists:
  `src/query_vlm_goatbench.py`;
- evaluation integration and JSONL decision/fallback traces:
  `run_goatbench_evaluation.py`;
- smoke, 3x2 fast, and 12x2 development configs:
  `cfg/eval_goatbench_goal_topomap_v7_dual_layer_*.yaml`.

The strict revisit gate is bound to the exact selected topology node.  A VLM
crop mismatch therefore retains original HGR execution instead of inheriting
another node's topology-route authorization.

V7.2 adds a stable Frontier commitment to this branch: a Layer-2-selected
`EXPLORE` node persists until arrival, source invalidation, navigation failure,
or a small no-progress budget is exhausted.  This prevents `choose_every_step`
from repeatedly replacing a valid topological long-horizon target.

V7.2-B additionally treats Frontier identity as geometry-aware: when TSDF
re-extraction replaces a transient Frontier ID, the commitment rebinds to a
nearby reachable Frontier. If re-extraction temporarily yields no equivalent
node but the prior physical target remains live, it continues toward that
anchor and still releases on true navigation failure or lack of progress.

V7.3 keeps topology in control of macro exploration without treating its
dynamic score as a complete local-motion oracle: it presents the primary
goal-conditioned Frontier and one topology-ranked fallback to the VLM, which
selects the local Frontier within that closed shortlist. The selected target
then follows the V7.2-B commitment policy.

## Objective

Build a training-free hierarchical navigation layer on top of HGR without
replacing its TSDF, object map, Snapshot representation, semantic hypothesis
graph, or Habitat Pathfinder.

## Representation

### Layer 0: HGR metric substrate

The existing TSDF/free-space map remains the source of collision checks,
precise observation points, geodesic distance, and local motion.

### Layer 1: Persistent Scene Topology

The existing `PersistentBeliefTopology` stores goal-independent episode facts:

- `VISITED` place/pose anchors;
- `OBSERVED` Snapshot evidence;
- transient executable `FRONTIER` nodes;
- `NAVIGABLE`, `OBSERVED_AT`, `ANCHORED_TO`, and dependency edges;
- metric length, accessibility, traversal outcome, and stable source identity.

This layer answers: *where are the stable places and how can the agent move
between them?*

### Layer 2: Dynamic Goal Topology

For each GOAT subtask, `DynamicGoalTopology` projects Layer 1 into goal-local
nodes carrying:

- raw and accumulated goal confidence;
- uncertainty and independent-view coverage;
- freshness;
- reachability, budget ratio, and route cost;
- typed one-hop support and edge reliability/description.

This layer is cleared on every goal switch and never mutates Layer 1 geometry.
It answers: *which currently executable place is useful for this goal?*

## Unified high-level decision

The same policy is used for Object, Description, and Image goals:

1. choose `DIRECT` only when the best current observation exceeds every
   non-direct alternative by the existing strict relevance margin;
2. otherwise choose the exact `REVISIT` node accepted by the existing strict
   semantic/budget/opportunity-cost gate;
3. otherwise choose the reachable `EXPLORE` frontier ordered by dynamic goal
   confidence, uncertainty, and metric cost;
4. if no valid topological action exists, use the untouched original HGR path.

No goal-type-specific threshold or learned component is introduced.

For a Snapshot action, Layer 2 selects the place/Snapshot but Qwen3-VL still
selects the concrete object crop.  The prompt is reduced to the selected
Snapshot plus one selected Frontier fallback.  For a Frontier-only decision,
the selected frontier is executed directly without an unnecessary VLM call.

## Execution

- `DIRECT`: original HGR object observation-point navigation;
- `REVISIT`: Layer-1 capture-anchor/direct-or-safe-topology route, followed by
  original HGR object targeting;
- `EXPLORE`: selected Frontier passed to original TSDF/Pathfinder execution;
- mapping, reachability, or prompt failure: trace the reason and fall back to
  original HGR for that step.

## Isolation and evaluation

The implementation is enabled only by a new config.  V6, frozen V7,
C-strict, revised selected-ID gate, and all prior result directories remain
unchanged.  Required checks before any development evaluation:

1. deterministic unit tests for all three action branches and fallbacks;
2. candidate/source-index preservation through the restricted VLM prompt;
3. every executed revisit matches the Layer-2 selected node and strict gate;
4. finite metrics and serializable traces;
5. full existing test-suite regression.

## V7.4 implemented region-level execution

V7.4 makes the hierarchy operational without adding a learned scene
segmentation.  A **Region** is the stable `VISITED` anchor already present in
Layer 1.  Historical snapshots join the Region through their immutable capture
anchor; live frontiers join it through `ANCHORED_TO`.

When the existing high-level policy reaches `EXPLORE`, Layer 2 ranks reachable
Regions lexicographically by their best member's dynamic relevance,
uncertainty, and metric cost.  The top two Regions are passed to Qwen3-VL with
**all** their current snapshots and frontiers, so the VLM makes a local action
choice with multi-view context rather than choosing between two isolated
frontiers.  Only a selected local Frontier is executed, and it retains the
V7.2-B geometric commitment/rebinding behavior.  Snapshot answers from this
regional prompt do not bypass the existing strict revisit gate; an empty,
invalid, or non-frontier local result falls back to the original HGR prompt.

This applies uniformly to object, description, and image goals, and has no
scene names, object-specific rules, training, or tuned per-scene thresholds.
