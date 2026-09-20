"""Unified goal-conditioned topology projection and route execution for HGR V7.

V7.1-A computes finite-safe costs and source mappings for DIRECT, REVISIT, and
EXPLORE.  The final V7 wraps that shared action space as a deliberately small
goal-conditioned overlay.  It exposes only four paper-level concepts:
evidence, belief, cost, and validity.  It does not maintain a second graph or
change the frozen C-strict scores.  V7.1-B adds a direct-first executor for an
already selected historical Snapshot; it does not choose or rank the target.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np

from src.persistent_belief_topology import (
    GoalTopoAction,
    GoalTopoActionType,
    GoalTopoTaskView,
    PersistentBeliefTopology,
)


class MetricRouteMode(str, Enum):
    CURRENT = "current"
    DIRECT_PATH = "direct_path"
    SAFE_TOPO = "safe_topology"
    UNREACHABLE = "unreachable"


class DirectFirstRouteMode(str, Enum):
    DIRECT_PATH = "direct_path"
    SAFE_TOPO = "safe_topology"
    ANCHOR_REACHED = "anchor_reached"
    ORIGINAL_HGR_FALLBACK = "original_hgr_fallback"


@dataclass(frozen=True)
class RevisitGateDecision:
    """Conservative V7.1-C decision for permitting historical reuse."""

    allowed: bool
    reason: str
    best_revisit_topo_id: Optional[str]
    best_alternative_topo_id: Optional[str]
    cheapest_explore_topo_id: Optional[str]
    relevance_margin: Optional[float]
    revisit_cost_m: Optional[float]
    remaining_budget_m: Optional[float]
    budget_ratio: Optional[float]
    revisit_to_explore_cost_ratio: Optional[float]
    semantic_clear: bool
    budget_ok: bool
    opportunity_cost_ok: bool
    source_mapping_ok: bool

    def to_trace_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TargetCommitment:
    """Serializable V7.1-D commitment to one historical capture anchor.

    Deliberately stores stable identifiers only.  In particular it never owns a
    ``SnapShot`` instance: scene cleanup may replace those Python objects while
    the capture anchor remains a valid metric navigation target.
    """

    goal_key: str
    topo_node_id: str
    capture_anchor_id: str
    snapshot_image: str
    created_step: int
    last_evaluated_step: int
    initial_cost_m: float
    last_cost_m: float
    best_cost_m: float
    relevance: float
    segments_executed: int = 0
    no_progress_segments: int = 0

    def to_trace_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TargetCommitmentDecision:
    """Finite, explainable decision made after one navigation segment."""

    action: str
    reason: str
    committed_topo_id: str
    challenger_topo_id: Optional[str]
    committed_cost_m: Optional[float]
    challenger_cost_m: Optional[float]
    committed_relevance: Optional[float]
    challenger_relevance: Optional[float]
    progress_m: Optional[float]
    no_progress_segments: int

    def to_trace_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DirectFirstRoutePlan:
    """Execution plan for one HGR-selected historical Snapshot."""

    snapshot_image: str
    observed_topo_id: Optional[str]
    capture_anchor_id: Optional[str]
    route_mode: str
    target_voxel: Optional[np.ndarray]
    look_at_voxel: Optional[np.ndarray]
    route: List[str]
    next_waypoint_id: Optional[str]
    direct_geodesic_m: Optional[float]
    topology_route_m: Optional[float]
    fallback_reason: Optional[str]

    @property
    def uses_override(self) -> bool:
        return self.target_voxel is not None

    def to_trace_dict(self):
        return {
            "snapshot_image": self.snapshot_image,
            "observed_topo_id": self.observed_topo_id,
            "capture_anchor_id": self.capture_anchor_id,
            "route_mode": self.route_mode,
            "target_voxel": (
                None if self.target_voxel is None
                else np.asarray(self.target_voxel, dtype=float).tolist()
            ),
            "look_at_voxel": (
                None if self.look_at_voxel is None
                else np.asarray(self.look_at_voxel, dtype=float).tolist()
            ),
            "route": list(self.route),
            "next_waypoint_id": self.next_waypoint_id,
            "direct_geodesic_m": self.direct_geodesic_m,
            "topology_route_m": self.topology_route_m,
            "fallback_reason": self.fallback_reason,
            "uses_override": self.uses_override,
        }


@dataclass(frozen=True)
class MetricAction:
    action_type: str
    topo_node_id: str
    source_kind: str
    source_frontier_index: Optional[int]
    snapshot_image: Optional[str]
    object_id: Optional[Any]
    current_observation: bool
    relevance: float
    capture_anchor_id: Optional[str]
    direct_geodesic_m: Optional[float]
    topology_route_m: Optional[float]
    effective_cost_m: Optional[float]
    route_mode: str
    reachable: bool
    remaining_budget_m: float
    budget_ratio: Optional[float]

    @property
    def source_key(self):
        if self.source_kind == "frontier":
            return ("frontier", self.source_frontier_index)
        return ("snapshot", self.snapshot_image, self.object_id)

    def to_trace_dict(self):
        payload = asdict(self)
        if isinstance(payload.get("object_id"), np.integer):
            payload["object_id"] = int(payload["object_id"])
        return payload


@dataclass(frozen=True)
class MetricActionView:
    goal_type: str
    subtask_id: str
    step: int
    actions: List[MetricAction]

    def to_trace_dict(self):
        counts = {
            action_type.value: sum(
                action.action_type == action_type.value for action in self.actions
            )
            for action_type in GoalTopoActionType
        }
        return {
            "goal_type": self.goal_type,
            "subtask_id": self.subtask_id,
            "step": self.step,
            "action_counts": counts,
            "reachable_actions": sum(action.reachable for action in self.actions),
            "actions": [action.to_trace_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class GoalOverlayCandidate:
    """One goal-conditioned projection of a persistent topology source.

    ``goal_evidence`` describes where the score came from; ``belief`` is the
    frozen goal relevance already computed by ``PersistentBeliefTopology``.
    Keeping those two notions separate avoids pretending that recency or
    repeated absence changes whether an object still exists in a static GOAT
    episode.  Cost and validity are derived from the metric action projection.
    """

    action: MetricAction
    goal_evidence: dict
    belief: float
    cost_m: Optional[float]
    validity: dict
    topology_relations: List[str]

    @property
    def action_type(self):
        return self.action.action_type

    @property
    def topo_node_id(self):
        return self.action.topo_node_id

    @property
    def relevance(self):
        return self.belief

    @property
    def effective_cost_m(self):
        return self.cost_m

    @property
    def reachable(self):
        return bool(self.validity.get("reachable"))

    @property
    def snapshot_image(self):
        return self.action.snapshot_image

    @property
    def capture_anchor_id(self):
        return self.action.capture_anchor_id

    @property
    def remaining_budget_m(self):
        return self.action.remaining_budget_m

    def to_trace_dict(self):
        return {
            "action_type": self.action.action_type,
            "topo_node_id": self.action.topo_node_id,
            "goal_evidence": dict(self.goal_evidence),
            "belief": float(self.belief),
            "cost_m": self.cost_m,
            "validity": dict(self.validity),
            "topology_relations": list(self.topology_relations),
        }


@dataclass(frozen=True)
class GoalConditionedTopologyOverlay:
    """Behavior-preserving projection of the scene topology for one goal.

    The conservative gate consumes ``actions`` from this object directly.
    Returning the original immutable ``MetricAction`` instances makes the
    final overlay exactly score-equivalent to the validated C-strict method.
    """

    metric_view: MetricActionView
    candidates: List[GoalOverlayCandidate]

    @property
    def goal_type(self):
        return self.metric_view.goal_type

    @property
    def subtask_id(self):
        return self.metric_view.subtask_id

    @property
    def step(self):
        return self.metric_view.step

    @property
    def actions(self):
        return self.metric_view.actions

    def to_trace_dict(self):
        counts = {
            action_type.value: sum(
                item.action.action_type == action_type.value
                for item in self.candidates
            )
            for action_type in GoalTopoActionType
        }
        return {
            "goal_type": self.goal_type,
            "subtask_id": self.subtask_id,
            "step": self.step,
            "action_counts": counts,
            "valid_candidates": sum(
                bool(item.validity.get("valid")) for item in self.candidates
            ),
            "candidates": [item.to_trace_dict() for item in self.candidates],
        }


def build_goal_conditioned_topology_overlay(
    metric_view: MetricActionView,
) -> GoalConditionedTopologyOverlay:
    """Expose the common candidate space without adding learned/dynamic state.

    The three edge roles are intentionally causal and compact:

    * ``OBSERVED_AT`` maps a Snapshot to its executable capture anchor;
    * ``NAVIGABLE`` supplies cost/reachability for REVISIT and EXPLORE;
    * spatial evidence deduplication is not activated in the final method,
      because the corresponding dynamic-evidence ablation did not validate.
    """

    candidates = []
    for action in metric_view.actions:
        if action.action_type == GoalTopoActionType.DIRECT.value:
            evidence_source = "current_observation"
            source_mapping_valid = bool(
                action.current_observation and action.topo_node_id
            )
            relations = ["OBSERVED_AT_CURRENT"]
        elif action.action_type == GoalTopoActionType.REVISIT.value:
            evidence_source = "historical_observation"
            source_mapping_valid = bool(
                action.snapshot_image and action.capture_anchor_id
            )
            relations = ["OBSERVED_AT", "NAVIGABLE"]
        else:
            evidence_source = "frontier_projection"
            source_mapping_valid = action.source_frontier_index is not None
            relations = ["NAVIGABLE"]

        finite_cost = _finite(action.effective_cost_m)
        reachable = bool(action.reachable)
        candidates.append(GoalOverlayCandidate(
            action=action,
            goal_evidence={
                "source": evidence_source,
                "score": float(action.relevance),
                "current_observation": bool(action.current_observation),
            },
            # Final V7 deliberately performs no temporal/Beta score update.
            belief=float(action.relevance),
            cost_m=finite_cost,
            validity={
                "valid": bool(reachable and source_mapping_valid),
                "reachable": reachable,
                "source_mapping_valid": source_mapping_valid,
                "route_mode": action.route_mode,
            },
            topology_relations=relations,
        ))
    return GoalConditionedTopologyOverlay(metric_view, candidates)


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _direct_path_cost(
    planner, current_voxel, target_voxel, floor_height, pathfinder,
):
    try:
        distance, path_points = planner.get_direct_geodesic_distance(
            current_voxel,
            target_voxel,
            height=floor_height,
            pathfinder=pathfinder,
        )
    except Exception:
        return None, False
    distance = _finite(distance)
    return distance, bool(distance is not None and path_points is not None)


def _metric_action(
    action: GoalTopoAction,
    topology: PersistentBeliefTopology,
    frontiers,
    planner,
    pathfinder,
    current_voxel,
    floor_height,
    remaining_budget_m,
) -> MetricAction:
    anchor_id = None
    direct_cost = None
    topology_cost = None
    effective_cost = None
    reachable = False
    route_mode = MetricRouteMode.UNREACHABLE

    if action.action_type == GoalTopoActionType.DIRECT:
        direct_cost = 0.0
        effective_cost = 0.0
        reachable = True
        route_mode = MetricRouteMode.CURRENT
    elif action.action_type == GoalTopoActionType.EXPLORE:
        index = action.source_frontier_index
        if index is not None and 0 <= index < len(frontiers):
            direct_cost, reachable = _direct_path_cost(
                planner,
                current_voxel,
                np.asarray(frontiers[index].position, dtype=float)[:2],
                floor_height,
                pathfinder,
            )
        if reachable:
            effective_cost = direct_cost
            route_mode = MetricRouteMode.DIRECT_PATH
    elif action.action_type == GoalTopoActionType.REVISIT:
        if action.snapshot_image is not None:
            anchor_id = topology.capture_anchor_for_snapshot(action.snapshot_image)
        anchor = topology.nodes.get(anchor_id)
        if anchor is not None:
            direct_cost, direct_reachable = _direct_path_cost(
                planner,
                current_voxel,
                anchor.position,
                floor_height,
                pathfinder,
            )
            route, _ = topology.plan_safe_route_to_anchor(
                anchor_id, record_stats=False
            )
            topology_cost = (
                _finite(topology.route_length_m(route)) if route else None
            )
            if direct_reachable:
                effective_cost = direct_cost
                reachable = True
                route_mode = MetricRouteMode.DIRECT_PATH
            elif topology_cost is not None:
                effective_cost = topology_cost
                reachable = True
                route_mode = MetricRouteMode.SAFE_TOPO

    budget_ratio = (
        effective_cost / remaining_budget_m
        if effective_cost is not None and remaining_budget_m > 0.0 else None
    )
    return MetricAction(
        action_type=action.action_type.value,
        topo_node_id=action.topo_node_id,
        source_kind=action.source_kind,
        source_frontier_index=action.source_frontier_index,
        snapshot_image=action.snapshot_image,
        object_id=action.object_id,
        current_observation=bool(action.current_observation),
        relevance=float(np.clip(action.relevance, 0.0, 1.0)),
        capture_anchor_id=anchor_id,
        direct_geodesic_m=direct_cost,
        topology_route_m=topology_cost,
        effective_cost_m=effective_cost,
        route_mode=route_mode.value,
        reachable=reachable,
        remaining_budget_m=float(max(0.0, remaining_budget_m)),
        budget_ratio=_finite(budget_ratio),
    )


def build_metric_action_view(
    task_view: GoalTopoTaskView,
    topology: PersistentBeliefTopology,
    frontiers,
    planner,
    pathfinder,
    current_voxel,
    floor_height: float,
    remaining_steps: int,
    planner_step_m: float,
) -> MetricActionView:
    """Build a finite-safe, behavior-free metric view for all goal types."""
    remaining_budget_m = max(0, int(remaining_steps)) * max(
        0.0, float(planner_step_m)
    )
    actions = [
        _metric_action(
            action,
            topology,
            frontiers,
            planner,
            pathfinder,
            np.asarray(current_voxel, dtype=float)[:2],
            float(floor_height),
            float(remaining_budget_m),
        )
        for action in task_view.actions
    ]
    return MetricActionView(
        goal_type=str(task_view.goal.goal_type),
        subtask_id=str(task_view.goal.subtask_id),
        step=int(task_view.created_at_step),
        actions=actions,
    )


def evaluate_revisit_intervention_gate(
    metric_view: Union[MetricActionView, GoalConditionedTopologyOverlay],
    min_relevance_margin: float,
    max_budget_fraction: float,
    max_revisit_to_explore_ratio: float,
    planner_step_m: float,
) -> RevisitGateDecision:
    """Decide whether V7.1-B may reuse historical Snapshot evidence.

    This is deliberately conservative: a failed check asks the caller to use
    the untouched HGR prompt and navigation path for this step. It never
    changes rankings, evidence, or source mappings itself.  Final V7 passes a
    ``GoalConditionedTopologyOverlay`` here, so all three action types share
    the same evidence/belief/cost/validity decision interface.
    """
    # In final V7 this reads belief/cost/validity from overlay candidates.
    # Legacy ablations may still pass a MetricActionView directly.
    decision_actions = (
        metric_view.candidates
        if isinstance(metric_view, GoalConditionedTopologyOverlay)
        else metric_view.actions
    )
    reachable = [action for action in decision_actions if action.reachable]
    revisits = [
        action for action in reachable
        if action.action_type == GoalTopoActionType.REVISIT.value
        and action.effective_cost_m is not None
    ]
    alternatives = [
        action for action in reachable
        if action.action_type != GoalTopoActionType.REVISIT.value
    ]
    explores = [
        action for action in alternatives
        if action.action_type == GoalTopoActionType.EXPLORE.value
        and action.effective_cost_m is not None
    ]
    best_revisit = max(revisits, key=lambda action: action.relevance) if revisits else None
    best_alternative = (
        max(alternatives, key=lambda action: action.relevance)
        if alternatives else None
    )
    cheapest_explore = (
        min(explores, key=lambda action: action.effective_cost_m)
        if explores else None
    )
    revisit_cost = (
        _finite(best_revisit.effective_cost_m) if best_revisit is not None else None
    )
    budget = (
        _finite(best_revisit.remaining_budget_m)
        if best_revisit is not None else None
    )
    source_mapping_ok = bool(
        best_revisit is not None
        and best_revisit.snapshot_image
        and best_revisit.capture_anchor_id
    )
    relevance_margin = (
        float(best_revisit.relevance - best_alternative.relevance)
        if best_revisit is not None and best_alternative is not None else None
    )
    semantic_clear = bool(
        relevance_margin is not None
        and relevance_margin >= float(min_relevance_margin)
    )
    budget_ratio = (
        revisit_cost / budget
        if revisit_cost is not None and budget is not None and budget > 0.0
        else None
    )
    budget_ok = bool(
        revisit_cost is not None
        and budget is not None
        and budget > 0.0
        and revisit_cost <= float(max_budget_fraction) * budget
    )
    revisit_to_explore_ratio = None
    if cheapest_explore is None:
        opportunity_cost_ok = True
    elif revisit_cost is None:
        opportunity_cost_ok = False
    else:
        explore_cost = max(
            float(cheapest_explore.effective_cost_m),
            max(0.0, float(planner_step_m)),
        )
        revisit_to_explore_ratio = revisit_cost / explore_cost
        opportunity_cost_ok = bool(
            revisit_cost <= float(max_revisit_to_explore_ratio) * explore_cost
        )

    if best_revisit is None:
        reason = "no_reachable_revisit"
    elif not source_mapping_ok:
        reason = "snapshot_anchor_mapping_unavailable"
    elif best_alternative is None:
        reason = "no_reachable_nonhistorical_alternative"
    elif not semantic_clear:
        reason = "relevance_margin_below_threshold"
    elif not budget_ok:
        reason = "revisit_exceeds_budget_fraction"
    elif not opportunity_cost_ok:
        reason = "revisit_cost_exceeds_explore_ratio"
    else:
        reason = "accepted"
    return RevisitGateDecision(
        allowed=reason == "accepted",
        reason=reason,
        best_revisit_topo_id=(
            best_revisit.topo_node_id if best_revisit is not None else None
        ),
        best_alternative_topo_id=(
            best_alternative.topo_node_id if best_alternative is not None else None
        ),
        cheapest_explore_topo_id=(
            cheapest_explore.topo_node_id if cheapest_explore is not None else None
        ),
        relevance_margin=_finite(relevance_margin),
        revisit_cost_m=revisit_cost,
        remaining_budget_m=budget,
        budget_ratio=_finite(budget_ratio),
        revisit_to_explore_cost_ratio=_finite(revisit_to_explore_ratio),
        semantic_clear=semantic_clear,
        budget_ok=budget_ok,
        opportunity_cost_ok=opportunity_cost_ok,
        source_mapping_ok=source_mapping_ok,
    )


def revisit_gate_authorizes_selected_action(
    decision: RevisitGateDecision,
    selected_action: Optional[GoalTopoAction],
) -> bool:
    """Bind a global revisit decision to the historical node it evaluated.

    ``evaluate_revisit_intervention_gate`` compares the best reachable REVISIT
    with the best non-historical alternative.  The VLM may subsequently choose
    another historical Snapshot.  That different Snapshot has not passed the
    semantic, budget, or opportunity-cost checks and must therefore retain
    original HGR navigation instead of inheriting the best node's approval.
    """
    return bool(
        decision.allowed
        and selected_action is not None
        and selected_action.action_type == GoalTopoActionType.REVISIT
        and selected_action.topo_node_id == decision.best_revisit_topo_id
    )


def create_target_commitment(
    action: MetricAction,
    goal_key: str,
    step: int,
) -> TargetCommitment:
    """Create a V7.1-D commitment from one executable REVISIT action."""
    if (
        action.action_type != GoalTopoActionType.REVISIT.value
        or not action.snapshot_image
        or not action.capture_anchor_id
        or action.effective_cost_m is None
        or not np.isfinite(float(action.effective_cost_m))
    ):
        raise ValueError("Target commitment requires a finite mapped REVISIT")
    cost = float(action.effective_cost_m)
    return TargetCommitment(
        goal_key=str(goal_key),
        topo_node_id=str(action.topo_node_id),
        capture_anchor_id=str(action.capture_anchor_id),
        snapshot_image=str(action.snapshot_image),
        created_step=int(step),
        last_evaluated_step=int(step),
        initial_cost_m=cost,
        last_cost_m=cost,
        best_cost_m=cost,
        relevance=float(action.relevance),
    )


def evaluate_target_commitment(
    commitment: TargetCommitment,
    metric_view: MetricActionView,
    step: int,
    switch_relevance_margin: float,
    tie_margin: float,
    min_switch_saving_m: float,
    min_switch_saving_ratio: float,
    min_progress_m: float,
    max_no_progress_segments: int,
) -> tuple[TargetCommitmentDecision, Optional[TargetCommitment]]:
    """Re-evaluate a commitment after a segment without querying a VLM.

    A challenger is only used when it has a genuine semantic lead, or when the
    two candidates are semantically tied and it saves a material metric cost.
    The caller resolves a returned ``switch`` challenger through the current
    reversible action view; this function itself never mutates rankings or
    source mappings.
    """
    committed = next(
        (
            action for action in metric_view.actions
            if action.action_type == GoalTopoActionType.REVISIT.value
            and action.topo_node_id == commitment.topo_node_id
            and action.snapshot_image == commitment.snapshot_image
            and action.capture_anchor_id == commitment.capture_anchor_id
            and action.reachable
            and action.effective_cost_m is not None
        ),
        None,
    )
    if committed is None:
        return TargetCommitmentDecision(
            action="cancel",
            reason="target_and_safe_route_unreachable",
            committed_topo_id=commitment.topo_node_id,
            challenger_topo_id=None,
            committed_cost_m=None,
            challenger_cost_m=None,
            committed_relevance=None,
            challenger_relevance=None,
            progress_m=None,
            no_progress_segments=commitment.no_progress_segments,
        ), None

    cost = float(committed.effective_cost_m)
    progress = float(commitment.last_cost_m - cost)
    no_progress = (
        0 if progress >= float(min_progress_m)
        else int(commitment.no_progress_segments) + 1
    )
    candidates = [
        action for action in metric_view.actions
        if action.reachable
        and action.effective_cost_m is not None
        and not (
            action.action_type == GoalTopoActionType.REVISIT.value
            and action.topo_node_id == commitment.topo_node_id
            and action.snapshot_image == commitment.snapshot_image
        )
    ]
    challenger = max(candidates, key=lambda action: action.relevance) if candidates else None
    common = dict(
        committed_topo_id=commitment.topo_node_id,
        challenger_topo_id=(challenger.topo_node_id if challenger else None),
        committed_cost_m=cost,
        challenger_cost_m=(
            float(challenger.effective_cost_m) if challenger else None
        ),
        committed_relevance=float(committed.relevance),
        challenger_relevance=(float(challenger.relevance) if challenger else None),
        progress_m=_finite(progress),
        no_progress_segments=no_progress,
    )
    if no_progress >= int(max_no_progress_segments):
        return TargetCommitmentDecision(
            action="cancel", reason="repeated_no_progress", **common
        ), None
    if challenger is not None:
        relevance_gain = float(challenger.relevance - committed.relevance)
        if relevance_gain >= float(switch_relevance_margin):
            return TargetCommitmentDecision(
                action="switch", reason="semantic_challenger", **common
            ), None
        challenger_cost = float(challenger.effective_cost_m)
        saving = cost - challenger_cost
        saving_ratio = saving / max(cost, 1e-6)
        if (
            abs(relevance_gain) <= float(tie_margin)
            and saving >= float(min_switch_saving_m)
            and saving_ratio >= float(min_switch_saving_ratio)
        ):
            return TargetCommitmentDecision(
                action="switch", reason="cheaper_tied_challenger", **common
            ), None
    updated = TargetCommitment(
        goal_key=commitment.goal_key,
        topo_node_id=commitment.topo_node_id,
        capture_anchor_id=commitment.capture_anchor_id,
        snapshot_image=commitment.snapshot_image,
        created_step=commitment.created_step,
        last_evaluated_step=int(step),
        initial_cost_m=commitment.initial_cost_m,
        last_cost_m=cost,
        best_cost_m=min(float(commitment.best_cost_m), cost),
        relevance=float(committed.relevance),
        segments_executed=int(commitment.segments_executed) + 1,
        no_progress_segments=no_progress,
    )
    return TargetCommitmentDecision(action="keep", reason="hysteresis_keep", **common), updated


def plan_direct_first_revisit(
    snapshot_image: str,
    topology: PersistentBeliefTopology,
    planner,
    pathfinder,
    current_voxel,
    floor_height: float,
    observed_topo_id: Optional[str] = None,
    arrival_tolerance_m: float = 0.15,
) -> DirectFirstRoutePlan:
    """Plan V7.1-B execution without changing the selected Snapshot.

    The fixed snapshot capture anchor is queried directly first. Only when
    Pathfinder cannot reach it do we use a VISITED-only topology route. If
    neither route exists, the caller must execute the original HGR Snapshot.
    """
    snapshot_image = str(snapshot_image)
    if observed_topo_id is None:
        observed_topo_id = topology.observed_node_for_snapshot(snapshot_image)
    anchor_id = topology.capture_anchor_for_snapshot(snapshot_image)
    anchor = topology.nodes.get(anchor_id)
    observed = topology.nodes.get(observed_topo_id)
    look_at = (
        np.asarray(observed.position, dtype=float).copy()
        if observed is not None else None
    )

    if anchor is None:
        return DirectFirstRoutePlan(
            snapshot_image=snapshot_image,
            observed_topo_id=observed_topo_id,
            capture_anchor_id=None,
            route_mode=DirectFirstRouteMode.ORIGINAL_HGR_FALLBACK.value,
            target_voxel=None,
            look_at_voxel=look_at,
            route=[],
            next_waypoint_id=None,
            direct_geodesic_m=None,
            topology_route_m=None,
            fallback_reason="capture_anchor_unavailable",
        )

    direct_cost, direct_reachable = _direct_path_cost(
        planner,
        np.asarray(current_voxel, dtype=float)[:2],
        anchor.position,
        float(floor_height),
        pathfinder,
    )
    if direct_reachable:
        if direct_cost <= max(0.0, float(arrival_tolerance_m)):
            return DirectFirstRoutePlan(
                snapshot_image=snapshot_image,
                observed_topo_id=observed_topo_id,
                capture_anchor_id=anchor_id,
                route_mode=DirectFirstRouteMode.ANCHOR_REACHED.value,
                target_voxel=None,
                look_at_voxel=look_at,
                route=[anchor_id],
                next_waypoint_id=None,
                direct_geodesic_m=direct_cost,
                topology_route_m=0.0,
                fallback_reason=None,
            )
        return DirectFirstRoutePlan(
            snapshot_image=snapshot_image,
            observed_topo_id=observed_topo_id,
            capture_anchor_id=anchor_id,
            route_mode=DirectFirstRouteMode.DIRECT_PATH.value,
            target_voxel=np.asarray(anchor.position, dtype=float).copy(),
            look_at_voxel=look_at,
            route=[anchor_id],
            next_waypoint_id=None,
            direct_geodesic_m=direct_cost,
            topology_route_m=None,
            fallback_reason=None,
        )

    route, waypoint_id = topology.plan_safe_route_to_anchor(
        anchor_id, record_stats=True
    )
    topology_cost = _finite(topology.route_length_m(route)) if route else None
    if waypoint_id is not None:
        waypoint = topology.nodes.get(waypoint_id)
        if waypoint is not None:
            return DirectFirstRoutePlan(
                snapshot_image=snapshot_image,
                observed_topo_id=observed_topo_id,
                capture_anchor_id=anchor_id,
                route_mode=DirectFirstRouteMode.SAFE_TOPO.value,
                target_voxel=np.asarray(waypoint.position, dtype=float).copy(),
                look_at_voxel=look_at,
                route=route,
                next_waypoint_id=waypoint_id,
                direct_geodesic_m=direct_cost,
                topology_route_m=topology_cost,
                fallback_reason=None,
            )
    if route:
        return DirectFirstRoutePlan(
            snapshot_image=snapshot_image,
            observed_topo_id=observed_topo_id,
            capture_anchor_id=anchor_id,
            route_mode=DirectFirstRouteMode.ANCHOR_REACHED.value,
            target_voxel=None,
            look_at_voxel=look_at,
            route=route,
            next_waypoint_id=None,
            direct_geodesic_m=direct_cost,
            topology_route_m=topology_cost,
            fallback_reason=None,
        )
    return DirectFirstRoutePlan(
        snapshot_image=snapshot_image,
        observed_topo_id=observed_topo_id,
        capture_anchor_id=anchor_id,
        route_mode=DirectFirstRouteMode.ORIGINAL_HGR_FALLBACK.value,
        target_voxel=None,
        look_at_voxel=look_at,
        route=[],
        next_waypoint_id=None,
        direct_geodesic_m=direct_cost,
        topology_route_m=None,
        fallback_reason="direct_and_safe_topology_unreachable",
    )
