"""Unified high-level policy over HGR's persistent and goal topology layers.

The persistent layer provides stable places, observations, and frontiers.  The
dynamic layer supplies goal-conditioned relevance and uncertainty.  This
module only decides which *kind* of HGR source should be considered next; it
does not replace TSDF planning, Habitat path finding, or the VLM's object-crop
selection inside a chosen Snapshot.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Iterable, Optional

from src.dynamic_goal_topology import DynamicGoalTopologyView
from src.metric_goal_execution import (
    MetricAction,
    MetricActionView,
    RevisitGateDecision,
)
from src.persistent_belief_topology import GoalTopoActionType


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class DualLayerTopologyDecision:
    """One explainable high-level choice with an exploration safety fallback."""

    action: Optional[MetricAction]
    fallback_explore: Optional[MetricAction]
    reason: str
    direct_margin: Optional[float]
    reachable_directs: int
    reachable_revisits: int
    reachable_explores: int

    @property
    def action_type(self) -> Optional[str]:
        return None if self.action is None else self.action.action_type

    def to_trace_dict(self):
        payload = asdict(self)
        payload["action"] = (
            None if self.action is None else self.action.to_trace_dict()
        )
        payload["fallback_explore"] = (
            None
            if self.fallback_explore is None
            else self.fallback_explore.to_trace_dict()
        )
        payload["action_type"] = self.action_type
        return payload


@dataclass(frozen=True)
class TopoExploreCommitment:
    """A stable high-level Frontier target for one navigation segment.

    The source Frontier object's list index is intentionally refreshed from
    the persistent topology each step.  The commitment identity itself is the
    stable topology node ID, not a transient Python object or list position.
    """

    topo_node_id: str
    source_frontier_index: int
    created_step: int
    last_step: int
    best_distance_voxels: float
    target_position_xy: tuple
    no_progress_steps: int = 0

    def to_trace_dict(self):
        return asdict(self)


def begin_topo_explore_commitment(
    action: MetricAction,
    current_voxel,
    frontier_position,
    step: int,
) -> TopoExploreCommitment:
    """Create a commitment only for a resolved executable Frontier action."""
    if (
        action.action_type != GoalTopoActionType.EXPLORE.value
        or action.source_frontier_index is None
    ):
        raise ValueError("Topo explore commitment requires an EXPLORE action")
    distance = float(
        math.dist(
            [float(value) for value in current_voxel[:2]],
            [float(value) for value in frontier_position[:2]],
        )
    )
    return TopoExploreCommitment(
        topo_node_id=str(action.topo_node_id),
        source_frontier_index=int(action.source_frontier_index),
        created_step=int(step),
        last_step=int(step),
        best_distance_voxels=distance,
        target_position_xy=tuple(
            float(value) for value in frontier_position[:2]
        ),
    )


def refresh_topo_explore_commitment(
    commitment: TopoExploreCommitment,
    metric_view: MetricActionView,
    current_voxel,
    frontier_positions: Dict[int, object],
    step: int,
    min_progress_voxels: float,
    max_no_progress_steps: int,
    geometric_rebind_radius_voxels: float = 0.0,
    live_target_position=None,
):
    """Refresh a commitment or release it with an explicit safe reason.

    A Frontier may be re-extracted with a new topology ID after a TSDF update.
    When enabled, geometric rebinding transfers the commitment to a nearby,
    reachable current Frontier.  If no such node exists but the original
    physical target remains live in the planner, it stays committed until the
    usual progress/failure checks decide otherwise.
    """
    matching = next(
        (
            action for action in metric_view.actions
            if action.action_type == GoalTopoActionType.EXPLORE.value
            and str(action.topo_node_id) == commitment.topo_node_id
            and action.reachable
            and action.source_frontier_index is not None
        ),
        None,
    )
    refresh_event = None
    if matching is None and float(geometric_rebind_radius_voxels) > 0.0:
        candidates = [
            action for action in metric_view.actions
            if action.action_type == GoalTopoActionType.EXPLORE.value
            and action.reachable
            and action.source_frontier_index is not None
            and int(action.source_frontier_index) in frontier_positions
        ]
        if candidates:
            def anchor_distance(action):
                position = frontier_positions[int(action.source_frontier_index)]
                return math.dist(
                    [float(value) for value in commitment.target_position_xy],
                    [float(value) for value in position[:2]],
                )
            candidate = min(candidates, key=anchor_distance)
            if anchor_distance(candidate) <= float(
                geometric_rebind_radius_voxels
            ):
                matching = candidate
                refresh_event = "geometric_rebind"
    if matching is None:
        if live_target_position is None:
            return None, "topology_action_missing"
        source_index = int(commitment.source_frontier_index)
        position = live_target_position
        refresh_event = "geometric_anchor_keep"
    else:
        source_index = int(matching.source_frontier_index)
        position = frontier_positions.get(source_index)
        if position is None:
            return None, "source_frontier_missing"
    distance = float(
        math.dist(
            [float(value) for value in current_voxel[:2]],
            [float(value) for value in position[:2]],
        )
    )
    if not _finite(distance):
        return None, "frontier_distance_nonfinite"
    improved = distance <= (
        float(commitment.best_distance_voxels) - float(min_progress_voxels)
    )
    updated = replace(
        commitment,
        topo_node_id=(
            str(matching.topo_node_id)
            if matching is not None else commitment.topo_node_id
        ),
        source_frontier_index=source_index,
        last_step=int(step),
        best_distance_voxels=min(
            float(commitment.best_distance_voxels), distance
        ),
        target_position_xy=tuple(float(value) for value in position[:2]),
        no_progress_steps=(
            0 if improved else int(commitment.no_progress_steps) + 1
        ),
    )
    if updated.no_progress_steps >= int(max_no_progress_steps):
        return None, "no_progress_budget_exhausted"
    return updated, refresh_event


def choose_dual_layer_topology_action(
    metric_view: MetricActionView,
    dynamic_view: Optional[DynamicGoalTopologyView],
    revisit_gate: Optional[RevisitGateDecision],
    min_direct_margin: float,
) -> DualLayerTopologyDecision:
    """Choose DIRECT, an authorized exact REVISIT, or EXPLORE.

    Dynamic relevance is already projected into ``metric_view`` by
    ``DynamicGoalTopology``.  Uncertainty is used only to break ties between
    exploration candidates; it is intentionally not mixed into a new weighted
    score.  Historical reuse remains governed by the existing conservative
    revisit gate.
    """

    uncertainty_by_id: Dict[str, float] = {
        str(node.topo_node_id): float(node.uncertainty)
        for node in (dynamic_view.nodes if dynamic_view is not None else [])
        if _finite(node.uncertainty)
    }
    reachable = [
        action
        for action in metric_view.actions
        if action.reachable and _finite(action.relevance)
    ]
    directs = [
        action for action in reachable
        if action.action_type == GoalTopoActionType.DIRECT.value
    ]
    revisits = [
        action for action in reachable
        if action.action_type == GoalTopoActionType.REVISIT.value
    ]
    explores = [
        action for action in reachable
        if action.action_type == GoalTopoActionType.EXPLORE.value
        and action.source_frontier_index is not None
    ]

    def stable_best(actions: Iterable[MetricAction]) -> Optional[MetricAction]:
        actions = list(actions)
        return max(
            actions,
            key=lambda action: (float(action.relevance), str(action.topo_node_id)),
        ) if actions else None

    best_direct = stable_best(directs)
    best_non_direct = stable_best(revisits + explores)
    direct_margin = None
    if best_direct is not None:
        direct_margin = (
            float(best_direct.relevance - best_non_direct.relevance)
            if best_non_direct is not None
            else float(best_direct.relevance)
        )

    def explore_key(action: MetricAction):
        uncertainty = uncertainty_by_id.get(str(action.topo_node_id), 0.0)
        cost = (
            float(action.effective_cost_m)
            if _finite(action.effective_cost_m)
            else float("inf")
        )
        return (
            float(action.relevance),
            uncertainty,
            -cost,
            str(action.topo_node_id),
        )

    ordered_explores = sorted(explores, key=explore_key, reverse=True)
    best_explore = ordered_explores[0] if ordered_explores else None

    if (
        best_direct is not None
        and direct_margin is not None
        and direct_margin >= float(min_direct_margin)
    ):
        action = best_direct
        reason = "direct_margin_accepted"
    else:
        approved_revisit_id = (
            revisit_gate.best_revisit_topo_id
            if revisit_gate is not None and revisit_gate.allowed
            else None
        )
        action = next(
            (
                candidate for candidate in revisits
                if candidate.topo_node_id == approved_revisit_id
            ),
            None,
        )
        if action is not None:
            reason = "revisit_gate_accepted"
        elif best_explore is not None:
            action = best_explore
            reason = (
                "explore_after_direct_margin"
                if best_direct is not None
                else "explore_without_direct"
            )
        else:
            reason = "no_reachable_topology_action"

    fallback_explore = next(
        (candidate for candidate in ordered_explores if candidate is not action),
        None,
    )
    return DualLayerTopologyDecision(
        action=action,
        fallback_explore=fallback_explore,
        reason=reason,
        direct_margin=direct_margin,
        reachable_directs=len(directs),
        reachable_revisits=len(revisits),
        reachable_explores=len(explores),
    )
