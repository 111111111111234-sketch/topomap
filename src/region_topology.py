"""Region-level projection for the V7.4 hierarchical topology policy.

The persistent HGR topology already records where observations and frontiers
were acquired: historical snapshots have a capture ``VISITED`` anchor and
frontiers have an ``ANCHORED_TO`` edge to the current visited place.  This
module turns those stable anchors into lightweight Region nodes without
learning a scene-specific segmentation or adding a second metric map.

The dynamic goal topology remains responsible for relevance/uncertainty.  We
only aggregate its executable actions by Region, rank Regions
lexicographically, and return every local HGR source in the selected Regions.
"""

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Optional, Tuple

from src.dynamic_goal_topology import DynamicGoalTopologyView
from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import (
    PersistentBeliefTopology,
    TopoEdgeType,
)


def _finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class RegionTopologyDecision:
    """A small set of stable regions and their local executable HGR actions."""

    region_ids: Tuple[str, ...]
    actions: Tuple[MetricAction, ...]
    region_action_counts: Dict[str, int]
    reason: str

    @property
    def snapshot_images(self) -> List[str]:
        return list(dict.fromkeys(
            str(action.snapshot_image)
            for action in self.actions
            if action.snapshot_image
        ))

    @property
    def frontier_indices(self) -> List[int]:
        return list(dict.fromkeys(
            int(action.source_frontier_index)
            for action in self.actions
            if action.source_frontier_index is not None
        ))

    def to_trace_dict(self):
        payload = asdict(self)
        payload["actions"] = [action.to_trace_dict() for action in self.actions]
        payload["snapshot_images"] = self.snapshot_images
        payload["frontier_indices"] = self.frontier_indices
        return payload


def _anchor_for_action(
    action: MetricAction,
    topology: PersistentBeliefTopology,
) -> Optional[str]:
    """Resolve the stable VISITED anchor associated with one HGR action."""
    if action.capture_anchor_id and action.capture_anchor_id in topology.nodes:
        return str(action.capture_anchor_id)
    if action.snapshot_image:
        anchor_id = topology.capture_anchor_for_snapshot(action.snapshot_image)
        if anchor_id in topology.nodes:
            return str(anchor_id)
    # Frontier identities are transient across TSDF updates, but their
    # ANCHORED_TO edge points at the stable place at which they were exposed.
    for edge in topology.edges.values():
        if (
            edge.source_id == action.topo_node_id
            and edge.edge_type == TopoEdgeType.ANCHORED_TO
            and edge.target_id in topology.nodes
        ):
            return str(edge.target_id)
    return None


def build_region_topology_decision(
    metric_view: MetricActionView,
    dynamic_view: Optional[DynamicGoalTopologyView],
    topology: PersistentBeliefTopology,
    max_regions: int = 2,
    require_frontier: bool = False,
) -> RegionTopologyDecision:
    """Select stable Regions then expose all valid local HGR candidates.

    There is deliberately no learned or manually weighted score: a Region is
    ordered by its best executable action's relevance, uncertainty, path cost,
    and stable ID.  This keeps the policy training-free and scene agnostic.
    """
    if int(max_regions) < 1:
        raise ValueError("max_regions must be positive")
    uncertainty_by_id = {
        str(node.topo_node_id): float(node.uncertainty)
        for node in (dynamic_view.nodes if dynamic_view is not None else [])
        if _finite(node.uncertainty)
    }
    groups: Dict[str, List[MetricAction]] = {}
    for action in metric_view.actions:
        if not action.reachable:
            continue
        # Only sources that the HGR VLM prompt can show are meaningful local
        # Region members.  Direct observations keep the pre-existing direct
        # first path and never need regional arbitration.
        if not (action.snapshot_image or action.source_frontier_index is not None):
            continue
        anchor_id = _anchor_for_action(action, topology)
        if anchor_id is not None:
            groups.setdefault(anchor_id, []).append(action)

    def action_key(action: MetricAction):
        cost = (
            float(action.effective_cost_m)
            if _finite(action.effective_cost_m) else float("inf")
        )
        return (
            float(action.relevance),
            uncertainty_by_id.get(str(action.topo_node_id), 0.0),
            -cost,
            str(action.topo_node_id),
        )

    rankable_regions = [
        region_id for region_id, actions in groups.items()
        if not require_frontier
        or any(action.source_frontier_index is not None for action in actions)
    ]
    ranked_regions = sorted(
        rankable_regions,
        key=lambda region_id: (action_key(max(groups[region_id], key=action_key)), region_id),
        reverse=True,
    )[:int(max_regions)]
    selected_actions = []
    for region_id in ranked_regions:
        selected_actions.extend(sorted(groups[region_id], key=action_key, reverse=True))
    return RegionTopologyDecision(
        region_ids=tuple(ranked_regions),
        actions=tuple(selected_actions),
        region_action_counts={key: len(groups[key]) for key in ranked_regions},
        reason=("ranked_stable_regions" if ranked_regions else "no_anchored_candidates"),
    )


def select_local_explore_for_snapshot(
    decision: RegionTopologyDecision,
    topology: PersistentBeliefTopology,
    snapshot_image: str,
) -> Optional[MetricAction]:
    """Continue through the selected snapshot's own stable Region.

    A snapshot answer is evidence, not an authorization to revisit it.  This
    helper therefore chooses the already ranked, reachable local Frontier in
    the same capture-anchor Region.  It never creates a cross-region shortcut
    or executes the historical snapshot itself.
    """
    anchor_id = next(
        (
            _anchor_for_action(action, topology)
            for action in decision.actions
            if action.snapshot_image == str(snapshot_image)
        ),
        None,
    )
    if anchor_id is None:
        return None
    return next(
        (
            action for action in decision.actions
            if action.source_frontier_index is not None
            and _anchor_for_action(action, topology) == anchor_id
        ),
        None,
    )
