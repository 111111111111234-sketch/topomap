"""Route-only navigation over the Phase-A Place graph.

This module never selects a semantic target.  It receives the Place associated
with an already frozen HGR choice and returns either the next valid graph hop
or an explicit reason why the graph cannot route that same target.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Mapping, Optional

import numpy as np
import copy

from src.place_topology import PlaceEdgeStatus, PlaceTopology


class PlaceRouteExecution:
    """Keep waypoint identity alive across all local executor steps.

    Local arrival at any graph waypoint (including the final Place) must be
    consumed here before HGR can interpret arrival as semantic completion.
    The next decision resumes the same source and eventually installs HGR's
    original terminal segment.
    """

    def __init__(self):
        self.active_plan = None
        self.reached_place_id = None

    def reset(self):
        self.active_plan = None
        self.reached_place_id = None

    def start(self, plan):
        self.active_plan = plan if plan is not None and plan.uses_override else None
        if self.active_plan is None:
            self.reached_place_id = None

    def consume_waypoint_arrival(self, local_arrived):
        if not local_arrived or self.active_plan is None:
            return None
        completed = self.active_plan
        self.active_plan = None
        # Execution reached this edge's verified endpoint. Geometric Place
        # assignment may still prefer the source in an overlapping region.
        # Keep progress local to this intent; never mutate map ownership.
        self.reached_place_id = completed.next_place_id
        return completed


def restore_selected_snapshot(snapshot, object_ids, objects):
    """Rebind the live image while preserving the exact selected object set."""
    if snapshot is None or not object_ids or any(i not in objects for i in object_ids):
        return None
    choice = copy.copy(snapshot)
    choice.cluster = list(object_ids)
    return choice


class SnapshotSourceCommitment:
    """Freeze selected evidence; resolve identity only through recorded merges.

    Snapshot clustering may replace a representative image without invalidating
    its observation. Keep its pose and selected object set for this intent only.
    Never match a replacement object by category, proximity, or ground truth.
    """

    def __init__(self, snapshot):
        self.snapshot = copy.deepcopy(snapshot)

    def resolve(self, objects, aliases, snapshots, frames):
        original_ids = list(self.snapshot.cluster)
        resolved_ids = []
        diagnostic = {
            "source_image": self.snapshot.image,
            "source_in_snapshots": any(
                s.image == self.snapshot.image for s in snapshots.values()
            ),
            "source_in_frames": self.snapshot.image in frames,
            "original_object_ids": original_ids,
            "resolved_object_ids": resolved_ids,
            "merge_chains": [],
        }
        if not original_ids:
            diagnostic["reason"] = "empty_selected_object_set"
            return None, diagnostic
        for object_id in original_ids:
            chain = [object_id]
            while object_id in aliases:
                object_id = aliases[object_id]
                if object_id in chain:
                    diagnostic["reason"] = "object_merge_alias_cycle"
                    return None, diagnostic
                chain.append(object_id)
            diagnostic["merge_chains"].append(chain)
            if object_id not in objects:
                diagnostic["reason"] = "selected_object_removed_without_live_successor"
                return None, diagnostic
            if object_id not in resolved_ids:
                resolved_ids.append(object_id)
        choice = copy.deepcopy(self.snapshot)
        choice.cluster = list(resolved_ids)
        # The original evidence remains frozen; only execution IDs are resolved.
        choice.full_obj_list = {
            resolved: max(
                self.snapshot.full_obj_list.get(chain[0], 0.0)
                for chain in diagnostic["merge_chains"] if chain[-1] == resolved
            ) for resolved in resolved_ids
        }
        diagnostic["reason"] = (
            "recorded_object_merge" if any(len(c) > 1 for c in diagnostic["merge_chains"])
            else "live_snapshot" if diagnostic["source_in_snapshots"]
            else "retained_selected_observation"
        )
        return choice, diagnostic


class PlaceRouteMode(str, Enum):
    SAME_PLACE = "same_place"
    NEXT_HOP = "next_hop"
    UNMAPPED_TARGET = "unmapped_target"
    MISSING_CURRENT_PLACE = "missing_current_place"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True)
class PlaceRoutePlan:
    target_kind: str
    target_id: str
    current_place_id: Optional[str]
    target_place_id: Optional[str]
    route: List[str]
    route_length_m: Optional[float]
    next_place_id: Optional[str]
    target_voxel: Optional[np.ndarray]
    look_at_voxel: Optional[np.ndarray]
    route_mode: str
    reason: str

    @property
    def uses_override(self) -> bool:
        return self.route_mode == PlaceRouteMode.NEXT_HOP.value

    @property
    def route_valid(self) -> bool:
        return self.route_mode in (
            PlaceRouteMode.SAME_PLACE.value,
            PlaceRouteMode.NEXT_HOP.value,
        )

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["target_voxel"] = (
            None if self.target_voxel is None
            else np.asarray(self.target_voxel, dtype=float).tolist()
        )
        payload["look_at_voxel"] = (
            None if self.look_at_voxel is None
            else np.asarray(self.look_at_voxel, dtype=float).tolist()
        )
        payload["uses_override"] = self.uses_override
        payload["route_valid"] = self.route_valid
        return payload


def plan_place_route(
    topology: PlaceTopology,
    target_place_id: Optional[str],
    target_kind: str,
    target_id: str,
    terminal_position=None,
    *,
    reached_place_id=None,
) -> PlaceRoutePlan:
    """Plan one next hop without changing the supplied HGR target identity."""
    current = (
        topology.current_place_id if reached_place_id is None
        else str(reached_place_id)
    )
    target = None if target_place_id is None else str(target_place_id)
    common = {
        "target_kind": str(target_kind),
        "target_id": str(target_id),
        "current_place_id": current,
        "target_place_id": target,
    }
    if current is None or current not in topology.nodes:
        return PlaceRoutePlan(
            **common, route=[], route_length_m=None, next_place_id=None,
            target_voxel=None, look_at_voxel=None,
            route_mode=PlaceRouteMode.MISSING_CURRENT_PLACE.value,
            reason="current_pose_has_no_place",
        )
    if target is None or target not in topology.nodes:
        return PlaceRoutePlan(
            **common, route=[], route_length_m=None, next_place_id=None,
            target_voxel=None, look_at_voxel=None,
            route_mode=PlaceRouteMode.UNMAPPED_TARGET.value,
            reason="hgr_target_has_no_place_mapping",
        )
    if current == target:
        return PlaceRoutePlan(
            **common, route=[current], route_length_m=0.0,
            next_place_id=None, target_voxel=None, look_at_voxel=None,
            route_mode=PlaceRouteMode.SAME_PLACE.value,
            reason="execute_hgr_terminal_segment_in_current_place",
        )

    route, route_length = topology.shortest_path(current, target)
    if not route:
        return PlaceRoutePlan(
            **common, route=[], route_length_m=None, next_place_id=None,
            target_voxel=None, look_at_voxel=None,
            route_mode=PlaceRouteMode.DISCONNECTED.value,
            reason="no_valid_directed_place_route",
        )
    for source, destination in zip(route, route[1:]):
        edge = topology.edges.get((source, destination))
        if edge is None or edge.status != PlaceEdgeStatus.VALID:
            raise AssertionError("Place route contains a non-valid edge")

    next_place_id = route[1]
    next_edge = topology.edges[(current, next_place_id)]
    target_voxel = next_edge.target_anchor.copy()
    if len(route) > 2:
        look_at = topology.edges[(next_place_id, route[2])].target_anchor.copy()
    elif terminal_position is not None:
        look_at = np.asarray(terminal_position, dtype=float)[:2].copy()
    else:
        look_at = topology.nodes[target].position.copy()
    return PlaceRoutePlan(
        **common,
        route=route,
        route_length_m=route_length,
        next_place_id=next_place_id,
        target_voxel=target_voxel,
        look_at_voxel=look_at,
        route_mode=PlaceRouteMode.NEXT_HOP.value,
        reason="next_hop_on_valid_place_route",
    )


def select_reachable_approach_place(
    topology: PlaceTopology,
    approach_terminal_costs_m: Mapping[str, float],
) -> Optional[str]:
    """Choose minimum graph-route plus terminal cost, with stable tie-break."""
    current = topology.current_place_id
    if current is None or current not in topology.nodes:
        return None
    candidates = []
    for place_id, terminal_cost in approach_terminal_costs_m.items():
        place_id = str(place_id)
        try:
            terminal_cost = float(terminal_cost)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(terminal_cost) or terminal_cost < 0.0:
            continue
        if place_id == current:
            route_cost = 0.0
        else:
            route, route_cost = topology.shortest_path(current, place_id)
            if not route or route_cost is None:
                continue
        candidates.append((route_cost + terminal_cost, place_id))
    return min(candidates)[1] if candidates else None
