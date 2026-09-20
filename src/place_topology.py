"""Minimal, scene-agnostic Place topology for the HGR dual-layer design.

The graph deliberately separates four facts that the legacy persistent graph
mixed together:

* a Place is a persistent, local spatial identity;
* an observation records where it was captured;
* a frontier records every Place from which it was observed;
* an approach Place is selected only from a verified local path.

Phase A uses this module in ``shadow`` stage.  It records graph state and audit
information but is not allowed to influence HGR candidate selection or motion.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import copy
import heapq
import math
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np


def _xy(value) -> np.ndarray:
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        raise ValueError("Place topology positions must contain two finite values")
    return point[:2].copy()


def _finite_nonnegative(value) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0.0 else None


class PlaceEdgeEvidence(str, Enum):
    TRAVERSED_TRAJECTORY = "traversed_trajectory"
    KNOWN_FREE_PATH = "known_free_path"


class PlaceEdgeStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass
class PlaceNode:
    place_id: str
    position: np.ndarray
    created_step: int
    last_observed_step: int
    observation_ids: Set[str] = field(default_factory=set)
    visit_count: int = 1

    def to_trace_dict(self) -> dict:
        return {
            "place_id": self.place_id,
            "position": self.position.tolist(),
            "created_step": self.created_step,
            "last_observed_step": self.last_observed_step,
            "observation_ids": sorted(self.observation_ids),
            "visit_count": self.visit_count,
        }


@dataclass
class PlaceEdge:
    source_place_id: str
    target_place_id: str
    path_length_m: float
    evidence: PlaceEdgeEvidence
    status: PlaceEdgeStatus
    last_verified_step: int
    supporting_observation_ids: Set[str] = field(default_factory=set)
    local_path_reference: Optional[str] = None
    source_anchor: Optional[np.ndarray] = None
    target_anchor: Optional[np.ndarray] = None

    def to_trace_dict(self) -> dict:
        return {
            "source_place_id": self.source_place_id,
            "target_place_id": self.target_place_id,
            "path_length_m": self.path_length_m,
            "evidence": self.evidence.value,
            "status": self.status.value,
            "last_verified_step": self.last_verified_step,
            "supporting_observation_ids": sorted(
                self.supporting_observation_ids
            ),
            "local_path_reference": self.local_path_reference,
            "source_anchor": (
                None if self.source_anchor is None else self.source_anchor.tolist()
            ),
            "target_anchor": (
                None if self.target_anchor is None else self.target_anchor.tolist()
            ),
        }


@dataclass(frozen=True)
class PlaceAssignment:
    place_id: str
    created: bool
    reason: str
    path_length_m: Optional[float]

    def to_trace_dict(self) -> dict:
        return asdict(self)


@dataclass
class ObservationBinding:
    observation_id: str
    observed_from_place_ids: Set[str]
    capture_place_id: str
    capture_pose: np.ndarray
    created_step: int
    last_observed_step: int

    def to_trace_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "observed_from_place_ids": sorted(self.observed_from_place_ids),
            "capture_place_id": self.capture_place_id,
            "capture_pose": self.capture_pose.tolist(),
            "created_step": self.created_step,
            "last_observed_step": self.last_observed_step,
        }


@dataclass
class FrontierBinding:
    frontier_id: str
    frontier_position: np.ndarray
    observed_from_place_ids: Set[str]
    approach_place_id: Optional[str]
    approach_path_length_m: Optional[float]
    approach_reason: str
    source_frontier_index: Optional[int]
    last_observed_step: int
    approach_candidates_m: Dict[str, float] = field(default_factory=dict)

    def to_trace_dict(self) -> dict:
        return {
            "frontier_id": self.frontier_id,
            "frontier_position": self.frontier_position.tolist(),
            "observed_from_place_ids": sorted(self.observed_from_place_ids),
            "approach_place_id": self.approach_place_id,
            "approach_path_length_m": self.approach_path_length_m,
            "approach_reason": self.approach_reason,
            "source_frontier_index": self.source_frontier_index,
            "last_observed_step": self.last_observed_step,
            "approach_candidates_m": dict(sorted(
                self.approach_candidates_m.items()
            )),
        }


class PlaceTopology:
    """Episode-scoped Place graph with one geometric scale parameter.

    With no external local-path lookup, association is deliberately
    conservative: only the active Place may be reused, and only while both the
    Euclidean displacement and actually travelled distance remain within
    ``place_spacing_m``.  An optional mapping of verified path lengths enables
    deterministic loop closure without ever merging through a wall.
    """

    def __init__(
        self, place_spacing_m: float, voxel_size: float, stage: str = "shadow"
    ):
        spacing = _finite_nonnegative(place_spacing_m)
        if spacing is None or spacing <= 0.0:
            raise ValueError("place_spacing_m must be finite and positive")
        voxel = _finite_nonnegative(voxel_size)
        if voxel is None or voxel <= 0.0:
            raise ValueError("voxel_size must be finite and positive")
        self.place_spacing_m = spacing
        self.voxel_size = voxel
        self.stage = str(stage).strip().lower()
        if self.stage not in ("shadow", "route_only"):
            raise ValueError("PlaceTopology stage must be shadow or route_only")
        self.nodes: Dict[str, PlaceNode] = {}
        self.edges: Dict[Tuple[str, str], PlaceEdge] = {}
        self.observations: Dict[str, ObservationBinding] = {}
        self.frontiers: Dict[str, FrontierBinding] = {}
        self.current_place_id: Optional[str] = None
        self._next_place_index = 0
        self._last_pose: Optional[np.ndarray] = None
        self._trajectory_since_place_m = 0.0
        self.stats = {
            "places_created": 0,
            "place_reuses": 0,
            "verified_loop_closures": 0,
            "traversable_edges_created": 0,
            "observations_bound": 0,
            "frontiers_bound": 0,
            "frontier_approach_updates": 0,
        }

    def _distance_m(self, lhs, rhs) -> float:
        return float(np.linalg.norm(_xy(lhs) - _xy(rhs)) * self.voxel_size)

    def distance_m(self, lhs, rhs) -> float:
        """Public metric distance helper for the HGR entry adapter."""
        return self._distance_m(lhs, rhs)

    def local_connection_targets(self, position) -> Dict[str, np.ndarray]:
        """Return only local Place candidates for known-grid verification.

        The previously active Place is always retained so that a just-finished
        transition can verify its reverse direction even when discretization
        puts the segment slightly beyond the node spacing.  Existing graph
        neighbours are deliberately not inherited: repeatedly copying them to
        a newly active Place would turn a sparse Place graph into a metric
        closure of the explored free-space component.
        """
        point = _xy(position)
        targets = {
            place_id: node.position
            for place_id, node in self.nodes.items()
            if self._distance_m(node.position, point) <= self.place_spacing_m
        }
        active = self.current_place_id
        if active in self.nodes:
            targets[active] = self.nodes[active].position
        return targets

    def _new_place(self, position: np.ndarray, step: int) -> PlaceNode:
        place_id = f"place_{self._next_place_index}"
        self._next_place_index += 1
        node = PlaceNode(place_id, position.copy(), int(step), int(step))
        self.nodes[place_id] = node
        self.stats["places_created"] += 1
        return node

    @staticmethod
    def known_grid_path_lengths_m(
        traversable,
        start,
        targets: Mapping[str, np.ndarray],
        voxel_size: float,
    ) -> Dict[str, float]:
        """Shortest paths restricted to HGR's currently known free-space grid.

        This routine intentionally has no Habitat ``PathFinder`` argument.  A
        target is returned only when its rounded voxel and the start voxel are
        already marked traversable.  Diagonal corner cutting is forbidden.
        """
        mask = np.asarray(traversable, dtype=bool)
        if mask.ndim != 2 or not targets:
            return {}
        scale = _finite_nonnegative(voxel_size)
        if scale is None or scale <= 0.0:
            raise ValueError("voxel_size must be finite and positive")

        def cell(point):
            value = _xy(point)
            index = tuple(np.rint(value).astype(int))
            if (
                index[0] < 0 or index[0] >= mask.shape[0]
                or index[1] < 0 or index[1] >= mask.shape[1]
                or not mask[index]
            ):
                return None
            return index

        start_cell = cell(start)
        if start_cell is None:
            return {}
        target_cells: Dict[Tuple[int, int], List[str]] = {}
        for target_id, position in targets.items():
            target_cell = cell(position)
            if target_cell is not None:
                target_cells.setdefault(target_cell, []).append(str(target_id))
        if not target_cells:
            return {}

        distances = {start_cell: 0.0}
        queue = [(0.0, start_cell)]
        result: Dict[str, float] = {}
        cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
        diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        while queue and target_cells:
            distance, current = heapq.heappop(queue)
            if distance != distances.get(current):
                continue
            if current in target_cells:
                for target_id in target_cells.pop(current):
                    result[target_id] = distance * scale
                if not target_cells:
                    break
            for dx, dy in cardinal + diagonal:
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    neighbor[0] < 0 or neighbor[0] >= mask.shape[0]
                    or neighbor[1] < 0 or neighbor[1] >= mask.shape[1]
                    or not mask[neighbor]
                ):
                    continue
                if dx and dy:
                    # Both adjacent cardinal cells must be free, otherwise
                    # the diagonal would cross an obstacle corner.
                    if not (
                        mask[current[0] + dx, current[1]]
                        and mask[current[0], current[1] + dy]
                    ):
                        continue
                    step_cost = math.sqrt(2.0)
                else:
                    step_cost = 1.0
                candidate = distance + step_cost
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        return result

    @property
    def current_place_offset_m(self) -> float:
        """Actual travelled distance since the active Place was established."""
        return float(self._trajectory_since_place_m)

    def add_known_free_connections(
        self,
        path_lengths_m: Mapping[str, float],
        step: int,
        observation_id: str,
    ) -> int:
        """Connect the current Place to peers verified in the known TSDF grid.

        Grid traversability is symmetric, so both directed edges are added.
        Candidate locality is controlled by the caller (nearby Places plus
        existing graph neighbors); verified edge length is not clipped by the
        node-creation spacing.
        """
        current = self.current_place_id
        if current is None or current not in self.nodes:
            return 0
        added = 0
        for target, raw_length in sorted(path_lengths_m.items()):
            target = str(target)
            length = _finite_nonnegative(raw_length)
            if (
                target == current
                or target not in self.nodes
                or length is None
            ):
                continue
            for source_id, target_id in ((current, target), (target, current)):
                existed = (source_id, target_id) in self.edges
                if existed and self.edges[
                    (source_id, target_id)
                ].status == PlaceEdgeStatus.VALID:
                    continue
                self.add_traversable_edge(
                    source_id,
                    target_id,
                    length,
                    PlaceEdgeEvidence.KNOWN_FREE_PATH,
                    step,
                    supporting_observation_ids=(observation_id,),
                    local_path_reference=(
                        f"known_tsdf_grid:{step}:{source_id}->{target_id}"
                    ),
                    source_anchor=(
                        self._last_pose
                        if source_id == current and self._last_pose is not None
                        else self.nodes[source_id].position
                    ),
                    target_anchor=(
                        self.nodes[target].position
                        if target_id == target
                        else self._last_pose
                        if self._last_pose is not None
                        else self.nodes[target_id].position
                    ),
                )
                added += int(not existed)
        return added

    @staticmethod
    def shadow_navigation_passthrough(frozen_decision: dict) -> dict:
        """Explicit Phase-A boundary: observe a decision without changing it."""
        return copy.deepcopy(frozen_decision)

    def observe_pose(
        self,
        position,
        step: int,
        observation_id: Optional[str] = None,
        verified_paths_m: Optional[Mapping[str, float]] = None,
        trajectory_traversable: bool = True,
    ) -> PlaceAssignment:
        """Associate a pose with a Place and add only auditable trajectory edges.

        ``verified_paths_m`` maps an existing Place ID to a path length obtained
        from the currently known free-space planner.  Invalid/non-finite values
        are ignored.  Candidate selection is by path length then stable ID, so
        dictionary or edge insertion order cannot change ownership.
        """
        pos = _xy(position)
        motion_m = (
            0.0 if self._last_pose is None
            else self._distance_m(self._last_pose, pos)
        )
        self._trajectory_since_place_m += motion_m
        previous_pose = None if self._last_pose is None else self._last_pose.copy()
        self._last_pose = pos.copy()

        candidates: List[Tuple[float, str, str]] = []
        for place_id, raw_path_m in (verified_paths_m or {}).items():
            node = self.nodes.get(str(place_id))
            path_m = _finite_nonnegative(raw_path_m)
            if (
                node is not None
                and path_m is not None
                and path_m <= self.place_spacing_m
                and self._distance_m(node.position, pos)
                <= self.place_spacing_m
            ):
                candidates.append((path_m, node.place_id, "verified_free_path"))

        active = self.nodes.get(self.current_place_id)
        if (
            active is not None
            and trajectory_traversable
            and self._distance_m(active.position, pos) <= self.place_spacing_m
            and self._trajectory_since_place_m <= self.place_spacing_m
        ):
            candidates.append((
                self._trajectory_since_place_m,
                active.place_id,
                "active_traversed_path",
            ))

        previous_place_id = self.current_place_id
        if candidates:
            path_m, place_id, reason = min(candidates, key=lambda item: item[:2])
            node = self.nodes[place_id]
            node.last_observed_step = int(step)
            node.visit_count += 1
            if observation_id:
                node.observation_ids.add(str(observation_id))
            if place_id != previous_place_id:
                self.stats["verified_loop_closures"] += 1
                if (
                    trajectory_traversable
                    and previous_place_id is not None
                    and previous_place_id in self.nodes
                ):
                    self.add_traversable_edge(
                        previous_place_id,
                        place_id,
                        self._trajectory_since_place_m,
                        PlaceEdgeEvidence.TRAVERSED_TRAJECTORY,
                        step,
                        supporting_observation_ids=(observation_id,),
                        local_path_reference=(
                            None if observation_id is None
                            else f"trajectory:{observation_id}"
                        ),
                        source_anchor=(
                            self.nodes[previous_place_id].position
                            if previous_pose is None else previous_pose
                        ),
                        target_anchor=pos,
                    )
            self.current_place_id = place_id
            # Keep accumulating the actually travelled distance while the
            # active Place is reused.  Otherwise a curved path could stay in
            # one Place forever merely because every individual control step
            # is short.  A verified loop closure starts a new local segment.
            if place_id != previous_place_id:
                self._trajectory_since_place_m = 0.0
            self.stats["place_reuses"] += 1
            return PlaceAssignment(place_id, False, reason, path_m)

        node = self._new_place(pos, step)
        if observation_id:
            node.observation_ids.add(str(observation_id))
        self.current_place_id = node.place_id
        trajectory_m = self._trajectory_since_place_m
        self._trajectory_since_place_m = 0.0
        if (
            trajectory_traversable
            and previous_place_id is not None
            and previous_place_id in self.nodes
        ):
            self.add_traversable_edge(
                previous_place_id,
                node.place_id,
                trajectory_m,
                PlaceEdgeEvidence.TRAVERSED_TRAJECTORY,
                step,
                supporting_observation_ids=(observation_id,),
                local_path_reference=(
                    None if observation_id is None
                    else f"trajectory:{observation_id}"
                ),
                source_anchor=(
                    self.nodes[previous_place_id].position
                    if previous_pose is None else previous_pose
                ),
                target_anchor=pos,
            )
        return PlaceAssignment(
            node.place_id,
            True,
            "no_verified_nearby_place",
            trajectory_m if previous_place_id is not None else None,
        )

    def add_traversable_edge(
        self,
        source_place_id: str,
        target_place_id: str,
        path_length_m: float,
        evidence: PlaceEdgeEvidence,
        step: int,
        supporting_observation_ids: Iterable[Optional[str]] = (),
        local_path_reference: Optional[str] = None,
        source_anchor=None,
        target_anchor=None,
    ) -> PlaceEdge:
        source = str(source_place_id)
        target = str(target_place_id)
        length = _finite_nonnegative(path_length_m)
        if source not in self.nodes or target not in self.nodes:
            raise KeyError("A traversable edge must connect existing Places")
        if source == target:
            raise ValueError("A traversable edge cannot be a self-loop")
        if length is None:
            raise ValueError("path_length_m must be finite and nonnegative")
        if not isinstance(evidence, PlaceEdgeEvidence):
            evidence = PlaceEdgeEvidence(str(evidence))
        support = {str(item) for item in supporting_observation_ids if item}
        source_anchor = (
            self.nodes[source].position.copy()
            if source_anchor is None else _xy(source_anchor)
        )
        target_anchor = (
            self.nodes[target].position.copy()
            if target_anchor is None else _xy(target_anchor)
        )
        key = (source, target)
        edge = self.edges.get(key)
        if edge is None:
            edge = PlaceEdge(
                source, target, length, evidence, PlaceEdgeStatus.VALID,
                int(step), support, local_path_reference,
                source_anchor, target_anchor,
            )
            self.edges[key] = edge
            self.stats["traversable_edges_created"] += 1
        else:
            edge.path_length_m = length
            edge.evidence = evidence
            edge.status = PlaceEdgeStatus.VALID
            edge.last_verified_step = int(step)
            edge.supporting_observation_ids.update(support)
            if local_path_reference is not None:
                edge.local_path_reference = str(local_path_reference)
            edge.source_anchor = source_anchor
            edge.target_anchor = target_anchor
        return edge

    def invalidate_edge(self, source_place_id: str, target_place_id: str, step: int):
        edge = self.edges.get((str(source_place_id), str(target_place_id)))
        if edge is None:
            return False
        edge.status = PlaceEdgeStatus.INVALID
        edge.last_verified_step = int(step)
        return True

    def bind_observation(
        self,
        observation_id: str,
        observed_from_place_id: str,
        capture_pose,
        step: int,
    ) -> ObservationBinding:
        """Bind a Snapshot without ever moving its immutable capture pose."""
        obs_id = str(observation_id)
        place_id = str(observed_from_place_id)
        if place_id not in self.nodes:
            raise KeyError(f"Unknown Place {place_id!r}")
        binding = self.observations.get(obs_id)
        if binding is None:
            binding = ObservationBinding(
                obs_id, {place_id}, place_id, _xy(capture_pose), int(step), int(step)
            )
            self.observations[obs_id] = binding
            self.stats["observations_bound"] += 1
        else:
            binding.observed_from_place_ids.add(place_id)
            binding.last_observed_step = int(step)
        self.nodes[place_id].observation_ids.add(obs_id)
        return binding

    def observe_frontier(
        self,
        frontier_id: str,
        frontier_position,
        observed_from_place_id: str,
        step: int,
        source_frontier_index: Optional[int] = None,
        verified_approach_path_m: Optional[float] = None,
    ) -> FrontierBinding:
        """Record observation and verified approach as separate relations."""
        stable_id = str(frontier_id)
        place_id = str(observed_from_place_id)
        if place_id not in self.nodes:
            raise KeyError(f"Unknown Place {place_id!r}")
        position = _xy(frontier_position)
        binding = self.frontiers.get(stable_id)
        if binding is None:
            binding = FrontierBinding(
                stable_id,
                position,
                set(),
                None,
                None,
                "no_verified_approach",
                source_frontier_index,
                int(step),
            )
            self.frontiers[stable_id] = binding
            self.stats["frontiers_bound"] += 1
        binding.frontier_position = position
        binding.observed_from_place_ids.add(place_id)
        binding.source_frontier_index = source_frontier_index
        binding.last_observed_step = int(step)

        path_m = _finite_nonnegative(verified_approach_path_m)
        if path_m is not None:
            old = binding.approach_candidates_m.get(place_id)
            binding.approach_candidates_m[place_id] = (
                path_m if old is None else min(old, path_m)
            )
        previous = binding.approach_place_id
        if binding.approach_candidates_m:
            best_place, best_path = min(
                binding.approach_candidates_m.items(),
                key=lambda item: (item[1], item[0]),
            )
            binding.approach_place_id = best_place
            binding.approach_path_length_m = best_path
            binding.approach_reason = "shortest_verified_known_path"
        if binding.approach_place_id != previous:
            self.stats["frontier_approach_updates"] += 1
        return binding

    def shortest_path(
        self, source_place_id: str, target_place_id: str
    ) -> Tuple[List[str], Optional[float]]:
        """Dijkstra route over valid directed traversability edges only."""
        source, target = str(source_place_id), str(target_place_id)
        if source not in self.nodes or target not in self.nodes:
            return [], None
        queue = [(0.0, source)]
        distances = {source: 0.0}
        parent: Dict[str, Optional[str]] = {source: None}
        while queue:
            distance, current = heapq.heappop(queue)
            if distance != distances.get(current):
                continue
            if current == target:
                route = [target]
                while parent[route[-1]] is not None:
                    route.append(parent[route[-1]])
                return list(reversed(route)), distance
            outgoing = sorted(
                (
                    edge for edge in self.edges.values()
                    if edge.source_place_id == current
                    and edge.status == PlaceEdgeStatus.VALID
                ),
                key=lambda edge: edge.target_place_id,
            )
            for edge in outgoing:
                candidate = distance + edge.path_length_m
                if candidate < distances.get(edge.target_place_id, float("inf")):
                    distances[edge.target_place_id] = candidate
                    parent[edge.target_place_id] = current
                    heapq.heappush(queue, (candidate, edge.target_place_id))
        return [], None

    def get_statistics(self) -> dict:
        payload = dict(self.stats)
        payload.update({
            "place_count": len(self.nodes),
            "valid_edge_count": sum(
                edge.status == PlaceEdgeStatus.VALID
                for edge in self.edges.values()
            ),
            "invalid_edge_count": sum(
                edge.status == PlaceEdgeStatus.INVALID
                for edge in self.edges.values()
            ),
            "observation_count": len(self.observations),
            "frontier_count": len(self.frontiers),
            "frontiers_with_verified_approach": sum(
                binding.approach_place_id is not None
                for binding in self.frontiers.values()
            ),
        })
        return payload

    def audit(self) -> dict:
        """Return machine-readable structural violations for experiment logs."""
        violations = []
        for key, edge in sorted(self.edges.items()):
            label = f"{key[0]}->{key[1]}"
            if edge.source_place_id not in self.nodes:
                violations.append(f"{label}:missing_source")
            if edge.target_place_id not in self.nodes:
                violations.append(f"{label}:missing_target")
            if _finite_nonnegative(edge.path_length_m) is None:
                violations.append(f"{label}:invalid_path_length")
            if not edge.local_path_reference:
                violations.append(f"{label}:missing_path_reference")
            if not edge.supporting_observation_ids:
                violations.append(f"{label}:missing_support")
            for anchor_name, anchor in (
                ("source_anchor", edge.source_anchor),
                ("target_anchor", edge.target_anchor),
            ):
                try:
                    _xy(anchor)
                except (TypeError, ValueError):
                    violations.append(f"{label}:invalid_{anchor_name}")
        for frontier_id, binding in sorted(self.frontiers.items()):
            approach = binding.approach_place_id
            if approach is None:
                continue
            if approach not in binding.observed_from_place_ids:
                violations.append(
                    f"{frontier_id}:approach_not_observed_from"
                )
            if approach not in binding.approach_candidates_m:
                violations.append(
                    f"{frontier_id}:approach_has_no_verified_path"
                )
        return {
            "valid": not violations,
            "violation_count": len(violations),
            "violations": violations,
        }

    def to_trace_dict(self, include_graph: bool = True) -> dict:
        payload = {
            "stage": self.stage,
            "current_place_id": self.current_place_id,
            "place_spacing_m": self.place_spacing_m,
            "statistics": self.get_statistics(),
            "audit": self.audit(),
        }
        if include_graph:
            payload.update({
                "places": [
                    self.nodes[key].to_trace_dict() for key in sorted(self.nodes)
                ],
                "traversable_edges": [
                    self.edges[key].to_trace_dict() for key in sorted(self.edges)
                ],
                "observations": [
                    self.observations[key].to_trace_dict()
                    for key in sorted(self.observations)
                ],
                "frontiers": [
                    self.frontiers[key].to_trace_dict()
                    for key in sorted(self.frontiers)
                ],
            })
        return payload


def replay_place_shadow_steps(records: Iterable[dict]) -> dict:
    """Replay frozen Phase-A deltas and prove shadow decision passthrough."""
    graph = None
    step_count = 0
    decision_count = 0
    for record in records:
        if not record:
            continue
        if graph is None:
            graph = PlaceTopology(
                record["place_spacing_m"], record["voxel_size"],
                stage=record.get("stage", "shadow"),
            )
        pose = record["pose"]
        assignment = graph.observe_pose(
            pose["position"],
            pose["step"],
            pose.get("observation_id"),
            verified_paths_m=pose.get("verified_paths_m", {}),
        )
        if assignment.to_trace_dict() != pose["expected_assignment"]:
            raise AssertionError(
                f"Place assignment mismatch at step {pose['step']}: "
                f"{assignment.to_trace_dict()} != "
                f"{pose['expected_assignment']}"
            )
        added = graph.add_known_free_connections(
            pose.get("verified_paths_m", {}),
            pose["step"],
            pose.get("observation_id") or f"pose:{pose['step']}",
        )
        if added != int(record.get("known_connections_added", added)):
            raise AssertionError(
                f"Known connection mismatch at step {pose['step']}"
            )
        for observation in record.get("observations", []):
            graph.bind_observation(
                observation["observation_id"],
                observation["expected_place_id"],
                observation["capture_pose"],
                pose["step"],
            )
        for frontier in record.get("frontiers", []):
            graph.observe_frontier(
                frontier["frontier_id"],
                frontier["frontier_position"],
                frontier["expected_place_id"],
                pose["step"],
                source_frontier_index=frontier.get(
                    "source_frontier_index"
                ),
                verified_approach_path_m=frontier.get(
                    "verified_approach_path_m"
                ),
            )

        disabled_output = record["frozen_navigation_decision"]
        enabled_output = graph.shadow_navigation_passthrough(disabled_output)
        if enabled_output != disabled_output:
            raise AssertionError(
                f"Shadow changed navigation decision at step {pose['step']}"
            )
        if enabled_output != record["shadow_navigation_decision"]:
            raise AssertionError(
                f"Recorded shadow output mismatch at step {pose['step']}"
            )
        if graph.get_statistics() != record["graph_statistics"]:
            raise AssertionError(
                f"Graph statistics mismatch at step {pose['step']}"
            )
        if graph.audit() != record["graph_audit"]:
            raise AssertionError(f"Graph audit mismatch at step {pose['step']}")
        step_count += 1
        decision_count += 1

    if graph is None:
        raise ValueError("No Place shadow replay records were provided")
    return {
        "valid": True,
        "steps_replayed": step_count,
        "decisions_compared": decision_count,
        "navigation_mismatches": 0,
        "graph_statistics": graph.get_statistics(),
        "graph_audit": graph.audit(),
    }
