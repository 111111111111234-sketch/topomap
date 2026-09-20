"""Phase C: goal-local evidence and intents over the existing Place graph.

No simulator truth, independent spatial graph, or category-specific policy.
Costs use valid Place edges plus known-grid connectors and terminal paths.
"""

import copy
import math
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from src.place_topology import PlaceTopology
from src.goal_topology_overlay import GoalTopologyOverlay
from src.goal_observation_coverage import ObservationCoverage
from src.topology_navigation import SnapshotSourceCommitment


@dataclass
class GoalCandidate:
    candidate_id: str
    intent: str
    entity_id: str
    source_image: str
    object_id: object
    frontier_index: object
    place_id: str
    observation_id: str
    capture_pose: list
    terminal: list
    look_at: list
    route: list
    graph_cost_m: float
    connector_cost_m: float
    terminal_cost_m: float
    support: object = None
    status: str = "unverified"
    evidence_place_id: object = None

    @property
    def cost_m(self):
        return self.graph_cost_m + self.connector_cost_m + self.terminal_cost_m

    def to_trace_dict(self):
        return {**asdict(self), "cost_m": self.cost_m}


def resolve_entity(object_id, objects, aliases):
    seen = set()
    while object_id in aliases:
        if object_id in seen:
            return None
        seen.add(object_id)
        object_id = aliases[object_id]
    return object_id if object_id in objects else None


def invalidate_blocked_hop(graph, plan, occupied, step):
    """Only positive obstacle evidence invalidates an executed graph edge."""
    if plan is None or not plan.uses_override or occupied is None:
        return False
    cell = np.rint(plan.target_voxel).astype(int)
    mask = np.asarray(occupied, dtype=bool)
    if (cell < 0).any() or (cell >= mask.shape).any() or not mask[tuple(cell)]:
        return False
    return graph.invalidate_edge(plan.current_place_id, plan.next_place_id, step)


class PlaceGoalNavigation:
    def __init__(self, goal_id, voxel_size, observe_distance_m=1.5, view_spacing_m=.5,
                 cache_paths=True, topological_actions=False,
                 event_driven_selection=False, evidence_coverage=False):
        self.goal_id = str(goal_id)
        self.voxel_size = float(voxel_size)
        self.observe_distance_m = float(observe_distance_m)
        self.view_spacing_m = float(view_spacing_m)
        self.cache_paths = bool(cache_paths)
        self.topological_actions = bool(topological_actions)
        self.event_driven_selection = bool(event_driven_selection)
        self.evidence_coverage = bool(evidence_coverage)
        self.coverage = ObservationCoverage(float(voxel_size), float(view_spacing_m))
        self.coverage_versions = {}
        self._coverage_signature = None
        self.active_coverage_version = None
        self._planning_cache = None
        self.planning_metrics = {}
        if min(self.voxel_size, self.observe_distance_m, self.view_spacing_m) <= 0:
            raise ValueError("Phase C geometry scales must be positive")
        self.evidence = {}
        self.checked_views = set()
        self.feedback = []
        self.active = None
        self.source = None
        self.candidates = []
        self.exclusions = []
        self.selection_reason = "new_goal"
        self.goal_topology = GoalTopologyOverlay(self.goal_id)

    def view_key(self, entity, position, look_at):
        position, look_at = np.asarray(position), np.asarray(look_at)
        xy = tuple(np.rint(position * self.voxel_size / self.view_spacing_m).astype(int))
        direction = look_at - position
        heading = int(round(math.atan2(direction[1], direction[0]) / (math.pi / 4))) % 8
        return (str(entity), *xy, heading)

    def record_support(self, candidate, score, step):
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Support must be a finite score in [0, 1]")
        # One representative per entity/view; repeated pipeline visits never sum.
        key = self.view_key(candidate.entity_id, candidate.capture_pose, candidate.look_at)
        previous = self.evidence.get(key)
        if previous is None or score > previous["support"]:
            self.evidence[key] = {
                "entity_id": candidate.entity_id, "observation_id": candidate.observation_id,
                "place_id": candidate.evidence_place_id, "capture_pose": candidate.capture_pose,
                "look_at": candidate.look_at, "support": float(score),
                "source": "vlm_candidate_match", "step": int(step),
            }

    def support(self, entity):
        values = [e["support"] for e in self.evidence.values() if e["entity_id"] == entity]
        return max(values) if values else None

    def _rebind_evidence(self, scene):
        # Reproject both checked views and positive evidence after actual merges.
        def canonical(entity):
            if not entity.startswith("object:"):
                return entity
            resolved = resolve_entity(int(entity.split(":")[1]), scene.objects, scene.object_id_aliases)
            return f"object:{resolved}" if resolved is not None else entity
        evidence = {}
        for key, value in self.evidence.items():
            entity = canonical(key[0])
            key = (entity, *key[1:])
            value = {**value, "entity_id": entity}
            if key not in evidence or value["support"] > evidence[key]["support"]:
                evidence[key] = value
        self.evidence = evidence
        self.checked_views = {(canonical(k[0]), *k[1:]) for k in self.checked_views}
        for old_entity in {key[1] for key in self.goal_topology.checked}:
            self.goal_topology.rebind_entity(old_entity, canonical(old_entity))
        for old_entity in list(self.coverage.sources):
            self.coverage.rebind(old_entity, canonical(old_entity))
        for feedback in self.feedback:
            feedback["entity_id"] = canonical(feedback["entity_id"])

    def _planning_query(self, key, operation):
        # Cache exists only during one synchronous candidate build. No reuse
        # across map updates, pose changes, edge invalidation or goal switches.
        self.planning_metrics["queries"] = self.planning_metrics.get("queries", 0) + 1
        if self._planning_cache is not None and key in self._planning_cache:
            self.planning_metrics["cache_hits"] = self.planning_metrics.get("cache_hits", 0) + 1
            return self._planning_cache[key]
        started = perf_counter()
        value = operation()
        self.planning_metrics["search_seconds"] = self.planning_metrics.get("search_seconds", 0.) + perf_counter() - started
        if self._planning_cache is not None:
            self._planning_cache[key] = value
        return value

    def _grid_paths(self, graph, mask, start, targets):
        key = ("grid", tuple(np.asarray(start)),
               tuple((name, tuple(np.asarray(point))) for name, point in targets.items()))
        return self._planning_query(key, lambda: graph.known_grid_path_lengths_m(
            mask, start, targets, self.voxel_size,
        ))

    def _route_to_terminal(self, graph, mask, current, terminals, look_at, entity,
                           approach_ids, intent):
        best = None
        origin = graph.current_place_id
        if origin not in graph.nodes:
            return None
        for place_id in sorted(set(approach_ids)):
            if (self.topological_actions and not self.evidence_coverage
                    and self.goal_topology.is_checked(intent, entity, place_id)):
                continue
            route, cost = self._planning_query(
                ("graph", origin, place_id), lambda: graph.shortest_path(origin, place_id),
            )
            if cost is None:
                continue
            # Current-Place terminal paths start at actual pose, not its center.
            start = current if place_id == origin else graph.nodes[place_id].position
            connector = 0.0
            if place_id != origin:
                connectors = self._grid_paths(
                    graph, mask, current, {origin: graph.nodes[origin].position},
                )
                if origin not in connectors:
                    continue
                connector = connectors[origin]
            allowed = {
                str(i): np.asarray(t) for i, t in enumerate(terminals)
                if (self.coverage.available(entity, t, look_at, self.coverage_versions[entity])
                    if self.evidence_coverage else (self.topological_actions
                    or self.view_key(entity, t, look_at) not in self.checked_views))
            }
            lengths = self._grid_paths(graph, mask, start, allowed)
            for key, length in lengths.items():
                # A terminal connector is local; long-range travel must use the
                # Place graph rather than silently replacing it with grid A*.
                if length > 2 * graph.place_spacing_m + self.observe_distance_m:
                    continue
                terminal = allowed[key].tolist()
                rank = (cost + connector + length, place_id, tuple(terminal))
                if best is None or rank < best[0]:
                    best = (rank, place_id, route, cost, connector, length, terminal)
        return best

    def build_candidates(self, graph, scene, planner, current):
        started = perf_counter()
        self.planning_metrics = {"queries": 0, "cache_hits": 0, "search_seconds": 0.}
        self._planning_cache = {} if self.cache_paths else None
        try:
            return self._build_candidates(graph, scene, planner, current)
        finally:
            self._planning_cache = None
            self.planning_metrics["build_seconds"] = perf_counter() - started

    def _build_candidates(self, graph, scene, planner, current):
        self._rebind_evidence(scene)
        self.coverage_versions = {}
        self.candidates, self.exclusions = [], []
        mask = getattr(planner, "island", None)
        if mask is None:
            self.exclusions.append({"reason": "known_free_grid_unavailable"})
            self.goal_topology.project(graph, self.candidates, self.evidence)
            return []
        mask = np.asarray(mask, dtype=bool)
        # Clusters are disjoint in HGR; explicitly deduplicate stable object IDs.
        representatives = {}
        for image, snapshot in sorted(scene.snapshots.items()):
            for obj_id in sorted(snapshot.cluster):
                if obj_id in scene.objects:
                    representatives.setdefault(obj_id, snapshot)
                    if self.evidence_coverage:
                        center = planner.habitat2voxel(scene.objects[obj_id]["bbox"].center)[:2]
                        self.coverage.observe_source(f"object:{obj_id}", snapshot.obs_point[:2], center)
        for obj_id, snapshot in sorted(representatives.items()):
            entity = f"object:{obj_id}"
            binding = graph.observations.get(f"snapshot:{snapshot.image}")
            if binding is None:
                self.exclusions.append({"entity_id": entity, "reason": "unmapped_observation"})
                continue
            center = np.asarray(planner.habitat2voxel(scene.objects[obj_id]["bbox"].center)[:2], dtype=float)
            radius = self.observe_distance_m / self.voxel_size
            if self.evidence_coverage:
                self.coverage.observe_source(entity, snapshot.obs_point[:2], center)
                self.coverage_versions[entity] = self.coverage.version(entity, center, mask, radius + 1)
            # Known-free cells on the safety ring, without a GT pathfinder.
            lo = np.maximum(0, np.floor(center - radius - 1).astype(int))
            hi = np.minimum(mask.shape, np.ceil(center + radius + 2).astype(int))
            cells = np.argwhere(mask[lo[0]:hi[0], lo[1]:hi[1]]) + lo
            cells = [cell for cell in cells if abs(np.linalg.norm(cell - center) - radius) <= .75]
            approaches = [pid for pid, node in graph.nodes.items()
                          if np.linalg.norm(node.position - center) * self.voxel_size
                          <= self.observe_distance_m + graph.place_spacing_m]
            approaches.extend([graph.current_place_id, binding.capture_place_id])
            best = self._route_to_terminal(graph, mask, current, cells, center, entity,
                                           [p for p in approaches if p in graph.nodes], "Verify")
            if best is None:
                self.exclusions.append({"entity_id": entity, "reason": "no_unchecked_reachable_observation_point"})
                continue
            _, place, route, cost, connector, terminal_cost, terminal = best
            self.candidates.append(GoalCandidate(
                (f"verify:{obj_id}@{place}" if self.topological_actions else f"verify:{obj_id}"),
                "Verify", entity, snapshot.image, obj_id, None,
                place, f"snapshot:{snapshot.image}", np.asarray(snapshot.obs_point[:2]).tolist(),
                terminal, center.tolist(), route, cost, connector, terminal_cost,
                self.support(entity), evidence_place_id=binding.capture_place_id,
            ))
        for index, frontier in enumerate(planner.frontiers):
            entity = str(getattr(frontier, "topo_id", ""))
            binding = graph.frontiers.get(entity)
            if binding is None:
                self.exclusions.append({"entity_id": entity, "reason": "unmapped_frontier"})
                continue
            # Frontier center must itself have a known-free terminal connection.
            look_at = np.asarray(frontier.position[:2]) + np.asarray(frontier.orientation[:2]) * 3
            if self.evidence_coverage:
                self.coverage_versions[entity] = self.coverage.version(
                    entity, frontier.position[:2], mask, self.observe_distance_m/self.voxel_size + 1)
            best = self._route_to_terminal(
                graph, mask, current, [frontier.position[:2]], look_at, entity,
                binding.approach_candidates_m.keys(),
                "Explore",
            )
            if best is None:
                self.exclusions.append({"entity_id": entity, "reason": "no_unchecked_reachable_frontier"})
                continue
            _, place, route, cost, connector, terminal_cost, terminal = best
            self.candidates.append(GoalCandidate(
                (f"explore:{entity}@{place}" if self.topological_actions else f"explore:{entity}"),
                "Explore", entity, str(frontier.image), None, index,
                place, f"frontier:{entity}", np.asarray(current).tolist(), terminal,
                look_at.tolist(), route, cost, connector, terminal_cost, self.support(entity),
                evidence_place_id=graph.current_place_id,
            ))
        for candidate in self.candidates:
            if self.evidence_coverage:
                candidate.candidate_id += f"/view:{self.coverage.bearing(candidate.terminal, candidate.look_at)}"
            prior = next((f for f in reversed(self.feedback) if f["entity_id"] == candidate.entity_id), None)
            if prior:
                candidate.status = prior["outcome"]
        self.goal_topology.project(graph, self.candidates, self.evidence, semantic_events=self.evidence_coverage)
        if self.evidence_coverage:
            self.coverage.project(self.goal_topology)
            # Geometry is consumed by routing and failure recovery. Only source
            # evidence of currently executable actions affects semantic revision.
            signature = tuple(sorted((c.entity_id, self.coverage_versions[c.entity_id][0])
                                     for c in self.candidates))
            if signature != self._coverage_signature:
                self._coverage_signature = signature
                self.goal_topology.revision += 1
                self.goal_topology.last_change_reason = "verification_opportunity_or_source_changed"
        return self.candidates

    def select(self, candidate, choice, support, step):
        self.record_support(candidate, support, step)
        candidate.support = self.support(candidate.entity_id)
        candidate.status = "navigating"
        self.active = candidate
        self.active_coverage_version = self.coverage_versions.get(candidate.entity_id)
        self.source = SnapshotSourceCommitment(choice) if candidate.intent == "Verify" else None
        self.goal_topology.mark_decision()

    def resume(self, scene, planner):
        if self.active is None:
            return None, {"reason": "no_active_intent"}
        if self.source is not None:
            choice, info = self.source.resolve(scene.objects, scene.object_id_aliases, scene.snapshots, scene.frames)
            self._rebind_evidence(scene)
            if choice is not None:
                self.active.entity_id = f"object:{choice.cluster[0]}"
                self.active.object_id = choice.cluster[0]
                if self.evidence_coverage and self.active_coverage_version is not None:
                    self.active_coverage_version = (tuple(sorted(self.coverage.sources.get(self.active.entity_id, set()))),
                                                    *self.active_coverage_version[1:])
            return choice, info
        choice = next((f for f in planner.frontiers
                       if str(getattr(f, "topo_id", "")) == self.active.entity_id), None)
        return choice, {"reason": "live_frontier" if choice is not None else "frontier_source_disappeared"}

    def finish(self, outcome, step, position=None, look_at=None):
        if self.active is None:
            return None
        candidate = self.active
        if position is not None and (not self.evidence_coverage or outcome in (
                "verification_matched", "verification_rejected", "verification_uncertain", "frontier_observed")):
            self.checked_views.add(self.view_key(candidate.entity_id, position, look_at))
            # Do not immediately retry a planned site if executor arrival tolerance
            # placed the camera in an adjacent quantized cell.
            self.checked_views.add(self.view_key(candidate.entity_id, candidate.terminal, candidate.look_at))
        event = {"goal_id": self.goal_id, "candidate_id": candidate.candidate_id,
                 "entity_id": candidate.entity_id, "intent": candidate.intent,
                 "outcome": outcome, "step": step, "route": candidate.route,
                 "observation_id": candidate.observation_id,
                 "position": None if position is None else np.asarray(position).tolist()}
        self.feedback.append(event)
        if self.evidence_coverage:
            self.coverage.finish(candidate, outcome, self.active_coverage_version, step)
            self.goal_topology.revision += 1
            self.goal_topology.last_change_reason = outcome
        else:
            self.goal_topology.mark_checked(candidate, outcome, step)
        self.selection_reason = outcome
        self.active = self.source = None
        return event

    def trace(self):
        return {"goal_id": self.goal_id, "selection_reason": self.selection_reason,
                "candidates": [c.to_trace_dict() for c in self.candidates],
                "excluded": self.exclusions, "evidence": list(self.evidence.values()),
                "checked_view_count": len(self.checked_views), "feedback": self.feedback,
                "planning_metrics": dict(self.planning_metrics),
                "goal_topology": self.goal_topology.to_trace_dict(),
                "topological_actions": self.topological_actions,
                "event_driven_selection": self.event_driven_selection,
                "evidence_coverage": self.evidence_coverage,
                **({"coverage": self.coverage.trace()} if self.evidence_coverage else {}),
                "active": self.active.to_trace_dict() if self.active else None}
