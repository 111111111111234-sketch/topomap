"""HGR V2 contracts, goal state and certified route ownership.

No simulator, task ground truth or model requests live in this module. Legacy
experiments opt out by not installing a DualTopoRuntime on their Place graph.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from time import perf_counter
import copy
import json

import numpy as np

from src.route_guidance import RouteGuidance, grid_path, path_mask, visible


@dataclass
class Candidate:
    candidate_id: str
    entity_id: str
    kind: str
    place_id: str = None
    observed_evidence_refs: set = field(default_factory=set)
    hypothesis_refs: set = field(default_factory=set)
    route_id: str = None

    def context(self):
        return {"candidate_id": self.candidate_id, "entity_id": self.entity_id,
                "observed_evidence_refs": sorted(self.observed_evidence_refs),
                "hypothesis_refs": sorted(self.hypothesis_refs)}


@dataclass
class NavigationIntent:
    intent_id: str
    goal_id: str
    candidate_id: str
    entity_id: str
    kind: str
    observed_evidence_refs: set
    hypothesis_refs: set
    route_id: str
    status: str = "active"
    reason: str = "selected"


@dataclass(frozen=True)
class Feedback:
    event_id: str
    goal_id: str
    kind: str
    reason: str
    entity_id: str = None
    affected_hypotheses: tuple = ()


@dataclass
class RoutePlan:
    route_id: str
    route_places: list
    path: list
    transitions: list
    terminal: list
    segment_distances_m: list
    segment_kinds: list
    total_distance_m: float
    voxel_size: float = 1.0

    def __post_init__(self):
        path = np.asarray(self.path, float)
        if (path.ndim != 2 or path.shape[1] != 2 or not len(path)
                or not np.all(np.isfinite(path)) or not self.route_places):
            raise ValueError("RoutePlan requires a finite 2-D path and Place route")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("RoutePlan requires a positive voxel size")
        if not np.array_equal(path[-1], self.terminal):
            raise ValueError("RoutePlan path must end at its actual terminal")
        if [p for p, _ in self.transitions] != self.route_places[1:]:
            raise ValueError("Transitions must expand the ordered Place route")
        indices = [int(i) for _, i in self.transitions]
        if indices != sorted(indices) or any(i < 0 or i >= len(path) for i in indices):
            raise ValueError("Invalid transition path indices")
        actual_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*self.voxel_size)
        if (len(self.segment_distances_m) != len(self.segment_kinds)
                or any(not np.isfinite(d) or d < 0 for d in self.segment_distances_m)
                or not np.isclose(sum(self.segment_distances_m), self.total_distance_m)
                or not np.isclose(actual_length, self.total_distance_m)):
            raise ValueError("RoutePlan costs do not describe its certified path")

    def approach(self):
        return {"status": "valid", "route_id": self.route_id,
                "approach_place_id": self.route_places[-1],
                "route_places": self.route_places, "terminal": self.terminal,
                "certified_path": self.path, "transitions": self.transitions,
                "segment_distances_m": self.segment_distances_m,
                "segment_kinds": self.segment_kinds,
                "total_distance_m": self.total_distance_m,
                "cost_basis": "shared_certified_route",
                "route_plan": asdict(self)}


class GridPathCache:
    """Exact search-domain memoization, including failed queries.

    Hash the bounded search mask, not just the returned path: newly free cells
    may create a shorter route. Each call still searches current graph edges.
    """
    def __init__(self, enabled=False, capacity=256):
        self.enabled, self.capacity = enabled, capacity
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        self.context = (None, None, None)
        self.origins = {}
        self.hit_kinds = dict.fromkeys(("same_decision", "same_step", "cross_step", "cross_goal"), 0)
        self.search_seconds = self.hash_seconds = 0.0

    def query(self, mask, start, end):
        if not self.enabled:
            return grid_path(mask, start, end)
        started = perf_counter()
        key = (mask.shape, tuple(np.rint(start).astype(int)),
               tuple(np.rint(end).astype(int)), sha256(np.packbits(mask).tobytes()).digest())
        self.hash_seconds += perf_counter() - started
        if key in self.entries:
            self.hits += 1
            origin = self.origins[key]
            kind = ("cross_goal" if origin[0] != self.context[0] else
                    "cross_step" if origin[1] != self.context[1] else
                    "same_step" if origin[2] != self.context[2] else "same_decision")
            self.hit_kinds[kind] += 1
            self.entries.move_to_end(key)
            value = self.entries[key]
            return None if value is None else value.copy()
        self.misses += 1
        started = perf_counter()
        value = grid_path(mask, start, end)
        self.search_seconds += perf_counter() - started
        self.entries[key] = None if value is None else value.copy()
        self.origins[key] = self.context
        while len(self.entries) > self.capacity:
            old_key, _ = self.entries.popitem(last=False)
            self.origins.pop(old_key, None)
        return value

    def statistics(self):
        return {"hits": self.hits, "misses": self.misses, "hit_kinds": dict(self.hit_kinds),
                "search_seconds": self.search_seconds, "hash_seconds": self.hash_seconds,
                "scope": "episode cumulative; hit counts do not estimate time saved"}


def build_route(graph, mask, current, terminal, target, origin, observe_distance, cache):
    """Recover a continuous bounded route; retry alternate graph paths on failure."""
    excluded = set()
    routing = copy.copy(graph)
    while True:
        routing.edges = {k: e for k, e in graph.edges.items() if k not in excluded}
        route, _ = routing.shortest_path(origin, target)
        if not route:
            return None
        points = [np.rint(current).astype(float)]
        transitions, costs, kinds = [], [], []
        failed = None
        # Connector / edge / terminal segments have disjoint path intervals.
        segments = []
        for index, (a, b) in enumerate(zip(route, route[1:])):
            edge = graph.edges[(a, b)]
            if edge.source_anchor is None or edge.target_anchor is None:
                failed = (a, b)
                break
            bound = edge.path_length_m + 2 * graph.place_spacing_m
            segments.extend([(edge.source_anchor, bound, "connector" if index == 0 else "inter_edge", None, (a, b)),
                             (edge.target_anchor, bound, "transition", b, (a, b))])
        if failed is not None:
            excluded.add(failed)
            continue
        segments.append((terminal, 2 * graph.place_spacing_m + observe_distance, "terminal", None, None))
        for end, bound, kind, destination, edge_key in segments:
            start, end = points[-1], np.rint(end).astype(float)
            x, y = np.ogrid[:mask.shape[0], :mask.shape[1]]
            local = mask & ((np.hypot(x-start[0], y-start[1]) +
                             np.hypot(x-end[0], y-end[1])) * graph.voxel_size <= bound + 1e-6)
            segment = cache.query(local, start, end)
            if segment is None:
                failed = edge_key
                break
            cost = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum() * graph.voxel_size)
            if cost > bound + 1e-6:
                failed = edge_key
                break
            points.extend(segment[1:])
            costs.append(cost)
            kinds.append(kind)
            if destination is not None:
                transitions.append((destination, len(points)-1))
        else:
            path = np.asarray(points).tolist()
            identity = json.dumps([route, path, graph.voxel_size], separators=(",", ":"))
            return RoutePlan(sha256(identity.encode()).hexdigest()[:24], route, path,
                             transitions, np.rint(terminal).tolist(), costs, kinds, sum(costs), graph.voxel_size)
        if failed is None:  # Failed terminal: this approach cannot reach the actual goal.
            return None
        excluded.add(failed)


class CertifiedExecution(RouteGuidance):
    """RouteGuidance with proposal/acknowledgement separated from actual progress."""
    @classmethod
    def from_plan(cls, plan, shape):
        path = np.asarray(plan.path, float)
        result = cls(list(plan.route_places), path, list(plan.transitions),
                     path_mask(shape, path), np.asarray(plan.terminal, float))
        result.route_id = plan.route_id
        result.pending_cursor = None
        return result

    def step(self, current, max_distance_m, voxel_size, mask):
        before = self.cursor
        result = super().step(current, max_distance_m, voxel_size, mask)
        self.pending_cursor = self.cursor if result is not None else None
        self.cursor = before
        return result

    def acknowledge(self, actual):
        if self.pending_cursor is None:
            return False
        matched = np.allclose(actual, self.path[self.pending_cursor], atol=1e-5, rtol=0)
        if matched:
            self.cursor = self.pending_cursor
        self.pending_cursor = None
        return matched


class GoalState:
    def __init__(self, goal_id):
        self.goal_id = str(goal_id)
        self.candidates = {}
        self.active_intent = None
        self.observation_coverage = {}
        self.revoked_hypotheses = set()
        self.events = set()
        self.selection_count = 0

    def register(self, candidate):
        candidate.hypothesis_refs -= self.revoked_hypotheses
        old = self.candidates.get(candidate.candidate_id)
        if old:
            candidate.observed_evidence_refs |= old.observed_evidence_refs
        self.candidates[candidate.candidate_id] = candidate
        self._refresh_intent_evidence(candidate)
        return candidate

    def _refresh_intent_evidence(self, candidate):
        intent = self.active_intent
        if intent is not None and intent.candidate_id == candidate.candidate_id:
            # A new independent observation must protect the active intent too,
            # not only a candidate rebuilt at the next selector call.
            intent.observed_evidence_refs |= candidate.observed_evidence_refs
            intent.hypothesis_refs = set(candidate.hypothesis_refs)

    def select(self, candidate, route_id, intent_id=None):
        self.selection_count += 1
        self.active_intent = NavigationIntent(
            str(intent_id or f"{self.goal_id}:{self.selection_count}"), self.goal_id, candidate.candidate_id,
            candidate.entity_id, candidate.kind, set(candidate.observed_evidence_refs),
            set(candidate.hypothesis_refs), route_id)
        return self.active_intent

    def finish(self, reason):
        intent = self.active_intent
        if intent is not None:
            intent.status = "completed" if reason == "terminal_arrived" else "released"
            intent.reason = reason
        self.active_intent = None
        return None if intent is None else {"intent_id": intent.intent_id, "goal_id": intent.goal_id,
            "candidate_id": intent.candidate_id, "route_id": intent.route_id,
            "status": intent.status, "reason": reason}

    def apply(self, feedback):
        if feedback.goal_id != self.goal_id or feedback.event_id in self.events:
            return False
        self.events.add(feedback.event_id)
        if feedback.kind == "hypotheses_revoked":
            revoked = set(feedback.affected_hypotheses)
            self.revoked_hypotheses |= revoked
            for candidate in self.candidates.values():
                candidate.hypothesis_refs -= revoked
            intent = self.active_intent
            if intent is not None and intent.hypothesis_refs & revoked:
                intent.hypothesis_refs -= revoked
                if not intent.hypothesis_refs and not intent.observed_evidence_refs:
                    self.finish("hypothesis_basis_revoked")
                    return True
        return False

    def rebind(self, old_id, new_id):
        old_id, new_id = str(old_id), str(new_id)
        rebound = {}
        for candidate in self.candidates.values():
            if candidate.kind == "TARGET_APPROACH" and candidate.entity_id == old_id:
                candidate.entity_id = new_id
                candidate.candidate_id = f"{candidate.kind}:{new_id}"
            previous = rebound.get(candidate.candidate_id)
            if previous:
                candidate.observed_evidence_refs |= previous.observed_evidence_refs
                candidate.hypothesis_refs |= previous.hypothesis_refs
            rebound[candidate.candidate_id] = candidate
        self.candidates = rebound
        if (self.active_intent and self.active_intent.kind == "TARGET_APPROACH"
                and self.active_intent.entity_id == old_id):
            self.active_intent.entity_id = new_id
            self.active_intent.candidate_id = f"{self.active_intent.kind}:{new_id}"
        if self.active_intent and self.active_intent.candidate_id in self.candidates:
            self._refresh_intent_evidence(self.candidates[self.active_intent.candidate_id])


class DualTopoRuntime:
    def __init__(self, config):
        self.config = dict(config)
        self.cache = GridPathCache(bool(config.get("cache_paths", False)))
        self.goal = None
        self.execution = None
        self.selected_approach = None
        self.plan = None

    def begin_goal(self, goal_id):
        self.goal = GoalState(goal_id)
        self.execution = self.selected_approach = None
        self.plan = None
        self.logged_route_id = None
        self.cache_start = (self.cache.hits, self.cache.misses)
        self.detached_sources = set()
        self.cache.context = (str(goal_id), None, None)

    def synchronize(self, scene, planner):
        changes = []
        for old in scene.object_id_aliases:
            seen, new = {old}, scene.object_id_aliases[old]
            while new in scene.object_id_aliases and new not in seen:
                seen.add(new)
                new = scene.object_id_aliases[new]
            if new not in seen and new in scene.objects:
                if any(c.entity_id == str(old) for c in self.goal.candidates.values()):
                    changes.append({"kind": "entity_merged", "old_entity": str(old), "entity_id": str(new)})
                self.goal.rebind(old, new)
        live_objects = {str(x) for x in scene.objects}
        live_frontiers = {str(getattr(f, "topo_id", "")) for f in planner.frontiers}
        for key, candidate in list(self.goal.candidates.items()):
            live = live_objects if candidate.kind == "TARGET_APPROACH" else live_frontiers
            if candidate.entity_id not in live:
                intent = self.goal.active_intent
                if (candidate.kind == "EXPLORE" and self.config.get("persistent_intent", False)
                        and intent is not None and intent.candidate_id == key):
                    if key not in self.detached_sources:
                        changes.append({"kind": "source_detached", "candidate_id": key})
                        self.detached_sources.add(key)
                    continue
                del self.goal.candidates[key]
                changes.append({"kind": "source_removed", "candidate_id": key})
        return changes

    def active_hypothesis_refs(self):
        intent = None if self.goal is None else self.goal.active_intent
        return set() if intent is None else set(intent.hypothesis_refs)

    def resolve(self, graph, mask, current, terminal, observe_distance, origin=None, previous=None):
        started = perf_counter()
        origin = graph.current_place_id if origin is None else origin
        pinned = previous.get("approach_place_id") if previous and previous.get("status") == "valid" else None
        groups = [[pinned], sorted(p for p in graph.nodes if p != pinned)] if pinned else [sorted(graph.nodes)]
        for group in groups:
            plans = []
            for place in group:
                if place not in graph.nodes:
                    continue
                if place != origin and graph.distance_m(graph.nodes[place].position, terminal) > 2*graph.place_spacing_m + observe_distance:
                    continue
                plan = build_route(graph, mask, current, terminal, place, origin, observe_distance, self.cache)
                if plan:
                    plans.append(plan)
            if plans:
                best = min(plans, key=lambda p: (p.total_distance_m, len(p.route_places), p.route_places[-1]))
                return {**best.approach(), "certification": self.certification(graph, mask, best.route_places),
                        "observe_distance_m": float(observe_distance),
                        "route_query_seconds": perf_counter()-started,
                        "route_event": "approach_retained" if pinned == best.route_places[-1] else "approach_selected"}
        return {"status": "unavailable", "reason": "no_certified_approach_route",
                "route_query_seconds": perf_counter()-started}

    def candidate(self, choice, approach, observed=True):
        is_object = hasattr(choice, "cluster")
        entity = str(choice.cluster[0]) if is_object and len(choice.cluster) == 1 else str(getattr(choice, "topo_id", ""))
        kind = "TARGET_APPROACH" if is_object else "EXPLORE"
        evidence = {f"snapshot:{choice.image}:object:{entity}"} if is_object and observed else set()
        hypothesis = getattr(choice, "hypothesis_node_id", None)
        candidate = Candidate(f"{kind}:{entity}", entity, kind,
                              approach.get("approach_place_id"), evidence,
                              {str(hypothesis)} if hypothesis else set(), approach.get("route_id"))
        return self.goal.register(candidate)

    def install(self, choice, approach, shape, retain_intent=False, external_intent=None):
        from src.topology_navigation import SnapshotSourceCommitment
        if not retain_intent:
            self.goal.finish("selection_replaced")
        candidate = self.candidate(choice, approach)
        plan = RoutePlan(**approach["route_plan"])
        if retain_intent and self.goal.active_intent is not None:
            self.goal.active_intent.route_id = plan.route_id
        else:
            # In the rebuilt backend this object is an execution projection of
            # the semantic owner's intent. Sharing the ID lets every delayed
            # feedback be checked against one authoritative task identity.
            self.goal.select(
                candidate, plan.route_id,
                None if external_intent is None else external_intent.intent_id,
            )
        self.execution = CertifiedExecution.from_plan(plan, shape)
        self.plan = plan
        self.selected_approach = approach
        self.source = SnapshotSourceCommitment(choice) if candidate.kind == "TARGET_APPROACH" else None
        self.choice = copy.deepcopy(choice) if candidate.kind == "EXPLORE" else choice
        self.recovery_attempted = False

    def refresh_source(self, scene, planner):
        intent = self.goal.active_intent
        if intent is None:
            return None
        if self.source is not None:
            choice, _ = self.source.resolve(scene.objects, scene.object_id_aliases, scene.snapshots, scene.frames)
            if choice is not None:
                self.goal.rebind(intent.entity_id, choice.cluster[0])
                self.choice = choice
                self.selected_approach["object_id"] = int(choice.cluster[0])
            return choice
        live = next((f for f in planner.frontiers
                     if str(getattr(f, "topo_id", "")) == intent.entity_id), None)
        if self.config.get("persistent_intent", False) and self.execution is not None:
            # The installed terminal remains authoritative. Geometry is checked
            # by ensure_route before each motion, including detached sources.
            return self.choice
        return live

    def reusable(self, approach, graph, mask, current):
        if not approach or approach.get("status") != "valid" or "route_plan" not in approach:
            return False
        plan = RoutePlan(**approach["route_plan"])
        return (np.array_equal(np.rint(current), plan.path[0])
                and plan.voxel_size == graph.voxel_size
                and approach.get("certification") == self.certification(graph, mask, plan.route_places)
                and CertifiedExecution.from_plan(plan, mask.shape).valid(graph, mask))

    @staticmethod
    def certification(graph, mask, route):
        edges = []
        for a, b in zip(route, route[1:]):
            edge = graph.edges.get((a, b))
            if edge is None:
                edges.append([a, b, None])
            else:
                edges.append([a, b, str(edge.status), edge.path_length_m,
                              np.asarray(edge.source_anchor).tolist(), np.asarray(edge.target_anchor).tolist()])
        return {"mask": sha256(np.packbits(mask).tobytes()).hexdigest(), "shape": list(mask.shape),
                "edges": edges, "voxel_size": graph.voxel_size}

    def release(self, reason):
        result = self.goal.finish(reason)
        self.execution = self.selected_approach = None
        self.plan = None
        return result

    @property
    def uses_local_controller(self):
        """Whether an exact same-Place intent should use native TSDF motion."""
        return bool(
            self.config.get("local_same_place", False)
            and self.plan is not None
            and len(self.plan.route_places) == 1
        )

    def ensure_route(self, graph, mask, current, observe_distance):
        execution = self.execution
        if execution is None:
            return "missing_route"
        if self.uses_local_controller:
            path = self.cache.query(mask, current, execution.terminal)
            return "local_direct_retained" if path is not None else "route_unavailable"
        aligned = np.allclose(current, execution.path[execution.cursor], atol=1e-5, rtol=0)
        if aligned and execution.valid(graph, mask):
            return "retained"
        if aligned and execution.repair(mask, current) and execution.valid(graph, mask):
            boundaries = [0] + [int(i) for _, i in execution.transitions] + [len(execution.path)-1]
            costs = [float(np.linalg.norm(np.diff(execution.path[a:b+1], axis=0), axis=1).sum()*graph.voxel_size)
                     for a, b in zip(boundaries, boundaries[1:])]
            path = execution.path.tolist()
            route_id = sha256(json.dumps([execution.route, path, graph.voxel_size]).encode()).hexdigest()[:24]
            repaired_plan = RoutePlan(route_id, execution.route, path, execution.transitions,
                                      execution.terminal.tolist(), costs,
                                      ["repair_transition"]*(len(costs)-1)+["terminal"], sum(costs), graph.voxel_size)
            execution.route_id = route_id
            self.plan = repaired_plan
            self.goal.active_intent.route_id = route_id
            self.selected_approach = {**self.selected_approach, **repaired_plan.approach()}
            return "local_repair"
        remaining = execution.remaining_route()
        result = self.resolve(graph, mask, current, execution.terminal, observe_distance,
                              origin=remaining[0] if remaining else None, previous=self.selected_approach)
        if result["status"] != "valid":
            return "route_unavailable"
        self.plan = RoutePlan(**result["route_plan"])
        self.execution = CertifiedExecution.from_plan(self.plan, mask.shape)
        self.selected_approach = {**self.selected_approach, **result}
        self.goal.active_intent.route_id = result["route_id"]
        return "same_target_reroute"

    def acknowledge_motion(self, actual, target_arrived=False):
        """Acknowledge either certified guidance or native same-Place motion."""
        if self.execution is None:
            return False
        if self.uses_local_controller:
            if target_arrived:
                self.execution.cursor = len(self.execution.path) - 1
            self.execution.pending_cursor = None
            return True
        return self.execution.acknowledge(actual)

    def revoke(self, report, event_id):
        affected = tuple(dict.fromkeys([str(report["hypothesis_node_id"])] +
                                      [str(x) for x in report.get("cascade_deleted_nodes", [])]))
        feedback = Feedback(event_id, self.goal.goal_id, "hypotheses_revoked", "semantic_critic",
                            affected_hypotheses=affected)
        before = {key: value.context() for key, value in self.goal.candidates.items()}
        cancelled = self.goal.apply(feedback)
        if cancelled:
            self.execution = self.selected_approach = None
        return {**asdict(feedback), "intent_cancelled": cancelled,
                "candidates_before": before,
                "candidates_after": {key: value.context() for key, value in self.goal.candidates.items()}}
