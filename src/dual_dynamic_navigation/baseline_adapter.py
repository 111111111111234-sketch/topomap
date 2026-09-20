"""Baseline-preserving adapter for the rebuilt dual dynamic topology.

It observes original HGR state and selections. It never selects a semantic
target, verifies a goal, or requests task completion. Route-only behavior is
owned by the existing Place/TSDF geometry path in the runner.
"""

from dataclasses import dataclass
from collections import Counter
from typing import Optional

from src.hierarchical_navigator import SemanticCandidate

from .goal_graph import GoalGraph
from .scene_map import SceneMap


@dataclass
class BaselineSelectionRecord:
    step: int
    source_kind: str
    entity_id: str
    source_id: str
    approach_place_id: Optional[str]
    route_id: Optional[str]
    route_distance_m: Optional[float]

    def to_trace_dict(self):
        return {
            "step": self.step,
            "source_kind": self.source_kind,
            "entity_id": self.entity_id,
            "source_id": self.source_id,
            "approach_place_id": self.approach_place_id,
            "route_id": self.route_id,
            "route_distance_m": self.route_distance_m,
        }


class BaselineDualDynamicNavigator:
    """L1/L2 observer whose public contract preserves original HGR policy."""

    backend_name = "dual_dynamic"

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.stage = str(self.config.get("stage", "shadow")).strip().lower()
        if self.stage not in ("shadow", "route_only", "known_space"):
            raise ValueError("dual_dynamic_navigation.stage must be shadow or route_only")
        for flag in ("preserve_hgr_selector", "preserve_hgr_stop"):
            if self.config.get(flag, True) is not True:
                raise ValueError(f"{self.stage} requires {flag}=true")
        if self.stage in ("shadow", "known_space") and self.config.get("object_approach", False):
            raise ValueError("shadow cannot enable object_approach")
        self.scene_map = SceneMap()
        self.goal_graph = None
        self.goal_id = None
        self.selections = []
        self.stats = {
            "map_updates": 0,
            "hgr_selections_observed": 0,
            "semantic_requests_added": 0,
            "terminal_verification_requests_added": 0,
        }

    @property
    def preserves_hgr_policy(self):
        return True

    def validate_restore(self, config):
        """Reject policy/stage changes instead of silently importing old state."""
        expected = type(self)(config)
        if self.stage != expected.stage or self.config != expected.config:
            raise ValueError("Dual dynamic checkpoint configuration differs from this run")
        return self

    def begin_goal(self, goal_id):
        self.goal_id = str(goal_id)
        self.goal_graph = GoalGraph(self.goal_id)
        self.selections = []

    def sync(self, graph, scene, frontiers, event_id, min_detection=1):
        """Ingest current spatial facts without changing any HGR object."""
        topology_delta = self.scene_map.sync_place_topology(
            graph, f"{event_id}:places"
        )
        candidates = []
        evidence_positions = {}
        seen_entities = set()
        for snapshot_id, snapshot in scene.snapshots.items():
            binding = graph.observations.get(f"snapshot:{snapshot_id}")
            place_id = None if binding is None else binding.capture_place_id
            for object_id in snapshot.cluster:
                obj = scene.objects.get(object_id)
                if obj is None or int(obj.get("num_detections", 0)) < int(min_detection):
                    continue
                entity_id = self.scene_map.resolve_entity(str(object_id))
                evidence_id = f"snapshot:{snapshot_id}:object:{object_id}"
                candidates.append(SemanticCandidate(
                    candidate_id=f"HGR_ENTITY:{entity_id}:snapshot:{snapshot_id}",
                    source_kind="snapshot",
                    entity_id=entity_id,
                    place_id=place_id,
                    route_id=None,
                    route_distance_m=None,
                    route_reachable=False,
                    snapshot_image=str(snapshot_id),
                    object_id=int(object_id),
                    evidence_refs=(evidence_id,),
                ))
                observation_position = getattr(snapshot, "obs_point", None)
                if observation_position is not None:
                    evidence_positions[evidence_id] = tuple(observation_position[:2])
                seen_entities.add(entity_id)
        for source_index, frontier in enumerate(frontiers):
            frontier_id = str(getattr(
                frontier, "topo_id", getattr(frontier, "frontier_id", source_index)
            ))
            binding = graph.frontiers.get(frontier_id)
            place_id = None if binding is None else binding.approach_place_id
            candidates.append(SemanticCandidate(
                candidate_id=f"HGR_FRONTIER:{frontier_id}",
                source_kind="frontier",
                entity_id=frontier_id,
                place_id=place_id,
                route_id=None,
                route_distance_m=None,
                route_reachable=False,
                source_index=source_index,
                hypothesis_refs=((str(frontier.hypothesis_node_id),)
                    if getattr(frontier, "hypothesis_node_id", None) else ()),
            ))
        try:
            step = int(str(event_id).rsplit(":", 1)[-1])
        except ValueError:
            step = -1
        candidate_delta = self.scene_map.update_candidates(
            candidates, f"{event_id}:evidence", goal_id=self.goal_id,
            step=step, evidence_positions=evidence_positions,
        )
        if self.goal_graph is not None:
            self.goal_graph.project(candidates, {}, f"{event_id}:goal")
        self.stats["map_updates"] += 1
        return {
            "topology": topology_delta.to_trace_dict(),
            "evidence": candidate_delta.to_trace_dict(),
            "entity_count_observed": len(seen_entities),
            "candidate_count_observed": len(candidates),
        }

    def observe_hgr_selection(self, choice, approach, step):
        """Record, but never alter, the source returned by original HGR."""
        cluster = list(getattr(choice, "cluster", []) or [])
        if cluster:
            source_kind = "snapshot"
            entity_id = str(cluster[0])
            source_id = str(getattr(choice, "image", ""))
        else:
            source_kind = "frontier"
            entity_id = str(getattr(choice, "topo_id", ""))
            source_id = entity_id
        approach = approach or {}
        record = BaselineSelectionRecord(
            int(step), source_kind, entity_id, source_id,
            approach.get("approach_place_id"), approach.get("route_id"),
            approach.get("total_distance_m"),
        )
        self.selections.append(record)
        self.stats["hgr_selections_observed"] += 1
        return record

    def diagnostics(self):
        """Return bounded per-subtask diagnostics suitable for metric JSON."""
        goal = self.goal_graph
        last = getattr(self, "last_selection", None)
        last_selection = None
        if last is not None:
            last_selection = {"type": type(last).__name__}
            for name in ("reason", "kind", "source_id", "entity_id", "evidence_digest"):
                value = getattr(last, name, None)
                if value is not None:
                    last_selection[name] = getattr(value, "value", value)
        intent = getattr(self, "active_intent", None)
        active_intent = None
        if intent is not None:
            active_intent = {
                "intent_id": intent.intent_id,
                "state": intent.state,
                "kind": intent.task.kind.value,
                "source_id": intent.task.source_id,
                "entity_id": intent.task.entity_id,
                "recovery_attempts": intent.recovery_attempts,
                "verification_views": intent.verification_views,
                "motion_steps": intent.motion_steps,
                "requires_verification": bool(intent.task.requires_verification),
            }
        candidate_reasons = Counter(
            str(row.get("reason", "eligible"))
            for row in getattr(self, "candidate_diagnostics", ())
        )
        return {
            "backend": self.backend_name,
            "stage": self.stage,
            "goal_id": self.goal_id,
            "scene_map": self.scene_map.to_trace_dict(self.goal_id),
            "goal_graph": None if goal is None else {
                "revision": goal.revision,
                "places": len(goal.places),
                "entities": len(goal.entities),
                "revoked_hypotheses": len(goal.revoked_hypotheses),
            },
            "stats": dict(self.stats),
            "mechanisms": {
                name: getattr(self, name)
                for name in (
                    "semantic_selection_mode", "enable_candidate_annotations",
                    "enable_revisit_deduplication", "enable_revisit_cost_gate",
                    "enable_persistent_intent", "enable_incremental_preemption",
                    "enable_terminal_verification",
                    "enable_selective_verification",
                    "selective_verification_goal_types",
                    "max_frontier_intent_motion_steps",
                    "historical_confirmation_views",
                ) if hasattr(self, name)
            },
            "last_selection": last_selection,
            "active_intent": active_intent,
            "candidate_reasons": dict(sorted(candidate_reasons.items())),
        }

    def to_trace_dict(self):
        return {
            "backend": self.backend_name,
            "stage": self.stage,
            "preserves_hgr_selector": True,
            "preserves_hgr_stop": True,
            "goal_id": self.goal_id,
            "scene_map": self.scene_map.to_trace_dict(self.goal_id),
            "goal_graph": (
                None if self.goal_graph is None else self.goal_graph.to_trace_dict()
            ),
            "stats": dict(self.stats),
            "selections": [row.to_trace_dict() for row in self.selections],
        }
