from typing import Mapping, Optional, Sequence

from src.hierarchical_navigator import (
    HierarchicalNavigator, NavigationDecision, SemanticAssessment, SemanticCandidate,
    SemanticConfidence, TerminalVerification,
)
from .goal_graph import GoalGraph
from .scene_map import SceneMap


class DualDynamicNavigator(HierarchicalNavigator):
    """Single semantic intent owner backed by L1 SceneMap and L2 GoalGraph."""

    backend_name = "dual_dynamic"

    def __init__(self, config: Optional[Mapping] = None):
        super().__init__(config)
        self.scene_map = SceneMap()
        self.goal_graph = None
        self.last_map_delta = None
        self.last_projected_candidates = []
        self.decision_revision = 0

    def begin_goal(self, goal_id):
        event = super().begin_goal(goal_id)
        self.goal_graph = GoalGraph(str(goal_id))
        self.last_projected_candidates = []
        self.decision_revision = 0
        return event

    def decide(self, candidates: Sequence[SemanticCandidate],
               assessments: Mapping[str, SemanticAssessment], current_place_id: Optional[str]) -> NavigationDecision:
        if self.goal_graph is None: raise RuntimeError("begin_goal must be called before deciding")
        self.decision_revision += 1
        event_id = f"{self.goal_id}:decision:{self.decision_revision}"
        self.last_map_delta = self.scene_map.update_candidates(candidates, event_id)
        projected, projected_assessments = self.goal_graph.project(candidates, assessments, event_id)
        eligible = []
        for candidate in projected:
            assessment = projected_assessments.get(candidate.candidate_id)
            if (candidate.source_kind == "snapshot" and assessment is not None
                    and assessment.confidence in (SemanticConfidence.HIGH, SemanticConfidence.MEDIUM)
                    and not self.goal_graph.evidence_available(candidate.entity_id, candidate.evidence_refs)):
                continue
            eligible.append(candidate)
        self.last_projected_candidates = list(eligible)
        return super().decide(eligible, projected_assessments, current_place_id)

    def record_verification(self, result: TerminalVerification, step: int,
                            evidence_digest: Optional[str] = None):
        intent = self.active_intent
        if intent is not None and self.goal_graph is not None:
            self.goal_graph.record_verification(
                f"{intent.intent_id}:verification:{intent.verification_attempts + 1}",
                intent.entity_id, intent.evidence_refs, result.verdict)
        return super().record_verification(result, step, evidence_digest)

    def rebind_entity(self, old_id, new_id, event_id):
        delta = self.scene_map.rebind_entity(old_id, new_id, event_id)
        if self.goal_graph is not None: self.goal_graph.rebind_entity(old_id, new_id, event_id)
        if self.active_intent is not None and self.active_intent.entity_id == str(old_id):
            self.active_intent.entity_id = str(new_id)
        return delta

    def to_trace_dict(self):
        payload = super().to_trace_dict()
        payload.update({"backend": self.backend_name, "scene_map": self.scene_map.to_trace_dict(self.goal_id),
                        "goal_graph": None if self.goal_graph is None else self.goal_graph.to_trace_dict(),
                        "map_delta": None if self.last_map_delta is None else self.last_map_delta.to_trace_dict()})
        return payload
