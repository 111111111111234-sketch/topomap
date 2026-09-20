from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Set, Tuple

from src.hierarchical_navigator import SemanticAssessment, SemanticCandidate, VerificationVerdict


@dataclass
class GoalEntityState:
    entity_id: str
    candidate_ids: Set[str] = field(default_factory=set)
    evidence_ids: Set[str] = field(default_factory=set)
    checked_evidence: Dict[str, str] = field(default_factory=dict)
    unresolved: bool = False
    confirmed_evidence: Set[str] = field(default_factory=set)


@dataclass
class GoalPlaceState:
    place_id: str
    entity_ids: Set[str] = field(default_factory=set)
    frontier_ids: Set[str] = field(default_factory=set)
    state: str = "unobserved"


class GoalGraph:
    """Goal-scoped L2 state organized on L1 Place identities."""

    def __init__(self, goal_id):
        self.goal_id = str(goal_id)
        self.entities: Dict[str, GoalEntityState] = {}
        self.places: Dict[str, GoalPlaceState] = {}
        self.revoked_hypotheses = set()
        self.revision = 0
        self._events = set()

    def project(self, candidates: Iterable[SemanticCandidate],
                assessments: Mapping[str, SemanticAssessment], event_id: str) -> Tuple[list, dict]:
        candidates = list(candidates)
        event_id = str(event_id)
        if event_id in self._events:
            return candidates, dict(assessments)
        self._events.add(event_id)
        objects, others = {}, []
        confidence_order = {"high": 0, "medium": 1, "low": 2, "no_match": 3}
        for candidate in candidates:
            place = None if candidate.place_id is None else self.places.setdefault(
                str(candidate.place_id), GoalPlaceState(str(candidate.place_id)))
            if candidate.source_kind != "snapshot":
                others.append(candidate)
                if place is not None: place.frontier_ids.add(str(candidate.entity_id))
                continue
            entity_id = str(candidate.entity_id)
            state = self.entities.setdefault(entity_id, GoalEntityState(entity_id))
            state.candidate_ids.add(candidate.candidate_id)
            state.evidence_ids.update(map(str, candidate.evidence_refs))
            if place is not None: place.entity_ids.add(entity_id)
            assessment = assessments.get(candidate.candidate_id)
            distance = float("inf") if candidate.route_distance_m is None else float(candidate.route_distance_m)
            key = (99, 10**9, distance, candidate.candidate_id) if assessment is None else (
                confidence_order[assessment.confidence.value], assessment.rank_group, distance, candidate.candidate_id)
            if entity_id not in objects or key < objects[entity_id][0]:
                objects[entity_id] = (key, candidate, assessment)
        projected = list(others)
        projected_assessments = {item.candidate_id: assessments[item.candidate_id]
                                 for item in others if item.candidate_id in assessments}
        for entity_id, (_, representative, assessment) in objects.items():
            combined = replace(representative, evidence_refs=tuple(sorted(self.entities[entity_id].evidence_ids)))
            projected.append(combined)
            if assessment is not None: projected_assessments[combined.candidate_id] = assessment
        for place in self.places.values():
            place.state = "evidence_available" if place.entity_ids else (
                "frontier_available" if place.frontier_ids else place.state)
        self.revision += 1
        return projected, projected_assessments

    def evidence_available(self, entity_id, evidence_ids):
        state = self.entities.get(str(entity_id))
        return state is None or any(str(e) not in state.checked_evidence for e in evidence_ids)

    def record_verification(self, event_id, entity_id, evidence_ids, verdict):
        event_id = str(event_id)
        if event_id in self._events: return False
        self._events.add(event_id)
        state = self.entities.setdefault(str(entity_id), GoalEntityState(str(entity_id)))
        if verdict == VerificationVerdict.CONFIRMED:
            state.confirmed_evidence.update(map(str, evidence_ids))
            state.unresolved = False
        if verdict in (VerificationVerdict.REJECTED, VerificationVerdict.UNCERTAIN):
            for evidence in evidence_ids: state.checked_evidence[str(evidence)] = verdict.value
            state.unresolved = verdict == VerificationVerdict.UNCERTAIN
        self.revision += 1
        return True

    def rebind_entity(self, old_id, new_id, event_id):
        event_id, old_id, new_id = str(event_id), str(old_id), str(new_id)
        if event_id in self._events or old_id == new_id: return False
        self._events.add(event_id)
        old = self.entities.pop(old_id, None)
        if old is not None:
            new = self.entities.setdefault(new_id, GoalEntityState(new_id))
            new.candidate_ids |= old.candidate_ids; new.evidence_ids |= old.evidence_ids
            new.checked_evidence.update(old.checked_evidence); new.unresolved |= old.unresolved
            new.confirmed_evidence |= old.confirmed_evidence
            for place in self.places.values():
                if old_id in place.entity_ids: place.entity_ids.discard(old_id); place.entity_ids.add(new_id)
        self.revision += 1
        return True

    def to_trace_dict(self):
        return {"goal_id": self.goal_id, "revision": self.revision,
                "places": {k: v.state for k, v in self.places.items()},
                "entities": {k: {"evidence": len(v.evidence_ids), "checked": len(v.checked_evidence),
                                  "unresolved": v.unresolved,
                                  "confirmed_evidence": sorted(v.confirmed_evidence)} for k, v in self.entities.items()}}
