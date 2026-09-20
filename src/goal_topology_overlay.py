"""Goal-conditioned dynamic state projected onto the persistent Place graph."""

from dataclasses import dataclass, field


@dataclass
class GoalPlaceState:
    place_id: str
    verify_entities: set = field(default_factory=set)
    frontier_entities: set = field(default_factory=set)
    evidence_ids: set = field(default_factory=set)
    checked_actions: dict = field(default_factory=dict)
    search_state: str = "unseen"

    def to_trace_dict(self):
        return {
            "place_id": self.place_id,
            "verify_entities": sorted(self.verify_entities),
            "frontier_entities": sorted(self.frontier_entities),
            "evidence_ids": sorted(self.evidence_ids),
            "checked_actions": [
                {"intent": key[0], "entity_id": key[1], **value}
                for key, value in sorted(self.checked_actions.items())
            ],
            "search_state": self.search_state,
        }


class GoalTopologyOverlay:
    """Per-goal state on Place IDs; it never creates spatial nodes or edges."""

    def __init__(self, goal_id):
        self.goal_id = str(goal_id)
        self.places = {}
        self.checked = {}
        self.revision = 0
        self.last_decision_revision = -1
        self.last_change_reason = "new_goal"
        self._projection_signature = None

    @staticmethod
    def action_key(intent, entity_id, place_id):
        return str(intent), str(entity_id), str(place_id)

    def is_checked(self, intent, entity_id, place_id):
        return self.action_key(intent, entity_id, place_id) in self.checked

    def mark_checked(self, candidate, outcome, step):
        key = self.action_key(candidate.intent, candidate.entity_id, candidate.place_id)
        value = {"outcome": str(outcome), "step": int(step)}
        if self.checked.get(key) != value:
            self.checked[key] = value
            self.revision += 1
            self.last_change_reason = str(outcome)

    def rebind_entity(self, old_entity, new_entity):
        old_entity, new_entity = str(old_entity), str(new_entity)
        if old_entity == new_entity:
            return
        rebound = {}
        changed = False
        for (intent, entity, place_id), value in self.checked.items():
            if entity == old_entity:
                entity = new_entity
                changed = True
            key = self.action_key(intent, entity, place_id)
            previous = rebound.get(key)
            if previous is None or value["step"] > previous["step"]:
                rebound[key] = value
        if changed:
            self.checked = rebound
            self.revision += 1
            self.last_change_reason = "entity_rebound"

    def project(self, graph, candidates, evidence, semantic_events=False):
        places = {place_id: GoalPlaceState(place_id) for place_id in graph.nodes}
        for (intent, entity, place_id), value in self.checked.items():
            if place_id in places:
                places[place_id].checked_actions[(intent, entity)] = dict(value)
        for item in candidates:
            state = places[item.place_id]
            if item.intent == "Verify":
                state.verify_entities.add(item.entity_id)
                state.evidence_ids.add(item.observation_id)
            else:
                state.frontier_entities.add(item.entity_id)
        for item in evidence.values():
            place_id = item.get("place_id")
            if place_id in places:
                places[place_id].evidence_ids.add(str(item["observation_id"]))
        for state in places.values():
            if state.verify_entities:
                state.search_state = "evidence_available"
            elif state.frontier_entities:
                state.search_state = "frontier_available"
            elif state.checked_actions:
                state.search_state = "exhausted_for_current_topology"
        valid_edges = tuple(sorted(
            (source, target, edge.status.value, float(edge.path_length_m))
            for (source, target), edge in graph.edges.items()
        ))
        actions = tuple(sorted(
            (item.intent, item.entity_id, item.place_id, item.observation_id,
             tuple(item.route), float(item.cost_m), tuple(item.terminal))
            for item in candidates
        ))
        if semantic_events:
            # Position/cost jitter is not a semantic interrupt. Active route
            # geometry is independently checked by the executor every step.
            actions = tuple(sorted((item.candidate_id,item.intent,item.entity_id,item.place_id,item.support)
                                   for item in candidates))
        evidence_signature = tuple(sorted(
            (str(key), str(value.get("observation_id")) if not semantic_events else str(value.get("support")),
             str(value.get("place_id")))
            for key, value in evidence.items()
        ))
        signature = (() if semantic_events else tuple(sorted(graph.nodes)),
                     () if semantic_events else valid_edges, actions, evidence_signature,
                     tuple(sorted((key, value["outcome"]) for key, value in self.checked.items())))
        if signature != self._projection_signature:
            self._projection_signature = signature
            self.revision += 1
            self.last_change_reason = "topology_or_goal_evidence_changed"
        self.places = places

    def decision_required(self):
        return self.last_decision_revision != self.revision

    def mark_decision(self):
        self.last_decision_revision = self.revision

    def to_trace_dict(self):
        return {
            "goal_id": self.goal_id,
            "revision": self.revision,
            "last_decision_revision": self.last_decision_revision,
            "last_change_reason": self.last_change_reason,
            "decision_required": self.decision_required(),
            "places": [self.places[key].to_trace_dict() for key in sorted(self.places)],
        }
