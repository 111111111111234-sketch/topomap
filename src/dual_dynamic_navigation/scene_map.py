from typing import Iterable, Mapping, Optional, Tuple

from src.hierarchical_navigator import SemanticCandidate
from .models import EntityMemory, FrontierMemory, MapDelta, PlaceMemory, TraversableEdgeMemory


class SceneMap:
    """Episode-scoped L1 spatial and observation facts."""

    def __init__(self):
        self.places, self.entities, self.frontiers, self.edges, self.aliases = {}, {}, {}, {}, {}
        self.revision = 0
        self._events = set()

    def resolve_entity(self, entity_id):
        current, seen = str(entity_id), set()
        while current in self.aliases and current not in seen:
            seen.add(current); current = self.aliases[current]
        return current

    def rebind_entity(self, old_id, new_id, event_id):
        event_id = str(event_id)
        old_id, new_id = self.resolve_entity(old_id), self.resolve_entity(new_id)
        if event_id in self._events or old_id == new_id:
            return MapDelta(self.revision, event_id, reason="duplicate_or_noop")
        self._events.add(event_id); self.aliases[old_id] = new_id
        old, new = self.entities.pop(old_id, None), self.entities.get(new_id)
        if old is not None:
            if new is None:
                old.entity_id = new_id; self.entities[new_id] = old
            else:
                new.observed_from_place_ids |= old.observed_from_place_ids
                new.evidence_ids |= old.evidence_ids
                new.approach_place_id = new.approach_place_id or old.approach_place_id
                for evidence_id, seen in old.evidence_first_seen.items():
                    new.evidence_first_seen.setdefault(evidence_id, seen)
                new.evidence_last_seen.update(old.evidence_last_seen)
                new.evidence_observation_positions.update(
                    old.evidence_observation_positions
                )
            for place in self.places.values():
                if old_id in place.entity_ids:
                    place.entity_ids.discard(old_id); place.entity_ids.add(new_id)
        self.revision += 1
        return MapDelta(self.revision, event_id, (new_id,), aliases=((old_id, new_id),), reason="entity_rebound")

    def update_candidates(self, candidates: Iterable[SemanticCandidate], event_id: str,
                          aliases: Optional[Mapping[str, str]] = None,
                          goal_id: Optional[str] = None, step: int = -1,
                          evidence_positions: Optional[Mapping[str, Tuple[float, float]]] = None):
        event_id = str(event_id)
        if event_id in self._events:
            return MapDelta(self.revision, event_id, reason="duplicate_event")
        self._events.add(event_id)
        changed_entities, changed_frontiers, changed_places, alias_rows = set(), set(), set(), []
        for old, new in (aliases or {}).items():
            old, new = self.resolve_entity(old), self.resolve_entity(new)
            if old != new:
                self.aliases[old] = new; alias_rows.append((old, new))
        for candidate in candidates:
            place_id = None if candidate.place_id is None else str(candidate.place_id)
            place_new = place_id is not None and place_id not in self.places
            place = None if place_id is None else self.places.setdefault(place_id, PlaceMemory(place_id))
            place_before = None if place is None else (
                set(place.observation_ids), set(place.entity_ids), set(place.frontier_ids))
            if candidate.source_kind == "snapshot":
                entity_id = self.resolve_entity(candidate.entity_id)
                entity = self.entities.setdefault(entity_id, EntityMemory(entity_id))
                before = (
                    set(entity.evidence_ids), set(entity.observed_from_place_ids),
                    entity.approach_place_id, dict(entity.evidence_first_seen),
                    dict(entity.evidence_last_seen),
                    dict(entity.evidence_observation_positions),
                )
                entity.evidence_ids.update(map(str, candidate.evidence_refs))
                for evidence_id in map(str, candidate.evidence_refs):
                    seen = (str(goal_id or "unknown"), int(step))
                    entity.evidence_first_seen.setdefault(evidence_id, seen)
                    entity.evidence_last_seen[evidence_id] = seen
                    if evidence_id in (evidence_positions or {}):
                        position = evidence_positions[evidence_id]
                        entity.evidence_observation_positions[evidence_id] = (
                            float(position[0]), float(position[1])
                        )
                if place is not None:
                    entity.observed_from_place_ids.add(place_id); entity.approach_place_id = place_id
                    place.entity_ids.add(entity_id); place.observation_ids.update(map(str, candidate.evidence_refs))
                if before != (
                    entity.evidence_ids, entity.observed_from_place_ids,
                    entity.approach_place_id, entity.evidence_first_seen,
                    entity.evidence_last_seen,
                    entity.evidence_observation_positions,
                ):
                    changed_entities.add(entity_id)
            elif candidate.source_kind == "frontier":
                frontier_id = str(candidate.entity_id)
                frontier = self.frontiers.setdefault(frontier_id, FrontierMemory(frontier_id))
                before = (frontier.approach_place_id, set(frontier.hypothesis_refs))
                frontier.approach_place_id = place_id
                frontier.hypothesis_refs.update(map(str, candidate.hypothesis_refs))
                if place is not None: place.frontier_ids.add(frontier_id)
                if before != (frontier.approach_place_id, frontier.hypothesis_refs): changed_frontiers.add(frontier_id)
            if place is not None and (place_new or place_before != (
                    place.observation_ids, place.entity_ids, place.frontier_ids)):
                changed_places.add(place_id)
        if changed_entities or changed_frontiers or changed_places or alias_rows: self.revision += 1
        return MapDelta(self.revision, event_id, tuple(sorted(changed_entities)),
                        tuple(sorted(changed_frontiers)), tuple(sorted(changed_places)), (), tuple(sorted(alias_rows)))

    def sync_place_topology(self, graph, event_id):
        """Copy the current certified Place routing view into L1."""
        event_id = str(event_id)
        if event_id in self._events:
            return MapDelta(self.revision, event_id, reason="duplicate_event")
        self._events.add(event_id)
        changed_places, changed_edges = set(), set()
        for place_id, node in graph.nodes.items():
            place_id = str(place_id)
            place_new = place_id not in self.places
            place = self.places.setdefault(place_id, PlaceMemory(place_id))
            before = set(place.observation_ids)
            place.observation_ids.update(map(str, node.observation_ids))
            if place_new or before != place.observation_ids: changed_places.add(place_id)
        live_edges = set()
        for key, edge in graph.edges.items():
            edge_key = (str(key[0]), str(key[1])); live_edges.add(edge_key)
            value = TraversableEdgeMemory(
                edge_key[0], edge_key[1], float(edge.path_length_m), edge.status.value,
                edge.local_path_reference, int(edge.last_verified_step))
            if self.edges.get(edge_key) != value:
                self.edges[edge_key] = value; changed_edges.add(edge_key)
        for key in set(self.edges) - live_edges:
            del self.edges[key]; changed_edges.add(key)
        if changed_places or changed_edges: self.revision += 1
        return MapDelta(self.revision, event_id, changed_places=tuple(sorted(changed_places)),
                        changed_edges=tuple(sorted(changed_edges)), reason="place_topology_sync")

    def to_trace_dict(self, current_goal_id=None):
        current_goal_id = None if current_goal_id is None else str(current_goal_id)
        evidence_count = 0
        prior_goal_evidence_count = 0
        entities_with_prior_goal_evidence = 0
        for entity in self.entities.values():
            evidence_count += len(entity.evidence_ids)
            prior = sum(
                str(first_goal) != current_goal_id
                for first_goal, _ in entity.evidence_first_seen.values()
                if current_goal_id is not None
            )
            prior_goal_evidence_count += prior
            entities_with_prior_goal_evidence += int(prior > 0)
        return {
            "revision": self.revision,
            "places": len(self.places),
            "entities": len(self.entities),
            "frontiers": len(self.frontiers),
            "edges": len(self.edges),
            "aliases": dict(self.aliases),
            "evidence_count": evidence_count,
            "prior_goal_evidence_count": prior_goal_evidence_count,
            "entities_with_prior_goal_evidence": entities_with_prior_goal_evidence,
        }
