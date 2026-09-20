from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Set, Tuple


@dataclass
class PlaceMemory:
    place_id: str
    observation_ids: Set[str] = field(default_factory=set)
    entity_ids: Set[str] = field(default_factory=set)
    frontier_ids: Set[str] = field(default_factory=set)


@dataclass
class EntityMemory:
    entity_id: str
    observed_from_place_ids: Set[str] = field(default_factory=set)
    evidence_ids: Set[str] = field(default_factory=set)
    approach_place_id: Optional[str] = None
    live: bool = True
    evidence_first_seen: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    evidence_last_seen: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    evidence_observation_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class FrontierMemory:
    frontier_id: str
    approach_place_id: Optional[str] = None
    hypothesis_refs: Set[str] = field(default_factory=set)
    live: bool = True


@dataclass
class TraversableEdgeMemory:
    source_place_id: str
    target_place_id: str
    path_length_m: float
    status: str
    local_path_reference: Optional[str] = None
    last_verified_step: int = -1


@dataclass(frozen=True)
class MapDelta:
    revision: int
    event_id: str
    changed_entities: Tuple[str, ...] = ()
    changed_frontiers: Tuple[str, ...] = ()
    changed_places: Tuple[str, ...] = ()
    changed_edges: Tuple[Tuple[str, str], ...] = ()
    aliases: Tuple[Tuple[str, str], ...] = ()
    reason: str = "candidate_projection"

    def to_trace_dict(self):
        return asdict(self)
