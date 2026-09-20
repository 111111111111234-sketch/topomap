"""Typed tasks and source-bound authorization, without confidence propagation."""

from dataclasses import dataclass, field
from enum import Enum


class ActionKind(str, Enum):
    APPROACH = "APPROACH"
    REVISIT = "REVISIT"
    EXPLORE = "EXPLORE"


@dataclass
class AuthorizedTask:
    kind: ActionKind
    source_id: str
    entity_id: str
    evidence_digest: str
    route: dict
    choice: object
    hypothesis_refs: tuple = ()
    requires_verification: bool = False

    def trace(self):
        return {"kind": self.kind.value, "source_id": self.source_id,
                "entity_id": self.entity_id, "evidence_digest": self.evidence_digest,
                "route": self.route, "hypothesis_refs": list(self.hypothesis_refs),
                "requires_verification": bool(self.requires_verification)}


@dataclass
class NoAction:
    reason: str
    unresolved_ids: tuple = ()


@dataclass
class RequestError:
    reason: str


@dataclass
class NavigationIntent:
    intent_id: str
    goal_id: str
    task: AuthorizedTask
    state: str = "move"
    recovery_attempts: int = 0
    verification_views: int = 0
    last_distance_m: float = None
    stalled_steps: int = 0
    motion_steps: int = 0
    pin_refs: tuple = ()


def revisit_allowed(route, explore_costs, step_length_m):
    if route.get("status") != "valid":
        return False
    return (not explore_costs or route["total_distance_m"] <=
            max(min(explore_costs), float(step_length_m)) + 1e-6)
