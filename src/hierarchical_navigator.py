"""Unified semantic, topological and execution policy for GOAT navigation.

The module deliberately keeps semantic identity and navigation cost separate.
The VLM assigns semantic confidence groups without seeing route metadata; the
planner may use geometry only after the identity gate has admitted a source.
Legacy V7 and V2 policies do not import or call this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class SemanticConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_MATCH = "no_match"


class IntentKind(str, Enum):
    TARGET_APPROACH = "target_approach"
    EVIDENCE_REVISIT = "evidence_revisit"
    FRONTIER_EXPLORE = "frontier_explore"


class ExecutionMode(str, Enum):
    LOCAL_TSDF = "local_tsdf"
    TOPO_TSDF = "topo_tsdf"


class NavigationEventType(str, Enum):
    GOAL_STARTED = "goal_started"
    EVIDENCE_CHANGED = "evidence_changed"
    SOURCE_INVALIDATED = "source_invalidated"
    EDGE_INVALIDATED = "edge_invalidated"
    LOCAL_BLOCKED = "local_blocked"
    WAYPOINT_ARRIVED = "waypoint_arrived"
    TERMINAL_ARRIVED = "terminal_arrived"
    VERIFICATION_CONFIRMED = "verification_confirmed"
    VERIFICATION_REJECTED = "verification_rejected"
    VERIFICATION_UNCERTAIN = "verification_uncertain"
    VERIFICATION_ERROR = "verification_error"
    BUDGET_ENDED = "budget_ended"


class NavigatorCommandType(str, Enum):
    CONTINUE = "continue"
    MOVE_LOCAL = "move_local"
    MOVE_TOPO = "move_topo"
    VERIFY = "verify"
    RESELECT = "reselect"
    STOP = "stop"
    FAIL = "fail"


class VerificationVerdict(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    ERROR = "error"


@dataclass(frozen=True)
class SemanticCandidate:
    candidate_id: str
    source_kind: str
    entity_id: str
    place_id: Optional[str]
    route_id: Optional[str]
    route_distance_m: Optional[float]
    route_reachable: bool
    source_index: Optional[int] = None
    snapshot_image: Optional[str] = None
    object_id: Optional[Any] = None
    current_observation: bool = False
    information_gain: float = 0.0
    failure_risk: float = 0.0
    evidence_refs: Tuple[str, ...] = ()
    hypothesis_refs: Tuple[str, ...] = ()

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        if isinstance(payload.get("object_id"), np.integer):
            payload["object_id"] = int(payload["object_id"])
        return payload


@dataclass(frozen=True)
class SemanticAssessment:
    candidate_id: str
    confidence: SemanticConfidence
    rank_group: int
    evidence: str = ""

    def __post_init__(self):
        if int(self.rank_group) < 0:
            raise ValueError("rank_group must be nonnegative")

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["confidence"] = self.confidence.value
        return payload


@dataclass
class NavigationIntent:
    intent_id: str
    goal_id: str
    candidate_id: str
    entity_id: str
    kind: IntentKind
    execution_mode: ExecutionMode
    route_id: Optional[str]
    terminal: Optional[Tuple[float, float]] = None
    status: str = "active"
    local_repairs: int = 0
    topology_reroutes: int = 0
    verification_attempts: int = 0
    evidence_refs: Tuple[str, ...] = ()
    hypothesis_refs: Tuple[str, ...] = ()

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["execution_mode"] = self.execution_mode.value
        return payload


@dataclass(frozen=True)
class NavigationEvent:
    event_type: NavigationEventType
    step: int
    reason: str = ""
    evidence_digest: Optional[str] = None

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        return payload


@dataclass(frozen=True)
class NavigatorCommand:
    command: NavigatorCommandType
    reason: str
    intent_id: Optional[str] = None
    candidate_id: Optional[str] = None

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["command"] = self.command.value
        return payload


@dataclass(frozen=True)
class NavigationDecision:
    candidate: Optional[SemanticCandidate]
    assessment: Optional[SemanticAssessment]
    intent_kind: Optional[IntentKind]
    execution_mode: Optional[ExecutionMode]
    reason: str

    def to_trace_dict(self) -> dict:
        return {
            "candidate": None if self.candidate is None else self.candidate.to_trace_dict(),
            "assessment": None if self.assessment is None else self.assessment.to_trace_dict(),
            "intent_kind": None if self.intent_kind is None else self.intent_kind.value,
            "execution_mode": None if self.execution_mode is None else self.execution_mode.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TerminalVerification:
    verdict: VerificationVerdict
    confidence: Optional[float]
    reason: str
    attempt: int

    def to_trace_dict(self) -> dict:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        return payload


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _extract_json_object(response: Optional[str]) -> Optional[dict]:
    if not response:
        return None
    start, end = response.find("{"), response.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(response[start:end])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _extract_json_array(response: Optional[str]) -> Optional[list]:
    """Extract a fenced or plain top-level JSON array.

    Qwen3-VL consistently emits the requested assessment rows as a top-level
    array even when asked for an object wrapper. The row contract remains
    strict; this helper only normalizes the harmless envelope difference.
    """
    if not response:
        return None
    start, end = response.find("["), response.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(response[start:end])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, list) else None


def parse_semantic_assessments(
    response: Optional[str], candidate_ids: Iterable[str],
) -> Optional[Dict[str, SemanticAssessment]]:
    """Parse a complete assessment response and reject stale/duplicate labels.

    The protocol requires one row per displayed candidate. Treating an empty
    or partial list as success silently converts every omitted object to
    ``no_match`` and every omitted frontier to the safe fallback.
    """
    allowed = {str(item) for item in candidate_ids}
    # Read the row array directly for both accepted envelopes. This also
    # handles a valid one-row top-level array, whose inner object would
    # otherwise be mistaken for the outer payload.
    rows = _extract_json_array(response)
    if rows is None:
        payload = _extract_json_object(response)
        rows = None if payload is None else payload.get("assessments")
    if not isinstance(rows, list):
        return None
    parsed: Dict[str, SemanticAssessment] = {}
    try:
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in allowed or candidate_id in parsed:
                return None
            confidence = SemanticConfidence(str(row["confidence"]).strip().lower())
            rank_group = int(row["rank_group"])
            parsed[candidate_id] = SemanticAssessment(
                candidate_id, confidence, rank_group, str(row.get("evidence", "")),
            )
    except (KeyError, TypeError, ValueError):
        return None
    return parsed if set(parsed) == allowed else None


def parse_terminal_verification(
    response: Optional[str], attempt: int,
) -> TerminalVerification:
    payload = _extract_json_object(response)
    if payload is None:
        return TerminalVerification(VerificationVerdict.ERROR, None, "invalid_json", attempt)
    try:
        verdict = VerificationVerdict(str(payload["verdict"]).strip().lower())
    except (KeyError, TypeError, ValueError):
        return TerminalVerification(VerificationVerdict.ERROR, None, "invalid_verdict", attempt)
    confidence = _finite(payload.get("confidence"))
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        return TerminalVerification(VerificationVerdict.ERROR, None, "invalid_confidence", attempt)
    return TerminalVerification(verdict, confidence, str(payload.get("reason", "")), attempt)


class HierarchicalNavigator:
    """One event-driven owner for semantic decisions and navigation intents."""

    RESELECT_EVENTS = {
        NavigationEventType.GOAL_STARTED,
        NavigationEventType.EVIDENCE_CHANGED,
        NavigationEventType.SOURCE_INVALIDATED,
        NavigationEventType.EDGE_INVALIDATED,
        NavigationEventType.VERIFICATION_REJECTED,
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        self.goal_id: Optional[str] = None
        self.active_intent: Optional[NavigationIntent] = None
        self.pending_decision: Optional[NavigationDecision] = None
        self.pending_semantic_digest: Optional[str] = None
        self.last_assessments: Dict[str, SemanticAssessment] = {}
        self.last_semantic_protocol: Dict[str, Any] = {}
        self.selection_count = 0
        self.semantic_cache: Dict[str, Dict[str, SemanticAssessment]] = {}
        self.suppressed_evidence: set[str] = set()
        self.events: List[NavigationEvent] = []
        self.stats = {
            "semantic_requests": 0,
            "semantic_cache_hits": 0,
            "terminal_verification_requests": 0,
            "high_level_replans": 0,
            "waypoint_events": 0,
            "local_repairs": 0,
            "topology_reroutes": 0,
        }

    @property
    def shadow_only(self) -> bool:
        return bool(self.config.get("shadow_only", False))

    def begin_goal(self, goal_id: str) -> NavigationEvent:
        self.goal_id = str(goal_id)
        self.active_intent = None
        self.pending_decision = None
        self.pending_semantic_digest = None
        self.last_assessments = {}
        self.last_semantic_protocol = {}
        self.selection_count = 0
        self.suppressed_evidence.clear()
        # subtask_metrics diagnostics are per-goal; cumulative values caused
        # later rows to count earlier requests again during aggregation.
        self.stats = {key: 0 for key in self.stats}
        event = NavigationEvent(NavigationEventType.GOAL_STARTED, -1, "new_goal")
        self.events = [event]
        return event

    @staticmethod
    def evidence_digest(goal_key: str, candidates: Sequence[SemanticCandidate]) -> str:
        rows = [
            {
                "candidate_id": item.candidate_id,
                "entity_id": item.entity_id,
                "evidence_refs": list(item.evidence_refs),
                "hypothesis_refs": list(item.hypothesis_refs),
            }
            for item in sorted(candidates, key=lambda value: value.candidate_id)
        ]
        raw = json.dumps([str(goal_key), rows], sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode()).hexdigest()

    def cached_assessments(self, digest: str) -> Optional[Dict[str, SemanticAssessment]]:
        value = self.semantic_cache.get(str(digest))
        if value is not None:
            self.stats["semantic_cache_hits"] += 1
            return dict(value)
        return None

    def store_assessments(self, digest: str, values: Mapping[str, SemanticAssessment]) -> None:
        self.semantic_cache[str(digest)] = dict(values)

    @staticmethod
    def _route_key(candidate: SemanticCandidate) -> Tuple[float, float, str]:
        risk = float(np.clip(candidate.failure_risk, 0.0, 0.95))
        distance = _finite(candidate.route_distance_m)
        adjusted = float("inf") if distance is None else distance / max(0.05, 1.0 - risk)
        return risk, adjusted, candidate.candidate_id

    def decide(
        self,
        candidates: Sequence[SemanticCandidate],
        assessments: Mapping[str, SemanticAssessment],
        current_place_id: Optional[str],
    ) -> NavigationDecision:
        reachable = [item for item in candidates if item.route_reachable]

        def assessed(confidence: SemanticConfidence, source_kind: Optional[str] = None):
            values = [
                (item, assessments.get(item.candidate_id)) for item in reachable
                if source_kind is None or item.source_kind == source_kind
            ]
            return [pair for pair in values if pair[1] is not None and pair[1].confidence == confidence]

        high_targets = assessed(SemanticConfidence.HIGH, "snapshot")
        if high_targets:
            best_group = min(value.rank_group for _, value in high_targets)
            tied = [pair for pair in high_targets if pair[1].rank_group == best_group]
            candidate, assessment = min(tied, key=lambda pair: self._route_key(pair[0]))
            kind = IntentKind.TARGET_APPROACH
            reason = "high_identity_gate"
        else:
            medium_targets = assessed(SemanticConfidence.MEDIUM, "snapshot")
            if medium_targets:
                best_group = min(value.rank_group for _, value in medium_targets)
                tied = [pair for pair in medium_targets if pair[1].rank_group == best_group]
                candidate, assessment = min(
                    tied,
                    key=lambda pair: (
                        -int(pair[0].current_observation),
                        -float(pair[0].information_gain),
                        *self._route_key(pair[0]),
                    ),
                )
                kind = IntentKind.EVIDENCE_REVISIT
                reason = "medium_identity_revisit"
            else:
                frontiers = []
                # Exploration must remain live even when every visible
                # Frontier is semantically weak.  NO_MATCH forbids treating an
                # object as the target; for a Frontier it is the final safe
                # fallback after all more informative frontiers are exhausted.
                for confidence in (
                    SemanticConfidence.HIGH,
                    SemanticConfidence.MEDIUM,
                    SemanticConfidence.LOW,
                    SemanticConfidence.NO_MATCH,
                ):
                    frontiers = assessed(confidence, "frontier")
                    if frontiers:
                        break
                if not frontiers:
                    return NavigationDecision(None, None, None, None, "no_reachable_semantic_candidate")
                best_group = min(value.rank_group for _, value in frontiers)
                tied = [pair for pair in frontiers if pair[1].rank_group == best_group]
                candidate, assessment = min(
                    tied,
                    key=lambda pair: (
                        -float(pair[0].information_gain),
                        *self._route_key(pair[0]),
                    ),
                )
                kind = IntentKind.FRONTIER_EXPLORE
                reason = "frontier_after_identity_gate"

        mode = (
            ExecutionMode.LOCAL_TSDF
            if current_place_id is not None and candidate.place_id == current_place_id
            else ExecutionMode.TOPO_TSDF
        )
        return NavigationDecision(candidate, assessment, kind, mode, reason)

    def install(self, decision: NavigationDecision, terminal=None) -> NavigationIntent:
        if decision.candidate is None or decision.intent_kind is None or decision.execution_mode is None:
            raise ValueError("Cannot install an empty navigation decision")
        self.selection_count += 1
        candidate = decision.candidate
        self.active_intent = NavigationIntent(
            intent_id=f"{self.goal_id}:{self.selection_count}",
            goal_id=str(self.goal_id),
            candidate_id=candidate.candidate_id,
            entity_id=candidate.entity_id,
            kind=decision.intent_kind,
            execution_mode=decision.execution_mode,
            route_id=candidate.route_id,
            terminal=(None if terminal is None else tuple(float(x) for x in terminal[:2])),
            evidence_refs=tuple(candidate.evidence_refs),
            hypothesis_refs=tuple(candidate.hypothesis_refs),
        )
        return self.active_intent

    def on_event(self, event: NavigationEvent) -> NavigatorCommand:
        self.events.append(event)
        intent = self.active_intent
        if event.event_type == NavigationEventType.WAYPOINT_ARRIVED:
            self.stats["waypoint_events"] += 1
            return NavigatorCommand(NavigatorCommandType.CONTINUE, "route_progress_only", None if intent is None else intent.intent_id)
        if event.event_type == NavigationEventType.TERMINAL_ARRIVED:
            if intent is None:
                return NavigatorCommand(NavigatorCommandType.RESELECT, "arrival_without_intent")
            intent.status = "verification_pending"
            return NavigatorCommand(NavigatorCommandType.VERIFY, "terminal_requires_fresh_semantics", intent.intent_id, intent.candidate_id)
        if event.event_type == NavigationEventType.VERIFICATION_CONFIRMED:
            if intent is not None and intent.kind == IntentKind.TARGET_APPROACH:
                intent.status = "completed"
                return NavigatorCommand(NavigatorCommandType.STOP, "target_confirmed", intent.intent_id, intent.candidate_id)
            self.release("evidence_revisit_confirmed")
            return NavigatorCommand(NavigatorCommandType.RESELECT, "confirmed_evidence_requires_target_approach")
        if event.event_type == NavigationEventType.VERIFICATION_UNCERTAIN:
            if intent is not None and intent.verification_attempts < int(self.config.get("max_verification_attempts", 2)):
                return NavigatorCommand(NavigatorCommandType.VERIFY, "alternate_view_required", intent.intent_id, intent.candidate_id)
            self.release("verification_unresolved")
            return NavigatorCommand(NavigatorCommandType.RESELECT, "verification_unresolved")
        if event.event_type == NavigationEventType.VERIFICATION_ERROR:
            if intent is not None and event.evidence_digest:
                self.suppressed_evidence.add(event.evidence_digest)
            self.release("verification_error")
            return NavigatorCommand(NavigatorCommandType.RESELECT, "verification_error_without_semantic_update")
        if event.event_type in self.RESELECT_EVENTS:
            self.stats["high_level_replans"] += 1
            if event.event_type != NavigationEventType.GOAL_STARTED:
                self.release(event.reason or event.event_type.value)
            return NavigatorCommand(NavigatorCommandType.RESELECT, event.reason or event.event_type.value)
        if event.event_type == NavigationEventType.LOCAL_BLOCKED:
            if intent is not None and intent.local_repairs < 1:
                intent.local_repairs += 1
                self.stats["local_repairs"] += 1
                return NavigatorCommand(NavigatorCommandType.CONTINUE, "attempt_local_repair", intent.intent_id)
            if intent is not None and intent.topology_reroutes < 1:
                intent.topology_reroutes += 1
                self.stats["topology_reroutes"] += 1
                return NavigatorCommand(NavigatorCommandType.CONTINUE, "attempt_same_target_reroute", intent.intent_id)
            self.release("recovery_exhausted")
            return NavigatorCommand(NavigatorCommandType.RESELECT, "recovery_exhausted")
        if event.event_type == NavigationEventType.BUDGET_ENDED:
            self.release("budget_ended")
            return NavigatorCommand(NavigatorCommandType.FAIL, "budget_ended")
        return NavigatorCommand(NavigatorCommandType.CONTINUE, "intent_unchanged", None if intent is None else intent.intent_id)

    def record_verification(self, result: TerminalVerification, step: int, evidence_digest: Optional[str] = None) -> NavigatorCommand:
        if self.active_intent is not None:
            self.active_intent.verification_attempts += 1
        mapping = {
            VerificationVerdict.CONFIRMED: NavigationEventType.VERIFICATION_CONFIRMED,
            VerificationVerdict.REJECTED: NavigationEventType.VERIFICATION_REJECTED,
            VerificationVerdict.UNCERTAIN: NavigationEventType.VERIFICATION_UNCERTAIN,
            VerificationVerdict.ERROR: NavigationEventType.VERIFICATION_ERROR,
        }
        return self.on_event(NavigationEvent(mapping[result.verdict], step, result.reason, evidence_digest))

    def release(self, reason: str) -> Optional[dict]:
        if self.active_intent is None:
            return None
        result = self.active_intent.to_trace_dict()
        self.active_intent.status = "released"
        result.update(status="released", reason=str(reason))
        self.active_intent = None
        return result

    def to_trace_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "active_intent": None if self.active_intent is None else self.active_intent.to_trace_dict(),
            "stats": dict(self.stats),
            "last_semantic_protocol": dict(self.last_semantic_protocol),
            "suppressed_evidence": sorted(self.suppressed_evidence),
        }


def query_terminal_verification(
    image: np.ndarray,
    goal_type: str,
    question: str,
    category: str,
    model_name: str,
    extra_body=None,
    goal_image_base64: Optional[str] = None,
    attempt: int = 1,
    decision_confidence_threshold: float = 0.70,
    candidate_image: Optional[np.ndarray] = None,
    candidate_label: Optional[str] = None,
    require_candidate_identity: bool = False,
) -> TerminalVerification:
    """Run one fresh, goal-specific terminal verification request."""
    from src.eval_utils_gpt_goatbench import call_openai_api, encode_tensor2base64

    identity_rule = (
        " When a selected candidate reference is supplied, confirmed requires BOTH that "
        "the selected candidate itself matches the goal and that the fresh view supports "
        "the same physical candidate. A different target merely visible elsewhere in the "
        "fresh scene must be rejected."
        if require_candidate_identity else ""
    )
    system = (
        "You are the terminal semantic verifier for an indoor navigation robot. "
        "Judge only what is visibly supported by the fresh current RGB view. "
        "Return only JSON with verdict (confirmed, rejected, or uncertain), "
        "confidence (0 to 1), and reason. Use uncertain when visibility is inadequate."
        + identity_rule
    )
    prompt = (
        f"Goal type: {goal_type}. Goal question: {question}. Target label: {category}. "
        f"Selected candidate detector label: {candidate_label or 'unknown'}."
    )
    content = []
    if goal_type == "image" and goal_image_base64:
        content.append((prompt + " The first image is the goal reference.", goal_image_base64))
    if candidate_image is not None:
        content.append((
            "Selected candidate reference. Verify this exact entity, not another target in the room:",
            encode_tensor2base64(np.asarray(candidate_image)[..., :3].astype(np.uint8)),
        ))
    content.append((
        prompt + " Fresh terminal observation:",
        encode_tensor2base64(np.asarray(image)[..., :3].astype(np.uint8)),
    ))
    try:
        response = call_openai_api(
            system, content, model_name=model_name, extra_body=extra_body,
            purpose="terminal_verification", max_tries=1,
        )
    except Exception as exc:
        return TerminalVerification(VerificationVerdict.ERROR, None, type(exc).__name__, attempt)
    result = parse_terminal_verification(response, attempt)
    if (
        result.verdict in (VerificationVerdict.CONFIRMED, VerificationVerdict.REJECTED)
        and (
            result.confidence is None
            or result.confidence < float(decision_confidence_threshold)
        )
    ):
        return TerminalVerification(
            VerificationVerdict.UNCERTAIN,
            result.confidence,
            f"low_confidence_{result.verdict.value}: {result.reason}",
            attempt,
        )
    return result
