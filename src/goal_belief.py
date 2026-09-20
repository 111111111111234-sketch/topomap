"""Goal-conditioned evidence fusion and multi-hypothesis action selection.

The module is deliberately independent from Habitat and TSDF.  It stores
episode-scoped semantic evidence keyed by persistent topology node IDs while
callers provide the current executable source mappings and path facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class EvidenceSource(str, Enum):
    CLIP_GOAL = "clip_goal"
    VLM_ROOM_PRIOR = "vlm_room_prior"
    DETECTOR = "detector"
    CONTEXT_PRIOR = "context_prior"
    ACTIVE_VERIFY_VLM = "active_verify_vlm"


SOURCE_RELIABILITY = {
    EvidenceSource.ACTIVE_VERIFY_VLM: 1.0,
    EvidenceSource.DETECTOR: 0.8,
    EvidenceSource.VLM_ROOM_PRIOR: 0.7,
    EvidenceSource.CLIP_GOAL: 0.5,
    EvidenceSource.CONTEXT_PRIOR: 0.3,
}


class BeliefActionType(str, Enum):
    NAVIGATE = "navigate"
    VERIFY = "verify"
    CONFIRM_APPROACH = "confirm_approach"
    EXPLORE = "explore"
    UNKNOWN_EXPLORE = "unknown_explore"


CATEGORY_ALIAS_GROUPS = (
    frozenset({"plant", "potted plant"}),
    frozenset({"rug", "carpet", "floor mat", "mat"}),
    frozenset({"hanging clothes", "clothes", "clothing"}),
    frozenset({"television", "tv"}),
    frozenset({"couch", "sofa"}),
    frozenset({"refrigerator", "fridge"}),
)


def normalize_category_label(value: str) -> str:
    value = " ".join(str(value or "").strip().lower().replace("_", " ").split())
    if value.endswith("s") and not value.endswith(("ss", "clothes")):
        value = value[:-1]
    return value


def category_compatibility_score(target: str, detected: str) -> float:
    """Conservative, label-only GOAT/detector compatibility without GT access."""
    target_label = normalize_category_label(target)
    detected_label = normalize_category_label(detected)
    if not target_label or not detected_label:
        return 0.0
    if target_label == detected_label:
        return 1.0
    if any(
        target_label in group and detected_label in group
        for group in CATEGORY_ALIAS_GROUPS
    ):
        return 0.90
    return 0.0


def normalize_goal_key(category: str) -> str:
    normalized = " ".join(str(category or "").strip().lower().replace("_", " ").split())
    return f"category:{normalized}"


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), 0.05, 0.95))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return float(math.log(value / (1.0 - value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def binary_entropy(probability: float) -> float:
    probability = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
    return float(-probability * math.log2(probability) -
                 (1.0 - probability) * math.log2(1.0 - probability))


def calibrated_clip_probability(
    image_embedding: Optional[np.ndarray], goal_embedding: Optional[np.ndarray],
    floor: float = 0.10, ceiling: float = 0.35,
) -> Optional[float]:
    if image_embedding is None or goal_embedding is None or ceiling <= floor:
        return None
    left = np.asarray(image_embedding, dtype=float).reshape(-1)
    right = np.asarray(goal_embedding, dtype=float).reshape(-1)
    if left.shape != right.shape:
        return None
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return None
    cosine = float(np.dot(left / left_norm, right / right_norm))
    return float(np.clip((cosine - floor) / (ceiling - floor), 0.0, 1.0))


def clip_cosine_similarity(
    image_embedding: Optional[np.ndarray], goal_embedding: Optional[np.ndarray],
) -> Optional[float]:
    """Return the unlabelled raw CLIP cosine used by the v2 calibrator."""
    if image_embedding is None or goal_embedding is None:
        return None
    left = np.asarray(image_embedding, dtype=float).reshape(-1)
    right = np.asarray(goal_embedding, dtype=float).reshape(-1)
    if left.shape != right.shape:
        return None
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return None
    return float(np.clip(np.dot(left / left_norm, right / right_norm), -1.0, 1.0))


def angular_distance(lhs: Optional[float], rhs: Optional[float]) -> float:
    if lhs is None or rhs is None:
        return math.inf
    delta = abs(float(lhs) - float(rhs)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


@dataclass(frozen=True)
class GoalEvidence:
    evidence_id: str
    node_id: str
    goal_key: str
    source: EvidenceSource
    score: float
    reliability: float
    step: int
    viewpoint: Optional[np.ndarray] = field(default=None, compare=False)
    heading: Optional[float] = None
    observation_id: str = ""
    reason: str = ""
    raw_score: Optional[float] = None

    def __post_init__(self):
        object.__setattr__(self, "source", EvidenceSource(self.source))
        object.__setattr__(self, "score", _clip_probability(self.score))
        object.__setattr__(self, "reliability", float(np.clip(self.reliability, 0.0, 1.0)))
        if self.raw_score is not None:
            object.__setattr__(self, "raw_score", float(self.raw_score))
        if self.viewpoint is not None:
            object.__setattr__(self, "viewpoint", np.asarray(self.viewpoint, dtype=float)[:2])

    @property
    def weighted_log_bayes_factor(self) -> float:
        return self.reliability * _logit(self.score)

    @property
    def is_verify(self) -> bool:
        return self.source == EvidenceSource.ACTIVE_VERIFY_VLM


@dataclass
class EvidenceCluster:
    cluster_id: str
    node_id: str
    goal_key: str
    viewpoint: Optional[np.ndarray]
    heading: Optional[float]
    evidence: List[GoalEvidence] = field(default_factory=list)

    @property
    def has_verify(self) -> bool:
        return any(item.is_verify for item in self.evidence)

    @property
    def contribution(self) -> float:
        selected = [item for item in self.evidence if item.is_verify]
        if not selected:
            selected = list(self.evidence)
        if not selected:
            return 0.0
        # Average the reliability-weighted log Bayes factors so correlated
        # sources cannot accumulate like independent viewpoints while source
        # reliability still attenuates a single observation.
        value = sum(item.weighted_log_bayes_factor for item in selected) / len(selected)
        return float(value)

    @property
    def conflict(self) -> float:
        positive = sum(item.reliability for item in self.evidence if item.score > 0.5)
        negative = sum(item.reliability for item in self.evidence if item.score < 0.5)
        total = positive + negative
        return float(2.0 * min(positive, negative) / total) if total > 0 else 0.0


@dataclass(frozen=True)
class GoalBelief:
    node_id: str
    goal_key: str
    posterior: float
    structural_confidence: float
    reachability: float
    entropy: float
    independent_view_count: int
    negative_verify_views: int
    evidence_count: int
    conflict: float
    evidence_summary: Dict[str, int]
    eliminated: bool = False


@dataclass(frozen=True)
class ExecutableNode:
    node_id: str
    source_kind: str
    source_index: Any
    structural_confidence: float
    reachability: float
    path_cost: float
    revisit: float = 0.0
    map_information_gain: float = 0.0
    reachable: bool = True


@dataclass(frozen=True)
class VerificationOption:
    target_point: np.ndarray = field(compare=False)
    look_at_point: np.ndarray = field(compare=False)
    visibility: float = 1.0
    novelty: float = 1.0
    path_cost: float = 0.0


@dataclass
class BeliefActionCandidate:
    action_type: BeliefActionType
    node_id: str
    source_kind: str
    source_index: Any
    belief: GoalBelief
    utility: float
    normalized_path_cost: float
    verification: Optional[VerificationOption] = None
    action_weight: float = 0.0
    confirmation_source_node_id: Optional[str] = None


@dataclass
class GoalHypothesisSet:
    goal_key: str
    hypotheses: List[GoalBelief]
    unknown_probability: float
    action_weights: Dict[str, float]
    executable_node_ids: List[str]
    created_at_step: int
    previous_top_node_id: Optional[str] = None
    rank_switched: bool = False
    legacy_unknown_probability: Optional[float] = None
    unknown_method: str = "independent_product"
    unknown_evidence_node_count: int = 0
    unknown_weight_sum: float = 0.0
    unknown_node_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class UnknownPolicyState:
    goal_key: str
    consecutive_actions: int = 0
    last_top_posterior: Optional[float] = None
    executable_node_ids: set = field(default_factory=set)
    last_reset_reason: str = "goal_switch"


@dataclass(frozen=True)
class UnknownPolicyDecision:
    eligible: bool
    reason: str
    unknown_probability: float
    top_posterior: float
    consecutive_actions: int
    reset_reason: Optional[str] = None


@dataclass(frozen=True)
class EvidenceFusionSummary:
    non_direct: float
    direct: float
    negative_verify_views: int
    evidence_count: int
    source_counts: Dict[str, int]
    conflict: float
    independent_view_count: int


@dataclass
class BufferedClipEvidence:
    evidence_id: str
    node_id: str
    goal_key: str
    raw_cosine: float
    step: int
    viewpoint: Optional[np.ndarray] = field(default=None, compare=False)
    heading: Optional[float] = None
    observation_id: str = ""
    reason: str = ""


@dataclass
class ClipCalibrationState:
    goal_key: str
    buffered: List[BufferedClipEvidence] = field(default_factory=list)
    seen_observations: set = field(default_factory=set)
    frozen: bool = False
    median: Optional[float] = None
    scale: Optional[float] = None


class ClipGoalCalibrator:
    """Episode-local, label-free and frozen per-category CLIP calibrator."""

    def __init__(self, warmup_samples: int = 16, temperature: float = 1.5):
        self.warmup_samples = max(1, int(warmup_samples))
        self.temperature = max(1e-6, float(temperature))
        self._states: Dict[str, ClipCalibrationState] = {}
        self.stats = {
            "clip_warmup_buffered": 0,
            "clip_calibrators_frozen": 0,
            "clip_evidence_replayed": 0,
            "clip_observations_deduplicated": 0,
        }

    def state(self, goal_key: str) -> ClipCalibrationState:
        return self._states.setdefault(goal_key, ClipCalibrationState(goal_key=goal_key))

    @staticmethod
    def _observation_key(item: BufferedClipEvidence) -> str:
        return item.observation_id or item.evidence_id

    def _score(self, state: ClipCalibrationState, raw_cosine: float) -> float:
        value = _sigmoid(
            (float(raw_cosine) - float(state.median)) /
            (self.temperature * float(state.scale))
        )
        return float(np.clip(value, 0.20, 0.80))

    def _to_evidence(self, item: BufferedClipEvidence, state: ClipCalibrationState) -> GoalEvidence:
        return GoalEvidence(
            evidence_id=item.evidence_id,
            node_id=item.node_id,
            goal_key=item.goal_key,
            source=EvidenceSource.CLIP_GOAL,
            score=self._score(state, item.raw_cosine),
            reliability=SOURCE_RELIABILITY[EvidenceSource.CLIP_GOAL],
            step=item.step,
            viewpoint=item.viewpoint,
            heading=item.heading,
            observation_id=item.observation_id,
            reason=item.reason,
            raw_score=item.raw_cosine,
        )

    def observe(self, item: BufferedClipEvidence) -> Tuple[List[GoalEvidence], Dict[str, Any]]:
        state = self.state(item.goal_key)
        key = self._observation_key(item)
        if key in state.seen_observations:
            self.stats["clip_observations_deduplicated"] += 1
            return [], {"event": "duplicate", "goal_key": item.goal_key, "observation_id": key}
        state.seen_observations.add(key)
        if state.frozen:
            evidence = self._to_evidence(item, state)
            return [evidence], {
                "event": "calibrated", "goal_key": item.goal_key,
                "raw_cosine": item.raw_cosine, "median": state.median,
                "scale": state.scale, "calibrated_score": evidence.score,
            }

        state.buffered.append(item)
        self.stats["clip_warmup_buffered"] += 1
        if len(state.buffered) < self.warmup_samples:
            return [], {
                "event": "warmup", "goal_key": item.goal_key,
                "raw_cosine": item.raw_cosine,
                "buffered": len(state.buffered), "required": self.warmup_samples,
            }

        values = np.asarray([entry.raw_cosine for entry in state.buffered], dtype=float)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        state.median = float(median)
        state.scale = float(max((q75 - q25) / 1.349, 0.02))
        state.frozen = True
        replay = [self._to_evidence(entry, state) for entry in state.buffered]
        self.stats["clip_calibrators_frozen"] += 1
        self.stats["clip_evidence_replayed"] += len(replay)
        return replay, {
            "event": "freeze", "goal_key": item.goal_key,
            "buffered": len(replay), "median": state.median, "scale": state.scale,
            "raw_min": float(values.min()), "raw_max": float(values.max()),
            "calibrated_scores": [entry.score for entry in replay],
        }


@dataclass
class ConfirmationState:
    node_id: str
    goal_key: str
    created_step: int
    expires_step: int
    posterior_at_creation: float
    max_approaches: int = 2
    approach_count: int = 0
    failure_count: int = 0
    awaiting_observation: bool = False
    last_approach_step: Optional[int] = None
    awaiting_observation_step: Optional[int] = None
    positive_source: str = ""
    max_posterior: float = 0.0
    active: bool = True
    end_reason: Optional[str] = None


@dataclass
class RejectedSnapshotState:
    goal_key: str
    object_id: str
    rejected_step: int
    expires_step: int
    reason: str
    compatibility: float
    detector_confidence: float
    detection_count: int


@dataclass(frozen=True)
class SnapshotAcceptanceDecision:
    accepted: bool
    reason: str
    compatibility: float
    detector_confidence: float
    detection_count: int
    posterior: float
    candidate_score: float
    best_candidate_score: float


def evaluate_snapshot_acceptance(
    *, target_label: str, detected_label: str, detector_confidence: float,
    detection_count: int, posterior: float, candidate_score: float,
    best_candidate_score: float, confirmed: bool = False,
    min_compatibility: float = 0.90, min_detector_confidence: float = 0.50,
    min_detections: int = 2, min_posterior: float = 0.35,
    alias_min_detections: int = 3, alias_min_posterior: float = 0.85,
    best_candidate_margin: float = 0.05,
) -> SnapshotAcceptanceDecision:
    compatibility = category_compatibility_score(target_label, detected_label)
    common = dict(
        compatibility=compatibility,
        detector_confidence=float(detector_confidence),
        detection_count=int(detection_count), posterior=float(posterior),
        candidate_score=float(candidate_score),
        best_candidate_score=float(best_candidate_score),
    )
    if compatibility < min_compatibility:
        return SnapshotAcceptanceDecision(False, "incompatible_category", **common)
    if detector_confidence < min_detector_confidence:
        return SnapshotAcceptanceDecision(False, "low_detector_confidence", **common)
    if candidate_score + best_candidate_margin < best_candidate_score:
        return SnapshotAcceptanceDecision(False, "less_stable_instance_available", **common)
    if confirmed:
        return SnapshotAcceptanceDecision(True, "active_confirmation", **common)
    if compatibility < 1.0:
        if detection_count < alias_min_detections or posterior < alias_min_posterior:
            return SnapshotAcceptanceDecision(False, "alias_requires_stronger_evidence", **common)
    elif detection_count < min_detections:
        return SnapshotAcceptanceDecision(False, "insufficient_multiview_detections", **common)
    elif posterior < min_posterior:
        return SnapshotAcceptanceDecision(False, "posterior_below_gate", **common)
    return SnapshotAcceptanceDecision(True, "accepted", **common)


class GoalBeliefMemory:
    """Episode-scoped, source-aware semantic evidence memory."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = dict(cfg or {})
        self.prior = float(self.cfg.get("prior", 0.20))
        self.top_k = int(self.cfg.get("top_k", 3))
        self.elimination_threshold = float(self.cfg.get("elimination_threshold", 0.05))
        self.min_negative_verify_views = int(self.cfg.get("min_negative_verify_views", 2))
        self.max_clusters = int(self.cfg.get("max_view_clusters_per_goal", 64))
        self.viewpoint_distance = float(self.cfg.get("viewpoint_cluster_distance_m", 0.75))
        self.viewpoint_angle = math.radians(float(self.cfg.get("viewpoint_cluster_angle_deg", 45.0)))
        self.unknown_method = str(self.cfg.get("unknown_method", "independent_product"))
        self.fast_evidence_path = bool(self.cfg.get("fast_evidence_path", False))
        self._clusters: Dict[Tuple[str, str], List[EvidenceCluster]] = {}
        self._seen_evidence_ids = set()
        self._seen_observations = set()
        self._evidence_revision: Dict[Tuple[str, str], int] = {}
        self._belief_cache: Dict[Tuple[Any, ...], GoalBelief] = {}
        self._fusion_cache: Dict[Tuple[str, str, int], EvidenceFusionSummary] = {}
        self._cluster_counter = 0
        calibration_cfg = dict(self.cfg.get("clip_calibration", {}))
        self.clip_calibrator = ClipGoalCalibrator(
            warmup_samples=calibration_cfg.get("warmup_samples", 16),
            temperature=calibration_cfg.get("temperature", 1.5),
        )
        self.confirmations: Dict[str, ConfirmationState] = {}
        self._confirmation_events: List[Dict[str, Any]] = []
        self.unknown_policy: Dict[str, UnknownPolicyState] = {}
        self.rejected_snapshots: Dict[Tuple[str, str], RejectedSnapshotState] = {}
        self.verify_attempts: Dict[Tuple[str, str], int] = {}
        self.verify_last_step: Dict[Tuple[str, str], int] = {}
        self.stats = {
            "evidence_added": 0,
            "evidence_deduplicated": 0,
            "evidence_fast_skipped": 0,
            "view_clusters_created": 0,
            "view_clusters_merged": 0,
            "posterior_updates": 0,
            "posterior_cache_hits": 0,
            "posterior_cache_misses": 0,
            "evidence_fusion_cache_hits": 0,
            "evidence_fusion_cache_misses": 0,
            "hypothesis_rank_switches": 0,
            "planner_direct_choices": 0,
            "vlm_tiebreak_choices": 0,
            "verify_actions": 0,
            "verify_vlm_calls": 0,
            "confirmations_created": 0,
            "confirmations_selected": 0,
            "confirmations_expired": 0,
            "confirmations_reversed": 0,
            "confirmation_approaches": 0,
            "confirmation_failures": 0,
            "snapshot_gate_accepted": 0,
            "snapshot_gate_rejected": 0,
            "unknown_policy_eligible": 0,
            "unknown_policy_suppressed": 0,
            "unknown_policy_resets": 0,
            "category_deterministic_choices": 0,
            "rejected_snapshot_suppressions": 0,
            "rejected_snapshot_releases": 0,
            "cumulative_normalized_action_cost": 0.0,
        }
        self._last_top_node: Dict[str, Optional[str]] = {}
        self.snapshot_gate_reasons: Dict[str, int] = {}
        self.unknown_policy_reasons: Dict[str, int] = {}

    @staticmethod
    def source_reliability(source: EvidenceSource) -> float:
        return SOURCE_RELIABILITY[EvidenceSource(source)]

    def make_evidence(
        self, evidence_id: str, node_id: str, goal_key: str,
        source: EvidenceSource, score: float, step: int,
        viewpoint=None, heading: Optional[float] = None,
        observation_id: str = "", reason: str = "",
        raw_score: Optional[float] = None,
    ) -> GoalEvidence:
        return GoalEvidence(
            evidence_id=evidence_id,
            node_id=node_id,
            goal_key=goal_key,
            source=source,
            score=score,
            reliability=self.source_reliability(source),
            step=step,
            viewpoint=viewpoint,
            heading=heading,
            observation_id=observation_id,
            reason=reason,
            raw_score=raw_score,
        )

    @staticmethod
    def _observation_key(evidence: GoalEvidence) -> Tuple[str, str, str, str]:
        return (
            evidence.node_id, evidence.goal_key, evidence.source.value,
            evidence.observation_id,
        )

    def contains_evidence(self, evidence: GoalEvidence, count_fast_skip: bool = False) -> bool:
        duplicate = evidence.evidence_id in self._seen_evidence_ids or (
            bool(evidence.observation_id) and
            self._observation_key(evidence) in self._seen_observations
        )
        if duplicate and count_fast_skip:
            self.stats["evidence_fast_skipped"] += 1
        return duplicate

    def _correlated(self, cluster: EvidenceCluster, evidence: GoalEvidence) -> bool:
        if cluster.viewpoint is None or evidence.viewpoint is None:
            return cluster.viewpoint is None and evidence.viewpoint is None
        distance = float(np.linalg.norm(cluster.viewpoint - evidence.viewpoint))
        angle = angular_distance(cluster.heading, evidence.heading)
        angle_matches = angle == math.inf or angle < self.viewpoint_angle
        return distance < self.viewpoint_distance and angle_matches

    def _new_cluster(self, evidence: GoalEvidence) -> EvidenceCluster:
        cluster = EvidenceCluster(
            cluster_id=f"evidence_cluster_{self._cluster_counter}",
            node_id=evidence.node_id,
            goal_key=evidence.goal_key,
            viewpoint=None if evidence.viewpoint is None else evidence.viewpoint.copy(),
            heading=evidence.heading,
            evidence=[evidence],
        )
        self._cluster_counter += 1
        self.stats["view_clusters_created"] += 1
        return cluster

    @staticmethod
    def _cluster_distance(lhs: EvidenceCluster, rhs: EvidenceCluster) -> float:
        if lhs.viewpoint is None or rhs.viewpoint is None:
            return math.inf
        return float(np.linalg.norm(lhs.viewpoint - rhs.viewpoint))

    def _compact(self, clusters: List[EvidenceCluster]) -> None:
        while len(clusters) > self.max_clusters:
            pairs = [
                (self._cluster_distance(clusters[i], clusters[j]), i, j)
                for i in range(len(clusters)) for j in range(i + 1, len(clusters))
            ]
            _, keep_index, merge_index = min(pairs, default=(math.inf, 0, 1))
            keep, merged = clusters[keep_index], clusters[merge_index]
            keep.evidence.extend(merged.evidence)
            if keep.viewpoint is not None and merged.viewpoint is not None:
                keep.viewpoint = (keep.viewpoint + merged.viewpoint) / 2.0
            del clusters[merge_index]
            self.stats["view_clusters_merged"] += 1

    def add_evidence(self, evidence: GoalEvidence) -> bool:
        observation_key = self._observation_key(evidence)
        if self.contains_evidence(evidence):
            self.stats["evidence_deduplicated"] += 1
            return False
        self._seen_evidence_ids.add(evidence.evidence_id)
        if evidence.observation_id:
            self._seen_observations.add(observation_key)
        clusters = self._clusters.setdefault((evidence.node_id, evidence.goal_key), [])
        cluster = next((item for item in clusters if self._correlated(item, evidence)), None)
        independent_view = cluster is None
        if cluster is None:
            clusters.append(self._new_cluster(evidence))
        else:
            cluster.evidence.append(evidence)
        self._compact(clusters)
        pair = (evidence.node_id, evidence.goal_key)
        self._evidence_revision[pair] = self._evidence_revision.get(pair, 0) + 1
        self._belief_cache = {
            key: value for key, value in self._belief_cache.items()
            if key[:2] != pair
        }
        self._fusion_cache = {
            key: value for key, value in self._fusion_cache.items()
            if key[:2] != pair
        }
        if independent_view and evidence.source in (
            EvidenceSource.DETECTOR, EvidenceSource.ACTIVE_VERIFY_VLM,
        ):
            self.reset_unknown_policy(evidence.goal_key, "independent_direct_evidence")
        self.stats["evidence_added"] += 1
        return True

    def clusters(self, node_id: str, goal_key: str) -> List[EvidenceCluster]:
        return list(self._clusters.get((node_id, goal_key), []))

    def _fusion_summary(self, node_id: str, goal_key: str) -> EvidenceFusionSummary:
        revision = self._evidence_revision.get((node_id, goal_key), 0)
        cache_key = (node_id, goal_key, revision)
        cached = self._fusion_cache.get(cache_key)
        if cached is not None:
            self.stats["evidence_fusion_cache_hits"] += 1
            return cached
        self.stats["evidence_fusion_cache_misses"] += 1
        clusters = self.clusters(node_id, goal_key)
        source_counts: Dict[str, int] = {}
        evidence_count = 0
        for cluster in clusters:
            for item in cluster.evidence:
                source_counts[item.source.value] = source_counts.get(item.source.value, 0) + 1
                evidence_count += 1
        result = EvidenceFusionSummary(
            non_direct=sum(item.contribution for item in clusters if not item.has_verify),
            direct=sum(item.contribution for item in clusters if item.has_verify),
            negative_verify_views=sum(
                item.has_verify and item.contribution < 0.0 for item in clusters
            ),
            evidence_count=evidence_count,
            source_counts=source_counts,
            conflict=(
                float(np.mean([item.conflict for item in clusters])) if clusters else 0.0
            ),
            independent_view_count=len(clusters),
        )
        self._fusion_cache[cache_key] = result
        return result

    def belief(
        self, node_id: str, goal_key: str,
        structural_confidence: float, reachability: float,
    ) -> GoalBelief:
        pair = (node_id, goal_key)
        cache_key = (
            node_id, goal_key, self._evidence_revision.get(pair, 0),
            float(structural_confidence), float(reachability),
        )
        cached = self._belief_cache.get(cache_key)
        if cached is not None:
            self.stats["posterior_cache_hits"] += 1
            return cached
        self.stats["posterior_cache_misses"] += 1
        fusion = self._fusion_summary(node_id, goal_key)
        log_odds = (
            _logit(self.prior)
            + float(np.clip(structural_confidence, 0.0, 1.0)) * fusion.non_direct
            + fusion.direct
        )
        posterior = _sigmoid(float(np.clip(log_odds, -6.0, 6.0)))
        eliminated = (
            posterior < self.elimination_threshold and
            fusion.negative_verify_views >= self.min_negative_verify_views
        )
        self.stats["posterior_updates"] += 1
        result = GoalBelief(
            node_id=node_id,
            goal_key=goal_key,
            posterior=float(posterior),
            structural_confidence=float(structural_confidence),
            reachability=float(reachability),
            entropy=binary_entropy(posterior),
            independent_view_count=fusion.independent_view_count,
            negative_verify_views=int(fusion.negative_verify_views),
            evidence_count=fusion.evidence_count,
            conflict=fusion.conflict,
            evidence_summary=fusion.source_counts,
            eliminated=eliminated,
        )
        self._belief_cache[cache_key] = result
        return result

    def build_hypotheses(
        self, goal_key: str, executable_nodes: Sequence[ExecutableNode], step: int,
    ) -> GoalHypothesisSet:
        executable = [item for item in executable_nodes if item.reachable]
        facts = {item.node_id: item for item in executable}
        belief_facts = (
            list(facts.values())
            if self.unknown_method == "weighted_geometric" else executable
        )
        beliefs = [
            self.belief(item.node_id, goal_key, item.structural_confidence, item.reachability)
            for item in belief_facts
        ]
        beliefs = [item for item in beliefs if not item.eliminated]
        beliefs.sort(key=lambda item: item.posterior, reverse=True)
        top = beliefs[:max(1, self.top_k)]
        if beliefs:
            log_unknown = sum(math.log(max(1e-8, 1.0 - item.posterior)) for item in beliefs)
            legacy_unknown = float(np.clip(math.exp(log_unknown), 0.0, 1.0))
        else:
            legacy_unknown = 1.0
        unknown_evidence_count = 0
        unknown_weight_sum = 0.0
        unknown_node_weights = {}
        if self.unknown_method == "weighted_geometric":
            weighted_terms = []
            for belief in beliefs:
                fact = facts.get(belief.node_id)
                if fact is None or belief.evidence_count <= 0:
                    continue
                alpha = float(np.clip(
                    belief.structural_confidence * belief.reachability, 0.05, 1.0,
                ))
                weighted_terms.append((alpha, math.log(max(1e-8, 1.0 - belief.posterior))))
                unknown_node_weights[belief.node_id] = alpha
            unknown_evidence_count = len(weighted_terms)
            unknown_weight_sum = float(sum(item[0] for item in weighted_terms))
            unknown = (
                float(np.clip(math.exp(
                    sum(alpha * value for alpha, value in weighted_terms) /
                    unknown_weight_sum
                ), 0.0, 1.0))
                if unknown_weight_sum > 0 else 1.0
            )
        else:
            unknown = legacy_unknown
            unknown_evidence_count = sum(item.evidence_count > 0 for item in beliefs)
            unknown_weight_sum = float(unknown_evidence_count)
        denominator = unknown + sum(item.posterior for item in top)
        weights = {
            item.node_id: item.posterior / denominator for item in top
        } if denominator > 0 else {}
        weights["unknown"] = unknown / denominator if denominator > 0 else 1.0
        top_node = top[0].node_id if top else None
        previous_top = self._last_top_node.get(goal_key)
        rank_switched = goal_key in self._last_top_node and previous_top != top_node
        if rank_switched:
            self.stats["hypothesis_rank_switches"] += 1
        self._last_top_node[goal_key] = top_node
        return GoalHypothesisSet(
            goal_key=goal_key,
            hypotheses=top,
            unknown_probability=unknown,
            action_weights=weights,
            executable_node_ids=[item.node_id for item in belief_facts],
            created_at_step=step,
            previous_top_node_id=previous_top,
            rank_switched=rank_switched,
            legacy_unknown_probability=legacy_unknown,
            unknown_method=self.unknown_method,
            unknown_evidence_node_count=unknown_evidence_count,
            unknown_weight_sum=unknown_weight_sum,
            unknown_node_weights=unknown_node_weights,
        )

    def reset_unknown_policy(self, goal_key: str, reason: str) -> UnknownPolicyState:
        state = self.unknown_policy.setdefault(
            goal_key, UnknownPolicyState(goal_key=goal_key)
        )
        state.consecutive_actions = 0
        state.last_reset_reason = str(reason)
        self.stats["unknown_policy_resets"] += 1
        return state

    def _controlled_unknown_decision(
        self, hypotheses: GoalHypothesisSet,
        executable_nodes: Sequence[ExecutableNode], *, step: int,
        active_confirmation: bool, has_acceptable_snapshot: bool,
        unknown_threshold: float, top_posterior_max: float,
        max_consecutive_actions: int, reset_posterior_delta: float,
    ) -> UnknownPolicyDecision:
        state = self.unknown_policy.setdefault(
            hypotheses.goal_key, UnknownPolicyState(goal_key=hypotheses.goal_key)
        )
        current_ids = {
            item.node_id for item in executable_nodes if item.reachable
        }
        reset_reason = None
        if not state.executable_node_ids:
            state.executable_node_ids = set(current_ids)
        elif current_ids - state.executable_node_ids:
            self.reset_unknown_policy(hypotheses.goal_key, "new_executable_node")
            reset_reason = "new_executable_node"
            state.executable_node_ids.update(current_ids)
        top_posterior = max(
            (item.posterior for item in hypotheses.hypotheses), default=0.0
        )
        if (
            state.last_top_posterior is not None
            and abs(top_posterior - state.last_top_posterior)
            >= float(reset_posterior_delta)
        ):
            self.reset_unknown_policy(hypotheses.goal_key, "top_posterior_delta")
            reset_reason = "top_posterior_delta"
        state.last_top_posterior = top_posterior

        if active_confirmation:
            eligible, reason = False, "active_confirmation"
        elif has_acceptable_snapshot:
            eligible, reason = False, "acceptable_snapshot"
        elif hypotheses.unknown_probability < float(unknown_threshold):
            eligible, reason = False, "unknown_below_threshold"
        elif top_posterior >= float(top_posterior_max):
            eligible, reason = False, "top_posterior_sufficient"
        elif state.consecutive_actions >= max(1, int(max_consecutive_actions)):
            eligible, reason = False, "consecutive_budget"
        else:
            eligible, reason = True, "eligible"
        key = "unknown_policy_eligible" if eligible else "unknown_policy_suppressed"
        self.stats[key] += 1
        self.unknown_policy_reasons[reason] = self.unknown_policy_reasons.get(reason, 0) + 1
        return UnknownPolicyDecision(
            eligible=eligible,
            reason=reason,
            unknown_probability=hypotheses.unknown_probability,
            top_posterior=top_posterior,
            consecutive_actions=state.consecutive_actions,
            reset_reason=reset_reason,
        )

    def build_controlled_unknown_action(
        self, hypotheses: GoalHypothesisSet,
        executable_nodes: Sequence[ExecutableNode], existing_actions,
        *, step: int, active_confirmation: bool = False,
        unknown_threshold: float = 0.75, top_posterior_max: float = 0.30,
        max_consecutive_actions: int = 2,
        reset_posterior_delta: float = 0.05,
    ) -> Tuple[Optional[BeliefActionCandidate], UnknownPolicyDecision]:
        has_snapshot = any(
            action.action_type == BeliefActionType.NAVIGATE
            for action in existing_actions
        )
        decision = self._controlled_unknown_decision(
            hypotheses, executable_nodes, step=step,
            active_confirmation=active_confirmation,
            has_acceptable_snapshot=has_snapshot,
            unknown_threshold=unknown_threshold,
            top_posterior_max=top_posterior_max,
            max_consecutive_actions=max_consecutive_actions,
            reset_posterior_delta=reset_posterior_delta,
        )
        if not decision.eligible:
            return None, decision
        frontier_facts = [
            item for item in executable_nodes
            if item.reachable and item.source_kind == "frontier"
        ]
        if not frontier_facts:
            return None, UnknownPolicyDecision(
                False, "no_frontier", decision.unknown_probability,
                decision.top_posterior, decision.consecutive_actions,
                decision.reset_reason,
            )
        fact = max(
            frontier_facts,
            key=lambda item: (
                item.map_information_gain - 0.1 * item.path_cost - 0.2 * item.revisit,
                -item.path_cost,
                item.node_id,
            ),
        )
        belief = self.belief(
            fact.node_id, hypotheses.goal_key,
            fact.structural_confidence, fact.reachability,
        )
        return BeliefActionCandidate(
            action_type=BeliefActionType.UNKNOWN_EXPLORE,
            node_id=fact.node_id,
            source_kind=fact.source_kind,
            source_index=fact.source_index,
            belief=belief,
            utility=float(
                hypotheses.unknown_probability * fact.map_information_gain
                - 0.1 * fact.path_cost - 0.2 * fact.revisit
            ),
            normalized_path_cost=float(fact.path_cost),
            action_weight=hypotheses.action_weights.get("unknown", 0.0),
        ), decision

    def record_action_selection(self, action: BeliefActionCandidate) -> None:
        self.stats["cumulative_normalized_action_cost"] += float(
            max(0.0, action.normalized_path_cost)
        )
        if action.action_type == BeliefActionType.UNKNOWN_EXPLORE:
            state = self.unknown_policy.setdefault(
                action.belief.goal_key,
                UnknownPolicyState(goal_key=action.belief.goal_key),
            )
            state.consecutive_actions += 1

    def record_snapshot_gate(self, decision: SnapshotAcceptanceDecision) -> None:
        key = "snapshot_gate_accepted" if decision.accepted else "snapshot_gate_rejected"
        self.stats[key] += 1
        self.snapshot_gate_reasons[decision.reason] = (
            self.snapshot_gate_reasons.get(decision.reason, 0) + 1
        )

    def reject_snapshot(
        self, goal_key: str, object_id: Any, step: int,
        decision: SnapshotAcceptanceDecision, duration_steps: int = 5,
    ) -> RejectedSnapshotState:
        state = RejectedSnapshotState(
            goal_key=goal_key, object_id=str(object_id), rejected_step=int(step),
            expires_step=int(step) + max(1, int(duration_steps)),
            reason=decision.reason, compatibility=decision.compatibility,
            detector_confidence=decision.detector_confidence,
            detection_count=decision.detection_count,
        )
        self.rejected_snapshots[(goal_key, str(object_id))] = state
        return state

    def is_snapshot_rejected(
        self, goal_key: str, object_id: Any, step: int, *,
        compatibility: float = 0.0, detector_confidence: float = 0.0,
        detection_count: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        key = (goal_key, str(object_id))
        state = self.rejected_snapshots.get(key)
        if state is None:
            return False, None
        if int(step) > state.expires_step:
            del self.rejected_snapshots[key]
            self.stats["rejected_snapshot_releases"] += 1
            return False, "expired"
        stronger_compatible = (
            state.compatibility < 0.90 <= float(compatibility)
            and float(detector_confidence) > state.detector_confidence
            and int(detection_count) > state.detection_count
        )
        if stronger_compatible:
            del self.rejected_snapshots[key]
            self.stats["rejected_snapshot_releases"] += 1
            return False, "stronger_compatible_detection"
        self.stats["rejected_snapshot_suppressions"] += 1
        return True, state.reason

    def rejected_object_ids(self, goal_key: str, step: int) -> set:
        result = set()
        for (stored_goal, object_id), state in list(self.rejected_snapshots.items()):
            if stored_goal != goal_key:
                continue
            if int(step) > state.expires_step:
                del self.rejected_snapshots[(stored_goal, object_id)]
                self.stats["rejected_snapshot_releases"] += 1
            else:
                result.add(object_id)
        return result

    def can_verify(
        self, node_id: str, goal_key: str, step: int,
        max_attempts: int = 2, cooldown_steps: int = 5,
    ) -> bool:
        key = (node_id, goal_key)
        if self.verify_attempts.get(key, 0) >= max_attempts:
            return False
        return step - self.verify_last_step.get(key, -10**9) >= cooldown_steps

    def record_verify_attempt(self, node_id: str, goal_key: str, step: int) -> None:
        key = (node_id, goal_key)
        self.verify_attempts[key] = self.verify_attempts.get(key, 0) + 1
        self.verify_last_step[key] = step
        self.stats["verify_actions"] += 1

    def observe_clip(
        self, evidence_id: str, node_id: str, goal_key: str,
        raw_cosine: float, step: int, viewpoint=None,
        heading: Optional[float] = None, observation_id: str = "",
        reason: str = "",
    ) -> Tuple[List[GoalEvidence], Dict[str, Any]]:
        """Buffer/calibrate one unique raw CLIP observation for v2 callers."""
        return self.clip_calibrator.observe(BufferedClipEvidence(
            evidence_id=evidence_id,
            node_id=node_id,
            goal_key=goal_key,
            raw_cosine=float(raw_cosine),
            step=int(step),
            viewpoint=None if viewpoint is None else np.asarray(viewpoint, dtype=float)[:2],
            heading=heading,
            observation_id=observation_id,
            reason=reason,
        ))

    def create_confirmation(
        self, node_id: str, goal_key: str, step: int, posterior: float,
        duration_steps: int = 5, max_approaches: int = 2,
        positive_source: str = "", reuse_existing: bool = False,
    ) -> ConfirmationState:
        existing = self.confirmations.get(goal_key)
        if reuse_existing and existing is not None and existing.active:
            if existing.node_id == node_id:
                existing.max_posterior = max(
                    existing.max_posterior, float(posterior)
                )
                if positive_source:
                    existing.positive_source = str(positive_source)
                self._confirmation_events.append({
                    "event": "updated", "state": asdict(existing),
                })
                return existing
            self.resolve_confirmation(goal_key, "superseded")
        state = ConfirmationState(
            node_id=node_id,
            goal_key=goal_key,
            created_step=int(step),
            expires_step=int(step) + max(1, int(duration_steps)),
            posterior_at_creation=float(posterior),
            max_approaches=max(1, int(max_approaches)),
            positive_source=str(positive_source),
            max_posterior=float(posterior),
        )
        self.confirmations[goal_key] = state
        self.stats["confirmations_created"] += 1
        self._confirmation_events.append({"event": "created", "state": asdict(state)})
        return state

    def active_confirmation(self, goal_key: str, step: int) -> Optional[ConfirmationState]:
        state = self.confirmations.get(goal_key)
        if state is None or not state.active:
            return None
        if int(step) > state.expires_step:
            self.resolve_confirmation(goal_key, "expired")
            return None
        return state

    def resolve_confirmation(self, goal_key: str, reason: str) -> Optional[ConfirmationState]:
        state = self.confirmations.get(goal_key)
        if state is None or not state.active:
            return state
        state.active = False
        state.end_reason = str(reason)
        self._confirmation_events.append({
            "event": "resolved", "reason": str(reason), "state": asdict(state),
        })
        if reason == "snapshot_selected":
            self.stats["confirmations_selected"] += 1
        elif reason == "posterior_reversed":
            self.stats["confirmations_reversed"] += 1
        else:
            self.stats["confirmations_expired"] += 1
        return state

    def pop_confirmation_events(self) -> List[Dict[str, Any]]:
        events = list(self._confirmation_events)
        self._confirmation_events.clear()
        return events

    def record_confirmation_approach(
        self, goal_key: str, step: Optional[int] = None,
        wait_for_observation: bool = False,
    ) -> Optional[ConfirmationState]:
        state = self.confirmations.get(goal_key)
        if state is None or not state.active:
            return state
        if wait_for_observation and state.awaiting_observation:
            return state
        state.approach_count += 1
        state.awaiting_observation = bool(wait_for_observation)
        if wait_for_observation:
            state.last_approach_step = None if step is None else int(step)
            state.awaiting_observation_step = (
                None if step is None else int(step) + 1
            )
        self.stats["confirmation_approaches"] += 1
        return state

    def record_confirmation_observation(
        self, goal_key: str, step: int, snapshot_found: bool,
    ) -> Optional[ConfirmationState]:
        state = self.confirmations.get(goal_key)
        if state is None or not state.active or not state.awaiting_observation:
            return state
        if (
            state.awaiting_observation_step is not None
            and int(step) < state.awaiting_observation_step
        ):
            return state
        state.awaiting_observation = False
        state.awaiting_observation_step = None
        if snapshot_found:
            return state
        if state.approach_count >= state.max_approaches:
            self.resolve_confirmation(goal_key, "no_snapshot_after_observation")
        return state

    def record_confirmation_failure(self, goal_key: str) -> Optional[ConfirmationState]:
        state = self.confirmations.get(goal_key)
        if state is None or not state.active:
            return state
        state.failure_count += 1
        self.stats["confirmation_failures"] += 1
        if state.failure_count >= 2:
            self.resolve_confirmation(goal_key, "failure_budget")
        return state

    def build_confirmation_action(
        self, state: ConfirmationState, executable_nodes: Sequence[ExecutableNode],
    ) -> Optional[BeliefActionCandidate]:
        """Use only a currently executable raw frontier mapping; never ghost coordinates."""
        if (
            not state.active or state.awaiting_observation
            or state.approach_count >= state.max_approaches
        ):
            return None
        fact = next((
            item for item in executable_nodes
            if item.node_id == state.node_id and item.source_kind == "frontier" and item.reachable
        ), None)
        if fact is None:
            return None
        belief = self.belief(
            fact.node_id, state.goal_key,
            fact.structural_confidence, fact.reachability,
        )
        if belief.posterior < 0.30:
            self.resolve_confirmation(state.goal_key, "posterior_reversed")
            return None
        path = float(np.clip(fact.path_cost, 0.0, 1.0))
        return BeliefActionCandidate(
            action_type=BeliefActionType.CONFIRM_APPROACH,
            node_id=fact.node_id,
            source_kind=fact.source_kind,
            source_index=fact.source_index,
            belief=belief,
            utility=float(belief.posterior * belief.reachability - 0.1 * path),
            normalized_path_cost=path,
            confirmation_source_node_id=state.node_id,
        )

    def build_confirmation_snapshot_action(
        self, state: ConfirmationState, snapshot_fact: Optional[ExecutableNode],
        confirmed_belief: GoalBelief, navigate_threshold: float = 0.75,
    ) -> Optional[BeliefActionCandidate]:
        """Create a confirmation NAVIGATE only from a real current snapshot fact."""
        if (
            not state.active
            or snapshot_fact is None
            or not snapshot_fact.reachable
            or snapshot_fact.source_kind != "snapshot"
            or confirmed_belief.node_id != state.node_id
            or confirmed_belief.posterior < float(navigate_threshold)
        ):
            return None
        snapshot_belief = self.belief(
            snapshot_fact.node_id,
            state.goal_key,
            snapshot_fact.structural_confidence,
            snapshot_fact.reachability,
        )
        path = float(np.clip(snapshot_fact.path_cost, 0.0, 1.0))
        return BeliefActionCandidate(
            action_type=BeliefActionType.NAVIGATE,
            node_id=snapshot_fact.node_id,
            source_kind="snapshot",
            source_index=snapshot_fact.source_index,
            belief=snapshot_belief,
            utility=float(
                confirmed_belief.posterior * snapshot_fact.reachability
                - 0.1 * path - 0.2 * float(np.clip(snapshot_fact.revisit, 0.0, 1.0))
            ),
            normalized_path_cost=path,
            confirmation_source_node_id=state.node_id,
        )

    def build_actions(
        self,
        hypotheses: GoalHypothesisSet,
        executable_nodes: Sequence[ExecutableNode],
        verification_options: Optional[Dict[str, VerificationOption]] = None,
        posterior_low: float = 0.30,
        posterior_high: float = 0.70,
        navigate_threshold: float = 0.75,
        step: int = 0,
        max_verify_attempts: int = 2,
        verify_cooldown_steps: int = 5,
        include_unknown: bool = True,
    ) -> List[BeliefActionCandidate]:
        verification_options = verification_options or {}
        facts = {item.node_id: item for item in executable_nodes if item.reachable}
        actions: List[BeliefActionCandidate] = []
        for belief in hypotheses.hypotheses:
            fact = facts.get(belief.node_id)
            if fact is None:
                continue
            path = float(np.clip(fact.path_cost, 0.0, 1.0))
            revisit = float(np.clip(fact.revisit, 0.0, 1.0))
            option = verification_options.get(belief.node_id)
            if (posterior_low <= belief.posterior <= posterior_high and
                    belief.evidence_count > 0 and option is not None and
                    self.can_verify(belief.node_id, belief.goal_key, step,
                                    max_verify_attempts, verify_cooldown_steps)):
                utility = (
                    belief.reachability * belief.entropy * option.visibility * option.novelty
                    - 0.1 * float(np.clip(option.path_cost, 0.0, 1.0))
                )
                action_type = BeliefActionType.VERIFY
            elif fact.source_kind == "snapshot" and belief.posterior >= navigate_threshold:
                utility = belief.posterior * belief.reachability - 0.1 * path - 0.2 * revisit
                action_type = BeliefActionType.NAVIGATE
                option = None
            elif fact.source_kind == "frontier":
                utility = (
                    belief.posterior * belief.reachability - 0.1 * path - 0.2 * revisit
                    + 0.05 * float(np.clip(fact.map_information_gain, 0.0, 1.0))
                )
                action_type = BeliefActionType.EXPLORE
                option = None
            else:
                continue
            actions.append(BeliefActionCandidate(
                action_type=action_type,
                node_id=belief.node_id,
                source_kind=fact.source_kind,
                source_index=fact.source_index,
                belief=belief,
                utility=float(utility),
                normalized_path_cost=path,
                verification=option,
                action_weight=hypotheses.action_weights.get(belief.node_id, 0.0),
            ))

        frontier_facts = [item for item in executable_nodes if item.reachable and item.source_kind == "frontier"]
        best_posterior = max((item.posterior for item in hypotheses.hypotheses), default=0.0)
        if include_unknown and hypotheses.unknown_probability > best_posterior and frontier_facts:
            fact = max(
                frontier_facts,
                key=lambda item: item.map_information_gain - 0.1 * item.path_cost,
            )
            belief = self.belief(
                fact.node_id, hypotheses.goal_key,
                fact.structural_confidence, fact.reachability,
            )
            actions.append(BeliefActionCandidate(
                action_type=BeliefActionType.UNKNOWN_EXPLORE,
                node_id=fact.node_id,
                source_kind=fact.source_kind,
                source_index=fact.source_index,
                belief=belief,
                utility=float(
                    hypotheses.unknown_probability * fact.map_information_gain
                    - 0.1 * fact.path_cost
                ),
                normalized_path_cost=float(fact.path_cost),
                action_weight=hypotheses.action_weights.get("unknown", 0.0),
            ))
        actions.sort(key=lambda item: item.utility, reverse=True)
        return actions

    def choose_deterministic_action(
        self, actions: Sequence[BeliefActionCandidate],
    ) -> Optional[BeliefActionCandidate]:
        """Stable v3 category choice without a VLM tie-break."""
        if not actions:
            return None
        priority = {
            BeliefActionType.NAVIGATE: 4,
            BeliefActionType.VERIFY: 3,
            BeliefActionType.EXPLORE: 2,
            BeliefActionType.UNKNOWN_EXPLORE: 1,
            BeliefActionType.CONFIRM_APPROACH: 5,
        }
        selected = max(
            actions,
            key=lambda item: (
                priority[item.action_type],
                item.utility,
                item.belief.posterior,
                item.belief.reachability,
                -item.normalized_path_cost,
                item.node_id,
            ),
        )
        self.stats["planner_direct_choices"] += 1
        self.stats["category_deterministic_choices"] += 1
        return selected

    def choose_actions_for_tiebreak(
        self, actions: Sequence[BeliefActionCandidate], decision_margin: float = 0.15,
    ) -> Tuple[Optional[BeliefActionCandidate], List[BeliefActionCandidate]]:
        if not actions:
            return None, []
        if len(actions) == 1 or actions[0].utility - actions[1].utility >= decision_margin:
            self.stats["planner_direct_choices"] += 1
            return actions[0], []
        return None, list(actions[:2])

    def evidence_trace(self, node_id: str, goal_key: str) -> Dict[str, Any]:
        clusters = self.clusters(node_id, goal_key)
        return {
            "node_id": node_id,
            "goal_key": goal_key,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "viewpoint": cluster.viewpoint,
                    "heading": cluster.heading,
                    "has_verify": cluster.has_verify,
                    "contribution": cluster.contribution,
                    "conflict": cluster.conflict,
                    "evidence": [asdict(item) for item in cluster.evidence],
                }
                for cluster in clusters
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            **self.clip_calibrator.stats,
            "snapshot_gate_reasons": dict(self.snapshot_gate_reasons),
            "unknown_policy_reasons": dict(self.unknown_policy_reasons),
        }


# One canonical, training-free room/object table is shared by all category
# likelihood sources. Keep names normalized with spaces.
ROOM_OBJECT_ASSOCIATIONS: Dict[str, set] = {
    "bedroom": {"bed", "nightstand", "dresser", "wardrobe", "pillow"},
    "bathroom": {"toilet", "sink", "shower", "bathtub", "mirror"},
    "kitchen": {"stove", "oven", "refrigerator", "microwave", "sink"},
    "living room": {"sofa", "tv", "television", "coffee table", "lamp"},
    "dining room": {"dining table", "chair"},
    "hallway": {"door", "painting", "console table"},
    "closet": {"wardrobe", "shelf", "hanger"},
    "office": {"desk", "chair", "monitor", "keyboard"},
}


def room_support_for_category(semantic_dist: Any, category: str) -> float:
    if semantic_dist is None:
        return 0.0
    target = normalize_goal_key(category).split(":", 1)[1]
    support = 0.0
    for room, probability in zip(
        getattr(semantic_dist, "categories", []),
        getattr(semantic_dist, "probabilities", []),
    ):
        normalized_room = str(room).strip().lower().replace("_", " ")
        if target in ROOM_OBJECT_ASSOCIATIONS.get(normalized_room, set()):
            support += float(probability)
    return float(np.clip(support, 0.0, 1.0))
