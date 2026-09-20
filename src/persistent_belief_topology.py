"""Persistent, task-conditioned topology used by HGR-ActiveTopo.

The persistent graph stores episode-scoped environment facts (C/A and stable
identity).  Goal relevance and lifecycle state live only in ActiveTopoView.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
from enum import Enum
import json
import heapq
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


class TopoNodeType(str, Enum):
    VISITED = "visited"
    OBSERVED = "observed"
    FRONTIER = "frontier"
    HYPOTHESIS = "hypothesis"


class TopoEdgeType(str, Enum):
    NAVIGABLE = "navigable"
    OBSERVED_AT = "observed_at"
    ANCHORED_TO = "anchored_to"
    DEPENDS_ON = "depends_on"
    SPATIAL_NEAR = "spatial_near"


class ActiveTopoState(str, Enum):
    ACTIVE = "active"
    VERIFY = "verify"
    FOLDED = "folded"
    SUPPRESSED = "suppressed"


class GoalTopoActionType(str, Enum):
    """Executable high-level actions exposed by one goal-conditioned view."""

    DIRECT = "direct"
    REVISIT = "revisit"
    EXPLORE = "explore"


class ActiveTopologyMode(str, Enum):
    """Behavioral modes used by the GOAT ablation study."""

    FULL = "full"
    STABLE_ONLY = "stable_only"
    CONFIDENCE_ACCESSIBILITY = "confidence_accessibility"
    CATEGORY_ONLY = "category_only"
    BELIEF_CATEGORY = "belief_category"
    BELIEF_CATEGORY_V2 = "belief_category_v2"
    BELIEF_CATEGORY_V3 = "belief_category_v3"
    SIMPLE_MEMORY = "simple_memory"
    GOAL_TOPO_MAP = "goal_topo_map"


@dataclass(frozen=True)
class ActiveTopologyPolicy:
    mode: ActiveTopologyMode
    use_active_view: bool
    use_goal_relevance: bool
    use_belief_planner: bool = False


def resolve_active_topology_policy(mode: str, goal_type: str) -> ActiveTopologyPolicy:
    """Resolve one explicit ablation mode for the current GOAT subtask.

    ``object`` is accepted as an alias for GOAT's normalized ``category`` goal.
    Invalid modes fail fast so a misspelled experiment cannot silently run the
    full method.
    """

    try:
        resolved_mode = ActiveTopologyMode(str(mode or ActiveTopologyMode.FULL.value))
    except ValueError as exc:
        choices = ", ".join(item.value for item in ActiveTopologyMode)
        raise ValueError(f"Unknown active_topology.mode={mode!r}; expected one of: {choices}") from exc
    normalized_goal = "category" if str(goal_type) == "object" else str(goal_type)
    if resolved_mode == ActiveTopologyMode.STABLE_ONLY:
        return ActiveTopologyPolicy(resolved_mode, False, False)
    if resolved_mode == ActiveTopologyMode.SIMPLE_MEMORY:
        # Stable frontier identity and navigation-failure memory only.  This
        # mode deliberately leaves goal reasoning and snapshot selection to
        # the original HGR pipeline for every GOAT goal type.
        return ActiveTopologyPolicy(resolved_mode, True, False)
    if resolved_mode == ActiveTopologyMode.GOAL_TOPO_MAP:
        # The persistent graph is goal-independent.  Only its per-subtask
        # connected projection and the ordering of executable frontiers are
        # goal-conditioned.
        return ActiveTopologyPolicy(resolved_mode, True, True)
    if resolved_mode == ActiveTopologyMode.CONFIDENCE_ACCESSIBILITY:
        return ActiveTopologyPolicy(resolved_mode, True, False)
    if resolved_mode == ActiveTopologyMode.CATEGORY_ONLY:
        enabled = normalized_goal == "category"
        return ActiveTopologyPolicy(resolved_mode, enabled, enabled)
    if resolved_mode in (
        ActiveTopologyMode.BELIEF_CATEGORY, ActiveTopologyMode.BELIEF_CATEGORY_V2,
        ActiveTopologyMode.BELIEF_CATEGORY_V3,
    ):
        enabled = normalized_goal == "category"
        return ActiveTopologyPolicy(resolved_mode, False, enabled, enabled)
    return ActiveTopologyPolicy(resolved_mode, True, True)


def restrict_active_topology_policy(
    policy: ActiveTopologyPolicy,
    goal_type: str,
    behavior_goal_types: Optional[Iterable[str]],
) -> ActiveTopologyPolicy:
    """Disable topology *behavior* outside an explicit GOAT goal allow-list.

    The persistent episode graph is intentionally unaffected by this gate.  A
    disabled subtask can therefore keep contributing observations to the
    global map while retaining the original HGR candidate ordering, prompt,
    selection, and navigation behavior.

    ``object`` and ``category`` are treated as aliases so configurations can
    use GOAT's public task name without depending on the internal normalized
    goal name.  ``None`` preserves the legacy behavior of existing configs.
    """

    if behavior_goal_types is None:
        return policy

    normalize = lambda value: (
        "category" if str(value).strip().lower() in ("object", "category")
        else str(value).strip().lower()
    )
    enabled_types = {normalize(value) for value in behavior_goal_types}
    if normalize(goal_type) in enabled_types:
        return policy
    return ActiveTopologyPolicy(
        mode=policy.mode,
        use_active_view=False,
        use_goal_relevance=False,
        use_belief_planner=False,
    )


def active_topology_goal_type_is_listed(
    goal_type: str,
    configured_goal_types: Optional[Iterable[str]],
) -> bool:
    """Return whether a GOAT goal is in a normalized configuration list."""

    if configured_goal_types is None:
        return False

    def normalize(value: str) -> str:
        value = str(value).strip().lower()
        return "category" if value in ("object", "category") else value

    normalized_types = {normalize(value) for value in configured_goal_types}
    return normalize(goal_type) in normalized_types


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return math.log(value / (1.0 - value))


def _unit(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-8 else None


def _cosine01(lhs: Optional[np.ndarray], rhs: Optional[np.ndarray]) -> Optional[float]:
    left, right = _unit(lhs), _unit(rhs)
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip((float(np.dot(left, right)) + 1.0) / 2.0, 0.0, 1.0))


def _clip_relevance(
    lhs: Optional[np.ndarray], rhs: Optional[np.ndarray],
    floor: float = 0.10, ceiling: float = 0.35,
) -> Optional[float]:
    """Return positive CLIP cosine on the probability-like R scale.

    Frontier matching uses cosine mapped from [-1, 1] to [0, 1], where 0.5
    correctly means neutral similarity. Goal relevance is different: treating
    neutral CLIP similarity as R=0.5 made unrelated candidates impossible to
    fold with the frozen R<0.20 threshold. For R, negative similarity carries
    no positive evidence and neutral similarity maps to zero.
    """
    left, right = _unit(lhs), _unit(rhs)
    if left is None or right is None or left.shape != right.shape:
        return None
    if ceiling <= floor:
        raise ValueError("CLIP relevance ceiling must be greater than floor")
    cosine = float(np.dot(left, right))
    return float(np.clip((cosine - floor) / (ceiling - floor), 0.0, 1.0))


@dataclass
class BeliefTopoNode:
    node_id: str
    node_type: TopoNodeType
    position: np.ndarray
    map_log_odds: float = 0.0
    semantic_dist: Any = None
    observed_categories: Set[str] = field(default_factory=set)
    evidence_refs: List[str] = field(default_factory=list)
    visual_embedding: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    region: Optional[np.ndarray] = None
    view_count: int = 0
    positive_evidence: float = 0.0
    negative_evidence: float = 0.0
    negative_evidence_refs: Set[str] = field(default_factory=set)
    last_positive_step: int = -1
    last_observed_step: int = -1
    visit_count: int = 0
    nav_success_count: float = 0.0
    nav_failure_count: float = 0.0
    consecutive_failures: int = 0
    blocked_until_step: int = -1
    consumed: bool = False
    hypothesis_node_id: Optional[str] = None
    active_this_step: bool = False
    semantic_cache: Optional["FrontierSemanticCacheEntry"] = None

    @property
    def map_confidence(self) -> float:
        return _sigmoid(self.map_log_odds)

    @property
    def accessibility(self) -> float:
        return (self.nav_success_count + 1.0) / (
            self.nav_success_count + self.nav_failure_count + 2.0
        )


@dataclass
class TopoEdge:
    source_id: str
    target_id: str
    edge_type: TopoEdgeType
    distance: Optional[float] = None
    path_cost: Optional[float] = None
    traversability: Optional[float] = None
    dependency_confidence: float = 1.0
    visit_count: int = 0
    failure_count: int = 0
    last_updated_step: int = -1


@dataclass
class FrontierSemanticCacheEntry:
    semantic_dist: Any
    reference_embedding: Optional[np.ndarray]
    reference_position: np.ndarray
    prediction_step: int
    prediction_source: str


@dataclass(frozen=True)
class GoalContext:
    subtask_id: str
    goal_type: str
    category: Optional[str] = None
    description: Optional[str] = None
    image_embedding: Optional[np.ndarray] = field(default=None, compare=False)
    text_embedding: Optional[np.ndarray] = field(default=None, compare=False)


@dataclass
class ActiveTopoCandidate:
    topo_node_id: str
    source_frontier_index: int
    relevance: float
    confidence: float
    accessibility: float
    path_cost: float
    revisit_penalty: float
    information_gain: float
    utility: float
    state: ActiveTopoState
    raw_path_cost: float = 0.0
    match_score: Optional[float] = None


@dataclass
class ActiveTopoView:
    goal: GoalContext
    recovery_level: int
    candidates: List[ActiveTopoCandidate]
    created_at_step: int
    all_candidates: List[ActiveTopoCandidate] = field(default_factory=list)
    uses_goal_relevance: bool = True
    include_prompt_metadata: bool = True
    included_node_ids: List[str] = field(default_factory=list)
    connector_node_ids: List[str] = field(default_factory=list)
    evidence_node_ids: List[str] = field(default_factory=list)
    topology_context: Optional[Dict[str, Any]] = None

    def resolve_frontier(self, projected_index: int, frontiers: list):
        if projected_index < 0 or projected_index >= len(self.candidates):
            raise IndexError(f"Projected frontier index out of range: {projected_index}")
        source_index = self.candidates[projected_index].source_frontier_index
        if source_index < 0 or source_index >= len(frontiers):
            raise IndexError(f"Source frontier index out of range: {source_index}")
        return frontiers[source_index]


@dataclass(frozen=True)
class GoalTopoAction:
    action_type: GoalTopoActionType
    topo_node_id: str
    source_kind: str
    source_frontier_index: Optional[int] = None
    snapshot_image: Optional[str] = None
    object_id: Optional[Any] = None
    relevance: float = 0.5
    current_observation: bool = False


@dataclass
class GoalTopoTaskView:
    """Unified executable projection over objects, snapshots and frontiers."""

    goal: GoalContext
    created_at_step: int
    actions: List[GoalTopoAction]

    @property
    def direct_actions(self) -> List[GoalTopoAction]:
        return [
            action for action in self.actions
            if action.action_type == GoalTopoActionType.DIRECT
        ]

    def resolve_snapshot(
        self, snapshot_image: str, object_id: Optional[Any] = None,
    ) -> Optional[GoalTopoAction]:
        matches = [
            action for action in self.actions
            if action.source_kind == "snapshot"
            and action.snapshot_image == snapshot_image
        ]
        if object_id is not None:
            exact = [
                action for action in matches
                if action.object_id == object_id
            ]
            if exact:
                return exact[0]
        return matches[0] if matches else None

    def resolve_frontier(self, source_index: int) -> Optional[GoalTopoAction]:
        return next((
            action for action in self.actions
            if action.source_kind == "frontier"
            and action.source_frontier_index == source_index
        ), None)


@dataclass
class VerificationEvidence:
    node_id: str
    is_positive: bool
    strength: float
    residual: Optional[float]
    reason: str
    evidence_ref: str


@dataclass
class RecoveryState:
    level: int = 0
    max_level: int = 2
    no_progress_steps: int = 0
    last_selected_node_id: Optional[str] = None
    last_selected_path_cost: Optional[float] = None

    def reset(self) -> None:
        self.level = 0
        self.no_progress_steps = 0
        self.last_selected_node_id = None
        self.last_selected_path_cost = None

    def record_progress(self) -> None:
        self.level = 0
        self.no_progress_steps = 0

    def record_step(self, made_progress: bool, budget: int) -> bool:
        if made_progress:
            self.record_progress()
            return False
        self.no_progress_steps += 1
        if self.no_progress_steps >= budget and self.level < self.max_level:
            self.level += 1
            self.no_progress_steps = 0
            return True
        return False

    def escalate(self) -> bool:
        if self.level >= self.max_level:
            return False
        self.level += 1
        self.no_progress_steps = 0
        return True


class ActiveTopologyTraceWriter:
    """Append-only JSONL trace plus an episode summary."""

    def __init__(self, output_dir: str, episode_id: str):
        trace_dir = os.path.join(output_dir, "active_topology_traces")
        os.makedirs(trace_dir, exist_ok=True)
        safe_id = str(episode_id).replace(os.sep, "_")
        self.trace_path = os.path.join(trace_dir, f"{safe_id}.jsonl")
        self.summary_path = os.path.join(trace_dir, f"{safe_id}.summary.json")
        # Logger results are committed at episode completion.  If a process
        # stops midway, that episode is replayed on the next run; rebuild its
        # trace instead of appending a duplicate prefix.  Completed episodes
        # are skipped before a writer is constructed by the GOAT main loop.
        with open(self.trace_path, "w", encoding="utf-8"):
            pass
        self.events = 0

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (Enum,)):
            return value.value
        if isinstance(value, set):
            return sorted(value)
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return str(value)

    def write(self, event: Dict[str, Any]) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=self._json_default, ensure_ascii=False) + "\n")
        self.events += 1

    def save_summary(self, summary: Dict[str, Any]) -> None:
        payload = {**summary, "trace_events": self.events, "trace_path": self.trace_path}
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=self._json_default, ensure_ascii=False, indent=2)


class PersistentBeliefTopology:
    """Episode-scoped persistent graph and ActiveTopo projector."""

    def __init__(self, cfg: Dict[str, Any], voxel_size: float):
        self.cfg = cfg or {}
        self.voxel_size = float(voxel_size)
        self.nodes: Dict[str, BeliefTopoNode] = {}
        self.edges: Dict[Tuple[str, str, TopoEdgeType], TopoEdge] = {}
        self.frontier_source_map: Dict[int, str] = {}
        self.frontier_match_scores: Dict[int, float] = {}
        self.snapshot_anchor_map: Dict[str, str] = {}
        self._counters = {kind: 0 for kind in TopoNodeType}
        self._seen_evidence: Set[str] = set()
        self._last_visited_id: Optional[str] = None
        self.recovery = RecoveryState()
        self.goal: Optional[GoalContext] = None
        self.current_view: Optional[ActiveTopoView] = None
        self.current_task_view: Optional[GoalTopoTaskView] = None
        self.stats = {
            "nodes_created": 0,
            "observed_merges": 0,
            "frontier_matches": 0,
            "frontier_new": 0,
            "frontier_match_attempts": 0,
            "navigation_successes": 0,
            "navigation_failures": 0,
            "navigation_progresses": 0,
            "blocked_target_retries": 0,
            "suppressed_nodes": 0,
            "recovery_escalations": [0, 0, 0],
            "soft_corrections": 0,
            "hard_corrections": 0,
            "pruned_nodes": 0,
            "semantic_cache_hits": 0,
            "semantic_cache_misses": 0,
            "semantic_predictor_calls_saved": 0,
            "semantic_cache_stores": 0,
            "semantic_cache_invalid_source": 0,
            "semantic_cache_visual_miss": 0,
            "semantic_cache_position_miss": 0,
            "semantic_cache_empty": 0,
        }

    def _section(self, name: str) -> Dict[str, Any]:
        value = self.cfg.get(name, {})
        return dict(value) if value is not None else {}

    def _new_id(self, kind: TopoNodeType) -> str:
        value = self._counters[kind]
        self._counters[kind] += 1
        return f"{kind.value}_{value}"

    def _distance_m(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(lhs)[:2] - np.asarray(rhs)[:2]) * self.voxel_size)

    def _add_edge(self, edge: TopoEdge) -> None:
        self.edges[(edge.source_id, edge.target_id, edge.edge_type)] = edge

    def _remove_node(self, node_id: str) -> Optional[BeliefTopoNode]:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return None
        self.edges = {
            key: edge for key, edge in self.edges.items()
            if edge.source_id != node_id and edge.target_id != node_id
        }
        return node

    def set_goal(self, goal: GoalContext) -> None:
        self.goal = goal
        self.current_view = None
        self.current_task_view = None
        self.recovery.reset()

    def decay_confidence(self, step: int) -> None:
        decay = float(self._section("confidence").get("decay", 0.995))
        for node in self.nodes.values():
            if node.last_observed_step != step:
                node.map_log_odds = float(np.clip(node.map_log_odds * decay, -6.0, 6.0))

    def upsert_visited(self, position, step: int, evidence_ref: Optional[str] = None) -> str:
        pos = np.asarray(position, dtype=float)[:2]
        threshold = float(self.cfg.get("visited_merge_distance", 0.75))
        candidates = [n for n in self.nodes.values() if n.node_type == TopoNodeType.VISITED]
        nearest = min(candidates, key=lambda n: self._distance_m(n.position, pos), default=None)
        if nearest is not None and self._distance_m(nearest.position, pos) <= threshold:
            nearest.position = (nearest.position * nearest.view_count + pos) / (nearest.view_count + 1)
            nearest.view_count += 1
            nearest.visit_count += 1
            nearest.last_observed_step = step
            if evidence_ref and evidence_ref not in nearest.evidence_refs:
                nearest.evidence_refs.append(evidence_ref)
            if (self._last_visited_id is not None and
                    self._last_visited_id != nearest.node_id and
                    self._last_visited_id in self.nodes):
                previous = self.nodes[self._last_visited_id]
                self._add_edge(TopoEdge(
                    self._last_visited_id,
                    nearest.node_id,
                    TopoEdgeType.NAVIGABLE,
                    distance=self._distance_m(previous.position, nearest.position),
                    traversability=1.0,
                    last_updated_step=step,
                ))
            self._last_visited_id = nearest.node_id
            return nearest.node_id
        node_id = self._new_id(TopoNodeType.VISITED)
        self.nodes[node_id] = BeliefTopoNode(
            node_id=node_id, node_type=TopoNodeType.VISITED, position=pos,
            view_count=1, visit_count=1, last_observed_step=step,
            evidence_refs=[evidence_ref] if evidence_ref else [],
        )
        self.stats["nodes_created"] += 1
        if self._last_visited_id is not None and self._last_visited_id in self.nodes:
            previous = self.nodes[self._last_visited_id]
            self._add_edge(TopoEdge(
                self._last_visited_id,
                node_id,
                TopoEdgeType.NAVIGABLE,
                distance=self._distance_m(previous.position, pos),
                traversability=1.0,
                last_updated_step=step,
            ))
        self._last_visited_id = node_id
        return node_id

    def upsert_observed(
        self, snapshot, position, categories: Iterable[str], step: int,
        anchor_visited_id: str, evidence_ref: Optional[str] = None,
        visual_embedding: Optional[np.ndarray] = None,
        reanchor_existing_evidence: bool = True,
    ) -> str:
        pos = np.asarray(position, dtype=float)[:2]
        cats = {str(cat).strip().lower() for cat in categories if str(cat).strip()}
        snapshot_image = (
            str(evidence_ref).split("snapshot:", 1)[1]
            if str(evidence_ref or "").startswith("snapshot:") else None
        )
        if evidence_ref:
            for existing in self.nodes.values():
                if (existing.node_type == TopoNodeType.OBSERVED and
                        evidence_ref in existing.evidence_refs):
                    existing.last_observed_step = step
                    anchor = self.nodes.get(anchor_visited_id)
                    if anchor is not None and reanchor_existing_evidence:
                        self._add_edge(TopoEdge(
                            existing.node_id,
                            anchor_visited_id,
                            TopoEdgeType.OBSERVED_AT,
                            distance=self._distance_m(existing.position, anchor.position),
                            last_updated_step=step,
                        ))
                    elif anchor is not None:
                        # GoalTopo treats OBSERVED_AT as the place where this
                        # concrete snapshot was captured. Merely iterating over
                        # the persistent snapshot dictionary on a later step
                        # must not move that historical fact to the robot's
                        # current pose.
                        self.stats.setdefault("observed_reanchor_skips", 0)
                        self.stats["observed_reanchor_skips"] += 1
                    if snapshot_image and anchor is not None:
                        self.snapshot_anchor_map.setdefault(
                            snapshot_image, anchor_visited_id
                        )
                    return existing.node_id
        merge_cfg = self._section("observed_merge")
        max_dist = float(merge_cfg.get("max_distance", 0.75))
        empty_dist = float(merge_cfg.get("empty_category_max_distance", 0.35))
        min_jaccard = float(merge_cfg.get("min_category_jaccard", 0.30))
        best = None
        best_dist = float("inf")
        for node in self.nodes.values():
            if node.node_type != TopoNodeType.OBSERVED:
                continue
            dist = self._distance_m(node.position, pos)
            if not cats or not node.observed_categories:
                valid = dist <= empty_dist
            else:
                union = cats | node.observed_categories
                jaccard = len(cats & node.observed_categories) / max(len(union), 1)
                valid = dist <= max_dist and jaccard >= min_jaccard
            if valid and dist < best_dist:
                best, best_dist = node, dist
        if best is None:
            node_id = self._new_id(TopoNodeType.OBSERVED)
            best = BeliefTopoNode(
                node_id=node_id, node_type=TopoNodeType.OBSERVED, position=pos,
                observed_categories=cats, visual_embedding=_unit(visual_embedding),
                view_count=0,
            )
            self.nodes[node_id] = best
            self.stats["nodes_created"] += 1
        else:
            self.stats["observed_merges"] += 1
            best.position = (best.position * best.view_count + pos) / (best.view_count + 1)
            best.observed_categories |= cats
            if visual_embedding is not None:
                merged = _unit(visual_embedding)
                if best.visual_embedding is not None and merged is not None:
                    merged = _unit(best.visual_embedding * best.view_count + merged)
                best.visual_embedding = merged
        best.view_count += 1
        best.last_observed_step = step
        if evidence_ref and evidence_ref not in best.evidence_refs:
            best.evidence_refs.append(evidence_ref)
        self._update_evidence(best, True, 1.0, evidence_ref or f"observed:{step}:{best.node_id}", step)
        self._add_edge(TopoEdge(best.node_id, anchor_visited_id, TopoEdgeType.OBSERVED_AT,
                                distance=self._distance_m(best.position, self.nodes[anchor_visited_id].position),
                                last_updated_step=step))
        if snapshot_image:
            self.snapshot_anchor_map.setdefault(snapshot_image, anchor_visited_id)
        return best.node_id

    @staticmethod
    def _region_iou(lhs: Optional[np.ndarray], rhs: Optional[np.ndarray]) -> Optional[float]:
        if lhs is None or rhs is None or np.shape(lhs) != np.shape(rhs):
            return None
        union = np.logical_or(lhs, rhs).sum()
        return float(np.logical_and(lhs, rhs).sum() / union) if union else 0.0

    def _frontier_match_score(self, node: BeliefTopoNode, frontier, embedding) -> float:
        match_cfg = self._section("frontier_matching")
        weights = dict(match_cfg.get("weights", {}))
        weights = {"distance": 0.45, "region_iou": 0.25, "direction": 0.15,
                   "visual": 0.15, **weights}
        scale = float(match_cfg.get("distance_scale", 1.0))
        values = {"distance": math.exp(-self._distance_m(node.position, frontier.position) / scale)}
        values["region_iou"] = self._region_iou(node.region, getattr(frontier, "region", None))
        values["direction"] = _cosine01(node.orientation, getattr(frontier, "orientation", None))
        values["visual"] = _cosine01(node.visual_embedding, embedding)
        available = [(weights[name], value) for name, value in values.items() if value is not None]
        total = sum(weight for weight, _ in available)
        return sum(weight * value for weight, value in available) / total if total else 0.0

    def match_frontiers(
        self, frontiers: list, step: int,
        visual_embeddings: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, str]:
        visual_embeddings = visual_embeddings or {}
        self.frontier_source_map = {}
        self.frontier_match_scores = {}
        historical = [n for n in self.nodes.values() if n.node_type == TopoNodeType.FRONTIER]
        for node in historical:
            node.active_this_step = False
        threshold = float(self._section("frontier_matching").get("min_score", 0.60))
        pairs = []
        for index, frontier in enumerate(frontiers):
            for node in historical:
                score = self._frontier_match_score(node, frontier, visual_embeddings.get(index))
                self.stats["frontier_match_attempts"] += 1
                if score >= threshold:
                    pairs.append((score, index, node.node_id))
        used_indices, used_nodes = set(), set()
        for score, index, node_id in sorted(pairs, reverse=True):
            if index in used_indices or node_id in used_nodes:
                continue
            used_indices.add(index)
            used_nodes.add(node_id)
            self.frontier_source_map[index] = node_id
            self.frontier_match_scores[index] = float(score)
            self.stats["frontier_matches"] += 1
        for index, frontier in enumerate(frontiers):
            embedding = _unit(visual_embeddings.get(index))
            is_new = index not in self.frontier_source_map
            if index not in self.frontier_source_map:
                node_id = self._new_id(TopoNodeType.FRONTIER)
                self.nodes[node_id] = BeliefTopoNode(
                    node_id=node_id, node_type=TopoNodeType.FRONTIER,
                    position=np.asarray(frontier.position, dtype=float)[:2],
                    visual_embedding=embedding,
                )
                self.frontier_source_map[index] = node_id
                self.frontier_match_scores[index] = 0.0
                self.stats["frontier_new"] += 1
                self.stats["nodes_created"] += 1
            node = self.nodes[self.frontier_source_map[index]]
            node.position = np.asarray(frontier.position, dtype=float)[:2]
            node.orientation = np.asarray(getattr(frontier, "orientation", None), dtype=float) if getattr(frontier, "orientation", None) is not None else None
            node.region = np.asarray(getattr(frontier, "region", None), dtype=bool) if getattr(frontier, "region", None) is not None else None
            current_semantic_dist = getattr(frontier, "semantic_dist", None)
            if current_semantic_dist is not None:
                node.semantic_dist = current_semantic_dist
            node.visual_embedding = embedding if embedding is not None else node.visual_embedding
            node.hypothesis_node_id = getattr(frontier, "hypothesis_node_id", None)
            node.last_observed_step = step
            node.active_this_step = True
            # A first detection establishes an unknown-confidence frontier
            # (C=0.5). Persistence across extraction steps is the independent
            # positive evidence that raises map confidence. This keeps VERIFY
            # reachable for a newly observed, highly relevant frontier.
            if not is_new:
                confidence_cfg = self._section("confidence")
                match_evidence_scale = float(
                    confidence_cfg.get("frontier_match_evidence_strength", 0.25)
                )
                self._update_evidence(
                    node,
                    True,
                    self.frontier_match_scores.get(index, 0.0) * match_evidence_scale,
                    f"frontier_match:{step}:{node.node_id}",
                    step,
                )
            if self._last_visited_id is not None and self._last_visited_id in self.nodes:
                self._add_edge(TopoEdge(
                    node.node_id,
                    self._last_visited_id,
                    TopoEdgeType.ANCHORED_TO,
                    distance=self._distance_m(
                        node.position, self.nodes[self._last_visited_id].position
                    ),
                    last_updated_step=step,
                ))
            setattr(frontier, "topo_id", node.node_id)
        return dict(self.frontier_source_map)

    def lookup_frontier_semantic_cache(
        self, topo_node_id: str, current_embedding: Optional[np.ndarray],
        current_position: np.ndarray, min_visual_cosine: float = 0.85,
        max_displacement_m: float = 1.0,
    ) -> Tuple[Optional[Any], str]:
        node = self.nodes.get(topo_node_id)
        entry = None if node is None else node.semantic_cache
        if entry is None:
            self.stats["semantic_cache_misses"] += 1
            self.stats["semantic_cache_empty"] += 1
            return None, "empty"
        if entry.prediction_source != "vlm":
            self.stats["semantic_cache_misses"] += 1
            self.stats["semantic_cache_invalid_source"] += 1
            return None, "invalid_source"
        displacement = self._distance_m(entry.reference_position, current_position)
        if displacement > float(max_displacement_m):
            self.stats["semantic_cache_misses"] += 1
            self.stats["semantic_cache_position_miss"] += 1
            return None, "position"
        reference = _unit(entry.reference_embedding)
        current = _unit(current_embedding)
        if reference is None or current is None or reference.shape != current.shape:
            self.stats["semantic_cache_misses"] += 1
            self.stats["semantic_cache_visual_miss"] += 1
            return None, "visual_missing"
        cosine = float(np.dot(reference, current))
        if cosine < float(min_visual_cosine):
            self.stats["semantic_cache_misses"] += 1
            self.stats["semantic_cache_visual_miss"] += 1
            return None, "visual"
        self.stats["semantic_cache_hits"] += 1
        self.stats["semantic_predictor_calls_saved"] += 1
        return copy.deepcopy(entry.semantic_dist), "hit"

    def store_frontier_semantic_cache(
        self, topo_node_id: str, semantic_dist: Any,
        reference_embedding: Optional[np.ndarray], reference_position: np.ndarray,
        prediction_step: int, prediction_source: str,
    ) -> bool:
        node = self.nodes.get(topo_node_id)
        if node is None or semantic_dist is None or prediction_source != "vlm":
            return False
        node.semantic_cache = FrontierSemanticCacheEntry(
            semantic_dist=copy.deepcopy(semantic_dist),
            reference_embedding=None if reference_embedding is None else np.asarray(
                reference_embedding, dtype=float,
            ).copy(),
            reference_position=np.asarray(reference_position, dtype=float)[:2].copy(),
            prediction_step=int(prediction_step),
            prediction_source=str(prediction_source),
        )
        node.semantic_dist = semantic_dist
        self.stats["semantic_cache_stores"] += 1
        return True

    def bind_hypothesis(self, topo_frontier_id: str, hypothesis_node_id: str,
                        dependency_confidence: float) -> None:
        node = self.nodes.get(topo_frontier_id)
        if node is None:
            return
        node.hypothesis_node_id = hypothesis_node_id
        topo_hypothesis_id = f"hypothesis:{hypothesis_node_id}"
        if topo_hypothesis_id not in self.nodes:
            self.nodes[topo_hypothesis_id] = BeliefTopoNode(
                node_id=topo_hypothesis_id,
                node_type=TopoNodeType.HYPOTHESIS,
                position=node.position.copy(),
                semantic_dist=node.semantic_dist,
                visual_embedding=node.visual_embedding,
                hypothesis_node_id=hypothesis_node_id,
                last_observed_step=node.last_observed_step,
            )
            self.stats["nodes_created"] += 1
        self._add_edge(TopoEdge(topo_frontier_id, topo_hypothesis_id,
                                TopoEdgeType.DEPENDS_ON,
                                dependency_confidence=float(dependency_confidence)))

    def _update_evidence(self, node: BeliefTopoNode, positive: bool, strength: float,
                         evidence_ref: str, step: int) -> bool:
        if evidence_ref in self._seen_evidence:
            return False
        self._seen_evidence.add(evidence_ref)
        strength = float(np.clip(strength, 0.0, 1.0))
        conf = self._section("confidence")
        delta = float(conf.get("positive_delta", 1.0) if positive else -conf.get("negative_delta", 1.2))
        node.map_log_odds = float(np.clip(node.map_log_odds + delta * strength, -6.0, 6.0))
        if positive:
            node.positive_evidence += strength
            node.last_positive_step = step
        else:
            node.negative_evidence += strength
            node.negative_evidence_refs.add(evidence_ref)
        if evidence_ref not in node.evidence_refs:
            node.evidence_refs.append(evidence_ref)
        return True

    def record_selection(self, node_id: str, path_cost: Optional[float] = None,
                         step: Optional[int] = None) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        node.visit_count += 1
        if step is not None and node.blocked_until_step >= step:
            self.stats["blocked_target_retries"] += 1
        made_progress = False
        if path_cost is not None:
            if (self.recovery.last_selected_node_id == node_id and
                    self.recovery.last_selected_path_cost is not None and
                    self.recovery.last_selected_path_cost - path_cost >= 0.1):
                made_progress = True
            self.recovery.last_selected_node_id = node_id
            self.recovery.last_selected_path_cost = float(path_cost)
        if made_progress:
            progress_weight = float(
                self._section("accessibility").get("progress_success_weight", 0.25)
            )
            node.nav_success_count += max(0.0, progress_weight)
            self.stats["navigation_progresses"] += 1
            self.recovery.record_progress()
        return made_progress

    def record_navigation_result(self, node_id: str, success: bool, step: int,
                                 reason: Optional[str] = None,
                                 consume: bool = True) -> None:
        del reason
        node = self.nodes.get(node_id)
        if node is None:
            return
        access_cfg = self._section("accessibility")
        if success:
            node.nav_success_count += 1
            node.consecutive_failures = 0
            node.blocked_until_step = -1
            if consume:
                node.consumed = True
            self.stats["navigation_successes"] += 1
            self.recovery.record_progress()
        else:
            node.nav_failure_count += 1
            node.consecutive_failures += 1
            node.blocked_until_step = step + int(access_cfg.get("block_steps", 5))
            self.stats["navigation_failures"] += 1
            self.recovery.escalate()

    def _dependent_descendants(self, node_id: str) -> List[Tuple[str, float]]:
        result, queue, seen = [], [(node_id, 1.0)], set()
        while queue:
            parent, accumulated = queue.pop(0)
            for edge in self.edges.values():
                if edge.edge_type != TopoEdgeType.DEPENDS_ON or edge.source_id != parent:
                    continue
                child = edge.target_id
                if child in seen:
                    continue
                seen.add(child)
                confidence = accumulated * edge.dependency_confidence
                result.append((child, confidence))
                queue.append((child, confidence))
        return result

    def apply_verification(self, node_id: str, evidence: VerificationEvidence, step: int,
                           correction_mode: str = "soft", hypothesis_graph=None) -> List[str]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        strength = float(np.clip(evidence.strength, 0.0, 1.0))
        self._update_evidence(node, evidence.is_positive, strength, evidence.evidence_ref, step)
        if evidence.is_positive:
            self.recovery.record_progress()
            return []
        if correction_mode == "hard":
            self.stats["hard_corrections"] += 1
            if hypothesis_graph is not None and node.hypothesis_node_id in hypothesis_graph.nodes:
                hypothesis_graph.cascade_delete(node.hypothesis_node_id)
            return []
        self.stats["soft_corrections"] += 1
        factor = float(self._section("confidence").get("dependency_propagation", 0.70))
        for child_id, dep_conf in self._dependent_descendants(node_id):
            child = self.nodes.get(child_id)
            if child is None:
                continue
            new_conf = child.map_confidence * (1.0 - factor * dep_conf * strength)
            child.map_log_odds = float(np.clip(_logit(new_conf), -6.0, 6.0))
        conf_cfg = self._section("confidence")
        threshold = float(conf_cfg.get("prune_threshold", 0.10))
        min_negative = int(conf_cfg.get("min_negative_evidence", 2))
        grace = int(conf_cfg.get("positive_grace_steps", 5))
        pruned = []
        for candidate_id in [node_id] + [item[0] for item in self._dependent_descendants(node_id)]:
            candidate = self.nodes.get(candidate_id)
            if candidate is None:
                continue
            if (candidate.map_confidence < threshold and
                    len(candidate.negative_evidence_refs) >= min_negative and
                    step - candidate.last_positive_step >= grace):
                pruned.append(candidate_id)
        for candidate_id in pruned:
            candidate = self._remove_node(candidate_id)
            if (candidate is not None and hypothesis_graph is not None and
                    candidate.hypothesis_node_id in hypothesis_graph.nodes):
                hypothesis_graph._remove_node(candidate.hypothesis_node_id)
        self.stats["pruned_nodes"] += len(pruned)
        return pruned

    @staticmethod
    def _semantic_relevance(node: BeliefTopoNode, category: Optional[str]) -> Optional[float]:
        if not category or node.semantic_dist is None:
            return None
        category = category.strip().lower()
        categories = [str(item).strip().lower() for item in getattr(node.semantic_dist, "categories", [])]
        probabilities = getattr(node.semantic_dist, "probabilities", [])
        return float(probabilities[categories.index(category)]) if category in categories else 0.0

    def _relevance(self, node: BeliefTopoNode, goal: GoalContext) -> float:
        goal_embedding = goal.image_embedding if goal.goal_type == "image" else goal.text_embedding
        relevance_cfg = self._section("relevance")
        clip_score = _clip_relevance(
            node.visual_embedding,
            goal_embedding,
            floor=float(relevance_cfg.get("clip_floor", 0.10)),
            ceiling=float(relevance_cfg.get("clip_ceiling", 0.35)),
        )
        is_category_goal = goal.goal_type == "category"
        semantic = (
            self._semantic_relevance(node, goal.category)
            if is_category_goal else None
        )
        category = (
            str(goal.category or "").strip().lower().replace("_", " ")
            if is_category_goal else ""
        )
        observed_match = None
        if category and node.observed_categories:
            observed = {
                str(item).strip().lower().replace("_", " ")
                for item in node.observed_categories
            }
            observed_match = 1.0 if category in observed else 0.0
        available = [
            value for value in (clip_score, semantic, observed_match)
            if value is not None
        ]
        return float(max(available)) if available else 0.5

    def node_relevance(self, node_id: str, goal: Optional[GoalContext] = None) -> float:
        node = self.nodes.get(node_id)
        active_goal = goal or self.goal
        if node is None or active_goal is None:
            return 0.5
        return self._relevance(node, active_goal)

    def build_active_view(
        self, goal: GoalContext, frontiers: list, path_costs: Dict[int, float], step: int,
        recovery_level: Optional[int] = None, unreachable_indices: Optional[Set[int]] = None,
        use_goal_relevance: bool = True,
    ) -> ActiveTopoView:
        unreachable_indices = unreachable_indices or set()
        recovery_level = self.recovery.level if recovery_level is None else int(recovery_level)
        indices = list(range(len(frontiers)))
        finite_costs = [path_costs[i] for i in indices if i in path_costs and np.isfinite(path_costs[i])]
        min_cost = min(finite_costs) if finite_costs else 0.0
        max_cost = max(finite_costs) if finite_costs else min_cost
        areas = [float(np.sum(getattr(frontiers[i], "region", 0))) for i in indices]
        max_area = max(areas) if areas else 0.0
        all_candidates = []
        access_cfg = self._section("accessibility")
        view_cfg = self._section("active_view")
        utility_cfg = self._section("utility")
        suppress_after = int(access_cfg.get("suppress_after_failures", 3))
        for index, frontier in enumerate(frontiers):
            node_id = self.frontier_source_map.get(index)
            node = self.nodes.get(node_id) if node_id else None
            if node is None:
                continue
            # In the C/A ablation all candidates receive the same neutral R.
            # The constant remains in the frozen utility formula but cannot
            # affect relative ordering, and it is omitted from the VLM prompt.
            relevance = self._relevance(node, goal) if use_goal_relevance else 0.5
            confidence = node.map_confidence
            accessibility = node.accessibility
            raw_path = float(path_costs.get(index, np.inf))
            norm_path = ((raw_path - min_cost) / (max_cost - min_cost)
                         if np.isfinite(raw_path) and max_cost > min_cost else 0.0)
            revisit = min(node.visit_count / 3.0, 1.0)
            information = areas[index] / max_area if max_area > 0 else 0.0
            blocked = node.blocked_until_step >= step
            suppressed = (node.consumed or node.consecutive_failures >= suppress_after or
                          index in unreachable_indices)
            if suppressed or blocked:
                state = ActiveTopoState.SUPPRESSED
            elif (use_goal_relevance and
                  relevance >= float(view_cfg.get("verify_relevance", 0.55)) and
                  confidence < float(view_cfg.get("verify_confidence", 0.55))):
                state = ActiveTopoState.VERIFY
            elif (use_goal_relevance and
                  relevance < float(view_cfg.get("fold_relevance", 0.20))):
                state = ActiveTopoState.FOLDED
            else:
                state = ActiveTopoState.ACTIVE
            utility = (
                float(utility_cfg.get("relevance", 1.0)) * relevance
                + float(utility_cfg.get("confidence", 0.4)) * confidence
                + float(utility_cfg.get("accessibility", 0.4)) * accessibility
                - float(utility_cfg.get("path_cost", 0.1)) * norm_path
                - float(utility_cfg.get("revisit", 0.2)) * revisit
                + float(utility_cfg.get("information_gain", 0.05)) * information
            )
            all_candidates.append(ActiveTopoCandidate(
                topo_node_id=node_id, source_frontier_index=index,
                relevance=relevance, confidence=confidence, accessibility=accessibility,
                path_cost=norm_path, revisit_penalty=revisit,
                information_gain=information, utility=float(utility), state=state,
                raw_path_cost=raw_path, match_score=self.frontier_match_scores.get(index),
            ))
        if recovery_level == 0:
            allowed = {ActiveTopoState.ACTIVE, ActiveTopoState.VERIFY}
            candidates = [item for item in all_candidates if item.state in allowed]
        elif recovery_level == 1:
            candidates = [item for item in all_candidates if item.state != ActiveTopoState.SUPPRESSED and
                          (item.state != ActiveTopoState.FOLDED or
                           (item.confidence >= 0.55 and item.accessibility >= 0.50))]
        else:
            candidates = [item for item in all_candidates if item.state != ActiveTopoState.SUPPRESSED]
        candidates.sort(key=lambda item: item.utility, reverse=True)
        if not candidates and recovery_level < 2:
            if self.recovery.escalate():
                self.stats["recovery_escalations"][self.recovery.level] += 1
                return self.build_active_view(goal, frontiers, path_costs, step,
                                              self.recovery.level, unreachable_indices,
                                              use_goal_relevance)
        self.stats["suppressed_nodes"] += sum(item.state == ActiveTopoState.SUPPRESSED for item in all_candidates)
        view = ActiveTopoView(
            goal, recovery_level, candidates, step, all_candidates,
            uses_goal_relevance=use_goal_relevance,
        )
        self.current_view = view
        return view

    def build_simple_memory_view(
        self, goal: GoalContext, frontiers: list, step: int,
        unreachable_indices: Optional[Set[int]] = None,
    ) -> ActiveTopoView:
        """Project only stable identity and failure memory into HGR.

        Healthy frontiers retain their source order, so enabling this mode
        cannot silently replace HGR's semantic/VLM ranking.  A recently failed
        frontier is hidden for five steps and a frontier with three consecutive
        failures remains suppressed.  If every current frontier is hidden, the
        least-failed reachable one is exposed as a deterministic recovery so
        the agent is never left without an executable target.
        """

        unreachable_indices = unreachable_indices or set()
        healthy: List[ActiveTopoCandidate] = []
        recoverable: List[ActiveTopoCandidate] = []
        all_candidates: List[ActiveTopoCandidate] = []
        for index, _frontier in enumerate(frontiers):
            node_id = self.frontier_source_map.get(index)
            node = self.nodes.get(node_id) if node_id else None
            if node is None:
                continue
            failure_count = float(node.consecutive_failures)
            hidden = (
                node.consumed
                or node.blocked_until_step >= step
                or node.consecutive_failures >= 3
                or index in unreachable_indices
            )
            candidate = ActiveTopoCandidate(
                topo_node_id=node_id,
                source_frontier_index=index,
                relevance=0.5,
                confidence=0.5,
                accessibility=node.accessibility,
                path_cost=0.0,
                revisit_penalty=failure_count,
                information_gain=0.0,
                # Not used for healthy-candidate ordering or exposed to the
                # prompt. Keep only a coefficient-free diagnostic value.
                utility=-failure_count,
                state=(ActiveTopoState.SUPPRESSED if hidden else ActiveTopoState.ACTIVE),
                match_score=self.frontier_match_scores.get(index),
            )
            all_candidates.append(candidate)
            if not hidden:
                healthy.append(candidate)
            elif not node.consumed and index not in unreachable_indices:
                recoverable.append(candidate)

        candidates = healthy
        recovery_level = 0
        if not candidates and recoverable:
            recovery_level = 1
            # Prefer the frontier with the fewest failures and the best
            # reachability. Source index is the stable final tie-break.
            chosen = min(
                recoverable,
                key=lambda item: (
                    self.nodes[item.topo_node_id].consecutive_failures,
                    -item.accessibility,
                    item.source_frontier_index,
                ),
            )
            chosen = copy.copy(chosen)
            chosen.state = ActiveTopoState.ACTIVE
            candidates = [chosen]
            self.stats["recovery_escalations"][1] += 1

        self.stats["suppressed_nodes"] += sum(
            item.state == ActiveTopoState.SUPPRESSED for item in all_candidates
        )
        view = ActiveTopoView(
            goal=goal,
            recovery_level=recovery_level,
            candidates=candidates,
            created_at_step=step,
            all_candidates=all_candidates,
            uses_goal_relevance=False,
            include_prompt_metadata=False,
        )
        self.current_view = view
        return view

    def _shortest_topology_path(self, source_id: str, target_id: str) -> List[str]:
        """Return an undirected BFS path through the persistent fact graph."""
        if source_id == target_id:
            return [source_id]
        adjacency: Dict[str, Set[str]] = {}
        for edge in self.edges.values():
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                continue
            adjacency.setdefault(edge.source_id, set()).add(edge.target_id)
            adjacency.setdefault(edge.target_id, set()).add(edge.source_id)
        queue = [source_id]
        parent = {source_id: None}
        for current in queue:
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == target_id:
                    path = [target_id]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                queue.append(neighbor)
        return []

    def observed_node_for_snapshot(self, snapshot_image: str) -> Optional[str]:
        evidence_ref = f"snapshot:{snapshot_image}"
        matches = [
            node for node in self.nodes.values()
            if node.node_type == TopoNodeType.OBSERVED
            and evidence_ref in node.evidence_refs
        ]
        if not matches:
            return None
        return max(matches, key=lambda node: node.last_observed_step).node_id

    def capture_anchor_for_snapshot(self, snapshot_image: str) -> Optional[str]:
        """Return the immutable VISITED anchor captured for one snapshot."""
        anchor_id = self.snapshot_anchor_map.get(str(snapshot_image))
        node = self.nodes.get(anchor_id)
        if node is None or node.node_type != TopoNodeType.VISITED:
            return None
        return anchor_id

    def route_length_m(self, route: List[str]) -> float:
        """Return the geometric length of a topology route in metres."""
        if len(route) < 2:
            return 0.0
        if any(node_id not in self.nodes for node_id in route):
            return float("nan")
        return float(sum(
            self._distance_m(
                self.nodes[source_id].position,
                self.nodes[target_id].position,
            )
            for source_id, target_id in zip(route[:-1], route[1:])
        ))

    def plan_safe_route_to_anchor(
        self, anchor_node_id: str, record_stats: bool = False,
    ) -> Tuple[List[str], Optional[str]]:
        """Plan a VISITED-only route to one fixed snapshot capture anchor."""
        current_id = self._last_visited_id
        anchor = self.nodes.get(anchor_node_id)
        if (
            current_id not in self.nodes
            or anchor is None
            or anchor.node_type != TopoNodeType.VISITED
        ):
            return [], None
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for edge in self.edges.values():
            source = self.nodes.get(edge.source_id)
            target = self.nodes.get(edge.target_id)
            if (
                edge.edge_type != TopoEdgeType.NAVIGABLE
                or source is None
                or target is None
                or source.node_type != TopoNodeType.VISITED
                or target.node_type != TopoNodeType.VISITED
            ):
                continue
            weight = (
                float(edge.distance)
                if edge.distance is not None
                else self._distance_m(source.position, target.position)
            )
            adjacency.setdefault(edge.source_id, []).append(
                (edge.target_id, weight)
            )
            adjacency.setdefault(edge.target_id, []).append(
                (edge.source_id, weight)
            )
        distances = {current_id: 0.0}
        parent = {current_id: None}
        queue = [(0.0, current_id)]
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances.get(node_id):
                continue
            if node_id == anchor_node_id:
                break
            for neighbor, weight in adjacency.get(node_id, []):
                candidate = distance + max(weight, 1e-6)
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    parent[neighbor] = node_id
                    heapq.heappush(queue, (candidate, neighbor))
        if anchor_node_id not in parent:
            return [], None
        route = [anchor_node_id]
        while parent[route[-1]] is not None:
            route.append(parent[route[-1]])
        route.reverse()
        next_waypoint_id = next((
            node_id for node_id in route[1:]
            if self._distance_m(
                self.nodes[current_id].position,
                self.nodes[node_id].position,
            ) > 0.1
        ), None)
        if record_stats:
            self.stats.setdefault("topology_route_plans", 0)
            self.stats.setdefault("topology_route_waypoints", 0)
            self.stats["topology_route_plans"] += 1
            if next_waypoint_id is not None:
                self.stats["topology_route_waypoints"] += 1
        return route, next_waypoint_id

    def plan_safe_route_to_observed(
        self, observed_node_id: str,
    ) -> Tuple[List[str], Optional[str]]:
        """Plan through actually visited poses to one historical observation.

        Only NAVIGABLE VISITED-to-VISITED edges are traversed. OBSERVED_AT is
        used solely to identify safe terminal anchors. Historical frontier or
        hypothesis coordinates can therefore never become route waypoints.
        """
        current_id = self._last_visited_id
        target = self.nodes.get(observed_node_id)
        if (
            current_id not in self.nodes
            or target is None
            or target.node_type != TopoNodeType.OBSERVED
        ):
            return [], None
        anchors: Set[str] = set()
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
        for edge in self.edges.values():
            source = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)
            if source is None or target_node is None:
                continue
            if edge.edge_type == TopoEdgeType.OBSERVED_AT:
                if edge.source_id == observed_node_id and target_node.node_type == TopoNodeType.VISITED:
                    anchors.add(edge.target_id)
                elif edge.target_id == observed_node_id and source.node_type == TopoNodeType.VISITED:
                    anchors.add(edge.source_id)
            elif (
                edge.edge_type == TopoEdgeType.NAVIGABLE
                and source.node_type == TopoNodeType.VISITED
                and target_node.node_type == TopoNodeType.VISITED
            ):
                weight = float(edge.distance) if edge.distance is not None else self._distance_m(
                    source.position, target_node.position
                )
                adjacency.setdefault(edge.source_id, []).append((edge.target_id, weight))
                adjacency.setdefault(edge.target_id, []).append((edge.source_id, weight))
        if not anchors:
            return [], None

        distances = {current_id: 0.0}
        parent = {current_id: None}
        queue = [(0.0, current_id)]
        reached = None
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance != distances.get(node_id):
                continue
            if node_id in anchors:
                reached = node_id
                break
            for neighbor, weight in adjacency.get(node_id, []):
                candidate = distance + max(weight, 1e-6)
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    parent[neighbor] = node_id
                    heapq.heappush(queue, (candidate, neighbor))
        if reached is None:
            return [], None
        route = [reached]
        while parent[route[-1]] is not None:
            route.append(parent[route[-1]])
        route.reverse()
        route.append(observed_node_id)
        next_waypoint_id = next(
            (
                node_id for node_id in route[1:-1]
                if self.nodes[node_id].node_type == TopoNodeType.VISITED
                and self._distance_m(
                    self.nodes[current_id].position,
                    self.nodes[node_id].position,
                ) > 0.1
            ),
            None,
        )
        self.stats.setdefault("topology_route_plans", 0)
        self.stats.setdefault("topology_route_waypoints", 0)
        self.stats["topology_route_plans"] += 1
        if next_waypoint_id is not None:
            self.stats["topology_route_waypoints"] += 1
        return route, next_waypoint_id

    def build_goal_topology_view(
        self, goal: GoalContext, frontiers: list, step: int,
    ) -> ActiveTopoView:
        """Create a connected, goal-conditioned view of the persistent map.

        The global graph remains untouched. Historical OBSERVED nodes provide
        goal evidence, VISITED nodes provide connectors, and only frontiers
        with a current raw-source mapping become executable candidates.
        """
        current_id = self._last_visited_id
        observed = [
            node for node in self.nodes.values()
            if node.node_type == TopoNodeType.OBSERVED
        ]
        observed_ranked = sorted(
            ((self._relevance(node, goal), node) for node in observed),
            key=lambda item: (-item[0], item[1].node_id),
        )
        # Direct category matches are always evidence. For description/image,
        # or before an exact category is observed, retain the strongest three
        # observations as the task projection's evidence anchors.
        normalized_category = (
            str(goal.category or "").strip().lower().replace("_", " ")
            if goal.goal_type == "category" else ""
        )
        direct = [
            (score, node) for score, node in observed_ranked
            if normalized_category and normalized_category in {
                str(item).strip().lower().replace("_", " ")
                for item in node.observed_categories
            }
        ]
        evidence_nodes = [item[1] for item in (direct or observed_ranked[:3])]

        included: Set[str] = set()
        connectors: Set[str] = set()
        if current_id in self.nodes:
            included.add(current_id)
        for node in evidence_nodes:
            included.add(node.node_id)
            if current_id in self.nodes:
                path = self._shortest_topology_path(current_id, node.node_id)
                included.update(path)
                connectors.update(path[1:-1])

        candidates: List[ActiveTopoCandidate] = []
        frontier_hops: Dict[str, Optional[int]] = {}
        for index, _frontier in enumerate(frontiers):
            node_id = self.frontier_source_map.get(index)
            node = self.nodes.get(node_id) if node_id else None
            if node is None:
                continue
            relevance = self._relevance(node, goal)
            included.add(node_id)
            if current_id in self.nodes:
                path = self._shortest_topology_path(current_id, node_id)
                included.update(path)
                connectors.update(path[1:-1])
            hop_candidates = []
            for evidence_node in evidence_nodes:
                evidence_path = self._shortest_topology_path(
                    node_id, evidence_node.node_id
                )
                if evidence_path:
                    hop_candidates.append(len(evidence_path) - 1)
            topo_hops = min(hop_candidates) if hop_candidates else None
            frontier_hops[node_id] = topo_hops
            candidates.append(ActiveTopoCandidate(
                topo_node_id=node_id,
                source_frontier_index=index,
                relevance=relevance,
                confidence=0.5,
                accessibility=node.accessibility,
                path_cost=float(topo_hops) if topo_hops is not None else float("inf"),
                revisit_penalty=float(node.visit_count),
                information_gain=0.0,
                utility=relevance,
                state=ActiveTopoState.ACTIVE,
                match_score=self.frontier_match_scores.get(index),
            ))
        # Goal evidence is primary; graph-hop proximity is a coefficient-free
        # topology tie-break; source index is the stable final fallback.
        candidates.sort(key=lambda item: (
            -item.relevance,
            item.path_cost,
            item.source_frontier_index,
        ))
        evidence_ids = [node.node_id for node in evidence_nodes]
        related_categories = sorted({
            category
            for node in evidence_nodes
            for category in node.observed_categories
        })
        goal_label = (
            goal.category if goal.goal_type == "category"
            else goal.description if goal.goal_type == "description"
            else "image target"
        )
        context = {
            "goal_type": goal.goal_type,
            "goal": goal_label,
            "global_nodes": len(self.nodes),
            "global_edges": len(self.edges),
            "active_nodes": len(included),
            "evidence_nodes": evidence_ids,
            "connector_nodes": sorted(connectors),
            "related_observations": related_categories[:12],
            "frontier_order": [item.topo_node_id for item in candidates],
            "frontier_hops_to_evidence": frontier_hops,
        }
        self.stats.setdefault("goal_topology_views", 0)
        self.stats.setdefault("goal_topology_active_nodes", 0)
        self.stats["goal_topology_views"] += 1
        self.stats["goal_topology_active_nodes"] += len(included)
        view = ActiveTopoView(
            goal=goal,
            recovery_level=0,
            candidates=candidates,
            created_at_step=step,
            all_candidates=list(candidates),
            uses_goal_relevance=True,
            include_prompt_metadata=False,
            included_node_ids=sorted(included),
            connector_node_ids=sorted(connectors),
            evidence_node_ids=evidence_ids,
            topology_context=context,
        )
        self.current_view = view
        return view

    def build_goal_task_view(
        self,
        goal: GoalContext,
        frontiers: list,
        snapshot_images: Iterable[str],
        step: int,
        direct_snapshot: Optional[Tuple[str, str, Any]] = None,
        current_snapshot_images: Optional[Iterable[str]] = None,
        record_state: bool = True,
    ) -> GoalTopoTaskView:
        """Project every executable HGR target into one task-level view.

        ``direct_snapshot`` is ``(topo_node_id, image, object_id)`` for an
        exact-category object observed during the current RGB sweep. Historical
        snapshots become REVISIT actions and current raw frontiers become
        EXPLORE actions. The persistent graph remains the source of identity;
        this view only carries executable source mappings for the current step.
        """
        actions: List[GoalTopoAction] = []
        direct_image = None
        if direct_snapshot is not None:
            direct_node_id, direct_image, direct_object_id = direct_snapshot
            if direct_node_id in self.nodes:
                actions.append(GoalTopoAction(
                    action_type=GoalTopoActionType.DIRECT,
                    topo_node_id=direct_node_id,
                    source_kind="snapshot",
                    snapshot_image=str(direct_image),
                    object_id=direct_object_id,
                    relevance=self.node_relevance(direct_node_id, goal),
                    current_observation=True,
                ))

        current_images = {str(item) for item in (current_snapshot_images or [])}
        seen_images = set()
        for snapshot_image in snapshot_images:
            snapshot_image = str(snapshot_image)
            if snapshot_image in seen_images or snapshot_image == direct_image:
                continue
            seen_images.add(snapshot_image)
            node_id = self.observed_node_for_snapshot(snapshot_image)
            if node_id is None:
                continue
            is_current = snapshot_image in current_images
            actions.append(GoalTopoAction(
                action_type=(
                    GoalTopoActionType.DIRECT
                    if is_current else GoalTopoActionType.REVISIT
                ),
                topo_node_id=node_id,
                source_kind="snapshot",
                snapshot_image=snapshot_image,
                relevance=self.node_relevance(node_id, goal),
                current_observation=is_current,
            ))

        for frontier_index, _frontier in enumerate(frontiers):
            node_id = self.frontier_source_map.get(frontier_index)
            if node_id not in self.nodes:
                continue
            actions.append(GoalTopoAction(
                action_type=GoalTopoActionType.EXPLORE,
                topo_node_id=node_id,
                source_kind="frontier",
                source_frontier_index=frontier_index,
                relevance=self.node_relevance(node_id, goal),
            ))

        # DIRECT is the deterministic shortest semantic action. The remaining
        # order mirrors the goal projection: relevance first, stable identity
        # second. VLM selection can still choose among the projected sources.
        order = {
            GoalTopoActionType.DIRECT: 0,
            GoalTopoActionType.REVISIT: 1,
            GoalTopoActionType.EXPLORE: 2,
        }
        actions.sort(key=lambda action: (
            order[action.action_type], -action.relevance,
            action.topo_node_id, action.snapshot_image or "",
            action.source_frontier_index
            if action.source_frontier_index is not None else -1,
        ))
        view = GoalTopoTaskView(goal, int(step), actions)
        if record_state:
            self.current_task_view = view
            self.stats.setdefault("goal_task_views", 0)
            self.stats.setdefault("goal_task_actions", 0)
            self.stats["goal_task_views"] += 1
            self.stats["goal_task_actions"] += len(actions)
        return view

    def get_statistics(self) -> Dict[str, Any]:
        resolved_frontiers = self.stats["frontier_matches"] + self.stats["frontier_new"]
        current = {kind.value: 0 for kind in TopoNodeType}
        for node in self.nodes.values():
            current[node.node_type.value] += 1
        return {
            **self.stats,
            "current_nodes": current,
            "current_edges": len(self.edges),
            "stable_id_match_rate": self.stats["frontier_matches"] / max(1, resolved_frontiers),
            "recovery_level": self.recovery.level,
        }
