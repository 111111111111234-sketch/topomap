"""Goal-conditioned second topology layer for HGR V7.

The persistent topology records episode-scoped scene facts.  This module
projects those facts into a short-lived graph for the current GOAT subtask:
executable DIRECT/REVISIT/EXPLORE nodes carry Bayesian goal evidence, while
typed edges retain navigation/dependency context.  It deliberately does not
plan or execute motion; active mode only updates the relevance consumed by the
already validated C-strict revisit gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import GoalTopoActionType, TopoEdgeType


_EPS = 1e-3


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _clip01(value: float) -> float:
    finite = _finite(value)
    return 0.5 if finite is None else float(np.clip(finite, 0.0, 1.0))


@dataclass
class GoalEvidenceState:
    """Independent evidence accumulated for one goal/node pair."""

    prior_relevance: float
    alpha: float
    beta: float
    positive_views: int = 0
    negative_views: int = 0
    last_evidence_step: int = -1
    evidence_positions: List[Tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_relevance(cls, relevance: float, prior_strength: float):
        relevance = _clip01(relevance)
        strength = max(float(prior_strength), 2.0 * _EPS)
        return cls(
            prior_relevance=relevance,
            alpha=max(_EPS, strength * relevance),
            beta=max(_EPS, strength * (1.0 - relevance)),
        )

    @property
    def evidence_views(self) -> int:
        return int(self.positive_views + self.negative_views)

    @property
    def posterior_mean(self) -> float:
        return _clip01(self.alpha / max(_EPS, self.alpha + self.beta))

    @property
    def posterior_uncertainty(self) -> float:
        total = max(2.0 * _EPS, self.alpha + self.beta)
        variance = self.alpha * self.beta / (
            total * total * (total + 1.0)
        )
        # Twice the standard deviation maps the widest Beta posterior close
        # to one while remaining finite for highly skewed distributions.
        return _clip01(2.0 * math.sqrt(max(0.0, variance)))

    def to_trace_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "evidence_views": self.evidence_views,
            "posterior_mean": self.posterior_mean,
            "posterior_uncertainty": self.posterior_uncertainty,
        })
        return payload


@dataclass(frozen=True)
class DynamicGoalEdge:
    source_id: str
    target_id: str
    edge_type: str
    metric_length_m: Optional[float]
    path_cost_m: Optional[float]
    traversability: Optional[float]
    reliability: float
    freshness: float
    observation_direction_deg: Optional[float]
    status: str
    description: str

    def to_trace_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicGoalNode:
    topo_node_id: str
    persistent_node_type: str
    executable: bool
    action_type: Optional[str]
    raw_relevance: float
    confidence: float
    uncertainty: float
    freshness: float
    propagated_support: Optional[float]
    dynamic_relevance: float
    positive_views: int
    negative_views: int
    reachable: Optional[bool]
    effective_cost_m: Optional[float]
    budget_ratio: Optional[float]
    route_mode: Optional[str]

    def to_trace_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicGoalTopologyView:
    goal_type: str
    subtask_id: str
    step: int
    shadow_only: bool
    nodes: List[DynamicGoalNode]
    edges: List[DynamicGoalEdge]
    changed_actions: int

    def to_trace_dict(self) -> Dict[str, Any]:
        return {
            "goal_type": self.goal_type,
            "subtask_id": self.subtask_id,
            "step": self.step,
            "shadow_only": self.shadow_only,
            "behavior_applied": not self.shadow_only,
            "changed_actions": self.changed_actions,
            "executable_nodes": sum(node.executable for node in self.nodes),
            "context_nodes": sum(not node.executable for node in self.nodes),
            "nodes": [node.to_trace_dict() for node in self.nodes],
            "edges": [edge.to_trace_dict() for edge in self.edges],
        }


class DynamicGoalTopology:
    """Build a dynamic goal graph over one persistent scene topology."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = dict(cfg or {})
        self.shadow_only = bool(self.cfg.get("shadow_only", True))
        self.prior_strength = max(0.1, float(self.cfg.get("prior_strength", 4.0)))
        self.positive_relevance = _clip01(
            self.cfg.get("positive_relevance", 0.60)
        )
        self.coverage_radius_m = max(
            0.0, float(self.cfg.get("coverage_radius_m", 1.0))
        )
        self.negative_evidence_weight = max(
            0.0, float(self.cfg.get("negative_evidence_weight", 0.5))
        )
        self.min_negative_views = max(
            1, int(self.cfg.get("min_negative_views", 3))
        )
        self.min_step_gap = max(
            0, int(self.cfg.get("min_independent_step_gap", 2))
        )
        self.min_distance_voxels = max(
            0.0,
            float(self.cfg.get("min_independent_distance_voxels", 5.0)),
        )
        self.half_life_steps = max(
            1.0, float(self.cfg.get("freshness_half_life_steps", 20.0))
        )
        self.one_hop_discount = _clip01(
            self.cfg.get("one_hop_discount", 0.25)
        )
        allowed = self.cfg.get("propagation_edge_types", [
            "observed_at", "anchored_to", "depends_on",
            "same_capture_anchor",
        ])
        self.propagation_edge_types = {str(item) for item in allowed}
        self.subtask_id: Optional[str] = None
        self.states: Dict[str, GoalEvidenceState] = {}
        self.stats = {
            "views": 0,
            "active_views": 0,
            "changed_actions": 0,
            "positive_updates": 0,
            "negative_updates": 0,
            "deduplicated_updates": 0,
            "propagated_actions": 0,
        }

    def begin_goal(self, subtask_id: str) -> None:
        subtask_id = str(subtask_id)
        if self.subtask_id == subtask_id:
            return
        self.subtask_id = subtask_id
        self.states = {}

    def _state(self, action: MetricAction) -> GoalEvidenceState:
        state = self.states.get(str(action.topo_node_id))
        if state is None:
            state = GoalEvidenceState.from_relevance(
                action.relevance, self.prior_strength
            )
            self.states[str(action.topo_node_id)] = state
        return state

    def _is_independent(
        self, state: GoalEvidenceState, position: np.ndarray, step: int,
    ) -> bool:
        if (
            state.last_evidence_step >= 0
            and int(step) - int(state.last_evidence_step) < self.min_step_gap
        ):
            return False
        if not state.evidence_positions:
            return True
        return all(
            float(np.linalg.norm(position - np.asarray(previous, dtype=float)))
            >= self.min_distance_voxels
            for previous in state.evidence_positions
        )

    def _record(
        self,
        state: GoalEvidenceState,
        measurement: float,
        weight: float,
        position: np.ndarray,
        step: int,
        evidence_kind: str,
    ) -> bool:
        if not self._is_independent(state, position, step):
            self.stats["deduplicated_updates"] += 1
            return False
        measurement = _clip01(measurement)
        weight = max(0.0, float(weight))
        state.alpha += weight * measurement
        state.beta += weight * (1.0 - measurement)
        if evidence_kind == "positive":
            state.positive_views += 1
            self.stats["positive_updates"] += 1
        elif evidence_kind == "negative":
            state.negative_views += 1
            self.stats["negative_updates"] += 1
        else:
            raise ValueError(f"unknown goal evidence kind: {evidence_kind}")
        state.last_evidence_step = int(step)
        state.evidence_positions.append(tuple(map(float, position[:2])))
        return True

    def _freshness(self, last_step: int, step: int) -> float:
        if last_step < 0:
            return 0.0
        age = max(0, int(step) - int(last_step))
        return float(2.0 ** (-float(age) / self.half_life_steps))

    @staticmethod
    def _persistent_type(topology, node_id: str) -> str:
        node = topology.nodes.get(node_id)
        node_type = None if node is None else getattr(node, "node_type", None)
        return str(getattr(node_type, "value", node_type or "unknown"))

    def _edge_freshness(self, last_step: int, step: int) -> float:
        return self._freshness(last_step, step) if last_step >= 0 else 0.0

    @staticmethod
    def _edge_direction(topology, source_id: str, target_id: str):
        source = topology.nodes.get(source_id)
        target = topology.nodes.get(target_id)
        if source is None or target is None:
            return None
        delta = np.asarray(target.position, dtype=float)[:2] - np.asarray(
            source.position, dtype=float
        )[:2]
        if float(np.linalg.norm(delta)) <= _EPS:
            return 0.0
        return float((math.degrees(math.atan2(delta[1], delta[0])) + 360.0) % 360.0)

    @staticmethod
    def _edge_reliability(edge) -> float:
        edge_type = str(getattr(getattr(edge, "edge_type", None), "value", "unknown"))
        base = _finite(getattr(edge, "traversability", None))
        if base is None:
            base = _finite(getattr(edge, "dependency_confidence", None))
        if base is None:
            base = 1.0 if edge_type in ("observed_at", "anchored_to") else 0.5
        visits = max(0.0, float(getattr(edge, "visit_count", 0)))
        failures = max(0.0, float(getattr(edge, "failure_count", 0)))
        history = (visits + 1.0) / (visits + failures + 2.0)
        return _clip01(float(base) * (0.5 + 0.5 * history))

    def _describe_edge(
        self, topology, edge, step: int, edge_type_override: Optional[str] = None,
    ) -> DynamicGoalEdge:
        edge_type = edge_type_override or str(
            getattr(getattr(edge, "edge_type", None), "value", "unknown")
        )
        distance = _finite(getattr(edge, "distance", None))
        path_cost = _finite(getattr(edge, "path_cost", None))
        traversability = _finite(getattr(edge, "traversability", None))
        reliability = self._edge_reliability(edge)
        freshness = self._edge_freshness(
            int(getattr(edge, "last_updated_step", -1)), step
        )
        if reliability >= 0.67:
            status = "reliable"
        elif reliability >= 0.34:
            status = "uncertain"
        else:
            status = "degraded"
        length_text = "unknown"
        if distance is not None:
            length_text = f"{distance:.2f}m"
        description = (
            f"{edge_type}: length={length_text}, status={status}, "
            f"reliability={reliability:.2f}, freshness={freshness:.2f}"
        )
        return DynamicGoalEdge(
            source_id=str(edge.source_id),
            target_id=str(edge.target_id),
            edge_type=edge_type,
            metric_length_m=distance,
            path_cost_m=path_cost,
            traversability=traversability,
            reliability=reliability,
            freshness=freshness,
            observation_direction_deg=self._edge_direction(
                topology, str(edge.source_id), str(edge.target_id)
            ),
            status=status,
            description=description,
        )

    def _project_edges(
        self,
        metric_view: MetricActionView,
        topology,
        step: int,
    ) -> Tuple[List[DynamicGoalEdge], List[str]]:
        action_ids = {str(action.topo_node_id) for action in metric_view.actions}
        included_ids = set(action_ids)
        raw_edges = []
        for edge in topology.edges.values():
            source_id, target_id = str(edge.source_id), str(edge.target_id)
            if source_id in action_ids or target_id in action_ids:
                raw_edges.append(edge)
                included_ids.update((source_id, target_id))
        projected = [
            self._describe_edge(topology, edge, step) for edge in raw_edges
        ]

        # Two observations made at one immutable capture anchor are explicitly
        # related in the goal graph, without modifying the persistent facts.
        by_anchor: Dict[str, List[str]] = {}
        for action in metric_view.actions:
            if action.capture_anchor_id:
                by_anchor.setdefault(str(action.capture_anchor_id), []).append(
                    str(action.topo_node_id)
                )
        existing = {
            (edge.source_id, edge.target_id, edge.edge_type) for edge in projected
        }
        for anchor_id, node_ids in by_anchor.items():
            unique_ids = sorted(set(node_ids))
            for index, source_id in enumerate(unique_ids):
                for target_id in unique_ids[index + 1:]:
                    key = (source_id, target_id, "same_capture_anchor")
                    if key in existing:
                        continue
                    pseudo = type("DynamicEdge", (), {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": None,
                        "distance": 0.0,
                        "path_cost": 0.0,
                        "traversability": None,
                        "dependency_confidence": 0.5,
                        "visit_count": 0,
                        "failure_count": 0,
                        "last_updated_step": step,
                    })()
                    projected.append(self._describe_edge(
                        topology, pseudo, step,
                        edge_type_override="same_capture_anchor",
                    ))
        return projected, sorted(included_ids)

    def _base_estimate(
        self, action: MetricAction, step: int,
    ) -> Tuple[float, float, float, GoalEvidenceState]:
        state = self._state(action)
        freshness = self._freshness(state.last_evidence_step, step)
        posterior_uncertainty = state.posterior_uncertainty
        # Stale or uncertain evidence falls back continuously to the current
        # semantic relevance instead of creating a permanent hotspot.
        trust = freshness * (1.0 - posterior_uncertainty)
        estimate = float(
            action.relevance
            + trust * (state.posterior_mean - float(action.relevance))
        )
        uncertainty = _clip01(1.0 - trust)
        return _clip01(estimate), uncertainty, freshness, state

    def update_and_project(
        self,
        metric_view: MetricActionView,
        topology,
        current_voxel: Iterable[float],
        step: int,
    ) -> Tuple[MetricActionView, DynamicGoalTopologyView]:
        """Update evidence and return a shadow or active metric projection."""

        self.begin_goal(metric_view.subtask_id)
        position = np.asarray(current_voxel, dtype=float).reshape(-1)[:2]
        direct_actions = [
            action for action in metric_view.actions
            if action.action_type == GoalTopoActionType.DIRECT.value
            and action.current_observation
        ]
        def is_exact_object(action: MetricAction) -> bool:
            return (
                str(metric_view.goal_type) in ("category", "object")
                and action.object_id is not None
            )

        strong_direct = any(
            is_exact_object(action)
            or float(action.relevance) >= self.positive_relevance
            for action in direct_actions
        )
        for action in direct_actions:
            if not (
                is_exact_object(action)
                or float(action.relevance) >= self.positive_relevance
            ):
                continue
            measurement = 1.0 if is_exact_object(action) else action.relevance
            self._record(
                self._state(action), measurement, 1.0, position, step,
                evidence_kind="positive",
            )

        # Absence is evidence only after the agent has reached the immutable
        # snapshot anchor and no strong current observation supports the goal.
        if not strong_direct and self.negative_evidence_weight > 0.0:
            for action in metric_view.actions:
                if (
                    action.action_type == GoalTopoActionType.REVISIT.value
                    and action.effective_cost_m is not None
                    and float(action.effective_cost_m) <= self.coverage_radius_m
                ):
                    state = self._state(action)
                    # The first independent absences establish coverage only.
                    # A weak posterior update starts at the predeclared third
                    # view, preventing one missed frame from suppressing a
                    # historical goal hypothesis.
                    posterior_weight = (
                        self.negative_evidence_weight
                        if state.negative_views + 1 >= self.min_negative_views
                        else 0.0
                    )
                    self._record(
                        state,
                        0.0,
                        posterior_weight,
                        position,
                        step,
                        evidence_kind="negative",
                    )

        edges, included_ids = self._project_edges(metric_view, topology, step)
        action_by_id = {
            str(action.topo_node_id): action for action in metric_view.actions
        }
        base = {
            node_id: self._base_estimate(action, step)
            for node_id, action in action_by_id.items()
        }
        adjacency: Dict[str, List[Tuple[str, DynamicGoalEdge]]] = {}
        for edge in edges:
            if edge.edge_type not in self.propagation_edge_types:
                continue
            adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge))
            adjacency.setdefault(edge.target_id, []).append((edge.source_id, edge))

        dynamic_nodes: List[DynamicGoalNode] = []
        adjusted_by_id: Dict[str, float] = {}
        propagated_count = 0
        for node_id in included_ids:
            action = action_by_id.get(node_id)
            if action is None:
                dynamic_nodes.append(DynamicGoalNode(
                    topo_node_id=node_id,
                    persistent_node_type=self._persistent_type(topology, node_id),
                    executable=False,
                    action_type=None,
                    raw_relevance=_clip01(topology.node_relevance(node_id)),
                    confidence=_clip01(topology.node_relevance(node_id)),
                    uncertainty=1.0,
                    freshness=0.0,
                    propagated_support=None,
                    dynamic_relevance=_clip01(topology.node_relevance(node_id)),
                    positive_views=0,
                    negative_views=0,
                    reachable=None,
                    effective_cost_m=None,
                    budget_ratio=None,
                    route_mode=None,
                ))
                continue

            estimate, uncertainty, freshness, state = base[node_id]
            support_values = []
            for neighbor_id, edge in adjacency.get(node_id, []):
                neighbor = base.get(neighbor_id)
                if neighbor is None:
                    continue
                neighbor_estimate, neighbor_uncertainty, _, neighbor_state = neighbor
                if neighbor_state.positive_views <= 0:
                    continue
                support_values.append(
                    neighbor_estimate
                    * (1.0 - neighbor_uncertainty)
                    * edge.reliability
                    * edge.freshness
                )
            propagated = max(support_values) if support_values else None
            dynamic_relevance = estimate
            if propagated is not None and propagated > dynamic_relevance:
                dynamic_relevance += self.one_hop_discount * (
                    propagated - dynamic_relevance
                )
                propagated_count += 1
            dynamic_relevance = _clip01(dynamic_relevance)
            adjusted_by_id[node_id] = dynamic_relevance
            dynamic_nodes.append(DynamicGoalNode(
                topo_node_id=node_id,
                persistent_node_type=self._persistent_type(topology, node_id),
                executable=True,
                action_type=action.action_type,
                raw_relevance=_clip01(action.relevance),
                confidence=state.posterior_mean,
                uncertainty=uncertainty,
                freshness=freshness,
                propagated_support=propagated,
                dynamic_relevance=dynamic_relevance,
                positive_views=state.positive_views,
                negative_views=state.negative_views,
                reachable=bool(action.reachable),
                effective_cost_m=_finite(action.effective_cost_m),
                budget_ratio=_finite(action.budget_ratio),
                route_mode=str(action.route_mode),
            ))

        changed = sum(
            not math.isclose(
                float(action.relevance), adjusted_by_id[str(action.topo_node_id)],
                rel_tol=0.0, abs_tol=1e-12,
            )
            for action in metric_view.actions
        )
        self.stats["views"] += 1
        self.stats["changed_actions"] += changed
        self.stats["propagated_actions"] += propagated_count
        if not self.shadow_only:
            self.stats["active_views"] += 1
            adjusted_view = replace(metric_view, actions=[
                replace(
                    action,
                    relevance=adjusted_by_id[str(action.topo_node_id)],
                )
                for action in metric_view.actions
            ])
        else:
            adjusted_view = metric_view
        view = DynamicGoalTopologyView(
            goal_type=str(metric_view.goal_type),
            subtask_id=str(metric_view.subtask_id),
            step=int(step),
            shadow_only=self.shadow_only,
            nodes=sorted(dynamic_nodes, key=lambda item: (
                not item.executable, item.action_type or "", item.topo_node_id
            )),
            edges=sorted(edges, key=lambda item: (
                item.edge_type, item.source_id, item.target_id
            )),
            changed_actions=changed,
        )
        return adjusted_view, view

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "subtask_id": self.subtask_id,
            "goal_evidence_nodes": len(self.states),
            "shadow_only": self.shadow_only,
        }
