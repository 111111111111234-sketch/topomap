"""Small confidence overlay for the frozen HGR V7 reuse gate.

This intentionally combines the useful parts of the abandoned V7.1-D/V7.2-
V7.4 roadmap into one conservative mechanism.  It does not create a second
planner: online evidence only nudges REVISIT relevance before the existing
metric gate, and the strict cost/budget checks remain authoritative.
"""

from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import GoalTopoActionType, TopoEdgeType


@dataclass
class GoalEvidenceState:
    confidence: float
    positive_views: int = 0
    negative_views: int = 0
    last_evidence_step: int = -1
    last_position_x: Optional[float] = None
    last_position_y: Optional[float] = None
    capture_anchor_id: Optional[str] = None

    @property
    def independent_views(self) -> int:
        return self.positive_views + self.negative_views

    @property
    def uncertainty(self) -> float:
        return float(1.0 / np.sqrt(1.0 + self.independent_views))

    def to_trace_dict(self):
        return asdict(self)


class LightweightGoalConfidence:
    """Goal-local, finite-safe confidence memory shared by all GOAT modes."""

    def __init__(self, cfg):
        self.cfg = dict(cfg or {})
        self.states: Dict[Tuple[str, str], GoalEvidenceState] = {}

    def _independent(self, state, position, step) -> bool:
        if state.last_evidence_step < 0:
            return True
        step_gap = int(step) - int(state.last_evidence_step)
        if step_gap < int(self.cfg.get("min_independent_step_gap", 2)):
            return False
        if state.last_position_x is None or state.last_position_y is None:
            return True
        previous = np.asarray(
            [state.last_position_x, state.last_position_y], dtype=float
        )
        distance = float(np.linalg.norm(np.asarray(position, dtype=float)[:2] - previous))
        return distance >= float(self.cfg.get("min_independent_distance_voxels", 5.0))

    @staticmethod
    def _state_key(view, action):
        return str(view.subtask_id), str(action.topo_node_id)

    def _one_hop_support(self, view, action, topology) -> Optional[float]:
        allowed = {TopoEdgeType.OBSERVED_AT, TopoEdgeType.DEPENDS_ON}
        neighbor_ids = set()
        for edge in topology.edges.values():
            if edge.edge_type not in allowed:
                continue
            if edge.source_id == action.topo_node_id:
                neighbor_ids.add(edge.target_id)
            elif edge.target_id == action.topo_node_id:
                neighbor_ids.add(edge.source_id)
        # OBSERVED nodes captured at the same VISITED anchor share weak support.
        if action.capture_anchor_id:
            for candidate in view.actions:
                if (
                    candidate.topo_node_id != action.topo_node_id
                    and candidate.capture_anchor_id == action.capture_anchor_id
                ):
                    neighbor_ids.add(candidate.topo_node_id)
        values = []
        for node_id in neighbor_ids:
            state = self.states.get((str(view.subtask_id), str(node_id)))
            if state is not None and state.positive_views > 0:
                values.append(float(state.confidence))
        return max(values) if values else None

    def update_and_adjust(self, view, topology, current_voxel, step):
        """Update independent online evidence and return an adjusted view.

        Positive evidence comes only from current DIRECT observations.  A
        negative view is counted only when the agent is already close to a
        historical capture anchor and no strong current observation exists.
        Thus repeatedly listing a historical Snapshot cannot reinforce itself.
        """
        position = np.asarray(current_voxel, dtype=float)[:2]
        positive_threshold = float(self.cfg.get("positive_relevance", 0.60))
        min_negative_views = int(self.cfg.get("min_negative_views", 3))
        coverage_radius_m = float(self.cfg.get("coverage_radius_m", 1.0))
        memory_weight = float(self.cfg.get("memory_weight", 0.15))
        one_hop_weight = float(self.cfg.get("one_hop_weight", 0.05))
        negative_step = float(self.cfg.get("negative_step", 0.05))

        direct_actions = [
            action for action in view.actions
            if action.action_type == GoalTopoActionType.DIRECT.value
            and action.current_observation
        ]
        strong_direct = any(
            action.relevance >= positive_threshold for action in direct_actions
        )
        for action in direct_actions:
            key = self._state_key(view, action)
            state = self.states.get(key)
            if state is None:
                state = GoalEvidenceState(
                    confidence=float(action.relevance),
                    capture_anchor_id=action.capture_anchor_id,
                )
                self.states[key] = state
            if not self._independent(state, position, step):
                continue
            # EMA prevents repeated views from saturating confidence.
            state.confidence = float(np.clip(
                0.70 * state.confidence + 0.30 * action.relevance, 0.0, 1.0
            ))
            if action.relevance >= positive_threshold:
                state.positive_views += 1
            state.last_evidence_step = int(step)
            state.last_position_x, state.last_position_y = map(float, position)

        for action in view.actions:
            if (
                action.action_type != GoalTopoActionType.REVISIT.value
                or action.effective_cost_m is None
                or float(action.effective_cost_m) > coverage_radius_m
                or strong_direct
            ):
                continue
            key = self._state_key(view, action)
            state = self.states.get(key)
            if state is None:
                state = GoalEvidenceState(
                    confidence=float(action.relevance),
                    capture_anchor_id=action.capture_anchor_id,
                )
                self.states[key] = state
            if not self._independent(state, position, step):
                continue
            state.negative_views += 1
            if state.negative_views >= min_negative_views:
                state.confidence = float(np.clip(
                    state.confidence - negative_step, 0.0, 1.0
                ))
            state.last_evidence_step = int(step)
            state.last_position_x, state.last_position_y = map(float, position)

        adjusted_actions = []
        trace_actions = []
        for action in view.actions:
            raw = float(action.relevance)
            adjusted = raw
            state = self.states.get(self._state_key(view, action))
            freshness = None
            if state is not None and state.independent_views > 0:
                age = max(0, int(step) - int(state.last_evidence_step))
                freshness = float(0.98 ** age)
                adjusted += (
                    memory_weight * freshness
                    * (float(state.confidence) - raw)
                )
                if state.negative_views >= min_negative_views:
                    adjusted -= negative_step * min(
                        state.negative_views - min_negative_views + 1, 2
                    )
            propagated = self._one_hop_support(view, action, topology)
            if propagated is not None:
                adjusted += one_hop_weight * max(0.0, propagated - raw)
            adjusted = float(np.clip(adjusted, 0.0, 1.0))
            adjusted_actions.append(replace(action, relevance=adjusted))
            if state is not None or propagated is not None or adjusted != raw:
                trace_actions.append({
                    "topo_node_id": action.topo_node_id,
                    "action_type": action.action_type,
                    "raw_relevance": raw,
                    "adjusted_relevance": adjusted,
                    "one_hop_support": propagated,
                    "freshness": freshness,
                    "uncertainty": (
                        state.uncertainty if state is not None else None
                    ),
                    "state": state.to_trace_dict() if state is not None else None,
                })
        return replace(view, actions=adjusted_actions), {
            "goal_type": view.goal_type,
            "subtask_id": view.subtask_id,
            "step": int(step),
            "changed_actions": sum(
                item["raw_relevance"] != item["adjusted_relevance"]
                for item in trace_actions
            ),
            "actions": trace_actions,
        }
