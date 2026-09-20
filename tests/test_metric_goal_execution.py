import json
import unittest
from types import SimpleNamespace

import numpy as np

from src.metric_goal_execution import (
    DirectFirstRouteMode,
    MetricAction,
    MetricActionView,
    MetricRouteMode,
    build_goal_conditioned_topology_overlay,
    build_metric_action_view,
    create_target_commitment,
    evaluate_revisit_intervention_gate,
    evaluate_target_commitment,
    plan_direct_first_revisit,
    revisit_gate_authorizes_selected_action,
)
from src.persistent_belief_topology import (
    GoalContext,
    GoalTopoAction,
    GoalTopoActionType,
    PersistentBeliefTopology,
)


CFG = {
    "visited_merge_distance": 0.1,
    "observed_merge": {
        "max_distance": 0.75,
        "min_category_jaccard": 0.30,
        "empty_category_max_distance": 0.35,
    },
    "frontier_matching": {
        "min_score": 0.0,
        "distance_scale": 1.0,
        "weights": {"distance": 1.0},
    },
}


class FakePlanner:
    def __init__(self, reachable=True):
        self.reachable = reachable

    def get_direct_geodesic_distance(
        self, start, target, height, pathfinder
    ):
        del height, pathfinder
        if not self.reachable:
            return float("inf"), None
        distance = float(np.linalg.norm(np.asarray(start) - np.asarray(target)) * 0.1)
        return distance, [np.asarray(start), np.asarray(target)]


class MetricGoalExecutionTest(unittest.TestCase):
    @staticmethod
    def _metric_action(action_type, topo_id, relevance, cost, **kwargs):
        return MetricAction(
            action_type=action_type,
            topo_node_id=topo_id,
            source_kind=kwargs.pop("source_kind", "snapshot"),
            source_frontier_index=kwargs.pop("source_frontier_index", None),
            snapshot_image=kwargs.pop("snapshot_image", "history.png"),
            object_id=None,
            current_observation=False,
            relevance=relevance,
            capture_anchor_id=kwargs.pop("capture_anchor_id", "visited_0"),
            direct_geodesic_m=cost,
            topology_route_m=None,
            effective_cost_m=cost,
            route_mode="direct_path",
            reachable=kwargs.pop("reachable", True),
            remaining_budget_m=kwargs.pop("remaining_budget_m", 20.0),
            budget_ratio=None,
        )

    def _topology_and_sources(self):
        topology = PersistentBeliefTopology(CFG, voxel_size=0.1)
        anchor = topology.upsert_visited([0, 0], 0)
        topology.upsert_observed(
            SimpleNamespace(image="history.png"),
            [0, 0],
            ["chair"],
            0,
            anchor,
            "snapshot:history.png",
            reanchor_existing_evidence=False,
        )
        topology.upsert_visited([10, 0], 1)
        frontier = SimpleNamespace(
            position=np.array([20.0, 0.0]),
            orientation=np.array([1.0, 0.0]),
            region=np.ones((2, 2), dtype=bool),
            embedding=None,
        )
        topology.match_frontiers([frontier], 1)
        return topology, anchor, [frontier]

    def test_all_goal_types_share_metric_action_schema(self):
        topology, anchor, frontiers = self._topology_and_sources()
        schemas = []
        for goal_type in ("category", "description", "image"):
            goal = GoalContext(f"task-{goal_type}", goal_type, category="chair")
            task_view = topology.build_goal_task_view(
                goal,
                frontiers,
                ["history.png"],
                2,
                record_state=False,
            )
            metric_view = build_metric_action_view(
                task_view,
                topology,
                frontiers,
                FakePlanner(reachable=True),
                object(),
                np.array([10.0, 0.0]),
                0.0,
                remaining_steps=20,
                planner_step_m=1.0,
            )
            schemas.append([
                tuple(sorted(action.to_trace_dict()))
                for action in metric_view.actions
            ])
            revisit = next(
                action for action in metric_view.actions
                if action.action_type == "revisit"
            )
            self.assertEqual(revisit.capture_anchor_id, anchor)
            self.assertEqual(revisit.route_mode, MetricRouteMode.DIRECT_PATH.value)
            self.assertAlmostEqual(revisit.direct_geodesic_m, 1.0)
            self.assertAlmostEqual(revisit.topology_route_m, 1.0)
            self.assertTrue(revisit.reachable)
        self.assertEqual(schemas[0], schemas[1])
        self.assertEqual(schemas[1], schemas[2])

    def test_unreachable_direct_path_uses_safe_topology_in_shadow(self):
        topology, _, frontiers = self._topology_and_sources()
        goal = GoalContext("task", "image")
        task_view = topology.build_goal_task_view(
            goal, frontiers, ["history.png"], 2, record_state=False
        )
        metric_view = build_metric_action_view(
            task_view,
            topology,
            frontiers,
            FakePlanner(reachable=False),
            object(),
            np.array([10.0, 0.0]),
            0.0,
            remaining_steps=20,
            planner_step_m=1.0,
        )
        revisit = next(
            action for action in metric_view.actions
            if action.action_type == "revisit"
        )
        frontier = next(
            action for action in metric_view.actions
            if action.action_type == "explore"
        )
        self.assertEqual(revisit.route_mode, MetricRouteMode.SAFE_TOPO.value)
        self.assertAlmostEqual(revisit.effective_cost_m, 1.0)
        self.assertFalse(frontier.reachable)
        self.assertIsNone(frontier.effective_cost_m)
        json.dumps(metric_view.to_trace_dict(), allow_nan=False)

    def test_shadow_projection_does_not_mutate_task_view_or_route_stats(self):
        topology, _, frontiers = self._topology_and_sources()
        before_stats = dict(topology.stats)
        self.assertIsNone(topology.current_task_view)
        goal = GoalContext("task", "description", description="wooden chair")
        task_view = topology.build_goal_task_view(
            goal, frontiers, ["history.png"], 2, record_state=False
        )
        build_metric_action_view(
            task_view,
            topology,
            frontiers,
            FakePlanner(reachable=False),
            object(),
            np.array([10.0, 0.0]),
            0.0,
            remaining_steps=20,
            planner_step_m=1.0,
        )
        self.assertIsNone(topology.current_task_view)
        self.assertEqual(topology.stats, before_stats)

    def test_direct_first_prefers_pathfinder_to_fixed_capture_anchor(self):
        topology, anchor, _ = self._topology_and_sources()
        plan = plan_direct_first_revisit(
            "history.png",
            topology,
            FakePlanner(reachable=True),
            object(),
            np.array([10.0, 0.0]),
            0.0,
        )
        self.assertEqual(plan.route_mode, DirectFirstRouteMode.DIRECT_PATH.value)
        self.assertEqual(plan.capture_anchor_id, anchor)
        np.testing.assert_allclose(plan.target_voxel, [0.0, 0.0])
        self.assertAlmostEqual(plan.direct_geodesic_m, 1.0)
        self.assertIsNone(plan.topology_route_m)
        self.assertTrue(plan.uses_override)
        json.dumps(plan.to_trace_dict(), allow_nan=False)

    def test_direct_first_uses_visited_only_route_when_direct_unreachable(self):
        topology, anchor, _ = self._topology_and_sources()
        plan = plan_direct_first_revisit(
            "history.png",
            topology,
            FakePlanner(reachable=False),
            object(),
            np.array([10.0, 0.0]),
            0.0,
        )
        self.assertEqual(plan.route_mode, DirectFirstRouteMode.SAFE_TOPO.value)
        self.assertEqual(plan.capture_anchor_id, anchor)
        self.assertEqual(plan.next_waypoint_id, anchor)
        self.assertTrue(plan.uses_override)
        self.assertAlmostEqual(plan.topology_route_m, 1.0)

    def test_direct_first_falls_back_to_original_hgr_without_safe_route(self):
        topology, _, _ = self._topology_and_sources()
        topology.edges.clear()
        plan = plan_direct_first_revisit(
            "history.png",
            topology,
            FakePlanner(reachable=False),
            object(),
            np.array([10.0, 0.0]),
            0.0,
        )
        self.assertEqual(
            plan.route_mode,
            DirectFirstRouteMode.ORIGINAL_HGR_FALLBACK.value,
        )
        self.assertFalse(plan.uses_override)
        self.assertEqual(
            plan.fallback_reason,
            "direct_and_safe_topology_unreachable",
        )

    def test_direct_first_does_not_infer_success_at_capture_anchor(self):
        topology, anchor, _ = self._topology_and_sources()
        plan = plan_direct_first_revisit(
            "history.png",
            topology,
            FakePlanner(reachable=True),
            object(),
            np.array([0.0, 0.0]),
            0.0,
        )
        self.assertEqual(plan.route_mode, DirectFirstRouteMode.ANCHOR_REACHED.value)
        self.assertEqual(plan.capture_anchor_id, anchor)
        self.assertFalse(plan.uses_override)

    def test_revisit_gate_accepts_clear_affordable_revisit(self):
        view = MetricActionView("image", "task", 0, [
            self._metric_action("revisit", "history", 0.9, 2.0),
            self._metric_action(
                "explore", "frontier", 0.7, 2.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        decision = evaluate_revisit_intervention_gate(
            view, 0.1, 0.25, 1.25, 1.0
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "accepted")
        self.assertAlmostEqual(decision.relevance_margin, 0.2)
        self.assertAlmostEqual(decision.revisit_to_explore_cost_ratio, 1.0)

    def test_revisit_gate_authorization_is_bound_to_evaluated_node(self):
        view = MetricActionView("description", "task", 0, [
            self._metric_action("revisit", "approved", 0.9, 2.0),
            self._metric_action("revisit", "other", 0.8, 1.0),
            self._metric_action(
                "explore", "frontier", 0.6, 3.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        decision = evaluate_revisit_intervention_gate(
            view, 0.1, 0.25, 1.25, 1.0
        )
        approved = GoalTopoAction(
            GoalTopoActionType.REVISIT, "approved", "snapshot"
        )
        other = GoalTopoAction(
            GoalTopoActionType.REVISIT, "other", "snapshot"
        )
        direct = GoalTopoAction(
            GoalTopoActionType.DIRECT, "approved", "snapshot"
        )
        self.assertTrue(decision.allowed)
        self.assertTrue(revisit_gate_authorizes_selected_action(
            decision, approved
        ))
        self.assertFalse(revisit_gate_authorizes_selected_action(
            decision, other
        ))
        self.assertFalse(revisit_gate_authorizes_selected_action(
            decision, direct
        ))
        self.assertFalse(revisit_gate_authorizes_selected_action(
            decision, None
        ))

    def test_final_goal_overlay_has_four_core_fields_and_shared_actions(self):
        view = MetricActionView("image", "task", 0, [
            self._metric_action("revisit", "history", 0.9, 2.0),
            self._metric_action(
                "explore", "frontier", 0.7, 2.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        overlay = build_goal_conditioned_topology_overlay(view)
        self.assertIs(overlay.metric_view, view)
        self.assertIs(overlay.actions, view.actions)
        self.assertEqual(len(overlay.candidates), 2)
        revisit = overlay.candidates[0]
        self.assertEqual(revisit.goal_evidence["source"], "historical_observation")
        self.assertEqual(revisit.belief, 0.9)
        self.assertEqual(revisit.cost_m, 2.0)
        self.assertTrue(revisit.validity["valid"])
        self.assertEqual(revisit.topology_relations, ["OBSERVED_AT", "NAVIGABLE"])
        payload = overlay.to_trace_dict()
        self.assertEqual(
            set(payload["candidates"][0]),
            {
                "action_type", "topo_node_id", "goal_evidence", "belief",
                "cost_m", "validity", "topology_relations",
            },
        )
        json.dumps(payload, allow_nan=False)

    def test_final_goal_overlay_is_exactly_c_strict_gate_equivalent(self):
        view = MetricActionView("description", "task", 0, [
            self._metric_action("revisit", "history", 0.9, 2.0),
            self._metric_action(
                "explore", "frontier", 0.7, 2.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        overlay = build_goal_conditioned_topology_overlay(view)
        original = evaluate_revisit_intervention_gate(
            view, 0.1, 0.15, 1.0, 1.0
        )
        projected = evaluate_revisit_intervention_gate(
            overlay, 0.1, 0.15, 1.0, 1.0
        )
        self.assertEqual(projected, original)

    def test_revisit_gate_rejects_weak_or_overbudget_history(self):
        weak_view = MetricActionView("description", "task", 0, [
            self._metric_action("revisit", "history", 0.7, 2.0),
            self._metric_action(
                "explore", "frontier", 0.72, 1.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        weak = evaluate_revisit_intervention_gate(
            weak_view, 0.05, 0.25, 1.25, 1.0
        )
        self.assertFalse(weak.allowed)
        self.assertEqual(weak.reason, "relevance_margin_below_threshold")
        budget_view = MetricActionView("category", "task", 0, [
            self._metric_action(
                "revisit", "history", 0.9, 8.0, remaining_budget_m=20.0
            ),
            self._metric_action(
                "explore", "frontier", 0.5, 10.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        budget = evaluate_revisit_intervention_gate(
            budget_view, 0.05, 0.25, 1.25, 1.0
        )
        self.assertFalse(budget.allowed)
        self.assertEqual(budget.reason, "revisit_exceeds_budget_fraction")

    def test_revisit_gate_requires_snapshot_anchor_mapping(self):
        view = MetricActionView("image", "task", 0, [
            self._metric_action(
                "revisit", "history", 0.9, 1.0, capture_anchor_id=None
            ),
            self._metric_action(
                "direct", "current", 0.5, 0.0,
                snapshot_image="current.png", capture_anchor_id=None,
            ),
        ])
        decision = evaluate_revisit_intervention_gate(
            view, 0.05, 0.25, 1.25, 1.0
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "snapshot_anchor_mapping_unavailable")

    def test_target_commitment_keeps_under_small_challenger_jitter(self):
        revisit = self._metric_action("revisit", "history", 0.80, 4.0)
        commitment = create_target_commitment(revisit, "task", 1)
        view = MetricActionView("image", "task", 2, [
            self._metric_action("revisit", "history", 0.79, 3.5),
            self._metric_action(
                "explore", "frontier", 0.84, 3.2,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        decision, updated = evaluate_target_commitment(
            commitment, view, 2, 0.10, 0.10, 1.0, 0.25, 0.2, 2
        )
        self.assertEqual(decision.action, "keep")
        self.assertEqual(decision.reason, "hysteresis_keep")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.segments_executed, 1)

    def test_target_commitment_switches_or_cancels_explainably(self):
        revisit = self._metric_action("revisit", "history", 0.70, 4.0)
        commitment = create_target_commitment(revisit, "task", 1)
        semantic_view = MetricActionView("description", "task", 2, [
            self._metric_action("revisit", "history", 0.70, 3.5),
            self._metric_action(
                "explore", "frontier", 0.85, 3.0,
                source_kind="frontier", source_frontier_index=0,
                snapshot_image=None, capture_anchor_id=None,
            ),
        ])
        decision, updated = evaluate_target_commitment(
            commitment, semantic_view, 2, 0.10, 0.10, 1.0, 0.25, 0.2, 2
        )
        self.assertEqual(decision.action, "switch")
        self.assertEqual(decision.reason, "semantic_challenger")
        self.assertIsNone(updated)
        stalled_view = MetricActionView("description", "task", 2, [
            self._metric_action("revisit", "history", 0.70, 4.0),
        ])
        decision, updated = evaluate_target_commitment(
            commitment, stalled_view, 2, 0.10, 0.10, 1.0, 0.25, 0.2, 1
        )
        self.assertEqual(decision.action, "cancel")
        self.assertEqual(decision.reason, "repeated_no_progress")
        self.assertIsNone(updated)


if __name__ == "__main__":
    unittest.main()
