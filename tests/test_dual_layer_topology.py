import unittest

from src.dual_layer_topology import (
    begin_topo_explore_commitment,
    choose_dual_layer_topology_action,
    refresh_topo_explore_commitment,
)
from src.metric_goal_execution import (
    MetricAction,
    MetricActionView,
    RevisitGateDecision,
)


def _action(action_type, topo_id, relevance, *, frontier=None, snapshot=None):
    return MetricAction(
        action_type=action_type,
        topo_node_id=topo_id,
        source_kind="frontier" if frontier is not None else "snapshot",
        source_frontier_index=frontier,
        snapshot_image=snapshot,
        object_id=None,
        current_observation=action_type == "direct",
        relevance=relevance,
        capture_anchor_id=("visited_0" if snapshot else None),
        direct_geodesic_m=1.0,
        topology_route_m=1.0,
        effective_cost_m=1.0,
        route_mode="direct_path",
        reachable=True,
        remaining_budget_m=20.0,
        budget_ratio=0.05,
    )


def _gate(allowed, topo_id=None):
    return RevisitGateDecision(
        allowed=allowed,
        reason="accepted" if allowed else "rejected",
        best_revisit_topo_id=topo_id,
        best_alternative_topo_id=None,
        cheapest_explore_topo_id=None,
        relevance_margin=0.2,
        revisit_cost_m=1.0,
        remaining_budget_m=20.0,
        budget_ratio=0.05,
        revisit_to_explore_cost_ratio=1.0,
        semantic_clear=allowed,
        budget_ok=allowed,
        opportunity_cost_ok=allowed,
        source_mapping_ok=allowed,
    )


class DualLayerTopologyTest(unittest.TestCase):
    def _view(self, *actions):
        return MetricActionView("object", "subtask", 3, list(actions))

    def test_direct_requires_existing_strict_margin(self):
        direct = _action("direct", "observed_now", 0.80, snapshot="now.png")
        explore = _action("explore", "frontier_0", 0.65, frontier=0)
        decision = choose_dual_layer_topology_action(
            self._view(direct, explore), None, _gate(False), 0.10
        )
        self.assertIs(decision.action, direct)
        self.assertEqual(decision.reason, "direct_margin_accepted")

    def test_approved_exact_revisit_precedes_explore(self):
        denied = _action("revisit", "observed_wrong", 0.95, snapshot="a.png")
        approved = _action("revisit", "observed_ok", 0.75, snapshot="b.png")
        explore = _action("explore", "frontier_0", 0.60, frontier=0)
        decision = choose_dual_layer_topology_action(
            self._view(denied, approved, explore),
            None,
            _gate(True, "observed_ok"),
            0.10,
        )
        self.assertIs(decision.action, approved)
        self.assertEqual(decision.reason, "revisit_gate_accepted")

    def test_rejected_revisit_falls_back_to_exploration(self):
        revisit = _action("revisit", "observed_0", 0.95, snapshot="a.png")
        explore = _action("explore", "frontier_0", 0.40, frontier=0)
        decision = choose_dual_layer_topology_action(
            self._view(revisit, explore), None, _gate(False), 0.10
        )
        self.assertIs(decision.action, explore)
        self.assertEqual(decision.action_type, "explore")

    def test_explore_decision_exposes_one_topological_fallback(self):
        primary = _action("explore", "frontier_0", 0.80, frontier=0)
        fallback = _action("explore", "frontier_1", 0.70, frontier=1)
        decision = choose_dual_layer_topology_action(
            self._view(primary, fallback), None, _gate(False), 0.10
        )
        self.assertIs(decision.action, primary)
        self.assertIs(decision.fallback_explore, fallback)

    def test_unreachable_actions_are_never_selected(self):
        revisit = _action("revisit", "observed_0", 0.99, snapshot="a.png")
        object.__setattr__(revisit, "reachable", False)
        explore = _action("explore", "frontier_1", 0.20, frontier=1)
        decision = choose_dual_layer_topology_action(
            self._view(revisit, explore), None, _gate(True, "observed_0"), 0.10
        )
        self.assertIs(decision.action, explore)

    def test_explore_commitment_keeps_stable_topo_id_and_resets_on_progress(self):
        explore = _action("explore", "frontier_7", 0.5, frontier=1)
        commitment = begin_topo_explore_commitment(
            explore, [0.0, 0.0], [10.0, 0.0], 3
        )
        refreshed, reason = refresh_topo_explore_commitment(
            commitment,
            self._view(explore),
            [3.0, 0.0],
            {1: [10.0, 0.0]},
            4,
            min_progress_voxels=1.0,
            max_no_progress_steps=3,
        )
        self.assertIsNone(reason)
        self.assertEqual(refreshed.topo_node_id, "frontier_7")
        self.assertEqual(refreshed.no_progress_steps, 0)
        self.assertEqual(refreshed.best_distance_voxels, 7.0)

    def test_explore_commitment_releases_after_no_progress_budget(self):
        explore = _action("explore", "frontier_7", 0.5, frontier=1)
        commitment = begin_topo_explore_commitment(
            explore, [0.0, 0.0], [10.0, 0.0], 3
        )
        _, reason = refresh_topo_explore_commitment(
            commitment,
            self._view(explore),
            [0.0, 0.0],
            {1: [10.0, 0.0]},
            4,
            min_progress_voxels=1.0,
            max_no_progress_steps=1,
        )
        self.assertEqual(reason, "no_progress_budget_exhausted")

    def test_explore_commitment_geometrically_rebinds_reextracted_frontier(self):
        old = _action("explore", "frontier_old", 0.5, frontier=1)
        new = _action("explore", "frontier_new", 0.5, frontier=2)
        commitment = begin_topo_explore_commitment(
            old, [0.0, 0.0], [10.0, 0.0], 3
        )
        refreshed, reason = refresh_topo_explore_commitment(
            commitment,
            self._view(new),
            [2.0, 0.0],
            {2: [10.5, 0.0]},
            4,
            min_progress_voxels=1.0,
            max_no_progress_steps=3,
            geometric_rebind_radius_voxels=1.0,
        )
        self.assertEqual(reason, "geometric_rebind")
        self.assertEqual(refreshed.topo_node_id, "frontier_new")
        self.assertEqual(refreshed.source_frontier_index, 2)

    def test_explore_commitment_keeps_live_geometric_anchor_when_unmatched(self):
        old = _action("explore", "frontier_old", 0.5, frontier=1)
        commitment = begin_topo_explore_commitment(
            old, [0.0, 0.0], [10.0, 0.0], 3
        )
        refreshed, reason = refresh_topo_explore_commitment(
            commitment,
            self._view(),
            [2.0, 0.0],
            {},
            4,
            min_progress_voxels=1.0,
            max_no_progress_steps=3,
            geometric_rebind_radius_voxels=1.0,
            live_target_position=[10.0, 0.0],
        )
        self.assertEqual(reason, "geometric_anchor_keep")
        self.assertEqual(refreshed.topo_node_id, "frontier_old")
        self.assertEqual(refreshed.no_progress_steps, 0)


if __name__ == "__main__":
    unittest.main()
