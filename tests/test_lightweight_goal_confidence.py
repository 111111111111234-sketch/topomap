import unittest
from types import SimpleNamespace

from src.lightweight_goal_confidence import LightweightGoalConfidence
from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import TopoEdgeType


def action(kind, node, relevance, cost=0.0, current=False, anchor=None):
    return MetricAction(
        action_type=kind,
        topo_node_id=node,
        source_kind="snapshot",
        source_frontier_index=None,
        snapshot_image=f"{node}.png",
        object_id=None,
        current_observation=current,
        relevance=relevance,
        capture_anchor_id=anchor,
        direct_geodesic_m=cost,
        topology_route_m=None,
        effective_cost_m=cost,
        route_mode="current" if kind == "direct" else "direct_path",
        reachable=True,
        remaining_budget_m=20.0,
        budget_ratio=0.0,
    )


class LightweightGoalConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.memory = LightweightGoalConfidence({
            "memory_weight": 0.15,
            "one_hop_weight": 0.05,
            "positive_relevance": 0.60,
            "min_negative_views": 3,
            "coverage_radius_m": 1.0,
            "negative_step": 0.05,
            "min_independent_step_gap": 2,
            "min_independent_distance_voxels": 5.0,
        })
        self.topology = SimpleNamespace(edges={})

    def test_current_positive_evidence_later_boosts_same_revisit(self):
        direct = MetricActionView(
            "image", "task", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        self.memory.update_and_adjust(direct, self.topology, [0, 0], 0)
        revisit = MetricActionView(
            "image", "task", 2,
            [action("revisit", "observed_0", 0.6, cost=2.0)],
        )
        adjusted, trace = self.memory.update_and_adjust(
            revisit, self.topology, [10, 0], 2
        )
        self.assertGreater(adjusted.actions[0].relevance, 0.6)
        self.assertEqual(trace["changed_actions"], 1)

    def test_negative_support_requires_three_independent_close_views(self):
        view = MetricActionView(
            "description", "task", 0,
            [action("revisit", "observed_0", 0.8, cost=0.5)],
        )
        adjusted = None
        for step, position in ((0, [0, 0]), (2, [5, 0]), (4, [10, 0])):
            adjusted, _ = self.memory.update_and_adjust(
                view, self.topology, position, step
            )
        self.assertLess(adjusted.actions[0].relevance, 0.8)

    def test_typed_one_hop_support_is_weak_and_bounded(self):
        direct = MetricActionView(
            "category", "task", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        self.memory.update_and_adjust(direct, self.topology, [0, 0], 0)
        edge = SimpleNamespace(
            source_id="observed_0",
            target_id="frontier_0",
            edge_type=TopoEdgeType.DEPENDS_ON,
        )
        self.topology.edges = {("observed_0", "frontier_0", "depends_on"): edge}
        revisit = MetricActionView(
            "category", "task", 2,
            [action("revisit", "frontier_0", 0.5, cost=2.0)],
        )
        adjusted, _ = self.memory.update_and_adjust(
            revisit, self.topology, [10, 0], 2
        )
        self.assertGreater(adjusted.actions[0].relevance, 0.5)
        self.assertLessEqual(adjusted.actions[0].relevance, 1.0)


if __name__ == "__main__":
    unittest.main()
