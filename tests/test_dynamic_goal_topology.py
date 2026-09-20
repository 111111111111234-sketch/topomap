import unittest
from types import SimpleNamespace

from src.dynamic_goal_topology import DynamicGoalTopology
from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import TopoEdgeType, TopoNodeType


def action(
    kind, node, relevance, cost=0.0, current=False, anchor=None,
    source_kind="snapshot", object_id=None,
):
    return MetricAction(
        action_type=kind,
        topo_node_id=node,
        source_kind=source_kind,
        source_frontier_index=0 if source_kind == "frontier" else None,
        snapshot_image=None if source_kind == "frontier" else f"{node}.png",
        object_id=object_id,
        current_observation=current,
        relevance=relevance,
        capture_anchor_id=anchor,
        direct_geodesic_m=cost,
        topology_route_m=None,
        effective_cost_m=cost,
        route_mode="current" if kind == "direct" else "direct_path",
        reachable=True,
        remaining_budget_m=20.0,
        budget_ratio=cost / 20.0,
    )


class FakeTopology:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, node_type=TopoNodeType.OBSERVED, position=(0, 0)):
        self.nodes[node_id] = SimpleNamespace(
            node_type=node_type,
            position=position,
        )

    def node_relevance(self, node_id):
        return 0.5


class DynamicGoalTopologyTest(unittest.TestCase):
    def config(self, shadow_only=True):
        return {
            "shadow_only": shadow_only,
            "prior_strength": 4.0,
            "positive_relevance": 0.60,
            "coverage_radius_m": 1.0,
            "negative_evidence_weight": 0.5,
            "min_negative_views": 3,
            "min_independent_step_gap": 2,
            "min_independent_distance_voxels": 5.0,
            "freshness_half_life_steps": 20,
            "one_hop_discount": 0.25,
            "propagation_edge_types": [
                "observed_at", "anchored_to", "depends_on",
                "same_capture_anchor",
            ],
        }

    def test_shadow_builds_second_layer_without_changing_metric_view(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        metric = MetricActionView(
            "image", "task", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        model = DynamicGoalTopology(self.config(shadow_only=True))
        projected, view = model.update_and_project(metric, topology, [0, 0], 0)
        self.assertIs(projected, metric)
        self.assertTrue(view.shadow_only)
        self.assertEqual(len(view.nodes), 1)
        self.assertEqual(view.nodes[0].positive_views, 1)

    def test_active_positive_evidence_changes_later_revisit_only(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        model = DynamicGoalTopology(self.config(shadow_only=False))
        direct = MetricActionView(
            "image", "task", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        model.update_and_project(direct, topology, [0, 0], 0)
        revisit = MetricActionView(
            "image", "task", 2,
            [action("revisit", "observed_0", 0.6, cost=2.0)],
        )
        projected, view = model.update_and_project(
            revisit, topology, [10, 0], 2
        )
        self.assertGreater(projected.actions[0].relevance, 0.6)
        self.assertEqual(view.changed_actions, 1)

    def test_close_repeated_view_is_deduplicated(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        model = DynamicGoalTopology(self.config())
        direct = MetricActionView(
            "description", "task", 0,
            [action("direct", "observed_0", 0.8, current=True)],
        )
        model.update_and_project(direct, topology, [0, 0], 0)
        model.update_and_project(direct, topology, [0, 0], 2)
        self.assertEqual(model.states["observed_0"].positive_views, 1)
        self.assertEqual(model.stats["deduplicated_updates"], 1)

    def test_anchor_coverage_absence_is_conservative_negative_evidence(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        model = DynamicGoalTopology(self.config(shadow_only=False))
        revisit = MetricActionView(
            "object", "task", 0,
            [action("revisit", "observed_0", 0.8, cost=0.5, anchor="visited_0")],
        )
        projected = None
        for step, position in ((0, [0, 0]), (2, [5, 0]), (4, [10, 0])):
            projected, _ = model.update_and_project(
                revisit, topology, position, step
            )
        self.assertLess(projected.actions[0].relevance, 0.8)
        self.assertEqual(model.states["observed_0"].negative_views, 3)

    def test_first_two_negative_views_only_record_coverage(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        model = DynamicGoalTopology(self.config(shadow_only=False))
        revisit = MetricActionView(
            "description", "task", 0,
            [action("revisit", "observed_0", 0.8, cost=0.5)],
        )
        for step, position in ((0, [0, 0]), (2, [5, 0])):
            projected, _ = model.update_and_project(
                revisit, topology, position, step
            )
            self.assertAlmostEqual(projected.actions[0].relevance, 0.8)
        self.assertEqual(model.states["observed_0"].negative_views, 2)

    def test_weak_direct_is_not_negative_and_exact_object_is_positive(self):
        topology = FakeTopology()
        topology.add_node("observed_weak")
        topology.add_node("observed_exact")
        topology.add_node("observed_history")
        model = DynamicGoalTopology(self.config(shadow_only=False))
        view = MetricActionView(
            "category", "task", 0,
            [
                action(
                    "direct", "observed_weak", 0.55, current=True
                ),
                action(
                    "direct", "observed_exact", 0.40, current=True,
                    object_id=17,
                ),
                action(
                    "revisit", "observed_history", 0.8, cost=0.5,
                    anchor="visited_0",
                ),
            ],
        )
        model.update_and_project(view, topology, [0, 0], 0)
        self.assertEqual(model.states["observed_weak"].evidence_views, 0)
        self.assertEqual(model.states["observed_exact"].positive_views, 1)
        self.assertEqual(model.states["observed_history"].negative_views, 0)

    def test_typed_edge_has_description_and_one_hop_support(self):
        topology = FakeTopology()
        topology.add_node("observed_0", position=(0, 0))
        topology.add_node("frontier_0", TopoNodeType.FRONTIER, position=(3, 4))
        edge = SimpleNamespace(
            source_id="observed_0",
            target_id="frontier_0",
            edge_type=TopoEdgeType.DEPENDS_ON,
            distance=5.0,
            path_cost=5.5,
            traversability=0.8,
            dependency_confidence=1.0,
            visit_count=2,
            failure_count=0,
            last_updated_step=2,
        )
        topology.edges[("observed_0", "frontier_0", TopoEdgeType.DEPENDS_ON)] = edge
        model = DynamicGoalTopology(self.config(shadow_only=False))
        direct = MetricActionView(
            "object", "task", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        model.update_and_project(direct, topology, [0, 0], 0)
        joint = MetricActionView(
            "object", "task", 2,
            [
                action("revisit", "observed_0", 0.9, cost=2.0),
                action(
                    "explore", "frontier_0", 0.1, cost=3.0,
                    source_kind="frontier",
                ),
            ],
        )
        projected, view = model.update_and_project(joint, topology, [10, 0], 2)
        frontier = next(
            item for item in projected.actions
            if item.topo_node_id == "frontier_0"
        )
        edge_trace = view.edges[0]
        self.assertGreater(frontier.relevance, 0.1)
        self.assertEqual(edge_trace.edge_type, "depends_on")
        self.assertIn("length=5.00m", edge_trace.description)
        self.assertAlmostEqual(edge_trace.observation_direction_deg, 53.130102, places=5)

    def test_goal_switch_clears_goal_local_evidence(self):
        topology = FakeTopology()
        topology.add_node("observed_0")
        model = DynamicGoalTopology(self.config())
        first = MetricActionView(
            "image", "task_a", 0,
            [action("direct", "observed_0", 0.9, current=True)],
        )
        model.update_and_project(first, topology, [0, 0], 0)
        second = MetricActionView(
            "image", "task_b", 1,
            [action("revisit", "observed_0", 0.4, cost=2.0)],
        )
        model.update_and_project(second, topology, [10, 0], 1)
        self.assertEqual(model.subtask_id, "task_b")
        self.assertEqual(model.states["observed_0"].positive_views, 0)


if __name__ == "__main__":
    unittest.main()
