import unittest

import numpy as np

from src.metric_goal_execution import MetricAction, MetricActionView
from src.persistent_belief_topology import (
    BeliefTopoNode,
    PersistentBeliefTopology,
    TopoEdge,
    TopoEdgeType,
    TopoNodeType,
)
from src.region_topology import (
    build_region_topology_decision,
    select_local_explore_for_snapshot,
)


def _action(topo_id, relevance, *, frontier=None, snapshot=None, anchor=None):
    return MetricAction(
        action_type="explore" if frontier is not None else "revisit",
        topo_node_id=topo_id,
        source_kind="frontier" if frontier is not None else "snapshot",
        source_frontier_index=frontier,
        snapshot_image=snapshot,
        object_id=None,
        current_observation=False,
        relevance=relevance,
        capture_anchor_id=anchor,
        direct_geodesic_m=1.0,
        topology_route_m=1.0,
        effective_cost_m=1.0,
        route_mode="direct_path",
        reachable=True,
        remaining_budget_m=20.0,
        budget_ratio=0.05,
    )


class RegionTopologyTest(unittest.TestCase):
    def setUp(self):
        self.topology = PersistentBeliefTopology({}, voxel_size=0.1)
        for node_id in ("visited_a", "visited_b"):
            self.topology.nodes[node_id] = BeliefTopoNode(
                node_id=node_id,
                node_type=TopoNodeType.VISITED,
                position=np.zeros(2),
            )

    def test_groups_snapshot_and_frontier_by_stable_anchor(self):
        frontier = _action("frontier_0", 0.8, frontier=3)
        self.topology.nodes["frontier_0"] = BeliefTopoNode(
            node_id="frontier_0",
            node_type=TopoNodeType.FRONTIER,
            position=np.ones(2),
        )
        edge = TopoEdge(
            "frontier_0", "visited_a", TopoEdgeType.ANCHORED_TO
        )
        self.topology.edges[(edge.source_id, edge.target_id, edge.edge_type)] = edge
        snapshot = _action(
            "observed_0", 0.7, snapshot="history.png", anchor="visited_a"
        )
        decision = build_region_topology_decision(
            MetricActionView("object", "s", 1, [frontier, snapshot]),
            None,
            self.topology,
            max_regions=2,
        )
        self.assertEqual(decision.region_ids, ("visited_a",))
        self.assertEqual(decision.frontier_indices, [3])
        self.assertEqual(decision.snapshot_images, ["history.png"])
        self.assertEqual(set(decision.region_action_counts), {"visited_a"})
        self.assertIs(
            select_local_explore_for_snapshot(
                decision, self.topology, "history.png"
            ), frontier
        )

    def test_keeps_all_members_of_two_best_regions(self):
        high = _action("frontier_a", 0.9, frontier=0)
        companion = _action("frontier_a2", 0.3, frontier=1)
        medium = _action("frontier_b", 0.8, frontier=2)
        low = _action("frontier_c", 0.1, frontier=3)
        for action, anchor in (
            (high, "visited_a"), (companion, "visited_a"),
            (medium, "visited_b"), (low, "visited_c"),
        ):
            self.topology.nodes[anchor] = BeliefTopoNode(
                node_id=anchor, node_type=TopoNodeType.VISITED,
                position=np.zeros(2),
            )
            self.topology.nodes[action.topo_node_id] = BeliefTopoNode(
                node_id=action.topo_node_id, node_type=TopoNodeType.FRONTIER,
                position=np.zeros(2),
            )
            edge = TopoEdge(action.topo_node_id, anchor, TopoEdgeType.ANCHORED_TO)
            self.topology.edges[(edge.source_id, edge.target_id, edge.edge_type)] = edge
        decision = build_region_topology_decision(
            MetricActionView("object", "s", 1, [low, companion, medium, high]),
            None, self.topology, max_regions=2,
        )
        self.assertEqual(set(decision.region_ids), {"visited_a", "visited_b"})
        self.assertEqual(set(decision.frontier_indices), {0, 1, 2})

    def test_unanchored_actions_never_create_transient_regions(self):
        decision = build_region_topology_decision(
            MetricActionView("object", "s", 1, [_action("frontier", 0.9, frontier=0)]),
            None, self.topology, max_regions=2,
        )
        self.assertEqual(decision.reason, "no_anchored_candidates")
        self.assertEqual(decision.actions, ())


    def test_frontier_only_excludes_snapshot_only_region(self):
        snapshot = _action(
            "observed_high", 0.99, snapshot="history.png", anchor="visited_a"
        )
        frontier = _action("frontier_b", 0.40, frontier=2)
        self.topology.nodes["frontier_b"] = BeliefTopoNode(
            node_id="frontier_b", node_type=TopoNodeType.FRONTIER,
            position=np.zeros(2),
        )
        edge = TopoEdge("frontier_b", "visited_b", TopoEdgeType.ANCHORED_TO)
        self.topology.edges[(edge.source_id, edge.target_id, edge.edge_type)] = edge
        decision = build_region_topology_decision(
            MetricActionView("object", "s", 1, [snapshot, frontier]),
            None, self.topology, max_regions=2, require_frontier=True,
        )
        self.assertEqual(decision.region_ids, ("visited_b",))
        self.assertEqual(decision.snapshot_images, [])
        self.assertEqual(decision.frontier_indices, [2])

if __name__ == "__main__":
    unittest.main()
