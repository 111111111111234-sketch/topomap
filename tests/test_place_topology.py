import unittest

import numpy as np

from src.place_topology import (
    PlaceEdgeEvidence,
    PlaceEdgeStatus,
    PlaceTopology,
    replay_place_shadow_steps,
)


class PlaceTopologyTest(unittest.TestCase):
    def test_local_connection_targets_do_not_inherit_graph_neighbors(self):
        graph = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
        active = graph.observe_pose([0, 0], 0, "pose:0").place_id
        far = graph.observe_pose([20, 0], 1, "pose:1").place_id
        graph.add_traversable_edge(
            far,
            active,
            2.0,
            PlaceEdgeEvidence.KNOWN_FREE_PATH,
            1,
            local_path_reference="test:far-to-active",
        )

        targets = graph.local_connection_targets([20, 0])

        self.assertIn(far, targets)
        self.assertNotIn(active, targets)

    def test_known_grid_paths_do_not_cross_unknown_or_obstacles(self):
        free = np.ones((5, 5), dtype=bool)
        free[:, 2] = False
        paths = PlaceTopology.known_grid_path_lengths_m(
            free,
            [2, 1],
            {"same_side": [4, 1], "behind_wall": [2, 3]},
            voxel_size=0.1,
        )
        self.assertAlmostEqual(paths["same_side"], 0.2)
        self.assertNotIn("behind_wall", paths)

    def test_known_free_grid_adds_auditable_reverse_connection(self):
        graph = PlaceTopology(place_spacing_m=2.0, voxel_size=0.1)
        a = graph.observe_pose([0, 0], 0, "pose:0").place_id
        b = graph.observe_pose([21, 0], 1, "pose:1").place_id
        # Edge verification is not clipped by the node spacing; locality is
        # supplied by the caller's small candidate set.
        graph.add_known_free_connections({a: 2.1}, 2, "pose:2")
        reverse = graph.edges[(b, a)]
        self.assertEqual(reverse.evidence, PlaceEdgeEvidence.KNOWN_FREE_PATH)
        self.assertTrue(reverse.local_path_reference.startswith("known_tsdf_grid:"))
        self.assertTrue(graph.audit()["valid"])

    def test_frozen_shadow_replay_preserves_every_decision(self):
        online = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
        records = []
        for step, position in enumerate(([0, 0], [11, 0])):
            assignment = online.observe_pose(
                position, step, f"pose:{step}", verified_paths_m={}
            )
            frozen = {
                "choice": {"type": "frontier", "topo_id": f"f{step}"},
                "planner_target": [float(step), 0.0],
            }
            records.append({
                "voxel_size": 0.1,
                "place_spacing_m": 1.0,
                "pose": {
                    "position": list(position),
                    "step": step,
                    "observation_id": f"pose:{step}",
                    "verified_paths_m": {},
                    "expected_assignment": assignment.to_trace_dict(),
                },
                "known_connections_added": 0,
                "observations": [],
                "frontiers": [],
                "frozen_navigation_decision": frozen,
                "shadow_navigation_decision": dict(frozen),
                "graph_statistics": online.get_statistics(),
                "graph_audit": online.audit(),
            })
        result = replay_place_shadow_steps(records)
        self.assertTrue(result["valid"])
        self.assertEqual(result["steps_replayed"], 2)
        self.assertEqual(result["navigation_mismatches"], 0)

    def test_nearby_pose_is_not_merged_without_verified_connectivity(self):
        graph = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
        first = graph.observe_pose([0, 0], 0, "pose:0")
        second = graph.observe_pose([11, 0], 1, "pose:1")
        self.assertNotEqual(first.place_id, second.place_id)

        # Geometrically close to place_0, but the caller has no known free
        # path through the wall.  The active trajectory is also longer than
        # the Place scale, so this must remain a distinct Place.
        third = graph.observe_pose(
            [1, 0], 2, "pose:2",
            verified_paths_m={first.place_id: None},
            trajectory_traversable=False,
        )
        self.assertTrue(third.created)
        self.assertNotEqual(first.place_id, third.place_id)

    def test_verified_place_assignment_is_order_independent(self):
        graph = PlaceTopology(place_spacing_m=2.0, voxel_size=0.1)
        a = graph.observe_pose([0, 0], 0).place_id
        b = graph.observe_pose([21, 0], 1).place_id
        # Both candidates are spatially close to the new pose.  The shorter
        # verified path wins regardless of mapping insertion order.
        left = graph.observe_pose(
            [10, 0], 2, verified_paths_m={b: 1.2, a: 0.9}
        )

        replay = PlaceTopology(place_spacing_m=2.0, voxel_size=0.1)
        a2 = replay.observe_pose([0, 0], 0).place_id
        b2 = replay.observe_pose([21, 0], 1).place_id
        right = replay.observe_pose(
            [10, 0], 2, verified_paths_m={a2: 0.9, b2: 1.2}
        )
        self.assertEqual(left.place_id, a)
        self.assertEqual(right.place_id, a2)

    def test_frontier_approach_does_not_depend_on_insertion_order(self):
        def result(order):
            graph = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
            a = graph.observe_pose([0, 0], 0).place_id
            b = graph.observe_pose([11, 0], 1).place_id
            paths = {a: 1.4, b: 0.7}
            for place_id in order(a, b):
                graph.observe_frontier(
                    "frontier_0", [6, 0], place_id, 2,
                    verified_approach_path_m=paths[place_id],
                )
            binding = graph.frontiers["frontier_0"]
            return binding.approach_place_id, binding.approach_reason

        self.assertEqual(
            result(lambda a, b: (a, b)),
            result(lambda a, b: (b, a)),
        )
        self.assertEqual(result(lambda a, b: (a, b))[1],
                         "shortest_verified_known_path")

    def test_snapshot_capture_pose_is_immutable(self):
        graph = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
        a = graph.observe_pose([0, 0], 0).place_id
        graph.bind_observation("snapshot:s0", a, [0, 0], 0)
        b = graph.observe_pose([11, 0], 1).place_id
        binding = graph.bind_observation("snapshot:s0", b, [99, 99], 1)
        np.testing.assert_array_equal(binding.capture_pose, [0, 0])
        self.assertEqual(binding.capture_place_id, a)
        self.assertEqual(binding.observed_from_place_ids, {a, b})

    def test_routes_use_valid_auditable_edges_only(self):
        graph = PlaceTopology(place_spacing_m=1.0, voxel_size=0.1)
        a = graph.observe_pose([0, 0], 0, "pose:0").place_id
        b = graph.observe_pose([11, 0], 1, "pose:1").place_id
        c = graph.observe_pose([22, 0], 2, "pose:2").place_id
        route, length = graph.shortest_path(a, c)
        self.assertEqual(route, [a, b, c])
        self.assertAlmostEqual(length, 2.2)
        for edge in graph.edges.values():
            self.assertEqual(edge.status, PlaceEdgeStatus.VALID)
            self.assertEqual(
                edge.evidence, PlaceEdgeEvidence.TRAVERSED_TRAJECTORY
            )
            self.assertTrue(edge.local_path_reference)
            self.assertIsNotNone(edge.source_anchor)
            self.assertIsNotNone(edge.target_anchor)
        self.assertTrue(graph.audit()["valid"])

        graph.invalidate_edge(b, c, 3)
        self.assertEqual(graph.shortest_path(a, c), ([], None))


if __name__ == "__main__":
    unittest.main()
