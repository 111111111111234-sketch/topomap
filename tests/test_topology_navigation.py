import unittest
from types import SimpleNamespace

import numpy as np

from src.place_topology import PlaceTopology
from src.topology_navigation import (
    PlaceRouteExecution,
    SnapshotSourceCommitment,
    restore_selected_snapshot,
    PlaceRouteMode,
    plan_place_route,
    select_reachable_approach_place,
)


class TopologyNavigationTest(unittest.TestCase):
    def _selected_source(self, ids=(2,)):
        return SimpleNamespace(
            image="old.png", cluster=list(ids), obs_point=np.array([3., 4.]),
            full_obj_list={1: .7, 2: .9},
        )

    def test_reclustering_keeps_original_evidence_and_approach(self):
        snapshot = self._selected_source()
        source = SnapshotSourceCommitment(snapshot)
        snapshot.cluster[:] = [99]
        snapshot.obs_point[:] = 100
        replacement = SimpleNamespace(image="new.png", cluster=[2, 99])
        choice, info = source.resolve({2: {}, 99: {}}, {}, {"new": replacement}, {})
        self.assertEqual(choice.image, "old.png")
        self.assertEqual(choice.cluster, [2])
        np.testing.assert_array_equal(choice.obs_point, [3., 4.])
        self.assertEqual(info["reason"], "retained_selected_observation")
        # A returned execution choice must not mutate the frozen evidence.
        choice.cluster.append(99)
        self.assertEqual(source.resolve({2: {}}, {}, {}, {})[0].cluster, [2])

    def test_recorded_merge_chain_rebinds_only_selected_identity(self):
        source = SnapshotSourceCommitment(self._selected_source())
        choice, info = source.resolve({8: {}, 99: {}}, {2: 5, 5: 8}, {}, {})
        self.assertEqual(choice.cluster, [8])
        self.assertEqual(choice.full_obj_list, {8: .9})
        self.assertEqual(info["merge_chains"], [[2, 5, 8]])
        self.assertEqual(info["reason"], "recorded_object_merge")
        self.assertEqual(source.snapshot.cluster, [2])

    def test_deleted_object_does_not_rebind_to_same_class_or_partial_set(self):
        for ids in [(2,), (1, 2)]:
            source = SnapshotSourceCommitment(self._selected_source(ids))
            choice, info = source.resolve({1: {}, 99: {"class_name": "chair"}}, {}, {}, {})
            self.assertIsNone(choice)
            self.assertEqual(info["reason"], "selected_object_removed_without_live_successor")

    def test_missing_merge_successor_and_alias_cycle_release(self):
        source = SnapshotSourceCommitment(self._selected_source())
        self.assertIsNone(source.resolve({}, {2: 5}, {}, {})[0])
        choice, info = source.resolve({2: {}}, {2: 5, 5: 2}, {}, {})
        self.assertIsNone(choice)
        self.assertEqual(info["reason"], "object_merge_alias_cycle")

    def test_selected_objects_merged_together_are_deduplicated(self):
        source = SnapshotSourceCommitment(self._selected_source((1, 2)))
        choice, _ = source.resolve({5: {}}, {1: 5, 2: 5}, {}, {})
        self.assertEqual(choice.cluster, [5])
        self.assertEqual(choice.full_obj_list, {5: .9})

    def test_overlapping_anchor_advances_without_changing_map_assignment(self):
        for kind in ("snapshot", "frontier"):
            graph, a, b, c = self._chain()
            execution = PlaceRouteExecution()
            # The agent can reach an entry anchor while map association
            # continues to prefer A. Execution must still progress A->B->C.
            for expected_next in (b, c):
                plan = plan_place_route(
                    graph, c, kind, "goal",
                    reached_place_id=execution.reached_place_id,
                )
                self.assertEqual(plan.next_place_id, expected_next)
                execution.start(plan)
                self.assertIsNone(execution.consume_waypoint_arrival(False))
                self.assertIsNotNone(execution.consume_waypoint_arrival(True))
                self.assertEqual(graph.current_place_id, a)
            terminal = plan_place_route(
                graph, c, kind, "goal",
                reached_place_id=execution.reached_place_id,
            )
            self.assertFalse(terminal.uses_override)
            execution.start(terminal)
            self.assertIsNone(execution.consume_waypoint_arrival(True))
            execution.reset()
            fresh = plan_place_route(
                graph, c, kind, "new_goal",
                reached_place_id=execution.reached_place_id,
            )
            self.assertEqual(fresh.current_place_id, a)

    def test_edge_invalidation_after_arrival_respects_execution_origin(self):
        graph, a, b, c = self._chain()
        execution = PlaceRouteExecution()
        execution.start(plan_place_route(graph, c, "snapshot", "s"))
        execution.consume_waypoint_arrival(True)
        graph.invalidate_edge(b, c, 5)
        plan = plan_place_route(
            graph, c, "snapshot", "s",
            reached_place_id=execution.reached_place_id,
        )
        self.assertEqual(plan.route_mode, "disconnected")
        self.assertEqual(graph.current_place_id, a)
        execution.start(None)
        self.assertIsNone(execution.reached_place_id)

    def test_multihop_arrivals_cannot_finish_semantic_target(self):
        for kind in ("snapshot", "frontier"):
            graph, a, b, c = self._chain()
            executor = PlaceRouteExecution()
            first = plan_place_route(graph, c, kind, "target")
            executor.start(first)
            # Multiple local steps must retain the original segment identity.
            for _ in range(3):
                self.assertIsNone(executor.consume_waypoint_arrival(False))
                self.assertIs(executor.active_plan, first)
            self.assertIs(executor.consume_waypoint_arrival(True), first)
            graph.current_place_id = b
            last = plan_place_route(graph, c, kind, "target")
            executor.start(last)
            self.assertIs(executor.consume_waypoint_arrival(True), last)
            # Even arrival at final Place was consumed; only terminal arrival
            # may reach the original HGR completion/verification protocol.
            graph.current_place_id = c
            executor.start(plan_place_route(graph, c, kind, "target"))
            self.assertIsNone(executor.consume_waypoint_arrival(True))

    def test_failed_override_does_not_leave_a_waypoint_active(self):
        graph, _, _, c = self._chain()
        executor = PlaceRouteExecution()
        executor.start(plan_place_route(graph, c, "snapshot", "s"))
        executor.start(None)
        self.assertIsNone(executor.consume_waypoint_arrival(True))

    def test_snapshot_resume_keeps_selected_object_not_full_cluster(self):
        live = SimpleNamespace(image="s", cluster=[1, 2, 3])
        choice = restore_selected_snapshot(live, [2], {1: {}, 2: {}, 3: {}})
        self.assertEqual(choice.cluster, [2])
        self.assertEqual(live.cluster, [1, 2, 3])
        self.assertIsNone(restore_selected_snapshot(live, [2], {1: {}}))

    def _chain(self):
        graph = PlaceTopology(1.0, 0.1, stage="route_only")
        a = graph.observe_pose([0, 0], 0, "pose:0").place_id
        b = graph.observe_pose([11, 0], 1, "pose:1").place_id
        c = graph.observe_pose([22, 0], 2, "pose:2").place_id
        graph.current_place_id = a
        return graph, a, b, c

    def test_same_place_keeps_original_hgr_terminal_segment(self):
        graph, a, _, _ = self._chain()
        plan = plan_place_route(graph, a, "snapshot", "s0", [1, 0])
        self.assertEqual(plan.route_mode, PlaceRouteMode.SAME_PLACE.value)
        self.assertFalse(plan.uses_override)
        self.assertEqual(plan.route, [a])

    def test_cross_place_returns_first_valid_next_hop(self):
        graph, a, b, c = self._chain()
        plan = plan_place_route(graph, c, "snapshot", "s0", [24, 0])
        self.assertEqual(plan.route_mode, PlaceRouteMode.NEXT_HOP.value)
        self.assertEqual(plan.route, [a, b, c])
        self.assertEqual(plan.next_place_id, b)
        np.testing.assert_array_equal(plan.target_voxel, graph.nodes[b].position)
        np.testing.assert_array_equal(plan.look_at_voxel, graph.nodes[c].position)

    def test_cross_place_executes_edge_entry_anchor_not_place_center(self):
        graph, a, b, c = self._chain()
        graph.add_traversable_edge(
            a, b, 1.1, "traversed_trajectory", 3,
            source_anchor=[1, 0], target_anchor=[9, 0],
        )
        graph.add_traversable_edge(
            b, c, 1.1, "traversed_trajectory", 3,
            source_anchor=[12, 0], target_anchor=[20, 0],
        )
        plan = plan_place_route(graph, c, "snapshot", "s0", [24, 0])
        np.testing.assert_array_equal(plan.target_voxel, [9, 0])
        np.testing.assert_array_equal(plan.look_at_voxel, [20, 0])

    def test_invalid_edge_makes_route_explicitly_unavailable(self):
        graph, a, b, c = self._chain()
        graph.invalidate_edge(b, c, 3)
        plan = plan_place_route(graph, c, "frontier", "f0", [22, 0])
        self.assertEqual(plan.route_mode, PlaceRouteMode.DISCONNECTED.value)
        self.assertFalse(plan.route_valid)
        self.assertEqual(plan.route, [])

    def test_known_free_revalidation_restores_an_invalid_route(self):
        graph, a, b, c = self._chain()
        graph.invalidate_edge(b, c, 3)
        graph.current_place_id = b
        graph.add_known_free_connections({c: 1.1}, 4, "pose:4")
        graph.current_place_id = a
        plan = plan_place_route(graph, c, "frontier", "f0", [22, 0])
        self.assertEqual(plan.route_mode, PlaceRouteMode.NEXT_HOP.value)
        self.assertEqual(plan.route, [a, b, c])

    def test_missing_source_mapping_is_not_guessed(self):
        graph, _, _, _ = self._chain()
        plan = plan_place_route(graph, None, "snapshot", "missing")
        self.assertEqual(plan.route_mode, PlaceRouteMode.UNMAPPED_TARGET.value)
        self.assertFalse(plan.uses_override)

    def test_directed_edges_are_respected(self):
        graph, _, _, c = self._chain()
        # The actual trajectory established A->B->C, not the reverse route.
        graph.current_place_id = c
        plan = plan_place_route(graph, "place_0", "snapshot", "s0")
        self.assertEqual(plan.route_mode, PlaceRouteMode.DISCONNECTED.value)

    def test_frontier_approach_minimizes_route_plus_terminal_cost(self):
        graph, a, b, c = self._chain()
        # The historical C approach has a shorter terminal segment, but from
        # current A its total route is longer than approaching from B.
        chosen = select_reachable_approach_place(
            graph, {c: 0.1, b: 0.4}
        )
        self.assertEqual(chosen, b)


if __name__ == "__main__":
    unittest.main()
