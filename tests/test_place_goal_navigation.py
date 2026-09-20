import unittest
from types import SimpleNamespace

import numpy as np

from src.place_topology import PlaceTopology
from src.place_goal_navigation import GoalCandidate, PlaceGoalNavigation, invalidate_blocked_hop
from src.topology_navigation import PlaceRouteExecution, plan_place_route


def candidate(entity="object:1", pose=(1., 2.)):
    return GoalCandidate("verify:1@place_1", "Verify", entity, "old.png", 1, None,
                         "place_1", "snapshot:old.png", list(pose), [8., 2.],
                         [10., 2.], ["place_0", "place_1"], 5., 1., 2.,
                         evidence_place_id="place_0")


class PlaceGoalNavigationTest(unittest.TestCase):
    def test_explore_has_real_source_and_checked_frontier_is_not_repeated(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([2, 3], 0, "a").place_id
        graph.observe_frontier("f1", [4, 3], a, 0, 0, 2.)
        frontier = SimpleNamespace(topo_id="f1", position=np.array([4, 3]),
                                   orientation=np.array([1, 0]), image="f.png")
        planner = SimpleNamespace(island=np.ones((8, 8), bool), frontiers=[frontier])
        scene = SimpleNamespace(objects={}, snapshots={}, object_id_aliases={})
        state = PlaceGoalNavigation("g", 1., 1., topological_actions=True)
        c = state.build_candidates(graph, scene, planner, [2, 3])[0]
        self.assertEqual(c.intent, "Explore")
        self.assertEqual(c.cost_m, 2.)
        state.select(c, frontier, .4, 0)
        self.assertIs(state.resume(scene, planner)[0], frontier)
        state.finish("frontier_observed", 1, c.terminal, c.look_at)
        self.assertFalse(state.build_candidates(graph, scene, planner, [4, 3]))
        # A new goal has no inherited "already searched" assertion.
        self.assertTrue(PlaceGoalNavigation("next", 1., 1.).build_candidates(graph, scene, planner, [4, 3]))

    def test_only_observed_obstacle_invalidates_an_edge(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([1, 3], 0, "a").place_id
        b = graph.observe_pose([8, 3], 1, "b").place_id
        graph.current_place_id = a
        plan = plan_place_route(graph, b, "snapshot", "s")
        occupied = np.zeros((12, 8), bool)
        self.assertFalse(invalidate_blocked_hop(graph, plan, occupied, 2))
        occupied[tuple(plan.target_voxel.astype(int))] = True
        self.assertTrue(invalidate_blocked_hop(graph, plan, occupied, 2))
        self.assertIsNone(graph.shortest_path(a, b)[1])

    def test_same_and_nearby_view_support_does_not_accumulate(self):
        state = PlaceGoalNavigation("g", .1)
        for i in range(100):
            state.record_support(candidate(pose=(1. + i % 2, 2.)), .7, i)
        self.assertEqual(len(state.evidence), 1)
        self.assertEqual(state.support("object:1"), .7)
        self.assertEqual(next(iter(state.evidence.values()))["place_id"], "place_0")
        state.record_support(candidate(pose=(20., 20.)), .6, 101)
        self.assertEqual(state.support("object:1"), .7)

    def test_goal_switch_has_no_inherited_support_or_negative_state(self):
        old = PlaceGoalNavigation("old", .1)
        c = candidate()
        old.record_support(c, .8, 0)
        old.active = c
        old.finish("verification_uncertain", 1, c.terminal, c.look_at)
        fresh = PlaceGoalNavigation("new", .1)
        self.assertEqual(fresh.trace()["evidence"], [])
        self.assertEqual(fresh.trace()["feedback"], [])
        self.assertFalse(fresh.checked_views)
        self.assertIsNone(fresh.active)

    def test_uncertain_verification_checks_view_without_rejecting_place(self):
        state = PlaceGoalNavigation("g", .1)
        c = candidate()
        state.record_support(c, .8, 0)
        state.active = c
        feedback = state.finish("verification_uncertain", 1, c.terminal, c.look_at)
        self.assertEqual(feedback["outcome"], "verification_uncertain")
        self.assertEqual(state.support(c.entity_id), .8)
        self.assertIn(state.view_key(c.entity_id, c.terminal, c.look_at), state.checked_views)
        self.assertNotIn(state.view_key("object:2", c.terminal, c.look_at), state.checked_views)

    def test_cost_includes_known_connectors_and_disconnected_is_excluded(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([1, 3], 0, "a").place_id
        b = graph.observe_pose([8, 3], 1, "b").place_id
        graph.current_place_id = a
        state = PlaceGoalNavigation("g", 1., 1., topological_actions=True)
        best = state._route_to_terminal(graph, np.ones((12, 8), bool), [2, 3], [[9, 3]],
                                        [10, 3], "object:1", [b], "Verify")
        self.assertIsNotNone(best)
        _, _, route, cost, connector, terminal, _ = best
        self.assertEqual(route, [a, b])
        self.assertEqual(connector, 1.)
        self.assertEqual(terminal, 1.)
        self.assertEqual(cost, 7.)
        graph.invalidate_edge(a, b, 3)
        self.assertIsNone(state._route_to_terminal(graph, np.ones((12, 8), bool), [2, 3],
                                                 [[9, 3]], [10, 3], "object:1", [b], "Verify"))

    def test_unknown_space_cannot_supply_a_terminal_path(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([1, 3], 0, "a").place_id
        mask = np.ones((10, 8), bool)
        mask[4, :] = False
        state = PlaceGoalNavigation("g", 1., 1.)
        self.assertIsNone(state._route_to_terminal(graph, mask, [1, 3], [[6, 3]],
                                                 [7, 3], "object:1", [a], "Verify"))

    def test_long_grid_connector_does_not_bypass_topology(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([1, 3], 0, "a").place_id
        state = PlaceGoalNavigation("g", 1., 1.)
        self.assertIsNone(state._route_to_terminal(graph, np.ones((20, 8), bool), [1, 3],
                                                 [[15, 3]], [16, 3], "object:1", [a], "Verify"))

    def test_build_deduplicates_objects_and_keeps_real_safe_point(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([2, 3], 0, "a").place_id
        graph.bind_observation("snapshot:old.png", a, [2, 3], 0)
        snapshot = SimpleNamespace(image="old.png", cluster=[1, 1], obs_point=np.array([2, 3]))
        scene = SimpleNamespace(objects={1: {"bbox": SimpleNamespace(center=np.array([4, 3]))}},
                                snapshots={"old.png": snapshot}, object_id_aliases={})
        planner = SimpleNamespace(island=np.ones((8, 8), bool), habitat2voxel=lambda x: x, frontiers=[])
        state = PlaceGoalNavigation("g", 1., 1., topological_actions=True)
        choices = state.build_candidates(graph, scene, planner, [2, 3])
        self.assertEqual(len(choices), 1)
        c = choices[0]
        self.assertEqual(c.intent, "Verify")
        self.assertGreater(np.linalg.norm(np.asarray(c.terminal) - [4, 3]), 0.)
        self.assertTrue(planner.island[tuple(c.terminal)])
        state.active = c
        state.finish("verification_uncertain", 1, c.terminal, c.look_at)
        self.assertFalse(state.build_candidates(graph, scene, planner, [2, 3]))
        self.assertEqual(
            state.goal_topology.places[a].search_state,
            "exhausted_for_current_topology",
        )
        b = graph.observe_pose([7, 3], 2, "b").place_id
        graph.current_place_id = a
        alternate = state.build_candidates(graph, scene, planner, [2, 3])[0]
        self.assertEqual(alternate.entity_id, c.entity_id)
        self.assertEqual(alternate.place_id, b)
        self.assertNotEqual(alternate.candidate_id, c.candidate_id)

    def test_r3_legacy_mode_keeps_physical_view_behavior(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        place_id = graph.observe_pose([2, 3], 0, "a").place_id
        graph.bind_observation("snapshot:old.png", place_id, [2, 3], 0)
        snapshot = SimpleNamespace(image="old.png", cluster=[1], obs_point=np.array([2, 3]))
        scene = SimpleNamespace(objects={1: {"bbox": SimpleNamespace(center=np.array([4, 3]))}},
                                snapshots={"old.png": snapshot}, object_id_aliases={})
        planner = SimpleNamespace(island=np.ones((8, 8), bool),
                                  habitat2voxel=lambda x: x, frontiers=[])
        state = PlaceGoalNavigation("g", 1., 1.)
        first = state.build_candidates(graph, scene, planner, [2, 3])[0]
        self.assertEqual(first.candidate_id, "verify:1")
        state.active = first
        state.finish("verification_uncertain", 1, first.terminal, first.look_at)
        alternate = state.build_candidates(graph, scene, planner, [2, 3])[0]
        self.assertEqual(alternate.place_id, first.place_id)
        self.assertNotEqual(state.view_key(first.entity_id, first.terminal, first.look_at),
                            state.view_key(alternate.entity_id, alternate.terminal, alternate.look_at))

    def test_path_cache_preserves_candidates_and_expires_between_builds(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([2, 3], 0, "a").place_id
        for i in range(3):
            graph.observe_frontier(f"f{i}", [4, 3], a, 0, 0, 2.)
        planner = SimpleNamespace(island=np.ones((8, 8), bool), frontiers=[
            SimpleNamespace(topo_id=f"f{i}", position=np.array([4, 3]),
                            orientation=np.array([1, 0]), image="f.png") for i in range(3)])
        scene = SimpleNamespace(objects={}, snapshots={}, object_id_aliases={})
        cached = PlaceGoalNavigation("g", 1., 1., cache_paths=True)
        plain = PlaceGoalNavigation("g", 1., 1., cache_paths=False)
        def build(state):
            return [c.to_trace_dict() for c in state.build_candidates(graph, scene, planner, [2, 3])]
        self.assertEqual(build(cached), build(plain))
        self.assertGreater(cached.planning_metrics["cache_hits"], 0)
        self.assertEqual(plain.planning_metrics["cache_hits"], 0)
        planner.island[3, :] = False
        self.assertEqual(build(cached), build(plain))
        self.assertFalse(cached.candidates)

    def test_multihop_then_uncertain_then_new_goal_lifecycle(self):
        graph = PlaceTopology(1., .1, stage="route_only")
        a = graph.observe_pose([0, 0], 0, "a").place_id
        b = graph.observe_pose([11, 0], 1, "b").place_id
        c = graph.observe_pose([22, 0], 2, "c").place_id
        graph.current_place_id = a
        state = PlaceGoalNavigation("image-goal", .1)
        target = candidate()
        target.place_id = c
        snapshot = SimpleNamespace(image="old.png", cluster=[1], obs_point=np.array([22, 0]), full_obj_list={1: .9})
        state.select(target, snapshot, .8, 0)
        executor = PlaceRouteExecution()
        scene = SimpleNamespace(objects={2: {}}, object_id_aliases={1: 2}, snapshots={}, frames={})
        for nxt in (b, c):
            choice, _ = state.resume(scene, SimpleNamespace())
            self.assertEqual(choice.cluster, [2])
            plan = plan_place_route(graph, c, "snapshot", choice.image, reached_place_id=executor.reached_place_id)
            self.assertEqual(plan.next_place_id, nxt)
            executor.start(plan)
            self.assertIsNotNone(executor.consume_waypoint_arrival(True))
            self.assertEqual(state.active.status, "navigating")
        terminal = plan_place_route(graph, c, "snapshot", choice.image, reached_place_id=executor.reached_place_id)
        self.assertEqual(terminal.route_mode, "same_place")
        self.assertFalse(state.feedback)  # Waypoints did not confirm the goal.
        state.finish("verification_uncertain", 3, target.terminal, target.look_at)
        self.assertIsNone(state.active)
        self.assertEqual(state.support("object:2"), .8)
        self.assertFalse(PlaceGoalNavigation("next-goal", .1).checked_views)


if __name__ == "__main__":
    unittest.main()
