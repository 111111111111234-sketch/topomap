import unittest
from unittest.mock import patch

import numpy as np

from src.active_verification import (
    query_action_tiebreak,
    query_active_verify,
    sample_verification_viewpoint,
    viewpoint_novelty,
)
from src.goal_belief import (
    BeliefActionCandidate,
    BeliefActionType,
    EvidenceCluster,
    GoalBelief,
)
from src.tsdf_planner import Frontier, TSDFPlanner


class ActiveVerificationTest(unittest.TestCase):
    def test_ring_candidate_is_free_visible_and_reachable(self):
        occupied = np.zeros((50, 50), dtype=bool)
        unoccupied = np.ones((50, 50), dtype=bool)
        island = np.ones((50, 50), dtype=bool)

        def path_query(start, end):
            return float(np.linalg.norm(np.asarray(end) - np.asarray(start))), [start, end]

        option = sample_verification_viewpoint(
            center=np.array([25, 25]), current=np.array([20, 20]),
            occupied=occupied, unoccupied=unoccupied, island=island,
            voxel_size=0.5, path_query=path_query,
            radii_m=[1.0], samples_per_radius=16,
        )
        self.assertIsNotNone(option)
        radius = np.linalg.norm(option.target_point - option.look_at_point)
        self.assertAlmostEqual(radius, 2.0, delta=0.5)
        self.assertTrue(unoccupied[tuple(option.target_point)])

    def test_no_candidate_falls_back_cleanly(self):
        occupied = np.ones((20, 20), dtype=bool)
        option = sample_verification_viewpoint(
            center=np.array([10, 10]), current=np.array([5, 5]),
            occupied=occupied,
            unoccupied=np.zeros_like(occupied),
            island=np.ones_like(occupied),
            voxel_size=0.5,
            path_query=lambda start, end: (1.0, [start, end]),
            radii_m=[1.0], samples_per_radius=8,
        )
        self.assertIsNone(option)

    def test_navigation_override_keeps_execution_source_and_look_target(self):
        planner = TSDFPlanner.__new__(TSDFPlanner)
        planner.max_point = None
        planner.target_point = None
        planner.look_at_point = None
        planner.occupied = np.zeros((12, 12), dtype=bool)
        planner.unoccupied = np.ones((12, 12), dtype=bool)
        planner.island = np.ones((12, 12), dtype=bool)
        planner.normal2voxel = lambda point: np.asarray([0, 0, 0])
        planner.check_within_bnds = lambda point: (
            0 <= point[0] < 12 and 0 <= point[1] < 12
        )
        source = Frontier(
            np.array([8, 8]), np.array([1.0, 0.0]),
            np.zeros((12, 12), dtype=bool), 1,
        )
        success = planner.set_next_navigation_point(
            source, np.zeros(3), {}, None, None,
            target_point_override=np.array([3, 4]),
            look_at_point=np.array([8, 8]),
        )
        self.assertTrue(success)
        self.assertIs(planner.max_point, source)
        np.testing.assert_array_equal(planner.target_point, [3, 4])
        np.testing.assert_array_equal(planner.look_at_point, [8, 8])

    def test_viewpoint_novelty_penalizes_repeated_view(self):
        history = [EvidenceCluster(
            "c", "n", "g", np.array([4.0, 4.0]), 0.0, []
        )]
        repeated = viewpoint_novelty(
            np.array([4.0, 4.0]), 0.0, history, distance_scale_voxels=5.0
        )
        novel = viewpoint_novelty(
            np.array([10.0, 4.0]), np.pi, history, distance_scale_voxels=5.0
        )
        self.assertEqual(repeated, 0.0)
        self.assertEqual(novel, 1.0)

    @patch("src.active_verification.call_openai_api")
    def test_verify_is_single_image_one_attempt_and_parse_safe(self, call):
        call.return_value = (
            '{"target_present_probability": 0.7, "reason": "visible"}'
        )
        result = query_active_verify(
            np.zeros((8, 8, 3), dtype=np.uint8), "chair", "mock-model"
        )
        self.assertEqual(result["target_present_probability"], 0.7)
        args, kwargs = call.call_args
        self.assertEqual(len(args[1]), 1)
        self.assertEqual(kwargs["max_tries"], 1)
        self.assertEqual(kwargs["purpose"], "active_verify")

        call.return_value = "not json"
        self.assertIsNone(query_active_verify(
            np.zeros((8, 8, 3), dtype=np.uint8), "chair", "mock-model"
        ))

    @patch("src.active_verification.call_openai_api")
    def test_tiebreak_exposes_exactly_two_actions(self, call):
        belief = GoalBelief(
            "n", "category:chair", 0.6, 0.8, 0.7, 0.9,
            1, 0, 1, 0.0, {"clip_goal": 1}, False,
        )
        actions = [
            BeliefActionCandidate(
                BeliefActionType.EXPLORE, f"n{index}", "frontier", index,
                belief, 0.5 - index * 0.01, 0.2,
            )
            for index in range(2)
        ]
        call.return_value = '{"choice_index": 1, "reason": "clearer"}'
        choice = query_action_tiebreak(
            actions,
            [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
            "chair", "mock-model",
        )
        self.assertEqual(choice, 1)
        self.assertEqual(len(call.call_args.args[1]), 2)
        call.return_value = "bad"
        self.assertIsNone(query_action_tiebreak(
            actions,
            [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
            "chair", "mock-model",
        ))


if __name__ == "__main__":
    unittest.main()
