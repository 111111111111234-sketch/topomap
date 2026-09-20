import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.dual_dynamic_navigation.executor import KnownSpaceExecutor
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier


class KnownSpaceExecutionTests(unittest.TestCase):
    def planner(self, mask, current=(1, 1)):
        p = TSDFPlanner.__new__(TSDFPlanner)
        p._voxel_size, p._vol_origin = 1., np.zeros(3)
        p.island = p.unoccupied = np.asarray(mask, bool)
        p.occupied = np.zeros_like(mask, bool)
        p._explore_vol_cpu = np.zeros((*mask.shape, 1))
        p.habitat2voxel = lambda _: np.array([*current, 0.])
        p.normal2voxel = p.habitat2voxel
        p.normal2habitat = lambda v: v
        p.max_point = SimpleNamespace()
        p.target_point, p.look_at_point = np.array([6., 1.]), np.array([7., 1.])
        p.get_distance = Mock(side_effect=AssertionError("full Pathfinder forbidden"))
        p.check_within_bnds = lambda v: all(0 <= v[i] < mask.shape[i] for i in range(2))
        return p

    def cfg(self):
        return SimpleNamespace(max_dist_from_cur_phase_1=2., max_dist_from_cur_phase_2=2.,
                               surrounding_explored_radius=1., final_observe_distance=1.5)

    def test_actual_agent_step_and_terminal_arrival_never_query_pathfinder(self):
        mask = np.ones((8, 4), bool)
        p = self.planner(mask)
        executor = KnownSpaceExecutor()
        result = executor.step(p, np.zeros(3), 0., {}, {}, self.cfg())
        self.assertTrue(executor.last_audit["valid"])
        self.assertEqual(result[2].tolist(), [3., 1.])
        self.assertFalse(result[-1])
        p.habitat2voxel = p.normal2voxel = lambda _: np.array([5., 1., 0.])
        result = executor.step(p, np.zeros(3), 0., {}, {}, self.cfg())
        self.assertTrue(result[-1])
        self.assertIsNone(p.target_point)
        p.get_distance.assert_not_called()

    def test_unknown_barrier_blocks_without_fallback_or_stale_audit(self):
        mask = np.ones((8, 4), bool)
        p = self.planner(mask)
        executor = KnownSpaceExecutor()
        self.assertTrue(executor.prepare(p, np.zeros(3)))
        p.unoccupied = mask.copy()
        p.unoccupied[4, :] = False
        result = executor.step(p, np.zeros(3), 0., {}, {}, self.cfg())
        self.assertIsNone(result[0])
        self.assertEqual(executor.last_audit["reason"], "no_known_path")
        self.assertNotIn("actual_segment", executor.last_audit)
        p.get_distance.assert_not_called()

    def test_noninteger_pose_requires_known_connector(self):
        p = self.planner(np.ones((8, 4), bool), current=(1.2, 1.1))
        executor = KnownSpaceExecutor()
        result = executor.step(p, np.zeros(3), 0., {}, {}, self.cfg())
        self.assertIsNotNone(result[0])
        self.assertEqual(executor.last_audit["actual_segment"][0], [1.2, 1.1])

    def test_snapshot_setup_cannot_use_simulator_observation_fallback(self):
        p = self.planner(np.ones((8, 4), bool))
        p.max_point = p.target_point = p.look_at_point = None
        choice = SnapShot(image="a.png", color=(1., 0., 0.), obs_point=np.array([1, 1]),
                          position=np.array([4, 1]), cluster=[7])
        objects = {7: {"bbox": SimpleNamespace(center=np.array([4., 1., 0.]))}}
        with patch("src.tsdf_planner.get_proper_observe_point", return_value=None), \
             patch("src.tsdf_planner.get_proper_observe_point_with_pathfinder",
                   side_effect=AssertionError("simulator fallback forbidden")):
            ok = p.set_next_navigation_point(choice, np.zeros(3), objects, self.cfg(),
                                             None, known_space_only=True)
        self.assertFalse(ok)
        self.assertIsNone(p.max_point)


if __name__ == "__main__":
    unittest.main()
