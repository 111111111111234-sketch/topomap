import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.place_topology import PlaceTopology
from src.route_guidance import RouteGuidance, grid_path, visible, structural_route_view
from src.tsdf_planner import TSDFPlanner


class RouteGuidanceTest(unittest.TestCase):
    def scene(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        a = graph.observe_pose([1, 2], 0).place_id
        b = graph.observe_pose([4, 2], 1).place_id
        c = graph.observe_pose([7, 2], 2).place_id
        graph.current_place_id = a
        return graph, np.ones((12, 8), bool), a, b, c

    def test_continuous_passes_portal_without_false_arrival(self):
        graph, mask, a, b, c = self.scene()
        guide = RouteGuidance.build(graph, mask, [1, 2], [8, 2], c)
        self.assertIsNotNone(guide)
        point, arrived = guide.step([1, 2], 5., 1., mask)
        self.assertEqual(point.tolist(), [6., 2.])
        self.assertFalse(arrived)
        self.assertEqual(guide.remaining_route()[0], b)
        point, arrived = guide.step(point, 5., 1., mask)
        self.assertEqual(point.tolist(), [8., 2.])
        self.assertTrue(arrived)

    def test_obstacle_and_unknown_block_path_and_shortcut(self):
        mask = np.ones((8, 8), bool)
        mask[4, :] = False
        self.assertIsNone(grid_path(mask, [1, 2], [6, 2]))
        self.assertFalse(visible(mask, [1, 2], [6, 2]))
        mask[4, 5] = True
        path = grid_path(mask, [1, 2], [6, 2])
        self.assertTrue(any(np.array_equal(p,[4,5]) for p in path))
        self.assertFalse(visible(mask, [1, 2], [6, 2]))
        corner = np.array([[True,False],[False,True]])
        self.assertFalse(visible(corner, [0,0], [1,1]))
        self.assertIsNone(grid_path(corner, [0,0], [1,1]))

    def test_invalid_edge_or_changed_mask_invalidates_guidance(self):
        graph, mask, a, b, c = self.scene()
        guide = RouteGuidance.build(graph, mask, [1,2], [8,2], c)
        self.assertTrue(guide.valid(graph, mask))
        mask[3,2] = False
        self.assertFalse(guide.valid(graph, mask))
        mask[3,2] = True
        graph.invalidate_edge(a,b,3)
        self.assertFalse(guide.valid(graph, mask))
        self.assertIsNone(RouteGuidance.build(graph, mask, [1,2], [8,2], c))

    def test_chain_view_preserves_all_nodes_and_direction(self):
        graph, mask, a, b, c = self.scene()
        self.assertEqual(structural_route_view(graph,[a,b,c]),[[a,b,c]])
        graph.observe_frontier("f",[4,3],b,3,0,1.)
        self.assertEqual(structural_route_view(graph,[a,b,c]),[[a,b],[b,c]])

    def test_blocked_anchor_reroutes_without_permanent_edge_mutation(self):
        graph, mask, a, b, c = self.scene()
        d = graph.observe_pose([4,6],3).place_id
        graph.add_known_free_connections({a:6.,c:6.},4,"observed")
        graph.current_place_id = a
        mask[4,2] = False
        guide = RouteGuidance.build(graph,mask,[1,2],[8,2],c)
        self.assertIsNotNone(guide)
        self.assertEqual(guide.route,[a,d,c])
        self.assertEqual(graph.edges[(a,b)].status.value,"valid")

    def test_turn_does_not_teleport_through_wall(self):
        mask = np.zeros((9,9),bool)
        mask[1:7,2] = True
        mask[6,2:7] = True
        path = grid_path(mask,[1,2],[6,6])
        guide = RouteGuidance(["a"],path,[],mask,np.array([6,6]))
        point,arrived = guide.step([1,2],20.,1.,mask)
        self.assertEqual(point.tolist(),[6.,2.])
        self.assertFalse(arrived)
        point,arrived = guide.step(point,20.,1.,mask)
        self.assertTrue(arrived)

    def test_real_agent_step_uses_certified_path_without_pathfinder(self):
        graph, mask, a, b, c = self.scene()
        guide = RouteGuidance.build(graph, mask, [1,2], [8,2], c)
        planner = TSDFPlanner.__new__(TSDFPlanner)
        planner.max_point = SimpleNamespace()
        planner.target_point, planner.look_at_point = np.array([8,2]), np.array([9,2])
        planner._voxel_size, planner._vol_origin = 1., np.zeros(3)
        planner.island = planner.unoccupied = mask
        planner.occupied = np.zeros_like(mask)
        planner._explore_vol_cpu = np.zeros((*mask.shape, 1))
        planner.normal2voxel = lambda _: np.array([1,2,0])
        planner.normal2habitat = lambda p: p
        planner.get_distance = Mock(side_effect=AssertionError("must not use full-scene pathfinder"))
        cfg = SimpleNamespace(max_dist_from_cur_phase_1=5., max_dist_from_cur_phase_2=5.,
                              surrounding_explored_radius=1.)
        result = planner.agent_step(np.zeros(3), 0., {}, {}, None, cfg,
                                    save_visualization=False, route_guidance=guide)
        self.assertEqual(result[2].tolist(), [6.,2.])
        self.assertFalse(result[-1])
        planner.get_distance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
