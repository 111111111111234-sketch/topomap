import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.place_goal_navigation import PlaceGoalNavigation
from src.goal_observation_coverage import ObservationCoverage
from src.query_place_goal import validated_request, verify_place_goal
from src.place_topology import PlaceTopology
from tests.test_place_goal_navigation import candidate


class CoverageIntegrationTest(unittest.TestCase):
    def test_main_rebind_preserves_checked_identity(self):
        state = PlaceGoalNavigation("g", 1., evidence_coverage=True)
        item = candidate()
        state.goal_topology.mark_checked(item, "verification_uncertain", 1)
        state.coverage.observe_source("object:1", [1,2], [10,2])
        version = state.coverage.version("object:1", [10,2], np.ones((20,20),bool), 2)
        state.coverage.finish(item,"verification_uncertain",version,1)
        scene = SimpleNamespace(objects={2:{}},object_id_aliases={1:2})
        state._rebind_evidence(scene)
        self.assertTrue(state.goal_topology.is_checked("Verify","object:2","place_1"))
        self.assertIn(("object:2",4),state.coverage.records)

    def test_semantic_check_requires_new_source_not_geometry(self):
        coverage = ObservationCoverage(1., .5)
        mask = np.ones((20,20),bool)
        item = candidate()
        coverage.observe_source(item.entity_id,[1,2],item.look_at)
        version = coverage.version(item.entity_id,item.look_at,mask,2)
        coverage.finish(item,"verification_uncertain",version,1)
        self.assertFalse(coverage.available(item.entity_id,[8,2.1],item.look_at,version))
        self.assertTrue(coverage.available(item.entity_id,[10,4],item.look_at,version))
        mask[19,19] = False
        self.assertEqual(version,coverage.version(item.entity_id,item.look_at,mask,2))
        mask[10,2] = False
        changed = coverage.version(item.entity_id,item.look_at,mask,2)
        self.assertFalse(coverage.available(item.entity_id,item.terminal,item.look_at,changed))
        coverage.observe_source(item.entity_id,[10,5],item.look_at)
        changed = coverage.version(item.entity_id,item.look_at,mask,2)
        self.assertTrue(coverage.available(item.entity_id,item.terminal,item.look_at,changed))
        coverage.finish(item,"verification_rejected",changed,2)
        coverage.observe_source(item.entity_id,[12,5],item.look_at)
        changed = coverage.version(item.entity_id,item.look_at,mask,2)
        self.assertFalse(coverage.available(item.entity_id,item.terminal,item.look_at,changed))
        self.assertTrue(coverage.available(item.entity_id,[10,4],item.look_at,changed))
        self.assertEqual(coverage.records[(item.entity_id,4)]["semantic_state"],"contradicted_view")

    def test_geometry_restores_execution_failure_without_semantic_record(self):
        coverage = ObservationCoverage(1., .5)
        item = candidate()
        mask = np.ones((20,20),bool)
        version = coverage.version(item.entity_id,item.look_at,mask,2)
        coverage.finish(item,"local_execution_failed",version,1)
        self.assertFalse(coverage.available(item.entity_id,item.terminal,item.look_at,version))
        mask[10,2] = False
        changed = coverage.version(item.entity_id,item.look_at,mask,2)
        self.assertTrue(coverage.available(item.entity_id,item.terminal,item.look_at,changed))
        self.assertFalse(coverage.records)

    def test_execution_error_is_not_observation(self):
        state = PlaceGoalNavigation("g",1.,evidence_coverage=True)
        state.active = candidate()
        state.active_coverage_version = ((),(),(),"v")
        state.finish("local_execution_failed",1)
        self.assertFalse(state.coverage.records)
        self.assertTrue(state.coverage.failures)
        self.assertFalse(state.goal_topology.checked)
        self.assertFalse(state.checked_views)

    def test_candidate_build_exhausts_verify_and_retains_explore_after_geometry_change(self):
        graph = PlaceTopology(2.,1.,stage="route_only")
        a = graph.observe_pose([2,3],0).place_id
        graph.bind_observation("snapshot:old.png",a,[2,3],0)
        snapshot = SimpleNamespace(image="old.png",cluster=[1],obs_point=np.array([2,3]))
        scene = SimpleNamespace(objects={1:{"bbox":SimpleNamespace(center=np.array([4,3]))}},
                                snapshots={"old.png":snapshot},object_id_aliases={})
        planner = SimpleNamespace(island=np.ones((9,9),bool),frontiers=[],habitat2voxel=lambda x:x)
        state = PlaceGoalNavigation("g",1.,1.,topological_actions=True,evidence_coverage=True)
        first = state.build_candidates(graph,scene,planner,[2,3])[0]
        state.select(first,snapshot,.8,0)
        state.finish("verification_uncertain",1,first.terminal,first.look_at)
        second = state.build_candidates(graph,scene,planner,[2,3])[0]
        self.assertNotEqual(state.coverage.bearing(first.terminal,first.look_at),
                            state.coverage.bearing(second.terminal,second.look_at))
        self.assertEqual(first.place_id,second.place_id)
        revision = state.goal_topology.revision
        planner.island[4,3] = False
        third = state.build_candidates(graph,scene,planner,[2,3])[0]
        self.assertEqual(second.terminal,third.terminal)
        self.assertEqual(revision,state.goal_topology.revision)
        graph.observe_pose([8,8],9)
        graph.current_place_id = a
        state.build_candidates(graph,scene,planner,[2,3])
        self.assertEqual(revision,state.goal_topology.revision)
        graph.observe_frontier("f1", [2,4], a, 2, 0, 1.)
        planner.frontiers = [SimpleNamespace(topo_id="f1",position=np.array([2,4]),
                                            orientation=np.array([0,1]),image="f.png")]
        for step in range(8):
            items = state.build_candidates(graph,scene,planner,[2,3])
            verifies = [c for c in items if c.intent == "Verify"]
            if not verifies:
                break
            state.select(verifies[0],snapshot,.8,step+2)
            state.finish("verification_uncertain",step+3,verifies[0].terminal,verifies[0].look_at)
        items = state.build_candidates(graph,scene,planner,[2,3])
        self.assertEqual([c.intent for c in items],["Explore"])
        revision = state.goal_topology.revision
        planner.island[4,3] = True
        items = state.build_candidates(graph,scene,planner,[2,3])
        self.assertEqual([c.intent for c in items],["Explore"])
        self.assertEqual(revision,state.goal_topology.revision)

    def test_malformed_selection_is_retried_and_transport_error_not_multiplied(self):
        cfg = {"place_goal_navigation":{"evidence_coverage":True}}
        with patch("src.query_place_goal._request",side_effect=["bad",'{"candidate_id":"v","support":0.8}']) as request:
            result, diagnostic = validated_request("prompt",[],cfg,"select",{"v"})
            self.assertEqual(result["candidate_id"],"v")
            self.assertEqual(request.call_count,2)
            self.assertEqual(diagnostic["parse_failures"][0]["stage"],"parse")
        with patch("src.query_place_goal._goal_content",return_value=[]), patch("src.query_place_goal._request",return_value=None) as request:
            verdict, diagnostic = verify_place_goal({},np.zeros((8,8,3),np.uint8),"crop",cfg)
            self.assertEqual(verdict,"request_failed")
            self.assertEqual(diagnostic["action"],"error")
            self.assertEqual(request.call_count,1)


if __name__ == "__main__":
    unittest.main()
