import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.object_approach import resolve_object_approach, resolve_terminal_route
from src.place_topology import PlaceTopology


class ObjectApproachTest(unittest.TestCase):
    def setUp(self):
        self.graph = PlaceTopology(2., 1., stage="route_only")
        self.a = self.graph.observe_pose([2, 3], 0).place_id
        self.b = self.graph.observe_pose([9, 3], 1).place_id
        self.graph.current_place_id = self.a
        self.planner = SimpleNamespace(island=np.ones((15,15),bool),
            unoccupied=np.ones((15,15),bool), occupied=np.zeros((15,15),bool), habitat2voxel=lambda x:x)
        self.objects = {1:{"bbox":SimpleNamespace(center=np.array([4,3]))}}

    def test_object_terminal_does_not_visit_remote_capture_place(self):
        self.graph.bind_observation("snapshot:old",self.b,[9,3],1)
        with patch("src.object_approach.get_proper_observe_point",return_value=np.array([3,3])):
            result = resolve_object_approach(self.graph,self.planner,self.objects,1,[2,3],1.)
        self.assertEqual(result["approach_place_id"],self.a)
        self.assertEqual(result["terminal"],[3,3])
        self.assertEqual(result["total_distance_m"],1.)
        self.assertEqual(result["route_places"],[self.a])

    def test_multihop_cost_accounts_for_actual_path_and_final_segment(self):
        result=resolve_terminal_route(self.graph,self.planner.island,[2,3],[10,3],1.)
        self.assertEqual(result["route_places"],[self.a,self.b])
        path=np.asarray(result["certified_path"])
        self.assertAlmostEqual(result["total_distance_m"],np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        self.assertAlmostEqual(result["total_distance_m"],sum(result["segment_distances_m"]))
        resumed=resolve_terminal_route(self.graph,self.planner.island,[9,3],[10,3],1.,origin=self.b)
        self.assertEqual(resumed["route_places"],[self.b])
        self.assertEqual(resumed["total_distance_m"],1.)

    def test_wall_and_invalid_edge_cannot_create_reachable_approach(self):
        mask=self.planner.island.copy(); mask[6,:]=False
        self.assertEqual(resolve_terminal_route(self.graph,mask,[2,3],[10,3],1.)["status"],"unavailable")
        self.graph.invalidate_edge(self.a,self.b,2)
        self.assertEqual(resolve_terminal_route(self.graph,self.planner.island,[2,3],[10,3],1.)["status"],"unavailable")

    def test_terminal_persists_across_resume_but_geometry_is_revalidated(self):
        with patch("src.object_approach.get_proper_observe_point",return_value=np.array([3,3])) as resolve:
            previous=resolve_object_approach(self.graph,self.planner,self.objects,1,[2,3],1.)
            result=resolve_object_approach(self.graph,self.planner,self.objects,1,[2,4],1.,previous)
            self.assertEqual(resolve.call_count,1)
            self.assertEqual(result["terminal"],previous["terminal"])
            self.planner.occupied[3,3]=True
            result=resolve_object_approach(self.graph,self.planner,self.objects,1,[2,4],1.,previous)
            self.assertEqual(result["status"],"unavailable")

    def test_missing_object_is_explicit(self):
        self.assertEqual(resolve_object_approach(self.graph,self.planner,{},1,[2,3],1.)["reason"],"object_missing")

    def test_identity_rebind_preserves_terminal_without_capture_lookup(self):
        with patch("src.object_approach.get_proper_observe_point",return_value=np.array([3,3])):
            previous=resolve_object_approach(self.graph,self.planner,self.objects,1,[2,3],1.)
        with patch("src.object_approach.get_proper_observe_point",side_effect=AssertionError("must retain terminal")):
            result=resolve_object_approach(self.graph,self.planner,{2:self.objects[1]},2,[2,4],1.,previous)
        self.assertEqual(result["object_id"],2)
        self.assertEqual(result["terminal"],previous["terminal"])

    def test_real_explore_prompt_rebinds_crop_labels_after_filtering(self):
        from omegaconf import OmegaConf
        from src.eval_utils_gpt_goatbench import explore_step
        import json
        context={"object_approach":True,"current_place_id":"p", "frontiers":[],
                 "snapshots":{"image":{"objects":[{"object_id":11}, {"object_id":22,"total_distance_m":3.}]}}}
        cfg=OmegaConf.create({"prefiltering":True,"top_k_categories":5})
        info=("find",None,[],[],{"image":"pixels"},{"image":["chair"]},
              {"image":["crop"]},[4],{"image":[1]},None)
        with patch("src.eval_utils_gpt_goatbench.get_step_info",return_value=info), \
             patch("src.eval_utils_gpt_goatbench.call_openai_api",return_value="snapshot 0, object 0") as request:
            result=explore_step({"hgr_topology_context":context},cfg)
        text=next(c[0] for c in request.call_args.args[1] if c[0].startswith("Navigation topology"))
        rows=json.loads(text.split("\n",1)[1])
        selected = rows["candidates"]["Snapshot 0"]["objects"]["Object 0"]
        self.assertEqual(selected["source_object_id"],22)
        self.assertEqual(selected["route_status"],"unknown")
        self.assertNotIn("total_distance_m", text)
        self.assertEqual(result[1],[4])
        self.assertEqual(result[2],{"image":[1]})

    def test_v2_config_changes_only_approach_flag_and_output(self):
        from pathlib import Path
        from run_goatbench_evaluation import _load_config
        root=Path(__file__).resolve().parents[1]/"cfg"
        old=_load_config(str(root/"eval_goatbench_hgr_topology_fusion_smoke.yaml"))
        new=_load_config(str(root/"eval_goatbench_hgr_dual_topo_v2_a_smoke.yaml"))
        self.assertTrue(new.hgr_topology_fusion.object_approach)
        for key in old:
            if key not in ("exp_name","hgr_topology_fusion"):
                self.assertEqual(old[key],new[key],key)
