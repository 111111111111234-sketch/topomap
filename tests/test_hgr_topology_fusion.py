import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.hgr_topology_context import build_hgr_topology_context, format_hgr_topology_context
from src.place_topology import PlaceTopology
from src.topology_navigation import plan_place_route
from src.eval_utils_gpt_goatbench import format_explore_prompt
from src.query_vlm_goatbench import query_vlm_for_response
from run_goatbench_evaluation import _load_config


class HGRTopologyFusionTest(unittest.TestCase):
    def setUp(self):
        self.graph = PlaceTopology(2., 1., stage="route_only")
        self.a = self.graph.observe_pose([0, 0], 0).place_id
        self.b = self.graph.observe_pose([4, 0], 1).place_id
        self.graph.current_place_id = self.a
        self.graph.bind_observation("snapshot:b.png", self.b, [4, 0], 1)
        self.graph.observe_frontier("f", [5, 0], self.b, 1, 0, 1.)
        self.snapshots = {"b.png": SimpleNamespace(image="b.png")}
        self.frontier = SimpleNamespace(topo_id="f", feature="pixels", position=[5, 0], semantic_dist=None)

    def test_context_and_executor_use_same_valid_route(self):
        context = build_hgr_topology_context(self.graph, self.snapshots, [self.frontier])
        row = context["snapshots"]["b.png"]
        self.assertEqual(row["route_places"], [self.a, self.b])
        plan = plan_place_route(self.graph, row["approach_place_id"], "snapshot", "b.png", [4, 0])
        self.assertTrue(plan.uses_override)
        self.assertEqual(plan.next_place_id, self.b)
        self.assertEqual(context["frontiers"][0]["approach_place_id"], self.b)
        self.graph.invalidate_edge(self.a, self.b, 2)
        context = build_hgr_topology_context(self.graph, self.snapshots, [self.frontier])
        self.assertEqual(context["snapshots"]["b.png"]["connectivity"], "no_known_route")
        self.assertIsNone(context["frontiers"][0]["graph_distance_m"])

    def test_prompt_labels_follow_filtered_images_and_budget(self):
        context = build_hgr_topology_context(self.graph, self.snapshots, [self.frontier, self.frontier])
        context["snapshots"]["removed.png"] = {"approach_place_id": "wrong"}
        rendered = format_hgr_topology_context(context, ["b.png"])
        rows = json.loads(rendered.split("\n", 1)[1])["candidates"]
        self.assertEqual(rows["Snapshot 0"]["approach_place_id"], self.b)
        self.assertNotIn("Snapshot 1", rows)
        _, content = format_explore_prompt("find", [], ["frontier"], {}, {}, {},
                                           hgr_topology_context=context)
        text = " ".join(str(item[0]) for item in content)
        self.assertIn('"Frontier 0"', text)
        self.assertNotIn('"Frontier 1"', text)

    def test_object_selection_hides_snapshot_cost_but_keeps_frontier_cost(self):
        context = {
            "object_approach": True,
            "shared_route": True,
            "current_place_id": self.a,
            "snapshots": {"b.png": {"objects": {"Object 0": {
                "status": "valid", "object_id": 9, "total_distance_m": 0.1,
                "route_places": [self.a, self.b]}}}},
            "frontiers": [{"connectivity": "known_route", "total_distance_m": 2.0}],
        }
        rendered = format_hgr_topology_context(context, ["b.png"])
        instruction, payload = rendered.split("\n", 1)
        rows = json.loads(payload)["candidates"]
        self.assertIn("hard visual-semantic constraint", instruction)
        self.assertNotIn("total_distance_m", rows["Snapshot 0"]["objects"]["Object 0"])
        self.assertEqual(rows["Snapshot 0"]["objects"]["Object 0"]["source_object_id"], 9)
        self.assertEqual(rows["Frontier 0"]["total_distance_m"], 2.0)

    def test_original_query_uses_context_and_keeps_source_and_request_count(self):
        scene = SimpleNamespace(objects={}, snapshots={}, all_observations={})
        planner = SimpleNamespace(frontiers=[self.frontier])
        cfg = SimpleNamespace(use_full_obj_list=False, egocentric_views=False,
                              get=lambda key, default=None: default)
        for task_type in ("object", "description", "image"):
            with self.subTest(task_type=task_type), patch("src.query_vlm_goatbench.explore_step") as request:
                request.return_value = ("frontier 0", [], {}, "explore", 0)
                selected, _ = query_vlm_for_response(
                    {"question": "find", "task_type": task_type, "class": "target", "image": None},
                    scene, planner, [], cfg, place_topology=self.graph)
                self.assertIs(selected, self.frontier)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(request.call_args.args[0]["hgr_topology_context"]["frontiers"][0]
                                 ["route_places"], [self.a, self.b])

    def test_fusion_adds_only_text_to_original_visual_prompt(self):
        args = ("find", ["current_pixels"], ["frontier_pixels"],
                {"b.png": "snapshot_pixels"}, {"b.png": ["chair"]},
                {"b.png": ["crop_pixels"]})
        context = build_hgr_topology_context(self.graph, self.snapshots, [self.frontier])
        baseline_system, baseline = format_explore_prompt(*args, egocentric_view=True)
        fused_system, fused = format_explore_prompt(*args, egocentric_view=True,
                                                  hgr_topology_context=context)
        self.assertEqual(baseline_system, fused_system)
        self.assertEqual([item for item in baseline if len(item) == 2],
                         [item for item in fused if len(item) == 2])
        self.assertEqual([item for item in fused if not item[0].startswith("Navigation topology")], baseline)

    def test_config_retains_hgr_budgets_and_disables_separate_verification(self):
        root = Path(__file__).resolve().parents[1] / "cfg"
        baseline = _load_config(str(root / "eval_goatbench_place_route_only_qwen3vl_dashscope_train_r5.yaml"))
        fusion = _load_config(str(root / "eval_goatbench_hgr_topology_fusion_train.yaml"))
        self.assertTrue(fusion.hgr_topology_fusion.enabled)
        self.assertFalse(fusion.place_goal_navigation.enabled)
        for key in baseline:
            if key != "exp_name":
                self.assertEqual(baseline[key], fusion[key], key)


if __name__ == "__main__":
    unittest.main()
