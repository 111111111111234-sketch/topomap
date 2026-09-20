import copy
import base64
from io import BytesIO
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from run_goatbench_evaluation import (
    _best_current_object_snapshot,
    _snapshot_choice_mapping_available,
)
from src.tsdf_planner import SnapShot

from src.persistent_belief_topology import (
    ActiveTopoCandidate,
    ActiveTopoState,
    ActiveTopoView,
    GoalContext,
)
from src.query_vlm_goatbench import query_vlm_for_response
from src.eval_utils_gpt_goatbench import (
    _format_active_topology_metadata,
    apply_visual_image_budget,
    encode_tensor2base64,
    get_step_info,
    format_explore_prompt,
)


class ActiveTopologyCompatibilityTest(unittest.TestCase):
    def test_current_exact_object_detection_builds_direct_snapshot(self):
        frame = SnapShot(
            image="12-view_0.png",
            color=(0.0, 0.0, 0.0),
            obs_point=np.zeros(3),
            full_obj_list={7: 0.8, 9: 0.95},
            cluster=[],
        )
        scene = SimpleNamespace(
            frames={frame.image: frame},
            objects={
                7: {"class_name": "chair", "conf": 0.8,
                    "num_detections": 2},
                9: {"class_name": "table", "conf": 0.95,
                    "num_detections": 4},
            },
        )
        choice = _best_current_object_snapshot(
            scene, [frame.image], "chair", min_detections=1
        )
        self.assertEqual(choice.image, frame.image)
        self.assertEqual(choice.cluster, [7])

    def test_current_object_detection_ignores_alias_and_old_frame(self):
        current = SnapShot(
            image="12-view_0.png", color=(0.0, 0.0, 0.0),
            obs_point=np.zeros(3), full_obj_list={7: 0.9}, cluster=[],
        )
        old = SnapShot(
            image="2-view_0.png", color=(0.0, 0.0, 0.0),
            obs_point=np.zeros(3), full_obj_list={8: 0.99}, cluster=[],
        )
        scene = SimpleNamespace(
            frames={current.image: current, old.image: old},
            objects={
                7: {"class_name": "couch", "conf": 0.9,
                    "num_detections": 2},
                8: {"class_name": "sofa", "conf": 0.99,
                    "num_detections": 3},
            },
        )
        self.assertIsNone(_best_current_object_snapshot(
            scene, [current.image], "sofa", min_detections=1
        ))

    def test_committed_snapshot_requires_live_image_and_object_mapping(self):
        choice = SnapShot(
            image="history.png",
            color=(0.0, 0.0, 0.0),
            obs_point=np.zeros(3),
            full_obj_list={7: 0.9},
            cluster=[7],
        )
        live_snapshot = SimpleNamespace(image="history.png")
        scene = SimpleNamespace(
            snapshots={"snapshot-key": live_snapshot},
            objects={7: {"class_name": "chair"}},
        )
        self.assertTrue(_snapshot_choice_mapping_available(choice, scene))
        scene.objects.clear()
        self.assertFalse(_snapshot_choice_mapping_available(choice, scene))
        scene.objects[7] = {"class_name": "chair"}
        scene.snapshots.clear()
        self.assertFalse(_snapshot_choice_mapping_available(choice, scene))

    def test_ca_prompt_formatter_omits_r_without_key_error(self):
        rendered = _format_active_topology_metadata({
            "state": "active", "confidence": 0.8,
            "accessibility": 0.7, "utility": 1.0,
        })
        self.assertNotIn("R=", rendered)
        self.assertIn("C=0.80", rendered)

    def test_tiny_extreme_crop_is_padded_to_valid_qwen_canvas(self):
        encoded = encode_tensor2base64(np.ones((100, 1, 3), dtype=np.uint8))
        decoded = Image.open(BytesIO(base64.b64decode(encoded)))
        self.assertEqual(decoded.size, (256, 256))

    def test_separate_image_budget_preserves_snapshot_and_crop_mapping(self):
        result = apply_visual_image_budget(
            max_images=7,
            has_image_goal=True,
            has_egocentric_view=True,
            frontier_imgs=["frontier-0", "frontier-1"],
            frontier_semantics=[{"id": 0}, {"id": 1}],
            snapshot_imgs={"rgb-a": "full-a", "rgb-b": "full-b"},
            snapshot_classes={"rgb-a": ["chair", "table"], "rgb-b": ["bed"]},
            snapshot_crops={"rgb-a": ["crop-a0", "crop-a1"], "rgb-b": ["crop-b0"]},
            snapshot_id_mapping=[3, 7],
            snapshot_crop_mapping={"rgb-a": [5, 9], "rgb-b": [2]},
        )
        (frontiers, semantics, snapshots, classes, crops,
         snapshot_mapping, crop_mapping) = result
        self.assertEqual(frontiers, ["frontier-0", "frontier-1"])
        self.assertEqual(semantics, [{"id": 0}, {"id": 1}])
        self.assertEqual(snapshots, {"rgb-a": "full-a"})
        self.assertEqual(classes, {"rgb-a": ["chair", "table"]})
        self.assertEqual(crops, {"rgb-a": ["crop-a0", "crop-a1"]})
        self.assertEqual(snapshot_mapping, [3])
        self.assertEqual(crop_mapping, {"rgb-a": [5, 9]})

    def _planner_and_scene(self):
        frontiers = [
            SimpleNamespace(position=np.array([0, 0]), feature=np.zeros((4, 4, 3), dtype=np.uint8), semantic_dist=None),
            SimpleNamespace(position=np.array([1, 0]), feature=np.ones((4, 4, 3), dtype=np.uint8), semantic_dist=None),
        ]
        planner = SimpleNamespace(frontiers=frontiers)
        scene = SimpleNamespace(objects={}, snapshots={}, all_observations={})
        return planner, scene, frontiers

    @patch("src.query_vlm_goatbench.explore_step")
    def test_projected_index_maps_to_original_frontier(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, frontiers = self._planner_and_scene()
        goal = GoalContext("g", "category", category="refrigerator")
        candidate = ActiveTopoCandidate(
            "frontier_1", 1, 0.9, 0.8, 0.7, 0.0, 0.0, 0.1, 1.2,
            ActiveTopoState.ACTIVE,
        )
        view = ActiveTopoView(goal, 0, [candidate], 0, [candidate])
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        selected, _ = query_vlm_for_response(
            {"question": "find refrigerator", "task_type": "object", "class": "refrigerator", "image": None},
            scene, planner, [], cfg, active_topo_view=view,
        )
        self.assertIs(selected, frontiers[1])

    @patch("src.query_vlm_goatbench.explore_step")
    def test_disabled_view_preserves_original_order(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, frontiers = self._planner_and_scene()
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        selected, _ = query_vlm_for_response(
            {"question": "find chair", "task_type": "object", "class": "chair", "image": None},
            scene, planner, [], cfg, active_topo_view=None,
        )
        self.assertIs(selected, frontiers[0])

    @patch("src.query_vlm_goatbench.explore_step")
    def test_frontier_allowlist_keeps_original_source_mapping(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, frontiers = self._planner_and_scene()
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        selected, _ = query_vlm_for_response(
            {"question": "find chair", "task_type": "object", "class": "chair", "image": None},
            scene, planner, [], cfg, allowed_frontier_indices=[1],
        )
        prompt = explore_step.call_args.args[0]
        self.assertEqual(len(prompt["frontier_imgs"]), 1)
        self.assertIs(selected, frontiers[1])

    @patch("src.query_vlm_goatbench.explore_step")
    def test_frontier_only_rejects_snapshot_action(self, explore_step):
        explore_step.return_value = ("snapshot 0, object 0", [], {}, "", 0)
        planner, scene, _ = self._planner_and_scene()
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        selected = query_vlm_for_response(
            {"question": "find chair", "task_type": "object", "class": "chair", "image": None},
            scene, planner, [], cfg, allowed_frontier_indices=[1],
            frontier_only=True,
        )
        self.assertIsNone(selected)
        self.assertTrue(explore_step.call_args.args[0]["frontier_only"])

    @patch("src.query_vlm_goatbench.explore_step")
    def test_rejected_snapshot_object_is_removed_before_prompt(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, _ = self._planner_and_scene()
        scene.objects = {7: {"class_name": "towel"}}
        scene.snapshots = {
            "rgb": SimpleNamespace(cluster=[7])
        }
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        query_vlm_for_response(
            {"question": "find rug", "task_type": "object", "class": "rug", "image": None},
            scene, planner, [], cfg, excluded_object_ids={7},
        )
        prompt = explore_step.call_args.args[0]
        self.assertEqual(prompt["snapshot_objects"], {})
        self.assertEqual(prompt["snapshot_imgs"], {})
        self.assertEqual(prompt["snapshot_source_indices"], [])

    def test_filtered_snapshot_mapping_retains_original_scene_index(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        step = {
            "question": "find rug", "task_type": "object",
            "class": "rug", "image": None,
            "frontier_imgs": [], "obj_map": {8: "rug"},
            "snapshot_source_indices": [1],
            "snapshot_objects": {"kept.png": [8]},
            "snapshot_imgs": {"kept.png": {
                "full_img": image,
                "object_crop": [{"obj_class": "rug", "obj_id": 8, "crop": image}],
            }},
            "use_prefiltering": False,
            "use_full_obj_list": False,
        }
        result = get_step_info(step)
        self.assertEqual(result[7], [1])
        self.assertEqual(result[8], {"kept.png": [0]})

    @patch("src.query_vlm_goatbench.explore_step")
    def test_filtered_snapshot_choice_maps_back_without_key_error(self, explore_step):
        class VisualPrompt:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                if isinstance(index, list):
                    return SimpleNamespace(
                        xyxy=np.asarray([[0, 0, 4, 4]], dtype=float)
                    )
                return SimpleNamespace(data={"obj_id": [8]})

        explore_step.return_value = (
            "snapshot 0, object 0", [1], {"kept.png": [0]}, "", 1
        )
        planner, scene, _ = self._planner_and_scene()
        scene.objects = {
            7: {"class_name": "towel", "conf": 0.9},
            8: {"class_name": "rug", "conf": 0.9},
        }
        scene.snapshots = {
            "rejected.png": SimpleNamespace(cluster=[7]),
            "kept.png": SimpleNamespace(
                cluster=[8], image="kept.png", visual_prompt=VisualPrompt(),
                obs_point=np.array([1., 2., 3.]),
                full_obj_list={8: 0.9},
            ),
        }
        scene.all_observations = {
            "kept.png": np.zeros((8, 8, 3), dtype=np.uint8)
        }
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            prompt_h=8, prompt_w=8,
            get=lambda key, default=None: default,
        )
        selected, _ = query_vlm_for_response(
            {"question": "find rug", "task_type": "object", "class": "rug", "image": None},
            scene, planner, [], cfg, excluded_object_ids={7},
        )
        self.assertEqual(selected.cluster, [8])

    @patch("src.query_vlm_goatbench.explore_step")
    def test_confidence_accessibility_prompt_omits_relevance(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, _ = self._planner_and_scene()
        goal = GoalContext("g", "category", category="chair")
        candidate = ActiveTopoCandidate(
            "frontier_0", 0, 0.5, 0.8, 0.7, 0.0, 0.0, 0.1, 1.0,
            ActiveTopoState.ACTIVE,
        )
        view = ActiveTopoView(
            goal, 0, [candidate], 0, [candidate], uses_goal_relevance=False
        )
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: default,
        )
        query_vlm_for_response(
            {"question": "find chair", "task_type": "object", "class": "chair", "image": None},
            scene, planner, [], cfg, active_topo_view=view,
        )
        metadata = explore_step.call_args.args[0]["frontier_semantic_predictions"][0][
            "active_topology"
        ]
        self.assertNotIn("relevance", metadata)
        self.assertEqual(metadata["confidence"], 0.8)
        self.assertEqual(metadata["accessibility"], 0.7)

    @patch("src.query_vlm_goatbench.explore_step")
    def test_simple_memory_filters_without_adding_prompt_metadata(self, explore_step):
        explore_step.return_value = ("frontier 0", [], {}, "", 0)
        planner, scene, frontiers = self._planner_and_scene()
        planner.rank_frontiers_by_exploration_score = lambda category, pos: [
            (0, 0.2), (1, 0.8)
        ]
        goal = GoalContext("g", "category", category="refrigerator")
        candidate = ActiveTopoCandidate(
            "frontier_1", 1, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.15,
            ActiveTopoState.ACTIVE,
        )
        view = ActiveTopoView(
            goal, 0, [candidate], 0, [candidate],
            uses_goal_relevance=False, include_prompt_metadata=False,
        )
        cfg = SimpleNamespace(
            use_full_obj_list=False, egocentric_views=False,
            get=lambda key, default=None: (
                {"enable_hypothesis_refinement": True}
                if key == "hypothesis" else default
            ),
        )
        selected, _ = query_vlm_for_response(
            {"question": "find refrigerator", "task_type": "object", "class": "refrigerator", "image": None},
            scene, planner, [], cfg, active_topo_view=view,
        )
        prompt = explore_step.call_args.args[0]
        self.assertIs(selected, frontiers[1])
        self.assertNotIn(
            "active_topology", prompt["frontier_semantic_predictions"][0] or {}
        )
        self.assertEqual(prompt["frontier_scores"], {0: 0.8})

    def test_goal_topology_context_is_rendered_without_extra_images(self):
        _, content = format_explore_prompt(
            question="find the chair",
            egocentric_imgs=[], frontier_imgs=[], snapshot_imgs={},
            snapshot_classes={}, snapshot_crops={},
            goal_topology_context={
                "global_nodes": 20, "global_edges": 31, "active_nodes": 7,
                "related_observations": ["chair", "table"],
            },
        )
        rendered = "".join(item[0] for item in content)
        self.assertIn("Persistent goal-conditioned topology", rendered)
        self.assertIn("20 nodes", rendered)
        self.assertIn("chair, table", rendered)


if __name__ == "__main__":
    unittest.main()
