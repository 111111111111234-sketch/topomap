import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.scene_goatbench import Scene
from src.place_goal_navigation import PlaceGoalNavigation
from tests.test_place_goal_navigation import candidate


class SceneIdentityInitializationTest(unittest.TestCase):
    def test_first_scene_and_reset_initialize_identity_and_phase_c_can_rebind(self):
        cfg = OmegaConf.create({
            "scene_data_path": "unused", "camera_height": 1.5,
            "img_width": 64, "img_height": 64, "hfov": 90,
            "scene_dataset_config_path": "unused", "camera_tilt_deg": -30,
            "seed": 77, "class_set": "scannet200",
        })
        # Run the actual constructor; stub only external simulator/model setup.
        with patch("src.scene_goatbench.os.path.exists", return_value=True), patch(
            "src.scene_goatbench.make_semantic_cfg"
        ), patch("src.scene_goatbench.habitat_sim.Simulator"), patch(
            "src.scene_goatbench.ObjectClasses"
        ):
            scene = Scene("00062-test", cfg, {"bg_classes": [], "skip_bg": True},
                          MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        self.assertEqual(scene.object_id_aliases, {})
        scene.objects[1] = {}
        state = PlaceGoalNavigation("goal", .1)
        state.record_support(candidate(), .8, 0)
        state._rebind_evidence(scene)  # The operation that crashed at step 1.
        self.assertEqual(state.support("object:1"), .8)
        scene.objects[2] = scene.objects.pop(1)
        scene.object_id_aliases[1] = 2
        state._rebind_evidence(scene)
        self.assertEqual(state.support("object:2"), .8)
        scene.clear_up_detections()
        self.assertEqual(scene.object_id_aliases, {})
        self.assertEqual(len(scene.objects), 0)
        self.assertEqual(scene.object_id_counter, 1)


if __name__ == "__main__":
    unittest.main()
