import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.eval_utils_gpt_goatbench import encode_tensor2base64, make_labeled_contact_sheet
from src.place_visual_cache import PlaceVisualCache
from src.place_goal_navigation import PlaceGoalNavigation
from src.query_place_goal import candidate_visual, select_place_goal
from tests.test_place_goal_navigation import candidate


class PlaceVisualCacheTest(unittest.TestCase):
    def test_exact_pixels_labels_order_and_eviction(self):
        pixels = np.zeros((16, 16, 3), np.uint8)
        cache = PlaceVisualCache()
        with patch("src.place_visual_cache.encode_tensor2base64", wraps=encode_tensor2base64) as encode:
            first = cache.encode(pixels)
            self.assertEqual(first, cache.encode(pixels.copy()))
            self.assertEqual(encode.call_count, 1)
            pixels[:] = 255
            second = cache.encode(pixels)
            self.assertNotEqual(first, second)
        items = [("one", first), ("two", second)]
        for tiles in (items, items[::-1], [("changed", first), ("two", second)]):
            self.assertEqual(cache.sheet(tiles), make_labeled_contact_sheet(tiles))
            self.assertEqual(cache.sheet(tiles), make_labeled_contact_sheet(tiles))
        small = PlaceVisualCache(max_bytes=len(first))
        small.encode(np.zeros_like(pixels))
        small.encode(pixels)
        self.assertLessEqual(small.bytes, len(first))
        self.assertEqual(small.encode(np.zeros_like(pixels)), first)

    def test_full_prompt_equal_disabled_cold_warm_and_live_sources_checked(self):
        rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        prompt = [SimpleNamespace(xyxy=np.array([[i, i, i + 8, i + 8]]),
                                  data={"obj_id": [i]}) for i in range(1, 15)]
        snapshot = SimpleNamespace(image="old.png", cluster=list(range(1, 15)),
                                   obs_point=np.array([2, 3]), visual_prompt=prompt)
        scene = SimpleNamespace(snapshots={"old.png": snapshot},
                                all_observations={"old.png": rgb},
                                objects={i: {} for i in range(1, 15)})
        states = []
        for enabled in (False, True):
            state = PlaceGoalNavigation("g", .1)
            state.candidates = [candidate(f"object:{i}") for i in range(1, 15)]
            for i, c in enumerate(state.candidates, 1):
                c.object_id = i
                c.candidate_id = f"verify:{i}"
            states.append(state)
        with patch("src.query_place_goal._goal_content", side_effect=lambda _: [("goal",)]), patch(
            "src.query_place_goal._request", return_value='{"candidate_id":"verify:14","support":0.8}'
        ) as request:
            for state, enabled in ((states[0], False), (states[1], True)):
                selected, diagnostic = select_place_goal(state, {}, scene, None, [rgb],
                    {"place_goal_navigation": {"cache_visuals": enabled}})
                self.assertEqual(selected[0].candidate_id, "verify:14")
                self.assertEqual(diagnostic["visual_tiles"], 28)
                self.assertEqual(diagnostic["image_parts"], 4)
            payloads = [call.args[:2] for call in request.call_args_list]
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(len(payloads), 2)
        cache = states[1]._visual_cache
        c = states[1].candidates[0]
        old_crop = candidate_visual(c, scene, None, cache)[2]
        snapshot.visual_prompt[0].xyxy[:] = [0, 0, 4, 4]
        self.assertNotEqual(candidate_visual(c, scene, None, cache)[2], old_crop)
        del scene.objects[1]
        self.assertIsNone(candidate_visual(c, scene, None, cache))


if __name__ == "__main__":
    unittest.main()
