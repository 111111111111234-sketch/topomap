import json
import os
import tempfile
import unittest

from src.goatbench_manifest import build_episode_manifest, load_episode_manifest


class GoatBenchManifestTest(unittest.TestCase):
    def _dataset(self, root):
        for scene_name in ("scene_a", "scene_b", "scene_c"):
            with open(os.path.join(root, f"{scene_name}.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "episodes": [
                        {"episode_id": 0},
                        {"episode_id": 1},
                    ]
                }, handle)

    def test_build_and_load_exact_split(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory)
            manifest = build_episode_manifest(directory, 77, 0.0, 1.0, [1, 2])
            path = os.path.join(directory, "manifest.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            first = load_episode_manifest(path, directory, 1)
            second = load_episode_manifest(path, directory, 2)
            self.assertEqual(len(first), 3)
            self.assertEqual([item["scene_file"] for item in first],
                             [item["scene_file"] for item in second])
            self.assertTrue(all(item["episode_id"] == "0" for item in first))
            self.assertTrue(all(item["episode_id"] == "1" for item in second))

    def test_rejects_episode_index_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            self._dataset(directory)
            manifest = build_episode_manifest(directory, 77, 0.0, 1.0, [1])
            manifest["splits"]["1"][0]["episode_index"] = 1
            path = os.path.join(directory, "manifest.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaises(ValueError):
                load_episode_manifest(path, directory, 1)


if __name__ == "__main__":
    unittest.main()
