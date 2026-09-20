"""Exercise merge provenance at the real producer, with geometry stubbed."""

import unittest
from unittest.mock import patch

import numpy as np

from src.conceptgraph.slam.utils import merge_objects


class ObjectMergeIdentityTest(unittest.TestCase):
    def _merge(self, aliases=None, matching=True):
        objects = {
            2: {"conf": .2, "clip_ft": np.array([1., 0.], dtype=np.float32)},
            5: {"conf": .5, "clip_ft": np.array([1., 0.], dtype=np.float32)},
            8: {"conf": .9, "clip_ft": np.array([1., 0.], dtype=np.float32)},
        }
        overlap = np.array([[0, .9, 0], [0, 0, .8], [0, 0, 0]])
        with patch("src.conceptgraph.slam.utils.compute_overlap_matrix_general", return_value=overlap), patch(
            "src.conceptgraph.slam.utils.merge_obj2_into_obj1",
            side_effect=lambda first, *args, **kwargs: first,
        ):
            return merge_objects(
                .5, .5 if matching else 1.1, .5, objects,
                .05, False, .1, 1, "iou", "cpu",
                object_id_aliases=aliases,
            )

    def test_aliases_follow_actual_survivors_through_multiple_merges(self):
        aliases = {}
        self.assertEqual(set(self._merge(aliases)), {8})
        self.assertEqual(aliases, {2: 5, 5: 8})
        # Optional tracking cannot change perception merge results.
        self.assertEqual(set(self._merge()), {8})

    def test_rejected_merge_never_creates_an_alias(self):
        aliases = {}
        self.assertEqual(set(self._merge(aliases, matching=False)), {2, 5, 8})
        self.assertEqual(aliases, {})


if __name__ == "__main__":
    unittest.main()
