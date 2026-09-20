import json
import os
import tempfile
import unittest

import numpy as np

from src.logger_goatbench import (Logger, _load_visualization_image, compute_spl,
                                  compute_spl_diagnostic)


class GoatBenchLoggerTest(unittest.TestCase):
    def test_restored_frontier_visualization_uses_in_memory_image(self):
        with tempfile.TemporaryDirectory() as directory:
            fallback = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
            loaded = _load_visualization_image(
                os.path.join(directory, "historical-frontier.png"), fallback)
            np.testing.assert_array_equal(loaded, fallback)
            self.assertIsNone(_load_visualization_image(
                os.path.join(directory, "missing-snapshot.png")))

    def test_failed_task_with_unreachable_ground_truth_has_zero_spl(self):
        self.assertEqual(compute_spl(False, float("inf"), 4.2), 0.0)

    def test_successful_task_uses_standard_spl_ratio(self):
        self.assertAlmostEqual(compute_spl(True, 4.0, 5.0), 0.8)
        self.assertEqual(compute_spl(True, 5.0, 4.0), 1.0)

    def test_success_with_undefined_distance_is_conservatively_zero(self):
        self.assertEqual(compute_spl(True, float("nan"), 4.0), 0.0)

    def test_spl_diagnostic_distinguishes_failure_from_invalid_distance(self):
        self.assertEqual(
            compute_spl_diagnostic(False, float("inf"), 4.0),
            (0.0, "failure"),
        )
        self.assertEqual(
            compute_spl_diagnostic(True, float("inf"), 4.0),
            (0.0, "invalid_shortest_distance"),
        )

    def test_logger_writes_finite_subtask_diagnostics_and_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(directory, 0.0, 0.09, 1, voxel_size=0.1)
            logger.subtask_explore_dist = 4.0
            logger.log_subtask_result(
                success_by_snapshot=False,
                success_by_distance=True,
                subtask_id="scene_0_0",
                gt_subtask_explore_dist=float("inf"),
                goal_type="object",
                n_filtered_snapshots=1,
                n_total_snapshots=2,
                n_total_frames=3,
                agent_subtask_distance=0.5,
                subtask_steps=4,
                subtask_frames=3,
                vlm_telemetry={"logical_calls": 2},
                final_choice={"type": "frontier", "source_index": 0},
                selected_topo_id="frontier:0",
                execution_outcome={
                    "execution_status": "error",
                    "termination_reason": "semantic_selection_failed",
                    "termination_detail": {"selection_reason": "request_exhausted"},
                    "error_counts": {"selection": 1},
                },
            )
            logger.save_results()
            logger.aggregate_results()
            with open(os.path.join(directory, "subtask_metrics.json")) as handle:
                record = json.load(handle)["scene_0_0"]
            self.assertEqual(record["spl_by_distance"], 0.0)
            self.assertEqual(
                record["spl_by_distance_status"], "invalid_shortest_distance"
            )
            self.assertIsNone(record["gt_shortest_distance_m"])
            with open(os.path.join(directory, "metric_validity.json")) as handle:
                validity = json.load(handle)
            self.assertEqual(validity["spl_by_distance"]["invalid"], 0)
            self.assertEqual(record["execution_status"], "error")
            self.assertEqual(record["termination_reason"], "semantic_selection_failed")
            with open(os.path.join(directory, "execution_quality.json")) as handle:
                quality = json.load(handle)
            self.assertEqual(quality["statuses"], {"error": 1})
            self.assertEqual(quality["error_counts"], {"selection": 1})
            self.assertFalse(quality["acceptance_ready"])


if __name__ == "__main__":
    unittest.main()
