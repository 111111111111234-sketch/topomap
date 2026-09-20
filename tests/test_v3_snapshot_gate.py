import unittest
from types import SimpleNamespace

import numpy as np

from run_goatbench_evaluation import _select_v3_frontier_fallback
from src.goal_belief import ExecutableNode


class V3SnapshotGateIntegrationTest(unittest.TestCase):
    def test_rejected_snapshot_uses_best_current_reachable_frontier(self):
        frontiers = [
            SimpleNamespace(region=np.ones((1, 1))),
            SimpleNamespace(region=np.ones((2, 2))),
        ]
        facts = [
            ExecutableNode(
                "frontier_0", "frontier", 0, 1.0, 1.0, 0.1,
                revisit=0.0, map_information_gain=0.2, reachable=True,
            ),
            ExecutableNode(
                "frontier_1", "frontier", 1, 1.0, 1.0, 0.1,
                revisit=0.0, map_information_gain=0.9, reachable=True,
            ),
        ]
        self.assertIs(
            _select_v3_frontier_fallback(facts, frontiers), frontiers[1]
        )

    def test_no_executable_mapping_never_creates_ghost_target(self):
        self.assertIsNone(_select_v3_frontier_fallback([], []))


if __name__ == "__main__":
    unittest.main()
