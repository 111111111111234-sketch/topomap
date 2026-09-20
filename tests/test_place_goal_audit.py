import json
from pathlib import Path
import tempfile
import unittest

from scripts.summarize_place_goal_traces import audit


class PlaceGoalAuditTest(unittest.TestCase):
    def test_guidance_audit_rejects_outside_corridor_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            traces = Path(directory) / "active_topology_traces"
            traces.mkdir()
            install = {"event":"route_guidance_installed","subtask_id":"s",
                       "guidance":{"certified_path":[[1,1],[2,1],[3,1]]}}
            movement = {"event":"route_guidance_step","subtask_id":"s",
                        "guidance":{"last_segment":[[1,1],[3,2]],"progress_index":2}}
            (traces/"episode.jsonl").write_text(json.dumps(install)+"\n"+json.dumps(movement)+"\n")
            self.assertEqual(audit(directory)["violations"][0]["reason"],"guidance_segment_outside_certificate")

    def test_success_requires_current_verification_event(self):
        with tempfile.TemporaryDirectory() as directory:
            traces = Path(directory) / "active_topology_traces"
            traces.mkdir()
            path = traces / "episode.jsonl"
            summary = {"event": "phase_c_subtask_summary", "subtask_id": "s", "task_success": True}
            path.write_text(json.dumps(summary) + "\n")
            self.assertEqual(audit(directory)["violations"][0]["reason"], "success_without_fresh_verification")
            matched = {"event": "phase_c_execution_feedback", "subtask_id": "s",
                       "verification": {"verdict": "matched"},
                       "feedback": {"outcome": "verification_matched"}}
            path.write_text(json.dumps(matched) + "\n" + json.dumps(summary) + "\n")
            self.assertFalse(audit(directory)["violations"])

    def test_repeated_topological_action_and_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            traces = Path(directory) / "active_topology_traces"
            traces.mkdir()
            path = traces / "episode.jsonl"
            decision = {
                "event": "phase_c_decision", "subtask_id": "s",
                "view": {"topological_actions": True, "candidates": [
                    {"candidate_id": "verify:1@place_0"}],
                    "goal_topology": {"revision": 2}},
                "selection": {"candidate_id": "verify:1@place_0", "model_called": True},
            }
            path.write_text(json.dumps(decision) + "\n" + json.dumps(decision) + "\n")
            reasons = {item["reason"] for item in audit(directory)["violations"]}
            self.assertIn("repeated_topological_action", reasons)
            self.assertIn("model_requeried_without_goal_topology_change", reasons)


if __name__ == "__main__":
    unittest.main()
