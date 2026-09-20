import unittest

from src.subtask_execution import (
    ExecutionStatus,
    SubtaskExecution,
    summarize_execution_quality,
)


class SubtaskExecutionTest(unittest.TestCase):
    def test_first_terminal_reason_is_locked_but_errors_accumulate(self):
        outcome = SubtaskExecution()
        self.assertTrue(outcome.terminate(
            ExecutionStatus.ERROR, "semantic_selection_failed",
            {"selection_reason": "request_exhausted"}, "selection",
        ))
        self.assertFalse(outcome.terminate(
            ExecutionStatus.EXHAUSTED, "step_budget_exhausted"
        ))
        outcome.record_error("feature_residual", 2)
        self.assertEqual(outcome.finalize(False), {
            "execution_status": "error",
            "termination_reason": "semantic_selection_failed",
            "termination_detail": {"selection_reason": "request_exhausted"},
            "error_counts": {"feature_residual": 2, "selection": 1},
        })

    def test_finalize_distinguishes_budget_from_success(self):
        self.assertEqual(
            SubtaskExecution().finalize(False)["termination_reason"],
            "step_budget_exhausted",
        )
        self.assertEqual(
            SubtaskExecution().finalize(True)["execution_status"],
            "completed",
        )

    def test_quality_summary_does_not_treat_legacy_record_as_ready(self):
        summary = summarize_execution_quality({
            "new": {
                "execution_status": "completed",
                "termination_reason": "verified_confirmation",
                "error_counts": {},
            },
            "legacy": {"success_by_distance": True},
        })
        self.assertEqual(summary["statuses"], {"completed": 1, "unknown": 1})
        self.assertEqual(summary["missing_outcome_ids"], ["legacy"])
        self.assertFalse(summary["acceptance_ready"])


if __name__ == "__main__":
    unittest.main()
