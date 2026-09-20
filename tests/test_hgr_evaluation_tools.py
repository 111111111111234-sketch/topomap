import json
import os
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts.run_hgr_stages import (
    ADAPTIVE_SCREEN_STAGES,
    REVISIT_FIX_STAGES,
    completed_result_matches,
    limited_cpu_affinity,
    load_manifest,
    resolve_splits,
)
from scripts.run_hgr_experiment_plan import audit_result, build_jobs, result_dir
from scripts.summarize_goatbench_results import render_markdown
from src.hgr_experiment_state import write_run_fingerprint
from run_goatbench_evaluation import _apply_diagnostic_limits


class HgrEvaluationToolsTest(unittest.TestCase):
    def test_adaptive_screen_has_only_required_comparison_stages(self):
        self.assertEqual(ADAPTIVE_SCREEN_STAGES, (
            "route-only", "memory-dedup", "adaptive-hybrid",
        ))

    def test_actual_runner_uses_task_scoped_terminal_verification(self):
        source = Path("run_goatbench_evaluation.py").read_text(encoding="utf-8")
        self.assertIn(
            "baseline_dual_dynamic.requires_terminal_verification(intent.task)",
            source,
        )

    def test_revisit_fix_group_contains_only_affected_c_lite_stages(self):
        self.assertEqual(REVISIT_FIX_STAGES, (
            "memory-dedup",
            "memory-cost-gate",
            "intent-persistent",
            "verified-ablation",
        ))

    def test_plan_audit_retains_classified_selection_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_quality.json").write_text(json.dumps({
                "total": 1,
                "statuses": {"error": 1},
                "error_counts": {"selection": 1},
                "missing_outcome_count": 0,
                "acceptance_ready": False,
            }))
            report = audit_result(root)
            self.assertFalse(report["acceptance_ready"])
            self.assertTrue(report["classified_online_failure"])

    def test_plan_audit_rejects_internal_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "execution_quality.json").write_text(json.dumps({
                "total": 1,
                "statuses": {"error": 1},
                "error_counts": {"feature_residual": 1},
                "missing_outcome_count": 0,
                "acceptance_ready": False,
            }))
            with self.assertRaises(RuntimeError):
                audit_result(root)

    def test_cpu_affinity_limit_selects_at_most_requested_cpus(self):
        hook, cpus = limited_cpu_affinity(3)
        if hasattr(os, "sched_getaffinity"):
            self.assertIsNotNone(hook)
            self.assertGreater(len(cpus), 0)
            self.assertLessEqual(len(cpus), 3)

    def test_reduced_development_plan_expands_only_five_3x2_stages(self):
        class Args:
            phase = "development"
            dev_manifest = "fast.json"
            final_manifest = "final.json"
            final_candidate = None
            matrix_prefix = "matrix"
            python = "/python"
            gpu = 4
            threads = 16
            results_root = "results"
            dry_run = False

        jobs = build_jobs(Args())
        self.assertEqual(
            [(job["stage"], job["manifest"], job["repeats"]) for job in jobs],
            [
                ("route-only", "fast.json", 1),
                ("memory-dedup", "fast.json", 1),
                ("memory-cost-gate", "fast.json", 1),
                ("intent-persistent", "fast.json", 1),
                ("verified-ablation", "fast.json", 1),
            ],
        )
        self.assertIn("--resume", jobs[0]["command"])
        self.assertEqual(
            result_dir(Path("results"), "route-only", 2, "matrix_final"),
            Path("results/exp_hgr_rebuild_route_only_r02_matrix_final"),
        )

    def test_final_plan_is_separate_and_repeats_frozen_candidate(self):
        class Args:
            phase = "final"
            dev_manifest = "fast.json"
            final_manifest = "final.json"
            final_candidate = "verified-ablation"
            matrix_prefix = "matrix"
            python = "/python"
            gpu = 4
            threads = 16
            results_root = "results"
            dry_run = False

        jobs = build_jobs(Args())
        self.assertEqual(
            [(job["stage"], job["repeats"]) for job in jobs],
            [("baseline", 3), ("route-only", 3), ("verified-ablation", 3)],
        )

    def test_final_plan_deduplicates_retained_route_only_candidate(self):
        class Args:
            phase = "final"
            dev_manifest = "fast.json"
            final_manifest = "final.json"
            final_candidate = "route-only"
            matrix_prefix = "matrix"
            python = "/python"
            gpu = 4
            threads = 16
            results_root = "results"
            dry_run = False

        jobs = build_jobs(Args())
        self.assertEqual(
            [(job["stage"], job["repeats"]) for job in jobs],
            [("baseline", 3), ("route-only", 3)],
        )

    def test_short_run_limits_subtasks_and_steps(self):
        kinds, goals, steps = _apply_diagnostic_limits(
            ["image", "object", "description"], [1, 2, 3],
            max_subtasks=1, step_budget=50, max_steps_per_subtask=25,
        )
        self.assertEqual(kinds, ["image"])
        self.assertEqual(goals, [1])
        self.assertEqual(steps, 25)

    def test_manifest_split_resolution_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"splits":{"1":[],"2":[]}}', encoding="utf-8")
            payload, checksum = load_manifest(path)
            self.assertEqual(resolve_splits(payload, "all"), ["1", "2"])
            self.assertEqual(resolve_splits(payload, "2"), ["2"])
            self.assertEqual(len(checksum), 64)
            with self.assertRaises(ValueError):
                resolve_splits(payload, "3")

    def test_fingerprint_is_idempotent_but_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src" / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text('{"splits":{"1":[]}}', encoding="utf-8")
            output = root / "output"
            cfg = OmegaConf.create({"episode_manifest": str(manifest)})
            first = write_run_fingerprint(output, cfg, root)
            self.assertEqual(write_run_fingerprint(output, cfg, root), first)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_run_fingerprint(output, cfg, root)

    def test_resume_requires_complete_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)
            marker_dir = result / "provenance" / "split_1"
            marker_dir.mkdir(parents=True)
            marker = {
                "split": 1, "subtask_count": 8,
                "orchestration_repeat": 2,
                "orchestration_config_sha256": "config",
                "fingerprint": {
                    "manifest_sha256": "manifest",
                    "source_sha256": {"src/a.py": "hash"},
                },
            }
            (marker_dir / "completed.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertEqual(completed_result_matches(
                result, "1", 2, "manifest", "config", {"src/a.py": "hash"}
            ), (True, "complete_and_matching"))
            self.assertEqual(completed_result_matches(
                result, "1", 2, "changed", "config", {"src/a.py": "hash"}
            )[1], "manifest_mismatch")

    def test_markdown_contains_metrics_cost_memory_and_pairing(self):
        result = {
            "metrics": {
                "success_by_snapshot": {"mean_percent": 50.0, "count": 2},
                "success_by_distance": {"mean_percent": 75.0, "count": 2},
                "spl_by_snapshot": {"mean_percent": 40.0, "count": 2},
                "spl_by_distance": {"mean_percent": 60.0, "count": 2},
            },
            "execution_quality": {
                "acceptance_ready": True, "statuses": {"completed": 2},
                "error_counts": {},
            },
            "operations": {
                "steps": 8, "distance_m": 3.5,
                "vlm": {"logical_calls": 4, "http_attempts": 5,
                        "elapsed_seconds": 2.0},
            },
            "routes": {"hypothesis_aware_navigation": {
                "selections_by_kind": {"REVISIT": 1},
                "historical_source_selections": 1,
                "repeated_source_selections": 0,
                "repeated_evidence_selections": 0,
                "completed_revisits": 1,
                "authorization_reselections": 1,
                "scene_map_goal_starts": [{"places": 1, "entities": 2,
                                           "frontiers": 3, "edges": 4}],
                "scene_map_goal_ends": [{"places": 2, "entities": 3,
                                         "frontiers": 4, "edges": 5}],
                "terminal_verifications": 1,
                "false_confirmation_count_by_snapshot": 0,
            }},
        }
        report = {
            "results": {"candidate": result},
            "paired_comparisons": [{
                "candidate": "candidate", "reference": "base",
                "metrics": {"success_by_snapshot": {
                    "paired_count": 2,
                    "candidate_minus_reference_percent_points": 25.0,
                    "fail_to_success": 1, "success_to_fail": 0,
                    "reference_only_count": 0,
                }},
            }],
        }
        rendered = render_markdown(report)
        self.assertIn("| candidate | — | — | 50.00 | 75.00 | 40.00 | 60.00 | 2 |", rendered)
        self.assertIn("Historical source reuse", rendered)
        self.assertIn("candidate vs base", rendered)


if __name__ == "__main__":
    unittest.main()
