import json
import os
import pickle
import tempfile
import unittest

from src.goatbench_results import (
    finite_metric_summary,
    paired_comparison,
    summarize_result_dir,
)


class GoatBenchResultsTest(unittest.TestCase):
    def test_hierarchical_events_are_summarized_by_intent_and_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            events = [
                {"event": "hierarchical_semantic_decision"},
                {"event": "hierarchical_intent_installed", "intent": {
                    "kind": "target_approach", "execution_mode": "local_tsdf",
                }},
                {"event": "hierarchical_waypoint_arrived"},
                {"event": "hierarchical_terminal_verification",
                 "result": {"verdict": "confirmed"},
                 "command": {"command": "stop"}},
                {"event": "hierarchical_goal_end"},
            ]
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            summary = report["routes"]["hierarchical_navigation"]
            self.assertEqual(summary["semantic_decisions"], 1)
            self.assertEqual(summary["intents_by_kind"], {"target_approach": 1})
            self.assertEqual(summary["execution_modes"], {"local_tsdf": 1})
            self.assertEqual(summary["verification_verdicts"], {"confirmed": 1})
            self.assertEqual(summary["verification_commands"], {"stop": 1})

    def test_verified_confirmation_is_correlated_with_evaluator_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            values = {"good": 1.0, "wrong": 0.0}
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump(values, handle)
            with open(os.path.join(directory, "subtask_metrics.json"), "w") as handle:
                json.dump({
                    "good": {"success_by_snapshot": 1.0, "success_by_distance": 1.0},
                    "wrong": {"success_by_snapshot": 0.0, "success_by_distance": 1.0},
                }, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                for subtask_id in ("good", "wrong"):
                    handle.write(json.dumps({
                        "event": "hypothesis_aware_goal_verification",
                        "subtask_id": subtask_id, "verdict": "confirmed", "action": "stop",
                    }) + "\n")
            report, _ = summarize_result_dir(directory)
            summary = report["routes"]["hypothesis_aware_navigation"]
            self.assertEqual(summary["confirmed_count"], 2)
            self.assertEqual(summary["false_confirmation_count_by_snapshot"], 1)
            self.assertEqual(summary["confirmed_snapshot_success_rate"], 0.5)
            self.assertEqual(summary["confirmed_distance_success_rate"], 1.0)

    def test_operational_memory_and_revisit_diagnostics_are_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"scene_0_0": 0.0}, handle)
            with open(os.path.join(directory, "subtask_metrics.json"), "w") as handle:
                json.dump({"scene_0_0": {
                    "goal_type": "image", "subtask_steps": 3,
                    "subtask_frames": 7, "traversed_distance_m": 2.5,
                    "vlm_telemetry": {
                        "logical_calls": 2, "http_attempts": 3, "successes": 2,
                        "failures": 1, "elapsed_seconds": 4.5,
                        "by_purpose": {"goat_explorer": {
                            "logical_calls": 2, "http_attempts": 3,
                            "successes": 2, "failures": 1, "elapsed_seconds": 4.5,
                        }},
                    },
                }}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            events = [
                {"event": "dual_dynamic_goal_started", "subtask_id": "scene_0_0",
                 "navigator": {"scene_map": {"places": 1, "entities": 2,
                                                "frontiers": 1, "edges": 0}}},
                {"event": "hypothesis_aware_selection", "subtask_id": "scene_0_0",
                 "navigator": {"active_intent": {"task": {
                     "kind": "REVISIT", "source_id": "snap:a",
                     "evidence_digest": "digest:a",
                     "historical_source": True}},
                     "stats": {"authorization_reselections": 1}}},
                {"event": "hypothesis_aware_selection", "subtask_id": "scene_0_0",
                 "navigator": {"active_intent": {"task": {
                     "kind": "REVISIT", "source_id": "snap:a",
                     "evidence_digest": "digest:a"}},
                     "stats": {"authorization_reselections": 2}}},
                {"event": "hypothesis_aware_revisit_completed",
                 "revisit": {"evidence_digest": "digest:a"},
                 "navigator": {"stats": {
                     "incremental_assessments": 3,
                     "successful_incremental_preemptions": 1,
                 }}},
                {"event": "hypothesis_aware_revisit_promoted",
                 "revisit": {"evidence_digest": "digest:b",
                              "transition": "REVISIT->APPROACH"}},
                {"event": "known_space_motion"},
                {"event": "dual_dynamic_goal_end", "subtask_id": "scene_0_0",
                 "navigator": {"scene_map": {"places": 2, "entities": 3,
                                                "frontiers": 1, "edges": 1}}},
            ]
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            self.assertEqual(report["operations"]["steps"], 3)
            self.assertEqual(report["operations"]["vlm"]["logical_calls"], 2)
            self.assertEqual(report["operations"]["vlm_by_purpose"]
                             ["goat_explorer"]["failures"], 1)
            navigation = report["routes"]["hypothesis_aware_navigation"]
            self.assertEqual(navigation["historical_revisit_selections"], 2)
            self.assertEqual(navigation["historical_source_selections"], 1)
            self.assertEqual(navigation["repeated_source_selections"], 1)
            self.assertEqual(navigation["repeated_evidence_selections"], 1)
            self.assertEqual(navigation["completed_revisits"], 2)
            self.assertEqual(navigation["promoted_revisits"], 1)
            self.assertEqual(navigation["distinct_completed_revisit_evidence"], 2)
            self.assertEqual(navigation["authorization_reselections"], 2)
            self.assertEqual(navigation["incremental_assessments"], 3)
            self.assertEqual(navigation["successful_incremental_preemptions"], 1)
            self.assertEqual(navigation["known_space_motion_events"], 1)
            self.assertEqual(navigation["scene_map_goal_ends"][0]["entities"], 3)

    def test_nonfinite_metric_is_reported_and_scored_as_zero(self):
        values, report = finite_metric_summary({"a": 1.0, "b": float("nan")})
        self.assertEqual(values, {"a": 1.0, "b": 0.0})
        self.assertEqual(report["invalid_ids"], ["b"])
        self.assertEqual(report["mean_percent"], 50.0)

    def test_paired_comparison_uses_only_identical_subtask_ids(self):
        reference = {
            "success_by_distance": {"a": 0.0, "b": 1.0},
            "success_by_snapshot": {}, "spl_by_snapshot": {}, "spl_by_distance": {},
        }
        candidate = {
            "success_by_distance": {"a": 1.0, "c": 1.0},
            "success_by_snapshot": {}, "spl_by_snapshot": {}, "spl_by_distance": {},
        }
        report = paired_comparison("base", reference, "v7", candidate)
        metric = report["metrics"]["success_by_distance"]
        self.assertEqual(metric["paired_count"], 1)
        self.assertEqual(metric["scene_count"], 1)
        self.assertEqual(metric["fail_to_success"], 1)
        self.assertEqual(metric["candidate_minus_reference_percent_points"], 100.0)
        self.assertEqual(metric["scene_block_bootstrap_ci95_percent_points"], [100.0, 100.0])

    def test_directory_summary_does_not_require_topology_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            report, _ = summarize_result_dir(directory)
            self.assertEqual(report["metrics"]["success_by_distance"]["count"], 1)
            self.assertEqual(report["routes"]["trace_files"], 0)
            self.assertEqual(report["execution_quality"]["total"], 0)
            self.assertFalse(report["execution_quality"]["acceptance_ready"])

    def test_directory_summary_reports_execution_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            with open(os.path.join(directory, "subtask_metrics.json"), "w") as handle:
                json.dump({"a": {
                    "goal_type": "object",
                    "execution_status": "completed",
                    "termination_reason": "verified_confirmation",
                    "error_counts": {"feature_residual": 2},
                }}, handle)
            report, _ = summarize_result_dir(directory)
            quality = report["execution_quality"]
            self.assertEqual(quality["statuses"], {"completed": 1})
            self.assertEqual(quality["error_counts"], {"feature_residual": 2})
            self.assertFalse(quality["acceptance_ready"])

    def test_goat_success_requires_completed_stop_and_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            values = {"exhausted_near": 1.0, "completed_near": 1.0,
                      "completed_far": 0.0}
            for name in ("success_by_snapshot", "success_by_distance",
                         "spl_by_snapshot", "spl_by_distance"):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump(values, handle)
            with open(os.path.join(directory, "subtask_metrics.json"), "w") as handle:
                json.dump({
                    "exhausted_near": {"goal_type": "image", "execution_status": "exhausted",
                                       "success_by_distance": True, "spl_by_distance": .7},
                    "completed_near": {"goal_type": "image", "execution_status": "completed",
                                       "success_by_distance": True, "spl_by_distance": .6},
                    "completed_far": {"goal_type": "object", "execution_status": "completed",
                                      "success_by_distance": False, "spl_by_distance": 0.0},
                }, handle)
            report, mappings = summarize_result_dir(directory)
            self.assertEqual(mappings["goat_success"], {
                "exhausted_near": 0.0, "completed_near": 1.0, "completed_far": 0.0,
            })
            self.assertEqual(report["metrics"]["goat_success"]["mean_percent"], 100 / 3)
            self.assertEqual(report["metrics"]["goat_spl"]["mean_percent"], 20.0)

    def test_metric_action_shadow_distribution_is_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            event = {
                "event": "metric_action_shadow_view",
                "view": {"goal_type": "image", "actions": [
                    {"action_type": "revisit", "route_mode": "direct_path",
                     "reachable": True, "relevance": 0.8,
                     "effective_cost_m": 2.0, "budget_ratio": 0.1},
                    {"action_type": "explore", "route_mode": "direct_path",
                     "reachable": True, "relevance": 0.5,
                     "effective_cost_m": 1.0, "budget_ratio": 0.05},
                ]},
            }
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            shadow = report["routes"]["metric_action_shadow"]
            self.assertEqual(shadow["views"], 1)
            self.assertEqual(shadow["views_by_goal_type"], {"image": 1})
            self.assertEqual(
                shadow["actions_by_goal_type"]["image"],
                {"revisit": 1, "explore": 1},
            )
            self.assertAlmostEqual(
                shadow["best_revisit_relevance_margin"]["mean"], 0.3
            )
            self.assertEqual(
                shadow["best_revisit_to_cheapest_explore_cost_ratio"]["mean"],
                2.0,
            )

    def test_direct_first_execution_is_summarized_by_goal_type(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            events = [
                {
                    "event": "direct_first_route_planned",
                    "goal_type": "image",
                    "plan": {
                        "route_mode": "direct_path",
                        "direct_geodesic_m": 2.0,
                        "topology_route_m": None,
                        "fallback_reason": None,
                    },
                },
                {
                    "event": "direct_first_route_planned",
                    "goal_type": "description",
                    "plan": {
                        "route_mode": "original_hgr_fallback",
                        "direct_geodesic_m": None,
                        "topology_route_m": None,
                        "fallback_reason": "capture_anchor_unavailable",
                    },
                },
            ]
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            direct_first = report["routes"]["direct_first_execution"]
            self.assertEqual(direct_first["plans"], 2)
            self.assertEqual(
                direct_first["route_modes"],
                {"direct_path": 1, "original_hgr_fallback": 1},
            )
            self.assertEqual(
                direct_first["route_modes_by_goal_type"]["image"],
                {"direct_path": 1},
            )
            self.assertEqual(
                direct_first["fallback_reasons"],
                {"capture_anchor_unavailable": 1},
            )

    def test_revisit_gate_acceptance_and_reasons_are_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            events = [
                {
                    "event": "revisit_intervention_gate",
                    "goal_type": "image",
                    "decision": {"reason": "accepted"},
                },
                {
                    "event": "revisit_intervention_gate",
                    "goal_type": "description",
                    "decision": {
                        "reason": "revisit_exceeds_budget_fraction"
                    },
                },
            ]
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            gate = report["routes"]["revisit_intervention_gate"]
            self.assertEqual(gate["decisions"], 2)
            self.assertEqual(gate["accepted"], 1)
            self.assertEqual(gate["rejected"], 1)
            self.assertEqual(
                gate["reasons_by_goal_type"]["description"],
                {"revisit_exceeds_budget_fraction": 1},
            )

    def test_goal_conditioned_overlay_relations_are_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in (
                "success_by_snapshot", "success_by_distance",
                "spl_by_snapshot", "spl_by_distance",
            ):
                with open(os.path.join(directory, f"{name}.pkl"), "wb") as handle:
                    pickle.dump({"a": 1.0}, handle)
            trace_dir = os.path.join(directory, "active_topology_traces")
            os.makedirs(trace_dir)
            event = {
                "event": "goal_conditioned_topology_overlay",
                "view": {"candidates": [
                    {
                        "action_type": "revisit",
                        "cost_m": 2.0,
                        "validity": {"valid": True},
                        "topology_relations": ["OBSERVED_AT", "NAVIGABLE"],
                    },
                    {
                        "action_type": "explore",
                        "cost_m": None,
                        "validity": {"valid": False},
                        "topology_relations": ["NAVIGABLE"],
                    },
                ]},
            }
            with open(os.path.join(trace_dir, "episode.jsonl"), "w") as handle:
                handle.write(json.dumps(event) + "\n")
            report, _ = summarize_result_dir(directory)
            overlay = report["routes"]["goal_conditioned_topology_overlay"]
            self.assertEqual(overlay["views"], 1)
            self.assertEqual(
                overlay["candidates_by_action_type"],
                {"revisit": 1, "explore": 1},
            )
            self.assertEqual(overlay["valid_candidates"], 1)
            self.assertEqual(overlay["invalid_candidates"], 1)
            self.assertEqual(
                overlay["causal_topology_relations"],
                {"OBSERVED_AT": 1, "NAVIGABLE": 2},
            )
            self.assertEqual(overlay["candidate_cost_m"]["mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
