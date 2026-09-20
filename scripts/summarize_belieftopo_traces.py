#!/usr/bin/env python3
"""Aggregate Persistent Goal-Conditioned Belief Topology diagnostics."""

import argparse
from collections import Counter
import glob
import json
import os
import pickle
import statistics
import math
import sys


def _delta(after, before, key):
    return float(after.get(key, 0)) - float(before.get(key, 0))


def _load_metric(result_dir, name):
    path = os.path.join(result_dir, f"{name}.pkl")
    if not os.path.exists(path):
        candidates = sorted(glob.glob(os.path.join(result_dir, f"{name}_*.pkl")))
        path = candidates[-1] if candidates else None
    if path is None:
        return None
    with open(path, "rb") as handle:
        values = pickle.load(handle)
    if isinstance(values, dict):
        values = list(values.values())
    flat = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(value)
        else:
            flat.append(value)
    finite_safe = []
    invalid = 0
    for value in flat:
        numeric = float(value)
        if not math.isfinite(numeric):
            numeric = 0.0
            invalid += 1
        finite_safe.append(numeric)
    if invalid:
        print(
            f"warning: {name} contained {invalid} non-finite values; scored as zero",
            file=sys.stderr,
        )
    return 100.0 * sum(finite_safe) / len(finite_safe) if finite_safe else None


def load_metrics(result_dir):
    return {
        "snapshot_sr": _load_metric(result_dir, "success_by_snapshot"),
        "distance_sr": _load_metric(result_dir, "success_by_distance"),
        "snapshot_spl": _load_metric(result_dir, "spl_by_snapshot"),
        "distance_spl": _load_metric(result_dir, "spl_by_distance"),
    }


def summarize(result_dir):
    trace_dir = os.path.join(result_dir, "active_topology_traces")
    summary_files = sorted(glob.glob(os.path.join(trace_dir, "*.summary.json")))
    trace_files = sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))
    if not summary_files and not trace_files:
        raise FileNotFoundError(f"no ActiveTopo traces found under {trace_dir}")

    totals = Counter()
    purpose_calls = Counter()
    decision_sources = Counter()
    action_types = Counter()
    goal_topology_action_types = Counter()
    goal_topology_decision_sources = Counter()
    posterior_changes = []
    entropy_reductions = []
    calibrated_clip_scores = []
    unknown_pairs = []
    confirmation_reasons = Counter()
    snapshot_gate_reasons = Counter()
    trace_snapshot_gate_reasons = Counter()
    unknown_policy_reasons = Counter()

    for path in summary_files:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        topology = payload.get("topology") or {}
        belief = payload.get("goal_belief") or {}
        for key, value in topology.items():
            if isinstance(value, (int, float)):
                totals[f"topology_{key}"] += value
        for key, value in belief.items():
            if isinstance(value, (int, float)):
                totals[f"belief_{key}"] += value
        snapshot_gate_reasons.update(belief.get("snapshot_gate_reasons") or {})
        before = payload.get("vlm_telemetry_start") or {}
        after = payload.get("vlm_telemetry_end") or {}
        for key in ("logical_calls", "http_attempts", "successes", "failures"):
            totals[f"vlm_{key}"] += _delta(after, before, key)
        for purpose, values in (after.get("by_purpose") or {}).items():
            old = (before.get("by_purpose") or {}).get(purpose, {})
            for metric in ("logical_calls", "http_attempts", "successes", "failures", "elapsed_seconds"):
                purpose_calls[f"{purpose}:{metric}"] += _delta(values, old, metric)

    for path in trace_files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event.get("event")
                totals[f"event_{kind}"] += 1
                if kind == "goal_evidence":
                    posterior_changes.append(
                        event.get("posterior_after", 0.0)
                        - event.get("posterior_before", 0.0)
                    )
                elif kind == "belief_decision":
                    decision_sources[event.get("decision_source", "unknown")] += 1
                    action = event.get("selected_action") or {}
                    action_types[action.get("action_type", "none")] += 1
                elif kind == "goal_topology_action_selected":
                    action = event.get("action") or {}
                    goal_topology_action_types[
                        action.get("action_type", "none")
                    ] += 1
                    goal_topology_decision_sources[
                        event.get("decision_source", "unknown")
                    ] += 1
                elif kind == "active_verify_result":
                    entropy_reductions.append(event.get("entropy_reduction", 0.0))
                elif kind in ("clip_calibration", "freeze", "calibrated"):
                    if event.get("calibrated_score") is not None:
                        calibrated_clip_scores.append(float(event["calibrated_score"]))
                    calibrated_clip_scores.extend(
                        float(value) for value in event.get("calibrated_scores", [])
                    )
                elif kind == "belief_projection":
                    hypotheses = event.get("hypotheses") or {}
                    legacy = hypotheses.get("legacy_unknown_probability")
                    current = hypotheses.get("unknown_probability")
                    if legacy is not None and current is not None:
                        unknown_pairs.append((float(legacy), float(current)))
                    for gate_event in event.get("snapshot_gate_events", []):
                        decision = gate_event.get("decision") or {}
                        trace_snapshot_gate_reasons[decision.get("reason", "unknown")] += 1
                    unknown_policy = event.get("unknown_policy") or {}
                    if unknown_policy:
                        unknown_policy_reasons[
                            unknown_policy.get("reason", "unknown")
                        ] += 1
                elif kind == "snapshot_gate":
                    decision = event.get("decision") or {}
                    trace_snapshot_gate_reasons[decision.get("reason", "unknown")] += 1
                for confirmation_event in event.get("confirmation_events", []):
                    if confirmation_event.get("event") == "resolved":
                        confirmation_reasons[confirmation_event.get("reason", "unknown")] += 1
                if kind in (
                    "episode_confirmation_events", "subtask_confirmation_events",
                ):
                    for confirmation_event in event.get("events", []):
                        if confirmation_event.get("event") == "resolved":
                            confirmation_reasons[confirmation_event.get("reason", "unknown")] += 1

    match_total = totals["topology_frontier_matches"] + totals["topology_frontier_new"]
    stable_rate = (
        totals["topology_frontier_matches"] / match_total if match_total else 0.0
    )
    report = {
        "episodes": len(summary_files),
        "stable_id_match_rate": stable_rate,
        "blocked_target_retries": totals["topology_blocked_target_retries"],
        "topology_routes": {
            "plans": totals["topology_topology_route_plans"],
            "waypoints_generated": totals[
                "topology_topology_route_waypoints"
            ],
            "waypoints_arrived": totals[
                "topology_topology_waypoint_arrivals"
            ],
            "fallbacks": totals["topology_topology_route_fallbacks"],
            "route_selected_events": totals[
                "event_topology_route_selected"
            ],
            "waypoint_arrived_events": totals[
                "event_topology_waypoint_arrived"
            ],
            "origin_reanchor_skips": totals[
                "topology_observed_reanchor_skips"
            ],
            "commitments_created": totals[
                "topology_topology_route_commitments_created"
            ],
            "commitment_resumes": totals[
                "topology_topology_route_commitment_resumes"
            ],
            "commitments_completed": totals[
                "topology_topology_route_commitments_completed"
            ],
            "commitments_cancelled": totals[
                "topology_topology_route_commitments_cancelled"
            ],
        },
        "evidence_events": totals["event_goal_evidence"],
        "belief_decisions": totals["event_belief_decision"],
        "active_verify_results": totals["event_active_verify_result"],
        "mean_posterior_change": (
            sum(posterior_changes) / len(posterior_changes)
            if posterior_changes else 0.0
        ),
        "mean_verify_entropy_reduction": (
            sum(entropy_reductions) / len(entropy_reductions)
            if entropy_reductions else 0.0
        ),
        "decision_sources": dict(decision_sources),
        "action_types": dict(action_types),
        "goal_topology_action_types": dict(goal_topology_action_types),
        "goal_topology_decision_sources": dict(
            goal_topology_decision_sources
        ),
        "vlm": {
            key.removeprefix("vlm_"): value
            for key, value in totals.items() if key.startswith("vlm_")
        },
        "vlm_by_purpose": dict(purpose_calls),
        "semantic_cache": {
            key.removeprefix("topology_semantic_"): value
            for key, value in totals.items()
            if key.startswith("topology_semantic_")
        },
        "clip_calibration": {
            "count": len(calibrated_clip_scores),
            "median": (
                statistics.median(calibrated_clip_scores)
                if calibrated_clip_scores else None
            ),
            "min": min(calibrated_clip_scores, default=None),
            "max": max(calibrated_clip_scores, default=None),
        },
        "unknown": {
            "samples": len(unknown_pairs),
            "legacy_mean": (
                sum(value[0] for value in unknown_pairs) / len(unknown_pairs)
                if unknown_pairs else None
            ),
            "v2_mean": (
                sum(value[1] for value in unknown_pairs) / len(unknown_pairs)
                if unknown_pairs else None
            ),
        },
        "confirmation_end_reasons": dict(confirmation_reasons),
        "confirmation_completion_rate": (
            sum(confirmation_reasons.values()) /
            totals["belief_confirmations_created"]
            if totals["belief_confirmations_created"] else 0.0
        ),
        "unknown_policy_reasons": dict(unknown_policy_reasons),
        "category_vlm_tiebreak_calls": decision_sources.get("vlm_tiebreak", 0),
        "snapshot_gate_reasons": dict(
            snapshot_gate_reasons or trace_snapshot_gate_reasons
        ),
        "confirmation_to_snapshot_rate": (
            totals["belief_confirmations_selected"] /
            totals["belief_confirmations_created"]
            if totals["belief_confirmations_created"] else 0.0
        ),
        "belief_stats": {
            key.removeprefix("belief_"): value
            for key, value in totals.items() if key.startswith("belief_")
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="experiment result directory")
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--compare", action="append", default=[], metavar="LABEL=RESULT_DIR",
        help="compare SR/SPL with v1, frozen baseline or full-v4 result directories",
    )
    args = parser.parse_args()
    report = summarize(args.result_dir)
    report["metrics"] = load_metrics(args.result_dir)
    comparisons = {}
    for spec in args.compare:
        if "=" not in spec:
            parser.error("--compare must use LABEL=RESULT_DIR")
        label, directory = spec.split("=", 1)
        metrics = load_metrics(directory)
        comparisons[label] = {
            "metrics": metrics,
            "delta_pp": {
                key: (
                    report["metrics"][key] - value
                    if value is not None and report["metrics"].get(key) is not None
                    else None
                )
                for key, value in metrics.items()
            },
        }
    report["comparisons"] = comparisons
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
