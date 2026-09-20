"""Finite-safe GOAT-Bench result loading and paired comparison utilities."""

import glob
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from src.subtask_execution import summarize_execution_quality


BASE_METRIC_NAMES = (
    "success_by_snapshot",
    "success_by_distance",
    "spl_by_snapshot",
    "spl_by_distance",
)
METRIC_NAMES = ("goat_success", "goat_spl", *BASE_METRIC_NAMES)


def load_metric_mapping(result_dir, name):
    aggregate_path = os.path.join(result_dir, f"{name}.pkl")
    paths = (
        [aggregate_path]
        if os.path.exists(aggregate_path)
        else sorted(glob.glob(os.path.join(result_dir, f"{name}_*.pkl")))
    )
    values = {}
    for path in paths:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a metric dictionary")
        values.update(payload)
    return values


def finite_metric_summary(values):
    sanitized = {}
    invalid_ids = []
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if not np.isfinite(numeric):
            numeric = 0.0
            invalid_ids.append(str(key))
        sanitized[str(key)] = numeric
    return sanitized, {
        "mean_percent": (
            100.0 * sum(sanitized.values()) / len(sanitized)
            if sanitized else None
        ),
        "count": len(sanitized),
        "invalid_count": len(invalid_ids),
        "invalid_ids": invalid_ids,
        "invalid_treatment": "replaced_with_zero",
    }


def _load_subtask_metrics(result_dir):
    aggregate_path = os.path.join(result_dir, "subtask_metrics.json")
    paths = (
        [aggregate_path]
        if os.path.exists(aggregate_path)
        else sorted(glob.glob(os.path.join(result_dir, "subtask_metrics_*.json")))
    )
    records = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            records.update(json.load(handle))
    return records


def _legacy_task_summary(result_dir):
    by_task = defaultdict(dict)
    for name in ("success_by_task", "spl_by_task"):
        aggregate_path = os.path.join(result_dir, f"{name}.pkl")
        paths = (
            [aggregate_path]
            if os.path.exists(aggregate_path)
            else sorted(glob.glob(os.path.join(result_dir, f"{name}_*.pkl")))
        )
        merged = defaultdict(list)
        for path in paths:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            for goal_type, values in payload.items():
                merged[str(goal_type)].extend(values)
        for goal_type, values in merged.items():
            _, summary = finite_metric_summary({
                str(index): value for index, value in enumerate(values)
            })
            by_task[goal_type][name] = summary["mean_percent"]
            by_task[goal_type]["count"] = summary["count"]
            by_task[goal_type][f"{name}_invalid_count"] = summary["invalid_count"]
    return dict(by_task)


def _operational_summary(records):
    purposes = defaultdict(Counter)
    goal_types = defaultdict(Counter)
    totals = Counter()
    for record in records.values():
        totals["subtasks"] += 1
        totals["steps"] += int(record.get("subtask_steps") or 0)
        totals["frames"] += int(record.get("subtask_frames") or 0)
        totals["distance_m"] += float(record.get("traversed_distance_m") or 0.0)
        telemetry = record.get("vlm_telemetry") or {}
        for field in ("logical_calls", "http_attempts", "successes", "failures"):
            totals[field] += int(telemetry.get(field) or 0)
        totals["vlm_elapsed_seconds"] += float(telemetry.get("elapsed_seconds") or 0.0)
        for purpose, values in (telemetry.get("by_purpose") or {}).items():
            for field in ("logical_calls", "http_attempts", "successes", "failures"):
                purposes[str(purpose)][field] += int(values.get(field) or 0)
            purposes[str(purpose)]["elapsed_seconds"] += float(
                values.get("elapsed_seconds") or 0.0
            )
        goal_type = str(record.get("goal_type", "unknown"))
        goal_types[goal_type]["subtasks"] += 1
        goal_types[goal_type]["steps"] += int(record.get("subtask_steps") or 0)
        goal_types[goal_type]["logical_calls"] += int(telemetry.get("logical_calls") or 0)
    return {
        "subtasks": int(totals["subtasks"]),
        "steps": int(totals["steps"]),
        "frames": int(totals["frames"]),
        "distance_m": float(totals["distance_m"]),
        "vlm": {
            "logical_calls": int(totals["logical_calls"]),
            "http_attempts": int(totals["http_attempts"]),
            "successes": int(totals["successes"]),
            "failures": int(totals["failures"]),
            "elapsed_seconds": float(totals["vlm_elapsed_seconds"]),
        },
        "vlm_by_purpose": {
            purpose: dict(values) for purpose, values in sorted(purposes.items())
        },
        "by_goal_type": {
            goal_type: dict(values) for goal_type, values in sorted(goal_types.items())
        },
    }


def _snapshot_mapping_summary(records):
    statuses = Counter()
    rows = []
    for subtask_id, record in records.items():
        diagnostic = record.get("snapshot_mapping_diagnostics") or {}
        status = str(diagnostic.get("status", "unknown"))
        statuses[status] += 1
        rows.append({
            "subtask_id": str(subtask_id),
            "status": status,
            "gt_object_ids": diagnostic.get("gt_object_ids", []),
            "raw_target_ids": diagnostic.get("raw_target_ids", []),
            "canonical_target_ids": diagnostic.get("canonical_target_ids", []),
            "raw_selected_ids": diagnostic.get("raw_selected_ids", []),
            "canonical_selected_ids": diagnostic.get("canonical_selected_ids", []),
            "mapping_votes": diagnostic.get("mapping_votes", {}),
            "aliases": diagnostic.get("aliases", {}),
        })
    return {"statuses": dict(sorted(statuses.items())), "subtasks": rows}


def _route_diagnostics(result_dir):
    trace_paths = sorted(glob.glob(os.path.join(
        result_dir, "active_topology_traces", "*.jsonl"
    )))
    counts = Counter()
    place_route_modes = Counter()
    cancellation_reasons = Counter()
    topology_lengths = []
    direct_lengths = []
    detour_ratios = []
    metric_action_types = Counter()
    metric_route_modes = Counter()
    metric_shadow_views = 0
    metric_views_by_goal_type = Counter()
    metric_actions_by_goal_type = defaultdict(Counter)
    metric_reachable_actions = 0
    metric_unreachable_actions = 0
    relevance_margins = []
    revisit_to_explore_ratios = []
    revisit_budget_ratios = []
    direct_first_modes = Counter()
    direct_first_by_goal_type = defaultdict(Counter)
    direct_first_fallback_reasons = Counter()
    direct_first_direct_lengths = []
    direct_first_topology_lengths = []
    revisit_gate_reasons = Counter()
    revisit_gate_by_goal_type = defaultdict(Counter)
    target_commitment_reasons = Counter()
    lightweight_confidence_views = 0
    lightweight_confidence_changed_actions = 0
    lightweight_positive_views = 0
    lightweight_negative_views = 0
    dynamic_goal_views = 0
    dynamic_goal_active_views = 0
    dynamic_goal_changed_actions = 0
    dynamic_goal_executable_nodes = 0
    dynamic_goal_context_nodes = 0
    dynamic_goal_propagated_nodes = 0
    dynamic_goal_edge_types = Counter()
    dynamic_goal_edge_statuses = Counter()
    dynamic_goal_uncertainties = []
    dynamic_goal_relevance_deltas = []
    goal_overlay_views = 0
    goal_overlay_candidates = Counter()
    goal_overlay_valid_candidates = 0
    goal_overlay_invalid_candidates = 0
    goal_overlay_relations = Counter()
    goal_overlay_costs = []
    hierarchical_intents = Counter()
    hierarchical_execution_modes = Counter()
    hierarchical_verdicts = Counter()
    hierarchical_commands = Counter()
    unified_verdicts = Counter()
    unified_actions = Counter()
    unified_confirmed_subtasks = set()
    unified_intents = Counter()
    unified_sources = Counter()
    unified_historical_sources = 0
    unified_historical_source_observations = 0
    unified_sources_by_subtask = defaultdict(Counter)
    unified_evidence_by_subtask = defaultdict(Counter)
    completed_revisit_evidence = set()
    unified_authorization_reselections = defaultdict(int)
    unified_incremental_assessments = defaultdict(int)
    unified_successful_incremental_preemptions = defaultdict(int)
    unified_selective_verification_tasks = defaultdict(int)
    unified_bounded_intent_expirations = defaultdict(int)
    scene_map_goal_starts = []
    scene_map_goal_ends = []
    for path in trace_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                event_name = str(event.get("event", "unknown"))
                counts[event_name] += 1
                navigator_stats = ((event.get("navigator") or {}).get("stats") or {})
                unified_incremental_assessments[path] = max(
                    unified_incremental_assessments[path],
                    int(navigator_stats.get("incremental_assessments", 0)),
                )
                unified_successful_incremental_preemptions[path] = max(
                    unified_successful_incremental_preemptions[path],
                    int(navigator_stats.get("successful_incremental_preemptions", 0)),
                )
                unified_selective_verification_tasks[path] = max(
                    unified_selective_verification_tasks[path],
                    int(navigator_stats.get("selective_verification_tasks", 0)),
                )
                unified_bounded_intent_expirations[path] = max(
                    unified_bounded_intent_expirations[path],
                    int(navigator_stats.get("bounded_intent_expirations", 0)),
                )
                if event_name == "place_route_planned":
                    place_route_modes[event["plan"]["route_mode"]] += 1
                if event_name == "topology_route_commitment_cancelled":
                    cancellation_reasons[str(event.get("reason", "unknown"))] += 1
                if event_name == "topology_route_selected":
                    for field, target in (
                        ("topology_route_length_m", topology_lengths),
                        ("direct_geodesic_distance_m", direct_lengths),
                        ("route_detour_ratio", detour_ratios),
                    ):
                        value = event.get(field)
                        if value is not None and np.isfinite(float(value)):
                            target.append(float(value))
                if event_name == "metric_action_shadow_view":
                    metric_shadow_views += 1
                    shadow_view = event.get("view") or {}
                    goal_type = str(shadow_view.get("goal_type", "unknown"))
                    metric_views_by_goal_type[goal_type] += 1
                    actions = shadow_view.get("actions") or []
                    for action in actions:
                        action_type = str(action.get("action_type", "unknown"))
                        metric_action_types[action_type] += 1
                        metric_actions_by_goal_type[goal_type][action_type] += 1
                        metric_route_modes[str(action.get("route_mode", "unknown"))] += 1
                        if action.get("reachable"):
                            metric_reachable_actions += 1
                        else:
                            metric_unreachable_actions += 1
                    reachable = [action for action in actions if action.get("reachable")]
                    revisits = [
                        action for action in reachable
                        if action.get("action_type") == "revisit"
                    ]
                    alternatives = [
                        action for action in reachable
                        if action.get("action_type") != "revisit"
                    ]
                    explores = [
                        action for action in reachable
                        if action.get("action_type") == "explore"
                        and action.get("effective_cost_m") is not None
                    ]
                    if revisits:
                        best_revisit = max(
                            revisits, key=lambda action: float(action.get("relevance", 0.0))
                        )
                        budget_ratio = best_revisit.get("budget_ratio")
                        if budget_ratio is not None:
                            revisit_budget_ratios.append(float(budget_ratio))
                        if alternatives:
                            best_alternative = max(
                                alternatives,
                                key=lambda action: float(action.get("relevance", 0.0)),
                            )
                            relevance_margins.append(
                                float(best_revisit.get("relevance", 0.0))
                                - float(best_alternative.get("relevance", 0.0))
                            )
                        if explores:
                            revisit_cost = best_revisit.get("effective_cost_m")
                            explore_cost = min(
                                float(action["effective_cost_m"])
                                for action in explores
                            )
                            if revisit_cost is not None and explore_cost > 0.0:
                                revisit_to_explore_ratios.append(
                                    float(revisit_cost) / explore_cost
                                )
                if event_name == "direct_first_route_planned":
                    plan = event.get("plan") or {}
                    route_mode = str(plan.get("route_mode", "unknown"))
                    goal_type = str(event.get("goal_type", "unknown"))
                    direct_first_modes[route_mode] += 1
                    direct_first_by_goal_type[goal_type][route_mode] += 1
                    fallback_reason = plan.get("fallback_reason")
                    if fallback_reason:
                        direct_first_fallback_reasons[str(fallback_reason)] += 1
                    for field, target in (
                        ("direct_geodesic_m", direct_first_direct_lengths),
                        ("topology_route_m", direct_first_topology_lengths),
                    ):
                        value = plan.get(field)
                        if value is not None and np.isfinite(float(value)):
                            target.append(float(value))
                if event_name == "revisit_intervention_gate":
                    decision = event.get("decision") or {}
                    reason = str(decision.get("reason", "unknown"))
                    goal_type = str(event.get("goal_type", "unknown"))
                    revisit_gate_reasons[reason] += 1
                    revisit_gate_by_goal_type[goal_type][reason] += 1
                if event_name in (
                    "target_commitment_kept",
                    "target_commitment_switched",
                    "target_commitment_completed",
                    "target_commitment_cancelled",
                    "target_commitment_preempted",
                ):
                    reason = event.get("reason")
                    if reason is None:
                        reason = (event.get("decision") or {}).get(
                            "reason", "unknown"
                        )
                    target_commitment_reasons[str(reason)] += 1
                if event_name == "lightweight_goal_confidence":
                    lightweight_confidence_views += 1
                    confidence_view = event.get("view") or {}
                    lightweight_confidence_changed_actions += int(
                        confidence_view.get("changed_actions", 0)
                    )
                    for item in confidence_view.get("actions") or []:
                        state = item.get("state") or {}
                        lightweight_positive_views += int(
                            state.get("positive_views", 0)
                        )
                        lightweight_negative_views += int(
                            state.get("negative_views", 0)
                        )
                if event_name == "dynamic_goal_topology_view":
                    dynamic_goal_views += 1
                    goal_view = event.get("view") or {}
                    dynamic_goal_active_views += int(
                        bool(goal_view.get("behavior_applied", False))
                    )
                    dynamic_goal_changed_actions += int(
                        goal_view.get("changed_actions", 0)
                    )
                    dynamic_goal_executable_nodes += int(
                        goal_view.get("executable_nodes", 0)
                    )
                    dynamic_goal_context_nodes += int(
                        goal_view.get("context_nodes", 0)
                    )
                    for node in goal_view.get("nodes") or []:
                        if not node.get("executable"):
                            continue
                        uncertainty = node.get("uncertainty")
                        if uncertainty is not None and np.isfinite(
                            float(uncertainty)
                        ):
                            dynamic_goal_uncertainties.append(float(uncertainty))
                        raw = node.get("raw_relevance")
                        dynamic = node.get("dynamic_relevance")
                        if (
                            raw is not None and dynamic is not None
                            and np.isfinite(float(raw))
                            and np.isfinite(float(dynamic))
                        ):
                            dynamic_goal_relevance_deltas.append(
                                float(dynamic) - float(raw)
                            )
                        if node.get("propagated_support") is not None:
                            dynamic_goal_propagated_nodes += 1
                    for edge in goal_view.get("edges") or []:
                        dynamic_goal_edge_types[str(
                            edge.get("edge_type", "unknown")
                        )] += 1
                        dynamic_goal_edge_statuses[str(
                            edge.get("status", "unknown")
                        )] += 1
                if event_name == "goal_conditioned_topology_overlay":
                    goal_overlay_views += 1
                    overlay_view = event.get("view") or {}
                    for candidate in overlay_view.get("candidates") or []:
                        goal_overlay_candidates[str(
                            candidate.get("action_type", "unknown")
                        )] += 1
                        validity = candidate.get("validity") or {}
                        if validity.get("valid"):
                            goal_overlay_valid_candidates += 1
                        else:
                            goal_overlay_invalid_candidates += 1
                        for relation in candidate.get(
                            "topology_relations", []
                        ):
                            goal_overlay_relations[str(relation)] += 1
                        cost = candidate.get("cost_m")
                        if cost is not None and np.isfinite(float(cost)):
                            goal_overlay_costs.append(float(cost))
                if event_name == "hierarchical_intent_installed":
                    intent = event.get("intent") or {}
                    hierarchical_intents[str(intent.get("kind", "unknown"))] += 1
                    hierarchical_execution_modes[str(
                        intent.get("execution_mode", "unknown")
                    )] += 1
                if event_name == "hierarchical_terminal_verification":
                    result = event.get("result") or {}
                    command = event.get("command") or {}
                    hierarchical_verdicts[str(result.get("verdict", "unknown"))] += 1
                    hierarchical_commands[str(command.get("command", "unknown"))] += 1
                if event_name == "hypothesis_aware_goal_verification":
                    verdict = str(event.get("verdict", "unknown"))
                    action = str(event.get("action", "unknown"))
                    unified_verdicts[verdict] += 1
                    unified_actions[action] += 1
                    if verdict == "confirmed" and event.get("subtask_id") is not None:
                        unified_confirmed_subtasks.add(str(event["subtask_id"]))
                if event_name == "hypothesis_aware_selection":
                    navigator = event.get("navigator") or {}
                    intent = navigator.get("active_intent") or {}
                    task = intent.get("task") or {}
                    kind = task.get("kind")
                    source = task.get("source_id")
                    if kind:
                        unified_intents[str(kind)] += 1
                    if source:
                        source = str(source)
                        unified_sources[source] += 1
                        unified_sources_by_subtask[str(event.get("subtask_id"))][source] += 1
                    digest = task.get("evidence_digest")
                    if digest:
                        unified_evidence_by_subtask[str(event.get("subtask_id"))][
                            str(digest)
                        ] += 1
                    if "historical_source" in task:
                        unified_historical_source_observations += 1
                        unified_historical_sources += int(bool(task["historical_source"]))
                    stats = navigator.get("stats") or {}
                    unified_authorization_reselections[path] = max(
                        unified_authorization_reselections[path],
                        int(stats.get("authorization_reselections", 0)),
                    )
                if event_name in (
                    "hypothesis_aware_revisit_completed",
                    "hypothesis_aware_revisit_promoted",
                ):
                    revisit = event.get("revisit") or {}
                    digest = revisit.get("evidence_digest")
                    if digest:
                        completed_revisit_evidence.add(str(digest))
                if event_name in ("dual_dynamic_goal_started", "dual_dynamic_goal_end"):
                    scene_map = ((event.get("navigator") or {}).get("scene_map") or {})
                    row = {name: int(scene_map.get(name, 0))
                           for name in ("places", "entities", "frontiers", "edges")}
                    for name in (
                        "evidence_count", "prior_goal_evidence_count",
                        "entities_with_prior_goal_evidence",
                    ):
                        row[name] = (None if name not in scene_map
                                     else int(scene_map[name]))
                    row["subtask_id"] = event.get("subtask_id")
                    (scene_map_goal_starts if event_name.endswith("started")
                     else scene_map_goal_ends).append(row)

    def distribution(values):
        if not values:
            return {"count": 0, "mean": None, "median": None,
                    "p25": None, "p75": None, "min": None, "max": None}
        array = np.asarray(values, dtype=float)
        return {
            "count": len(values),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p25": float(np.percentile(array, 25)),
            "p75": float(np.percentile(array, 75)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }
    return {
        "trace_files": len(trace_paths),
        "place_route_only": {
            "plans": counts["place_route_planned"],
            "route_modes": dict(place_route_modes),
            "waypoint_arrivals": counts["place_route_waypoint_arrived"],
            "commitments_created": counts["place_route_commitment_created"],
            "commitments_resumed": counts["place_route_commitment_resumed"],
            "terminal_segments_started": counts["place_route_commitment_completed"],
            "commitments_released": counts["place_route_commitment_released"],
            "waypoint_fallbacks": counts["place_route_waypoint_fallback"],
            "execution_failures": counts["place_route_execution_failure"],
        },
        "plans": counts["topology_route_selected"],
        "commitments_created": counts["topology_route_commitment_created"],
        "commitments_completed": counts["topology_route_commitment_completed"],
        "commitments_cancelled": counts["topology_route_commitment_cancelled"],
        "cancellation_reasons": dict(cancellation_reasons),
        "mean_topology_route_length_m": (
            sum(topology_lengths) / len(topology_lengths) if topology_lengths else None
        ),
        "mean_direct_geodesic_distance_m": (
            sum(direct_lengths) / len(direct_lengths) if direct_lengths else None
        ),
        "mean_route_detour_ratio": (
            sum(detour_ratios) / len(detour_ratios) if detour_ratios else None
        ),
        "metric_action_shadow": {
            "views": metric_shadow_views,
            "views_by_goal_type": dict(metric_views_by_goal_type),
            "actions_by_type": dict(metric_action_types),
            "actions_by_goal_type": {
                goal_type: dict(counts)
                for goal_type, counts in metric_actions_by_goal_type.items()
            },
            "route_modes": dict(metric_route_modes),
            "reachable_actions": metric_reachable_actions,
            "unreachable_actions": metric_unreachable_actions,
            "best_revisit_relevance_margin": distribution(relevance_margins),
            "best_revisit_to_cheapest_explore_cost_ratio": distribution(
                revisit_to_explore_ratios
            ),
            "best_revisit_budget_ratio": distribution(revisit_budget_ratios),
        },
        "direct_first_execution": {
            "plans": counts["direct_first_route_planned"],
            "anchor_reached": counts["direct_first_anchor_reached"],
            "post_anchor_original_hgr": counts[
                "direct_first_original_hgr_after_anchor"
            ],
            "route_modes": dict(direct_first_modes),
            "route_modes_by_goal_type": {
                goal_type: dict(mode_counts)
                for goal_type, mode_counts in direct_first_by_goal_type.items()
            },
            "fallback_reasons": dict(direct_first_fallback_reasons),
            "mean_direct_geodesic_m": (
                sum(direct_first_direct_lengths)
                / len(direct_first_direct_lengths)
                if direct_first_direct_lengths else None
            ),
            "mean_safe_topology_route_m": (
                sum(direct_first_topology_lengths)
                / len(direct_first_topology_lengths)
                if direct_first_topology_lengths else None
            ),
        },
        "revisit_intervention_gate": {
            "decisions": counts["revisit_intervention_gate"],
            "accepted": revisit_gate_reasons["accepted"],
            "rejected": (
                counts["revisit_intervention_gate"]
                - revisit_gate_reasons["accepted"]
            ),
            "reasons": dict(revisit_gate_reasons),
            "reasons_by_goal_type": {
                goal_type: dict(reason_counts)
                for goal_type, reason_counts in revisit_gate_by_goal_type.items()
            },
        },
        "target_commitment": {
            "created": counts["target_commitment_created"],
            "kept": counts["target_commitment_kept"],
            "switched": counts["target_commitment_switched"],
            "completed": counts["target_commitment_completed"],
            "cancelled": counts["target_commitment_cancelled"],
            "preempted": counts["target_commitment_preempted"],
            "reasons": dict(target_commitment_reasons),
        },
        "lightweight_goal_confidence": {
            "views": lightweight_confidence_views,
            "changed_actions": lightweight_confidence_changed_actions,
            "reported_positive_view_sum": lightweight_positive_views,
            "reported_negative_view_sum": lightweight_negative_views,
        },
        "dynamic_goal_topology": {
            "views": dynamic_goal_views,
            "active_views": dynamic_goal_active_views,
            "changed_actions": dynamic_goal_changed_actions,
            "executable_nodes": dynamic_goal_executable_nodes,
            "context_nodes": dynamic_goal_context_nodes,
            "nodes_with_propagated_support": dynamic_goal_propagated_nodes,
            "edge_types": dict(dynamic_goal_edge_types),
            "edge_statuses": dict(dynamic_goal_edge_statuses),
            "uncertainty": distribution(dynamic_goal_uncertainties),
            "dynamic_minus_raw_relevance": distribution(
                dynamic_goal_relevance_deltas
            ),
        },
        "goal_conditioned_topology_overlay": {
            "views": goal_overlay_views,
            "candidates_by_action_type": dict(goal_overlay_candidates),
            "valid_candidates": goal_overlay_valid_candidates,
            "invalid_candidates": goal_overlay_invalid_candidates,
            "causal_topology_relations": dict(goal_overlay_relations),
            "candidate_cost_m": distribution(goal_overlay_costs),
        },
        "hierarchical_navigation": {
            "semantic_decisions": counts["hierarchical_semantic_decision"],
            "intents": counts["hierarchical_intent_installed"],
            "intents_by_kind": dict(hierarchical_intents),
            "execution_modes": dict(hierarchical_execution_modes),
            "evidence_change_replans": counts["hierarchical_evidence_changed"],
            "waypoint_arrivals": counts["hierarchical_waypoint_arrived"],
            "terminal_verifications": counts["hierarchical_terminal_verification"],
            "verification_verdicts": dict(hierarchical_verdicts),
            "verification_commands": dict(hierarchical_commands),
            "alternate_viewpoints": counts["hierarchical_verification_viewpoint_installed"],
            "goals_completed": counts["hierarchical_goal_end"],
        },
        "hypothesis_aware_navigation": {
            "terminal_verifications": counts["hypothesis_aware_goal_verification"],
            "verification_verdicts": dict(unified_verdicts),
            "verification_actions": dict(unified_actions),
            "confirmed_subtask_ids": sorted(unified_confirmed_subtasks),
            "selections_by_kind": dict(unified_intents),
            "historical_revisit_selections": unified_intents["REVISIT"],
            "historical_source_selections": (
                unified_historical_sources
                if unified_historical_source_observations else None
            ),
            "selections_with_source_age": unified_historical_source_observations,
            "distinct_sources_selected": len(unified_sources),
            "repeated_source_selections": sum(
                max(0, count - 1)
                for source_counts in unified_sources_by_subtask.values()
                for count in source_counts.values()
            ),
            "repeated_evidence_selections": sum(
                max(0, count - 1)
                for evidence_counts in unified_evidence_by_subtask.values()
                for count in evidence_counts.values()
            ),
            "completed_revisits": (
                counts["hypothesis_aware_revisit_completed"]
                + counts["hypothesis_aware_revisit_promoted"]
            ),
            "promoted_revisits": counts["hypothesis_aware_revisit_promoted"],
            "distinct_completed_revisit_evidence": len(completed_revisit_evidence),
            "authorization_reselections": sum(
                unified_authorization_reselections.values()
            ),
            "incremental_assessments": sum(
                unified_incremental_assessments.values()
            ),
            "successful_incremental_preemptions": sum(
                unified_successful_incremental_preemptions.values()
            ),
            "selective_verification_tasks": sum(
                unified_selective_verification_tasks.values()
            ),
            "bounded_intent_expirations": sum(
                unified_bounded_intent_expirations.values()
            ),
            "preemption_events": counts["hypothesis_aware_preemption"],
            "stall_recoveries": counts["hypothesis_aware_stall_recovery"],
            "known_space_motion_events": counts["known_space_motion"],
            "scene_map_goal_starts": scene_map_goal_starts,
            "scene_map_goal_ends": scene_map_goal_ends,
        },
    }


def summarize_result_dir(result_dir):
    mappings = {}
    summaries = {}
    for name in BASE_METRIC_NAMES:
        mappings[name], summaries[name] = finite_metric_summary(
            load_metric_mapping(result_dir, name)
        )
    subtask_records = _load_subtask_metrics(result_dir)
    mappings["goat_success"], mappings["goat_spl"] = {}, {}
    for subtask_id, record in subtask_records.items():
        status = record.get("execution_status")
        distance_success = record.get("success_by_distance")
        if status in (None, "unknown") or distance_success is None:
            continue
        success = status == "completed" and bool(distance_success)
        mappings["goat_success"][str(subtask_id)] = float(success)
        mappings["goat_spl"][str(subtask_id)] = (
            float(record.get("spl_by_distance") or 0.0) if success else 0.0
        )
    for name in ("goat_success", "goat_spl"):
        mappings[name], summaries[name] = finite_metric_summary(mappings[name])
    task_metrics = defaultdict(lambda: defaultdict(list))
    for subtask_id, record in subtask_records.items():
        goal_type = str(record.get("goal_type", "unknown"))
        for name in METRIC_NAMES:
            value = (mappings[name].get(str(subtask_id)) if name.startswith("goat_")
                     else record.get(name))
            if value is not None:
                task_metrics[goal_type][name].append(float(value))
    by_task = {}
    for goal_type, metrics in task_metrics.items():
        by_task[goal_type] = {
            name: 100.0 * sum(values) / len(values)
            for name, values in metrics.items() if values
        }
        by_task[goal_type]["count"] = max(
            (len(values) for values in metrics.values()), default=0
        )
    if not by_task:
        by_task = _legacy_task_summary(result_dir)
    routes = _route_diagnostics(result_dir)
    verification = routes["hypothesis_aware_navigation"]
    confirmed = verification["confirmed_subtask_ids"]
    def succeeded(subtask_id, field):
        value = subtask_records.get(subtask_id, {}).get(field)
        try:
            return value is not None and float(value) >= 0.5
        except (TypeError, ValueError):
            return False
    snapshot_true = sum(succeeded(subtask_id, "success_by_snapshot")
                        for subtask_id in confirmed)
    distance_true = sum(succeeded(subtask_id, "success_by_distance")
                        for subtask_id in confirmed)
    snapshot_false = 0
    snapshot_mapping_unavailable = 0
    for subtask_id in confirmed:
        record = subtask_records.get(subtask_id, {})
        status = (record.get("snapshot_mapping_diagnostics") or {}).get("status")
        if status == "mapping_unavailable":
            snapshot_mapping_unavailable += 1
        elif not succeeded(subtask_id, "success_by_snapshot"):
            snapshot_false += 1
    verification.update({
        "confirmed_count": len(confirmed),
        "confirmed_snapshot_success_count": snapshot_true,
        "confirmed_distance_success_count": distance_true,
        "false_confirmation_count_by_snapshot": snapshot_false,
        "confirmed_snapshot_mapping_unavailable_count": snapshot_mapping_unavailable,
        "confirmed_snapshot_success_rate": (
            snapshot_true / len(confirmed) if confirmed else None
        ),
        "confirmed_distance_success_rate": (
            distance_true / len(confirmed) if confirmed else None
        ),
    })
    return {
        "result_dir": os.path.abspath(result_dir),
        "metrics": summaries,
        "by_task": by_task,
        "subtask_diagnostics_count": len(subtask_records),
        "execution_quality": summarize_execution_quality(subtask_records),
        "operations": _operational_summary(subtask_records),
        "snapshot_mapping": _snapshot_mapping_summary(subtask_records),
        "routes": routes,
    }, mappings


def paired_comparison(reference_label, reference, candidate_label, candidate):
    report = {
        "reference": reference_label,
        "candidate": candidate_label,
        "metrics": {},
    }
    for name in METRIC_NAMES:
        reference_values = reference.get(name, {})
        candidate_values = candidate.get(name, {})
        shared_ids = sorted(set(reference_values) & set(candidate_values))
        deltas = [
            candidate_values[key] - reference_values[key] for key in shared_ids
        ]
        by_scene = defaultdict(list)
        for key in shared_ids:
            parts = str(key).rsplit("_", 2)
            scene_id = parts[0] if len(parts) == 3 else str(key)
            by_scene[scene_id].append(candidate_values[key] - reference_values[key])
        scene_means = {
            scene_id: sum(values) / len(values)
            for scene_id, values in by_scene.items()
        }
        interval = None
        if scene_means:
            rng = np.random.default_rng(77)
            scene_ids = sorted(scene_means)
            bootstrap = []
            for _ in range(2000):
                sampled = rng.choice(scene_ids, size=len(scene_ids), replace=True)
                bootstrap.append(float(np.mean([
                    scene_means[str(scene_id)] for scene_id in sampled
                ])))
            interval = [
                100.0 * float(np.percentile(bootstrap, 2.5)),
                100.0 * float(np.percentile(bootstrap, 97.5)),
            ]
        metric = {
            "paired_count": len(shared_ids),
            "scene_count": len(scene_means),
            "candidate_minus_reference_percent_points": (
                100.0 * sum(deltas) / len(deltas) if deltas else None
            ),
            "scene_block_bootstrap_ci95_percent_points": interval,
            "scene_wins": sum(value > 0 for value in scene_means.values()),
            "scene_ties": sum(value == 0 for value in scene_means.values()),
            "scene_losses": sum(value < 0 for value in scene_means.values()),
            "reference_only_count": len(set(reference_values) - set(candidate_values)),
            "candidate_only_count": len(set(candidate_values) - set(reference_values)),
        }
        if name.startswith("success_"):
            metric["fail_to_success"] = sum(
                reference_values[key] < 0.5 <= candidate_values[key]
                for key in shared_ids
            )
            metric["success_to_fail"] = sum(
                candidate_values[key] < 0.5 <= reference_values[key]
                for key in shared_ids
            )
        report["metrics"][name] = metric
    return report
