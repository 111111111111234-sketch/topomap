#!/usr/bin/env python3
"""Audit Phase-C state transitions and compare identical IDs with frozen R5."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(result_dir, reference_manifest=None):
    result_dir = Path(result_dir)
    events = [json.loads(line) for path in sorted((result_dir / "active_topology_traces").glob("*.jsonl"))
              for line in path.read_text().splitlines() if line.strip()]
    summaries = [e for e in events if e["event"] == "phase_c_subtask_summary"]
    decisions = [e for e in events if e["event"] == "phase_c_decision"]
    outcomes = [e for e in events if e["event"] == "phase_c_execution_feedback"]
    violations = []
    for summary in summaries:
        if summary["task_success"] and not any(
            e["subtask_id"] == summary["subtask_id"]
            and (e.get("verification") or {}).get("verdict") == "matched"
            for e in outcomes
        ):
            violations.append({"subtask_id": summary["subtask_id"], "reason": "success_without_fresh_verification"})
    for event in events:
        if event["event"] == "place_route_waypoint_arrived" and event.get("semantic_target_arrived"):
            violations.append({"subtask_id": event["subtask_id"], "reason": "waypoint_marked_semantic_success"})
    for event in decisions:
        ids = [c["candidate_id"] for c in event["view"]["candidates"]]
        if len(ids) != len(set(ids)):
            violations.append({"subtask_id": event["subtask_id"], "reason": "duplicate_candidate_ids"})
    seen_topological_actions = set()
    seen_decision_revisions = set()
    for event in decisions:
        view = event.get("view", {})
        if not view.get("topological_actions") or view.get("evidence_coverage"):
            continue
        candidate_id = event.get("selection", {}).get("candidate_id")
        action_key = (event["subtask_id"], candidate_id)
        if candidate_id is not None and action_key in seen_topological_actions:
            violations.append({"subtask_id": event["subtask_id"],
                               "reason": "repeated_topological_action",
                               "candidate_id": candidate_id})
        if candidate_id is not None:
            seen_topological_actions.add(action_key)
        goal_topology = view.get("goal_topology", {})
        revision_key = (event["subtask_id"], goal_topology.get("revision"))
        if event.get("selection", {}).get("model_called") and revision_key in seen_decision_revisions:
            violations.append({"subtask_id": event["subtask_id"],
                               "reason": "model_requeried_without_goal_topology_change",
                               "revision": goal_topology.get("revision")})
        if event.get("selection", {}).get("model_called"):
            seen_decision_revisions.add(revision_key)
    # Guidance records carry the actual certified cell path. Replay continuous
    # segments against its corridor and require monotonically increasing progress.
    from src.route_guidance import path_mask, visible
    import numpy as np
    guidance_state = {}
    for event in events:
        typ, task = event["event"], event.get("subtask_id")
        if typ == "route_guidance_installed" or (typ == "route_guidance_local_repair" and event.get("repaired")):
            guide = event["guidance"]
            path = np.asarray(guide["certified_path"],float)
            shape = tuple(np.ceil(path.max(axis=0)).astype(int)+1)
            guidance_state[task] = (path_mask(shape,path),-1)
        elif typ == "route_guidance_step":
            guide = event["guidance"]
            segment = guide.get("last_segment")
            previous = guidance_state.get(task)
            if previous is None:
                violations.append({"subtask_id":task,"reason":"guidance_step_without_certificate"})
            elif segment is not None:
                mask, progress = previous
                if guide["progress_index"] < progress or not visible(mask,*segment):
                    violations.append({"subtask_id":task,"reason":"guidance_segment_outside_certificate"})
                guidance_state[task] = (mask,guide["progress_index"])
    report = {
        "result_dir": str(result_dir), "events": dict(Counter(e["event"] for e in events)),
        "completed_subtasks": len(summaries),
        "intent_outcomes": dict(Counter((e.get("feedback") or {}).get("outcome", "missing") for e in outcomes)),
        "decision_failures": dict(Counter(e["selection"]["reason"] for e in decisions if "candidate_id" not in e["selection"])),
        "violations": violations,
    }
    metrics_path = result_dir / "subtask_metrics.json"
    timing_groups = {
        "planning": [e.get("view", {}).get("planning_metrics", {}) for e in decisions],
        "selection": [e.get("selection", {}).get("timings", {}) for e in decisions],
        "verification": [(e.get("verification") or {}).get("timings", {}) for e in outcomes],
    }
    report["timings"] = {}
    for group, samples in timing_groups.items():
        report["timings"][group] = {}
        for key in sorted({key for sample in samples for key in sample}):
            values = [sample[key] for sample in samples if key in sample]
            report["timings"][group][key] = {
                "count": len(values), "total": sum(values), "median": statistics.median(values),
            }
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for typ in ("all", "object", "description", "image"):
        rows = [r for r in metrics.values() if typ == "all" or r["goal_type"] == typ]
        report.setdefault("metrics", {})[typ] = {
            "count": len(rows), **{key: statistics.mean(r[key] for r in rows) if rows else None
                for key in ("success_by_distance", "spl_by_distance", "subtask_steps", "traversed_distance_m")},
            "vlm_calls": sum(r.get("vlm_telemetry", {}).get("logical_calls", 0) for r in rows),
        }
    if reference_manifest:
        manifest_path = Path(reference_manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        root = manifest_path.parents[2]
        for file in manifest["files"]:
            path = root / file["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != file["sha256"]:
                raise ValueError(f"Frozen R5 reference changed: {path}")
        baseline = json.loads((root / manifest["subtask_metrics"]).read_text())
        ids = sorted(metrics.keys() & baseline.keys())
        report["paired_r5"] = {
            "common_subtasks": len(ids),
            "new_successes": sum(bool(metrics[s]["success_by_distance"]) and not baseline[s]["success_by_distance"] for s in ids),
            "lost_successes": sum(bool(baseline[s]["success_by_distance"]) and not metrics[s]["success_by_distance"] for s in ids),
            **{key + "_delta": statistics.mean(metrics[s][key] - baseline[s][key] for s in ids) if ids else None
               for key in ("success_by_distance", "spl_by_distance", "subtask_steps", "traversed_distance_m")},
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--reference-manifest", default="cfg/manifests/place_goal_phase_c_r5_reference.json")
    parser.add_argument("--expected-subtasks", type=int)
    args = parser.parse_args()
    report = audit(args.result_dir, args.reference_manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if report["violations"] or (args.expected_subtasks is not None and report["completed_subtasks"] != args.expected_subtasks):
        raise SystemExit(1)
