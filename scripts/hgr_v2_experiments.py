"""Prepare immutable V2 experiments, replay route records, and compare runs.

This script never starts navigation or makes model requests. Run from repo root.
"""
import argparse
import hashlib
import json
import random
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.route_guidance import visible, path_mask


def audit_retraction_sources(events):
    """Validate that semantic retraction changes only inferred goal state."""
    verifications = {}
    retractions = []
    for row in events:
        key = (row.get("subtask_id"), row.get("step"))
        if row.get("event") == "v2_hypothesis_verification":
            verifications.setdefault(key, []).append(row.get("result", {}))
        elif row.get("event") == "v2_hypothesis_retraction":
            retractions.append(row)

    violations = []

    def reject(row, reason):
        violations.append({
            "task": row.get("subtask_id"),
            "step": row.get("step"),
            "reason": reason,
        })

    for row in retractions:
        key = (row.get("subtask_id"), row.get("step"))
        matches = [
            result for result in verifications.get(key, [])
            if not result.get("skipped", True)
            and result.get("is_falsified", False)
        ]
        feedback = row.get("feedback")
        if not isinstance(feedback, dict):
            reject(row, "malformed_retraction_snapshot")
            continue
        affected = set(feedback.get("affected_hypotheses", []))
        if len(matches) != 1:
            reject(row, "retraction_without_matching_falsification")
        elif matches[0].get("hypothesis_node_id") not in affected:
            reject(row, "retraction_hypothesis_not_affected")
        if feedback.get("intent_cancelled") is not True:
            reject(row, "retraction_did_not_cancel_intent")
        if feedback.get("spatial_facts_preserved") is not True:
            reject(row, "retraction_modified_spatial_facts")

        before = feedback.get("candidates_before")
        after = feedback.get("candidates_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            reject(row, "malformed_retraction_snapshot")
            continue
        if set(before) != set(after):
            reject(row, "retraction_changed_candidate_set")
            continue
        for candidate_id in before:
            previous = before[candidate_id]
            current = after[candidate_id]
            if not isinstance(previous, dict) or not isinstance(current, dict):
                reject(row, "malformed_retraction_snapshot")
                break
            if set(previous.get("observed_evidence_refs", [])) != set(
                current.get("observed_evidence_refs", [])
            ):
                reject(row, "retraction_changed_observed_evidence")
                break
            expected = set(previous.get("hypothesis_refs", [])) - affected
            if set(current.get("hypothesis_refs", [])) != expected:
                reject(row, "retraction_reference_mismatch")
                break

    return {
        "retractions_checked": len(retractions),
        "violations": violations,
        "passed": bool(retractions) and not violations,
    }


def audit_events(events):
    """Check recorded route adherence and arrival ordering, not unseen geometry."""
    paths, progress, execution_contracts, counts, route_violations = {}, {}, {}, {}, []
    hierarchical_semantic_steps = {
        (row.get("subtask_id"), row.get("step"))
        for row in events
        if row.get("event") == "hierarchical_semantic_decision"
    }
    for row in events:
        event = row.get("event", "")
        if event == "hierarchical_waypoint_arrived" and (
            row.get("semantic_request_triggered") is not False
            or (row.get("subtask_id"), row.get("step")) in hierarchical_semantic_steps
        ):
            route_violations.append({
                "task": row.get("subtask_id"),
                "step": row.get("step"),
                "reason": "semantic_selection_triggered_by_waypoint",
            })
        if not event.startswith("v2_"):
            continue
        counts[event] = counts.get(event, 0)+1
        task, step = row.get("subtask_id"), row.get("step")
        if event == "v2_execution" and row.get("guidance"):
            guide, route_id = row["guidance"], row.get("route_id")
            key = (task, route_id)
            if "certified_path" in guide:
                paths[key] = np.asarray(guide["certified_path"], float)
            progress.setdefault((task, row.get("intent_id"), route_id), int(guide["progress_index"]))
            execution_contracts[(task, row.get("intent_id"))] = {
                "mode": row.get("execution_mode", "certified_route"),
                "reason": row.get("reason"),
            }
        if event != "v2_motion_acknowledged":
            continue
        contract = execution_contracts.get((task, row.get("intent_id")), {})
        if row.get("execution_mode") == "local_tsdf":
            reason = None
            if contract.get("reason") not in (
                "local_direct_retained", "verification_viewpoint_retained",
            ):
                reason = "local_motion_without_known_path_validation"
            elif not row.get("acknowledged"):
                reason = "local_motion_not_acknowledged"
            if reason:
                route_violations.append({"task": task, "step": step, "reason": reason})
            continue
        key = (task, row.get("route_id"))
        progress_key = (task, row.get("intent_id"), row.get("route_id"))
        path, guide = paths.get(key), row["guidance"]
        if path is None or progress_key not in progress:
            route_violations.append({"task": task, "step": step, "reason": "motion_without_recorded_route"})
            continue
        cursor, before = int(guide["progress_index"]), progress[progress_key]
        reason = None
        if cursor < before or cursor >= len(path):
            reason = "invalid_progress"
        elif not row["acknowledged"] and cursor != before:
            reason = "unacknowledged_progress"
        elif row.get("terminal_arrived") and (not row["acknowledged"] or cursor != len(path)-1):
            reason = "premature_terminal_arrival"
        elif row["acknowledged"]:
            segment = guide.get("last_segment")
            if segment is None or not np.allclose(segment[0], path[before]) or not np.allclose(segment[1], path[cursor]):
                reason = "motion_endpoint_mismatch"
            else:
                shape = tuple(np.ceil(np.max(path, axis=0)).astype(int)+2)
                corridor = path_mask(shape, path[before:cursor+1])
                if not visible(corridor, *segment):
                    reason = "motion_outside_certified_prefix"
                start, end = np.asarray(segment, float)
                delta = end-start
                norm = float(delta @ delta)
                last_projection = -float("inf")
                for _, index in guide.get("transitions", []):
                    if before < index <= cursor:
                        portal = path[index]
                        projection = float((portal-start) @ delta)/norm if norm else 0.
                        miss = float(np.linalg.norm(portal-(start+np.clip(projection, 0., 1.)*delta)))
                        if projection < last_projection-1e-9 or not -1e-9 <= projection <= 1+1e-9 or miss > .5:
                            reason = "ordered_transition_skipped"
                            break
                        last_projection = projection
        if reason:
            route_violations.append({"task": task, "step": step, "reason": reason})
        progress[progress_key] = cursor
    coverage = {"verification": 0, "falsification": 0, "retraction": 0, "local_repair": 0,
                "same_target_reroute": 0, "route_unavailable": 0}
    for row in events:
        if row.get("event") == "v2_hypothesis_verification" and not row.get("result", {}).get("skipped", True):
            coverage["verification"] += 1
            coverage["falsification"] += int(row["result"].get("is_falsified", False))
        if row.get("event") == "v2_hypothesis_retraction":
            coverage["retraction"] += 1
        if row.get("event") == "v2_execution" and row.get("reason") in coverage:
            coverage[row["reason"]] += 1
    source_audit = audit_retraction_sources(events)
    cascade_acceptance = (
        "passed" if source_audit["passed"]
        else "failed" if coverage["retraction"]
        else "not_covered"
    )
    violations = route_violations + source_audit["violations"]
    return {"events": counts, "violations": violations,
            "route_violations": route_violations, "source_audit": source_audit,
            "coverage": coverage, "cascade_acceptance": cascade_acceptance,
            "recovery_acceptance": "covered_requires_geometry_audit" if coverage["local_repair"] or coverage["same_target_reroute"] else "not_covered",
            "trace_contract_passed": bool(counts.get("v2_motion_acknowledged", 0)) and not route_violations,
            "scope": "recorded motion/arrival contract plus retraction source invariants; unseen geometry remains separate"}


def audit(args):
    events = []
    for file in sorted(Path(args.directory).rglob("*.jsonl")):
        with file.open() as handle:
            events.extend(json.loads(line) for line in handle if line.strip())
    result = audit_events(events)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if (not result["trace_contract_passed"]
            or result["cascade_acceptance"] == "failed"):
        raise SystemExit(1)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)


def holdout(args):
    excluded, sources = set(), []
    for path in sorted((ROOT / "cfg/manifests").glob("*.json")) + [Path(p) for p in args.exclude_manifest]:
        manifest = read_json(path)
        for entries in manifest.get("splits", {}).values():
            excluded.update(row["scene_name"] for row in entries)
        sources.append(str(path))
    for root in args.history:
        if not Path(root).is_dir():
            raise ValueError(f"History directory does not exist: {root}")
        for path in sorted(Path(root).rglob("subtask_metrics*.json")):
            for task_id in read_json(path):
                scene = task_id.rsplit("_", 2)[0]
                excluded.add(scene.split("-")[-1])
            sources.append(str(path))
    eligible = []
    for path in sorted(Path(args.data).glob("*.json")):
        if path.stem in excluded:
            continue
        payload = read_json(path)
        if len(payload.get("episodes", [])) >= 2:
            eligible.append(path)
    random.Random(77).shuffle(eligible)
    if len(eligible) < 12:
        raise ValueError(f"Only {len(eligible)} eligible scenes; need 12 without reusing development scenes")
    splits = {"1": [], "2": []}
    for path in eligible[:12]:
        episodes = read_json(path)["episodes"]
        for index in range(2):
            splits[str(index+1)].append({"scene_file": path.name, "scene_name": path.stem,
                "episode_index": index, "episode_id": str(episodes[index]["episode_id"])})
    write_json(args.output, {"schema_version": 1, "seed": 77, "dataset_dir": args.data,
        "splits": splits, "exclusion_sources": sources,
        "excluded_scene_names": sorted(excluded), "status": "frozen_before_evaluation"})
    print(hashlib.sha256(Path(args.output).read_bytes()).hexdigest(), args.output)


def commands(args):
    from omegaconf import OmegaConf
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest).resolve()
    if args.suite == "memory":
        methods = {f"{method}_{mode}": (base, {"controlled_memory": {
            "mode": mode, "directory": str(Path(args.checkpoints).resolve()),
            "preserve_hgr_memory": True, "single_subtask": True}})
            for method, base in (("baseline", "eval_goatbench_hgr_dual_topo_v2_baseline_train.yaml"),
                                 ("v2", "eval_goatbench_hgr_dual_topo_v2_d_train.yaml"))
            for mode in ("cold", "warm")}
    else:
        names = args.methods or (["baseline", "a", "b", "c_shadow", "c", "c_persistent", "d"]
                 if args.suite in ("development", "smoke") else ["baseline", "d"])
        suffix = "smoke" if args.suite == "smoke" else "train"
        methods = {name: (f"eval_goatbench_hgr_dual_topo_v2_{name}_{suffix}.yaml", {}) for name in names}
        if args.suite == "smoke" and "a" in methods:
            methods["a"] = ("eval_goatbench_hgr_dual_topo_v2_a_smoke_r2.yaml", {})
        if "a" in methods and args.suite == "smoke":
            methods["a"] = (methods["a"][0], {"hgr_topology_fusion": {"record_decisions": True}})
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             ': "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY before running}"',
             f"cd {shlex.quote(str(ROOT))}",
             "export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16"]
    for repeat in range(1, args.repeats+1):
        for method, (base, extra) in methods.items():
            for split in sorted(read_json(manifest)["splits"], key=int):
                name = f"{directory.name}_{method}_r{repeat}_s{split}"
                path = directory / f"{name}.yaml"
                config = {"extends": str(ROOT / "cfg" / base), "exp_name": name,
                          "episode_manifest": str(manifest), "record_run_fingerprint": True, **extra}
                with path.open("x") as handle:
                    handle.write(OmegaConf.to_yaml(OmegaConf.create(config)))
                lines.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.quote(args.python)} "
                             f"run_goatbench_evaluation.py -cf {shlex.quote(str(path))} --split {split}")
    with (directory / "run.sh").open("x") as handle:
        handle.write("\n".join(lines)+"\n")
    print(directory / "run.sh")


def replay(args):
    checked = legacy_rounded = 0
    for file in sorted(Path(args.directory).rglob("decision_*.npz")):
        with np.load(file, allow_pickle=False) as payload:
            mask = payload["known_free"]
            record = json.loads(str(payload["record"]))
        approach = record["approach"]
        path = np.asarray(approach["certified_path"], float)
        if not all(visible(mask, a, b) for a, b in zip(path, path[1:])):
            raise ValueError(f"Uncertified segment: {file}")
        terminal_matches = np.allclose(path[-1], approach["terminal"])
        legacy_grid_terminal = (approach.get("cost_basis") == "current_pose_to_edge_targets_to_terminal"
                                and np.array_equal(path[-1], np.rint(approach["terminal"])))
        if not visible(mask, path[0], path[0]) or not (terminal_matches or legacy_grid_terminal):
            raise ValueError(f"Invalid endpoints: {file}")
        legacy_rounded += int(legacy_grid_terminal and not terminal_matches)
        voxel = float(record["voxel_size"])
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*voxel)
        if not np.isclose(length, approach["total_distance_m"]) or not np.isclose(length, sum(approach["segment_distances_m"])):
            raise ValueError(f"Route cost mismatch: {file}")
        checked += 1
    if not checked:
        raise ValueError("No decision records found")
    print(json.dumps({"records_checked": checked, "legacy_rounded_terminals": legacy_rounded,
        "meaning": "stored route feasibility and cost; not policy evaluation"}))


def cache_replay(args):
    """Compare identical frozen route queries; no simulator or model calls."""
    from time import perf_counter
    from src.hgr_dual_topo import GridPathCache, build_route
    from src.place_topology import PlaceTopology, PlaceNode, PlaceEdge, PlaceEdgeEvidence, PlaceEdgeStatus
    files = sorted(Path(args.directory).rglob("decision_*.npz"))
    if not files:
        raise ValueError("No decision records found")
    rows = []
    for file in files:
        with np.load(file, allow_pickle=False) as payload:
            record = json.loads(str(payload["record"]))
            mask = payload["known_free"].copy()
        approach, raw = record["approach"], record["graph"]
        observe_distance = approach.get("observe_distance_m", args.observe_distance)
        if observe_distance is None:
            raise ValueError("Old records require --observe-distance from the run configuration")
        if approach.get("cost_basis") != "shared_certified_route":
            raise ValueError(f"Cache replay requires B or later records: {file}")
        graph = PlaceTopology(raw["place_spacing_m"], record["voxel_size"], stage="route_only")
        for node in raw["places"]:
            node = dict(node)
            node["position"] = np.asarray(node["position"])
            node["observation_ids"] = set(node["observation_ids"])
            graph.nodes[node["place_id"]] = PlaceNode(**node)
        for edge in raw["traversable_edges"]:
            edge = dict(edge)
            edge["evidence"], edge["status"] = PlaceEdgeEvidence(edge["evidence"]), PlaceEdgeStatus(edge["status"])
            edge["supporting_observation_ids"] = set(edge["supporting_observation_ids"])
            for name in ("source_anchor", "target_anchor"):
                edge[name] = None if edge[name] is None else np.asarray(edge[name])
            graph.edges[(edge["source_place_id"], edge["target_place_id"])] = PlaceEdge(**edge)
        cache = GridPathCache(True)
        outputs, seconds = [], []
        for active in (GridPathCache(False), cache, cache):
            started = perf_counter()
            plan = build_route(graph, mask, record["current"], approach["terminal"],
                approach["route_places"][-1], approach["route_places"][0], observe_distance, active)
            seconds.append(perf_counter()-started)
            outputs.append(None if plan is None else plan.approach()["route_plan"])
        if outputs[0] is None or not outputs[0] == outputs[1] == outputs[2]:
            raise ValueError(f"Cache changed route result: {file}")
        rows.append({"file": str(file), "uncached_seconds": seconds[0], "cold_cache_seconds": seconds[1],
                     "warm_cache_seconds": seconds[2], "cache": cache.statistics()})
    write_json(args.output, {"route_equivalence_passed": True, "records": rows,
        "meaning": "identical frozen queries; this does not measure navigation or cross-goal memory benefit"})


def metrics(paths):
    # One argument is one repetition; a directory may contain multiple split files.
    runs = []
    for raw in paths:
        files = []
        for group in raw.split(","):
            path = Path(group)
            files.extend([path] if path.is_file() else sorted(path.rglob("subtask_metrics_*.json")))
        rows = {}
        for file in files:
            for key, value in read_json(file).items():
                if key in rows:
                    raise ValueError(f"Duplicate subtask {key}; supply one repetition per argument")
                rows[key] = value
        if not rows:
            raise ValueError(f"No metrics: {path}")
        runs.append(rows)
    return runs


def memory_metadata(groups, expected_mode):
    result = []
    for group in groups:
        metadata = {}
        for raw in group.split(","):
            directory = Path(raw).parent if Path(raw).is_file() else Path(raw)
            for file in sorted(directory.rglob("memory_*.json")):
                row = read_json(file)
                if row["mode"] != expected_mode:
                    raise ValueError(f"Expected {expected_mode} checkpoint metadata: {file}")
                identity = {k: row[k] for k in ("checkpoint_sha256", "first_subtask", "start_position",
                                               "start_angle", "preserve_hgr_memory", "single_subtask")}
                if not identity["preserve_hgr_memory"] or not identity["single_subtask"]:
                    raise ValueError("Memory attribution requires preserved HGR memory and one checkpoint-start task")
                if file.name in metadata and metadata[file.name] != identity:
                    raise ValueError(f"Conflicting memory start: {file.name}")
                metadata[file.name] = identity
        if not metadata:
            raise ValueError(f"Missing checkpoint provenance for {group}")
        result.append(metadata)
    return result


def compare(args):
    baseline, candidate = metrics(args.baseline), metrics(args.candidate)
    if len(baseline) != len(candidate):
        raise ValueError("Repeat counts differ")
    ids = set(baseline[0])
    if any(set(run) != ids for run in baseline+candidate):
        raise ValueError("Subtask sets differ; complete missing runs instead of dropping failures")
    from src.goatbench_utils import prepare_goatbench_navigation_goals
    manifest = read_json(args.manifest)
    expected = set()
    for entries in manifest["splits"].values():
        for entry in entries:
            data = read_json(Path(manifest["dataset_dir"]) / entry["scene_file"])
            episode = next(e for e in data["episodes"] if str(e["episode_id"]) == entry["episode_id"])
            kinds, _ = prepare_goatbench_navigation_goals(entry["scene_name"], episode, data["goals"])
            single_subtask = bool(
                getattr(args, "single_subtask", False)
                or args.baseline_cold or args.candidate_cold
            )
            stop = min(args.start_subtask+1, len(kinds)) if single_subtask else len(kinds)
            expected.update((entry["scene_name"], entry["episode_id"], str(i)) for i in range(args.start_subtask, stop))
    normalized = {(scene.split("-")[-1], episode, task) for scene, episode, task in (key.rsplit("_", 2) for key in ids)}
    if normalized != expected:
        raise ValueError(f"Manifest mismatch: missing {len(expected-normalized)}, extra {len(normalized-expected)}")
    scenes = sorted({task.rsplit("_", 2)[0] for task in ids})
    access = {"sr": lambda x: float(x["success_by_distance"]),
              "spl": lambda x: float(x["spl_by_distance"]),
              "snapshot_sr": lambda x: float(x["success_by_snapshot"]),
              "snapshot_spl": lambda x: float(x["spl_by_snapshot"]),
              "path_m": lambda x: float(x["traversed_distance_m"]),
              "seconds": lambda x: float(x["timing"]["total_seconds"])}
    report = {"subtasks_per_repeat": len(ids), "repeats": len(baseline), "scenes": len(scenes), "metrics": {}}
    rng = np.random.default_rng(77)
    samples = rng.integers(0, len(scenes), size=(10000, len(scenes)))
    for name, get in access.items():
        sums, counts, before, after = [], [], [], []
        for scene in scenes:
            diffs = []
            for left, right in zip(baseline, candidate):
                for task in sorted(ids):
                    if task.rsplit("_", 2)[0] != scene:
                        continue
                    a, b = get(left[task]), get(right[task])
                    if not np.isfinite(a) or not np.isfinite(b):
                        raise ValueError(f"Invalid {name}: {task}; cannot silently exclude it")
                    before.append(a); after.append(b); diffs.append(b-a)
            sums.append(sum(diffs)); counts.append(len(diffs))
        boot = np.asarray(sums)[samples].sum(axis=1)/np.asarray(counts)[samples].sum(axis=1)
        report["metrics"][name] = {"baseline": float(np.mean(before)), "candidate": float(np.mean(after)),
            "difference": float(np.mean(after)-np.mean(before)), "ci95": np.quantile(boot, [.025, .975]).tolist()}
    m = report["metrics"]
    report["performance_gate"] = bool(m["sr"]["ci95"][0] >= -.02
        and m["snapshot_sr"]["ci95"][0] >= -.02 and m["spl"]["difference"] > 0 and
        (m["path_m"]["ci95"][1] < 0 or m["seconds"]["ci95"][1] < 0))
    report["structural_acceptance"] = "requires separate tests and online trace audit"
    report["diagnostics"] = {name: {
        "vlm_failures": sum(int(row.get("vlm_telemetry", {}).get("failures", 0)) for run in runs for row in run.values()),
        "spl_statuses": sorted({str(row.get("spl_by_distance_status")) for run in runs for row in run.values()})}
        for name, runs in (("baseline", baseline), ("candidate", candidate))}
    report["goal_types"] = {}
    for goal_type in sorted({r["goal_type"] for r in baseline[0].values()}):
        report["goal_types"][goal_type] = {
            name: {metric: float(np.mean([get(row) for run in runs for row in run.values() if row["goal_type"] == goal_type]))
                   for metric, get in access.items()} for name, runs in (("baseline", baseline), ("candidate", candidate))}
    if args.baseline_cold or args.candidate_cold:
        if not args.baseline_cold or not args.candidate_cold:
            raise ValueError("Supply both cold groups; primary baseline/candidate must be warm groups")
        cold_b, cold_c = metrics(args.baseline_cold), metrics(args.candidate_cold)
        if len(cold_b) != len(baseline) or len(cold_c) != len(candidate) or any(set(r) != ids for r in cold_b+cold_c):
            raise ValueError("Cold/warm repetitions and subtask IDs must match")
        memories = [memory_metadata(args.baseline, "warm"), memory_metadata(args.candidate, "warm"),
                    memory_metadata(args.baseline_cold, "cold"), memory_metadata(args.candidate_cold, "cold")]
        if any(items != memories[0] for items in memories[1:]):
            raise ValueError("Cold/warm groups did not restore identical checkpoint bytes and starting poses")
        if any(row["first_subtask"] != args.start_subtask for repeat in memories[0] for row in repeat.values()):
            raise ValueError("start-subtask does not match checkpoint prefix")
        report["memory_provenance_verified"] = True
        report["memory_interaction"] = {}
        for metric, get in access.items():
            sums, counts = [], []
            for scene in scenes:
                diffs = [get(wc[k])-get(wb[k])-get(cc[k])+get(cb[k])
                         for wb, wc, cb, cc in zip(baseline, candidate, cold_b, cold_c)
                         for k in sorted(ids) if k.rsplit("_", 2)[0] == scene]
                if not np.all(np.isfinite(diffs)):
                    raise ValueError(f"Invalid memory interaction metric: {metric}")
                sums.append(sum(diffs)); counts.append(len(diffs))
            boot = np.asarray(sums)[samples].sum(axis=1)/np.asarray(counts)[samples].sum(axis=1)
            report["memory_interaction"][metric] = {"difference_of_differences": sum(sums)/sum(counts),
                "ci95": np.quantile(boot, [.025, .975]).tolist()}
        report["memory_note"] = "(V2 warm - baseline warm) - (V2 cold - baseline cold); prefix cost is recorded in memory_*.json"
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("holdout")
    p.add_argument("--data", default="data/goat_bench/train/content")
    p.add_argument("--history", nargs="+", required=True)
    p.add_argument("--exclude-manifest", action="append", default=[])
    p.add_argument("--output", required=True)
    p.set_defaults(func=holdout)
    p = sub.add_parser("commands")
    p.add_argument("--suite", choices=["smoke", "development", "holdout", "memory"], required=True)
    p.add_argument("--methods", nargs="+", choices=["baseline", "a", "b", "c_shadow", "c", "c_persistent", "d"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu", default="0")
    p.add_argument("--python", default="/home/tangyuxin/miniconda3/envs/hgr/bin/python")
    p.add_argument("--checkpoints", default="results/hgr_v2_memory_checkpoints")
    p.set_defaults(func=commands)
    p = sub.add_parser("replay")
    p.add_argument("directory")
    p.set_defaults(func=replay)
    p = sub.add_parser("audit")
    p.add_argument("directory")
    p.add_argument("--output", required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser("cache-replay")
    p.add_argument("directory")
    p.add_argument("--observe-distance", type=float,
                   help="Use the run's planner.final_observe_distance in meters")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cache_replay)
    p = sub.add_parser("compare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--start-subtask", type=int, default=0, help="Use checkpoint prefix length for memory experiments")
    p.add_argument("--single-subtask", action="store_true",
                   help="Expect only the first task after start-subtask (warm-only memory validation)")
    p.add_argument("--baseline", nargs="+", required=True)
    p.add_argument("--candidate", nargs="+", required=True)
    p.add_argument("--baseline-cold", nargs="+")
    p.add_argument("--candidate-cold", nargs="+")
    p.add_argument("--output", required=True)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
