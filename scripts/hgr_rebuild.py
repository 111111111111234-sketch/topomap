"""Print manual staged evaluation commands or audit known-space motion traces.

This tool never starts an evaluation or calls a model provider.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex


STAGES = {
    "baseline": "eval_goatbench_hgr_baseline_qwen3vl_dashscope_train.yaml",
    "shadow": "eval_goatbench_dual_dynamic_shadow.yaml",
    "known-space": "eval_goatbench_dual_dynamic_known_space.yaml",
    "route-only": "eval_goatbench_dual_dynamic_smoke.yaml",
    "memory": "eval_goatbench_hgr_dual_topo_memory.yaml",
    "intent": "eval_goatbench_hgr_dual_topo_intent.yaml",
    "verified": "eval_goatbench_hgr_dual_topo_verified.yaml",
}

ABLATIONS = {
    "memory-semantic-first": "eval_goatbench_hgr_dual_topo_memory_semantic_first.yaml",
    "memory-dedup": "eval_goatbench_hgr_dual_topo_memory_dedup.yaml",
    "memory-cost-gate": "eval_goatbench_hgr_dual_topo_memory_cost_gate.yaml",
    "intent-persistent": "eval_goatbench_hgr_dual_topo_intent_persistent.yaml",
    "intent-preempt": "eval_goatbench_hgr_dual_topo_intent_preempt.yaml",
    "verified-ablation": "eval_goatbench_hgr_dual_topo_verified_ablation.yaml",
    "adaptive-hybrid": "eval_goatbench_hgr_dual_topo_adaptive_hybrid.yaml",
}


def commands(args):
    repo = Path(__file__).resolve().parents[1]
    print(shlex.join(["cd", str(repo)]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stages = ABLATIONS if args.stage == "ablations" else {**STAGES, **ABLATIONS}
    for stage, config in stages.items():
        if args.stage not in ("all", "ablations", stage):
            continue
        if args.stage == "all" and stage in ABLATIONS:
            continue
        print(shlex.join(["env", f"CUDA_VISIBLE_DEVICES={args.gpu}",
            "MPLCONFIGDIR=/tmp/hgr-matplotlib", "YOLO_CONFIG_DIR=/tmp/hgr-ultralytics",
            "OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8", args.python,
            "run_goatbench_evaluation.py", "-cf", f"cfg/{config}",
            "--run_name", f"exp_hgr_rebuild_{stage.replace('-', '_')}_{stamp}",
            "--record_run_fingerprint", "--split", "1", "--episode_manifest",
            "cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json"]))


def audit_events(events):
    checked, blocked, violations = 0, 0, []
    for event in events:
        if event.get("event") != "known_space_motion":
            continue
        checked += 1
        if event.get("blocked"):
            blocked += 1
            continue
        record = event.get("audit") or {}
        if not record.get("valid") or record.get("fallback") or not record.get("geometry_sha256"):
            violations.append({"step": event.get("step"), "reason": "uncertified_movement"})
    return {"motion_events": checked, "blocked_events": blocked, "violations": violations,
            "passed": bool(checked) and not violations,
            "scope": "Recorded known-space motion; not complete task/semantic/performance acceptance"}


def audit(args):
    paths = sorted(Path(args.path).rglob("*.jsonl")) if Path(args.path).is_dir() else [Path(args.path)]
    events = []
    for path in paths:
        events.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    result = audit_events(events)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    command = subs.add_parser("commands", help="print only; never run smoke")
    command.add_argument("--stage", choices=["all", "ablations", *STAGES, *ABLATIONS], default="all")
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--python", default="/home/tangyuxin/miniconda3/envs/hgr/bin/python")
    command.set_defaults(func=commands)
    command = subs.add_parser("audit")
    command.add_argument("path")
    command.set_defaults(func=audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
