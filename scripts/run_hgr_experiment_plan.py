"""Run the reduced development screen or a separately frozen final evaluation.

Development runs five configurations on the 3x2 manifest. Final evaluation is
an explicit command requiring a frozen candidate and runs Baseline, Route-only,
and that candidate three times on an independent 12x2 manifest. Individual
evaluations use run_hgr_stages.py, retaining its fingerprint and resume rules.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/home/tangyuxin/miniconda3/envs/hgr/bin/python"
DEFAULT_DEV_MANIFEST = "cfg/manifests/goat_train_fast_3x2_seed77.json"
DEFAULT_FINAL_MANIFEST = "cfg/manifests/goat_train_validation2_12x2_seed77.json"
FINAL_CANDIDATES = (
    "route-only", "memory-dedup", "memory-cost-gate", "intent-persistent",
    "verified-ablation", "adaptive-hybrid",
)
DEVELOPMENT_STAGES = (
    "route-only", "memory-dedup", "memory-cost-gate",
    "intent-persistent", "verified-ablation",
)


def result_dir(results_root, stage, repeat, matrix_id):
    return results_root / (
        f"exp_hgr_rebuild_{stage.replace('-', '_')}_r{repeat:02d}_{matrix_id}"
    )


def phase_definition(args):
    if args.phase == "development":
        return DEVELOPMENT_STAGES, args.dev_manifest, 1, "c_lite"
    stages = ["baseline", "route-only"]
    if args.final_candidate not in stages:
        stages.append(args.final_candidate)
    return (
        tuple(stages),
        args.final_manifest, 3, "final",
    )


def build_jobs(args):
    """Return serial runner jobs for exactly one experiment phase."""
    stages, manifest, repeats, suffix = phase_definition(args)
    matrix_id = f"{args.matrix_prefix}_{suffix}"
    jobs = []
    for stage in stages:
        command = [
            args.python, "scripts/run_hgr_stages.py",
            "--stage", stage,
            "--gpu", str(args.gpu),
            "--threads", str(args.threads),
            "--manifest", manifest,
            "--split", "all",
            "--repeats", str(repeats),
            "--matrix-id", matrix_id,
            "--results-root", args.results_root,
            "--resume", "--skip-offline",
        ]
        if args.dry_run:
            command.append("--dry-run")
        jobs.append({
            "phase": args.phase, "stage": stage, "manifest": manifest,
            "split": "all", "repeats": repeats, "matrix_id": matrix_id,
            "command": command,
        })
    return jobs


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def audit_result(path):
    """Check structural acceptance and every recorded known-space movement."""
    if not path.is_dir():
        raise RuntimeError(f"result directory is missing: {path}")
    quality_path = path / "execution_quality.json"
    quality = load_json(quality_path)
    statuses = quality.get("statuses", {})
    error_counts = quality.get("error_counts", {})
    if quality.get("missing_outcome_count", 0) or statuses.get("unknown", 0):
        raise RuntimeError(f"outcome coverage failed: {quality_path}")
    # Provider/selection failures are valid benchmark outcomes and must stay in
    # the main table. Internal computation/execution failures still stop the
    # experiment matrix because later comparisons would not be trustworthy.
    classified_online_failures = {
        "request", "selection", "verification_request", "parse",
    }
    fatal_errors = {
        name: count for name, count in error_counts.items()
        if count and name not in classified_online_failures
    }
    if fatal_errors:
        raise RuntimeError(
            f"internal execution errors in {quality_path}: {fatal_errors}"
        )

    motion_events = 0
    violations = []
    for trace_path in sorted((path / "active_topology_traces").glob("*.jsonl")):
        for line_number, line in enumerate(
                trace_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "known_space_motion":
                continue
            motion_events += 1
            if event.get("blocked"):
                continue
            record = event.get("audit") or {}
            if (not record.get("valid") or record.get("fallback")
                    or not record.get("geometry_sha256")):
                violations.append({
                    "trace": str(trace_path), "line": line_number,
                    "step": event.get("step"),
                })
    if violations:
        raise RuntimeError(
            f"known-space motion audit failed for {path}: {violations[:3]}"
        )
    return {
        "acceptance_ready": bool(quality.get("acceptance_ready")),
        "statuses": statuses,
        "error_counts": error_counts,
        "classified_online_failure": bool(error_counts) and not fatal_errors,
        "motion_events": motion_events,
        "motion_violations": 0,
    }


def run(command, dry_run=False):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO, check=True)


def write_reports(args, stages, repeats, matrix_id):
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO / results_root
    report_root = REPO / "artifacts" / "hgr_rebuild" / "experiment_plan"
    report_root.mkdir(parents=True, exist_ok=True)
    reference = stages[0]
    for repeat in range(1, repeats + 1):
        command = [
            args.python, "scripts/summarize_goatbench_results.py",
            str(result_dir(results_root, reference, repeat, matrix_id)),
            "--label", reference,
        ]
        for stage in stages[1:]:
            command.extend([
                "--compare",
                f"{stage}={result_dir(results_root, stage, repeat, matrix_id)}",
            ])
        stem = f"{matrix_id}_r{repeat:02d}"
        command.extend([
            "--output", str(report_root / f"{stem}.json"),
            "--markdown-output", str(report_root / f"{stem}.md"),
        ])
        run(command, args.dry_run)


def validate_args(parser, args):
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if Path(args.matrix_prefix).name != args.matrix_prefix or not args.matrix_prefix:
        parser.error("--matrix-prefix must be one directory-safe name")
    if args.phase == "final" and not args.final_candidate:
        parser.error(
            "--phase final requires --final-candidate after reviewing development results"
        )
    if args.final_candidate and args.final_candidate not in FINAL_CANDIDATES:
        parser.error(
            "--final-candidate must be one of: " + ", ".join(FINAL_CANDIDATES)
        )
    manifest = args.dev_manifest if args.phase == "development" else args.final_manifest
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("development", "final"), default="development",
        help="development is the default 3x2 screen; final must run separately",
    )
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--matrix-prefix", default="hgr_plan_v2")
    parser.add_argument("--dev-manifest", default=DEFAULT_DEV_MANIFEST)
    parser.add_argument("--final-manifest", default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--final-candidate", default=None,
                        help="candidate frozen after reviewing development report")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(parser, args)

    if not args.dry_run:
        if not Path(args.python).is_file():
            parser.error(f"Python interpreter not found: {args.python}")
        if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
            parser.error("Set DASHSCOPE_API_KEY before running online evaluations")

    if not args.skip_offline:
        offline = [
            args.python, "scripts/run_hgr_stages.py", "--stage", "offline",
            "--threads", str(args.threads), "--gpu", str(args.gpu),
        ]
        if args.dry_run:
            offline.append("--dry-run")
        run(offline, args.dry_run)

    jobs = build_jobs(args)
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO / results_root
    state = {
        "phase": args.phase, "matrix_prefix": args.matrix_prefix,
        "final_candidate": args.final_candidate, "jobs": [],
    }
    state_path = (
        REPO / "artifacts" / "hgr_rebuild" /
        f"{args.matrix_prefix}_{args.phase}_state.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {args.phase.capitalize()} phase ===", flush=True)
    for job in jobs:
        run(job["command"], args.dry_run)
        record = {key: value for key, value in job.items() if key != "command"}
        record["status"] = "planned" if args.dry_run else "completed"
        record["audits"] = []
        if not args.dry_run:
            for repeat in range(1, job["repeats"] + 1):
                path = result_dir(
                    results_root, job["stage"], repeat, job["matrix_id"]
                )
                record["audits"].append(audit_result(path))
        state["jobs"].append(record)
        if not args.dry_run:
            state_path.write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8"
            )

    stages, _, repeats, _ = phase_definition(args)
    write_reports(args, stages, repeats, jobs[0]["matrix_id"])
    if args.dry_run:
        print("\nDry run complete; no test, simulator, or model request was made.")
    else:
        print(f"\nPhase completed. State: {state_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Stopped after command failed with exit {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
    except RuntimeError as exc:
        print(f"Stopped by acceptance audit: {exc}", file=sys.stderr)
        raise SystemExit(1)
