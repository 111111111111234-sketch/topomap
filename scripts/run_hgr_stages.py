"""Run offline regression and the seven staged HGR evaluations sequentially."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .hgr_rebuild import ABLATIONS, STAGES
except ImportError:  # Direct execution places scripts/ on sys.path.
    from hgr_rebuild import ABLATIONS, STAGES


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/tangyuxin/miniconda3/envs/hgr/bin/python"
MANIFEST = "cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json"
REVISIT_FIX_STAGES = (
    "memory-dedup",
    "memory-cost-gate",
    "intent-persistent",
    "verified-ablation",
)
ADAPTIVE_SCREEN_STAGES = (
    "route-only",
    "memory-dedup",
    "adaptive-hybrid",
)


def load_manifest(path):
    with path.open("rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("manifest must contain a non-empty 'splits' mapping")
    return payload, hashlib.sha256(raw).hexdigest()


def resolve_splits(manifest, requested):
    available = sorted((str(key) for key in manifest["splits"]), key=lambda item: int(item))
    selected = available if requested == "all" else [str(requested)]
    missing = [split for split in selected if split not in manifest["splits"]]
    if missing:
        raise ValueError(
            f"manifest has splits {available}, requested {', '.join(missing)}"
        )
    return selected


def config_chain_sha256(config_path):
    """Hash a config and its simple extends chain without importing runtime deps."""
    rows = {}
    current = config_path.resolve()
    seen = set()
    while current not in seen:
        seen.add(current)
        raw = current.read_bytes()
        rows[str(current.relative_to(REPO))] = hashlib.sha256(raw).hexdigest()
        parent = None
        for line in raw.decode("utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("extends:"):
                parent = stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"")
                break
        if not parent:
            break
        parent_path = Path(parent)
        current = (parent_path if parent_path.is_absolute()
                   else current.parent / parent_path).resolve()
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True).encode("utf-8")
    ).hexdigest()


def current_source_sha256():
    paths = list((REPO / "src").rglob("*.py"))
    paths += list((REPO / "configs").rglob("*.py"))
    paths += list(REPO.glob("run_*evaluation.py"))
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def limited_cpu_affinity(limit):
    """Return a child pre-exec hook that enforces a real logical-CPU ceiling."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None, []
    allowed = sorted(os.sched_getaffinity(0))
    selected = allowed[:min(limit, len(allowed))]
    if not selected:
        return None, []

    def apply_limit():
        os.sched_setaffinity(0, selected)

    return apply_limit, selected


def completed_result_matches(result_dir, split, repeat, manifest_sha256,
                             config_sha256, source_sha256):
    marker = result_dir / "provenance" / f"split_{split}" / "completed.json"
    if not marker.is_file():
        return False, "completion_marker_missing"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "completion_marker_invalid"
    fingerprint = payload.get("fingerprint") or {}
    checks = (
        (payload.get("split") == int(split), "split_mismatch"),
        (payload.get("orchestration_repeat") == int(repeat), "repeat_mismatch"),
        (payload.get("orchestration_config_sha256") == config_sha256,
         "config_mismatch"),
        (fingerprint.get("manifest_sha256") == manifest_sha256,
         "manifest_mismatch"),
        (fingerprint.get("source_sha256") == source_sha256, "source_mismatch"),
        (int(payload.get("subtask_count") or 0) > 0, "empty_result"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "complete_and_matching"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[
        "all", "ablations", "revisit-fix", "adaptive-screen", "offline",
        *STAGES, *ABLATIONS
    ], default="all")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8,
                        help="OMP/MKL CPU thread limit")
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--split", default="1",
                        help="manifest split number, or 'all'")
    parser.add_argument("--repeats", type=int, default=1,
                        help="number of serial repeats for every stage/split")
    parser.add_argument("--matrix-id", default=None,
                        help="stable result suffix used with --resume")
    parser.add_argument("--resume", action="store_true",
                        help="skip only complete results with matching fingerprints")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--max-subtasks", type=int, default=None,
                        help="diagnostic cap on goals per episode")
    parser.add_argument("--max-steps-per-subtask", type=int, default=None,
                        help="diagnostic cap on navigation steps per goal")
    parser.add_argument("--skip-offline", action="store_true", help="skip regression before online stages")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running tests or models")
    args = parser.parse_args()
    if args.stage == "offline" and args.skip_offline:
        parser.error("--stage offline cannot be combined with --skip-offline")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.max_subtasks is not None and args.max_subtasks < 1:
        parser.error("--max-subtasks must be positive")
    if args.max_steps_per_subtask is not None and args.max_steps_per_subtask < 1:
        parser.error("--max-steps-per-subtask must be positive")
    if args.matrix_id is not None and (
        Path(args.matrix_id).name != args.matrix_id
        or args.matrix_id in ("", ".", "..")
    ):
        parser.error("--matrix-id must be a single directory-safe name")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    manifest_payload = None
    manifest_sha256 = None
    selected_splits = []
    if args.stage != "offline":
        if not manifest_path.is_file():
            parser.error(f"Manifest not found: {manifest_path}")
        try:
            manifest_payload, manifest_sha256 = load_manifest(manifest_path)
            selected_splits = resolve_splits(manifest_payload, args.split)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"Invalid manifest: {exc}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result_suffix = args.matrix_id or stamp
    jobs = []
    if not args.skip_offline:
        jobs.append(("offline", [args.python, "-m", "unittest", "discover", "-s", "tests", "-q"]))
    if args.stage != "offline":
        matrix = ({**STAGES, **ABLATIONS} if args.stage == "adaptive-screen"
                  else ABLATIONS if args.stage in ("ablations", "revisit-fix")
                  else {**STAGES, **ABLATIONS})
        for stage, config in matrix.items():
            if args.stage not in (
                "all", "ablations", "revisit-fix", "adaptive-screen", stage
            ):
                continue
            if args.stage == "all" and stage in ABLATIONS:
                continue
            if args.stage == "revisit-fix" and stage not in REVISIT_FIX_STAGES:
                continue
            if args.stage == "adaptive-screen" and stage not in ADAPTIVE_SCREEN_STAGES:
                continue
            for repeat in range(1, args.repeats + 1):
                for split in selected_splits:
                    job_id = f"{stage}_r{repeat:02d}_s{split}"
                    run_name = (
                        f"exp_hgr_rebuild_{stage.replace('-', '_')}_r{repeat:02d}_"
                        f"{result_suffix}"
                    )
                    config_path = REPO / "cfg" / config
                    config_digest = hashlib.sha256(json.dumps({
                        "config_chain": config_chain_sha256(config_path),
                        "max_subtasks": args.max_subtasks,
                        "max_steps_per_subtask": args.max_steps_per_subtask,
                    }, sort_keys=True).encode("utf-8")).hexdigest()
                    diagnostic_args = []
                    if args.max_subtasks is not None:
                        diagnostic_args.extend([
                            "--max-subtasks", str(args.max_subtasks)
                        ])
                    if args.max_steps_per_subtask is not None:
                        diagnostic_args.extend([
                            "--max-steps-per-subtask",
                            str(args.max_steps_per_subtask),
                        ])
                    jobs.append((job_id, stage, repeat, split, run_name, config_digest, [
                        args.python, "-u", "run_goatbench_evaluation.py",
                        "-cf", f"cfg/{config}", "--run_name",
                        run_name,
                        "--record_run_fingerprint", "--split", split,
                        "--episode_manifest", args.manifest,
                        "--orchestration-config-sha256", config_digest,
                        "--orchestration-repeat", str(repeat),
                        *diagnostic_args,
                    ]))

    env = os.environ.copy()
    thread_limit = str(args.threads)
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MPLCONFIGDIR="/tmp/hgr-matplotlib",
        YOLO_CONFIG_DIR="/tmp/hgr-ultralytics",
        OMP_NUM_THREADS=thread_limit,
        MKL_NUM_THREADS=thread_limit,
        OPENBLAS_NUM_THREADS=thread_limit,
        NUMEXPR_NUM_THREADS=thread_limit,
        VECLIB_MAXIMUM_THREADS=thread_limit,
        BLIS_NUM_THREADS=thread_limit,
    )
    affinity_hook, affinity_cpus = limited_cpu_affinity(args.threads)
    if args.dry_run:
        print(shlex.join(["cd", str(REPO)]))
        for job in jobs:
            job_id, command = (job[0], job[1]) if job[0] == "offline" else (job[0], job[6])
            print(f"# {job_id}")
            print(shlex.join(["env", *[f"{key}={env[key]}" for key in (
                "CUDA_VISIBLE_DEVICES", "MPLCONFIGDIR", "YOLO_CONFIG_DIR",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS")], *command]))
        return

    if not Path(args.python).is_file():
        parser.error(f"Python interpreter not found: {args.python}; use --python /absolute/path")
    if args.stage != "offline":
        if not env.get("DASHSCOPE_API_KEY", "").strip():
            parser.error("Set DASHSCOPE_API_KEY before running online stages")

    log_dir = REPO / "artifacts" / "hgr_rebuild" / "runs" / stamp
    log_dir.mkdir(parents=True, exist_ok=False)
    summary = {"gpu": args.gpu, "threads": args.threads,
               "manifest": args.manifest, "manifest_sha256": manifest_sha256,
               "splits": selected_splits, "repeats": args.repeats,
               "matrix_id": result_suffix,
               "max_subtasks": args.max_subtasks,
               "max_steps_per_subtask": args.max_steps_per_subtask,
               "jobs": []}
    print(f"Logs and summary: {log_dir}", flush=True)
    if affinity_cpus:
        print(
            f"CPU affinity: {len(affinity_cpus)} logical CPUs "
            f"({affinity_cpus[0]}-{affinity_cpus[-1]})",
            flush=True,
        )
    source_sha256 = current_source_sha256() if args.resume else None
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = REPO / results_root
    for job in jobs:
        if job[0] == "offline":
            job_id, command = job
            stage, repeat, split = "offline", None, None
            run_name = config_digest = None
        else:
            job_id, stage, repeat, split, run_name, config_digest, command = job
        if stage == "memory-semantic-first":
            print(
                "[memory-semantic-first] Diagnostic control: revisit deduplication "
                "and persistent intent are disabled; repeated selections may run "
                "to the full step budget.",
                flush=True,
            )
        log_path = log_dir / f"{job_id}.log"
        record = {"job_id": job_id, "stage": stage, "repeat": repeat,
                  "split": split, "command": command,
                  "log": str(log_path), "status": "running"}
        summary["jobs"].append(record)
        summary_path = log_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if args.resume and stage != "offline":
            matched, reason = completed_result_matches(
                results_root / run_name, split, repeat, manifest_sha256,
                config_digest, source_sha256,
            )
            record["resume_check"] = reason
            if matched:
                record["status"] = "skipped_complete"
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"[{job_id}] Skipped complete matching result", flush=True)
                continue
        print(f"\n[{job_id}] {shlex.join(command)}", flush=True)
        try:
            with log_path.open("w") as log:
                with subprocess.Popen(
                    command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace",
                    preexec_fn=affinity_hook,
                ) as process:
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        code = 130
        except OSError as exc:
            print(f"[{job_id}] Failed to start: {exc}", file=sys.stderr)
            record["error"] = str(exc)
            code = 1
        record.update(returncode=code, status="completed" if code == 0 else "failed")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if code:
            print(f"[{job_id}] Stopped (exit {code}). Log: {log_path}", file=sys.stderr)
            raise SystemExit(code if code > 0 else 1)
    print(f"\nAll selected commands completed. Summary: {log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
