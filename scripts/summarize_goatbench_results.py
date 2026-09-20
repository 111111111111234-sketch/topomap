#!/usr/bin/env python3
"""Summarize GOAT-Bench results and compare methods on identical subtasks."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.goatbench_results import paired_comparison, summarize_result_dir


METRIC_LABELS = {
    "goat_success": "GOAT SR",
    "goat_spl": "GOAT SPL",
    "success_by_snapshot": "Snapshot SR",
    "success_by_distance": "Distance SR",
    "spl_by_snapshot": "Snapshot SPL",
    "spl_by_distance": "Distance SPL",
}


def _cell(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _counter_cell(values):
    return ", ".join(f"{key}: {value}" for key, value in sorted(values.items())) or "—"


def render_markdown(report):
    """Render the machine-readable report as a compact review document."""
    lines = ["# GOAT-Bench HGR evaluation report", ""]
    lines.extend([
        "## Metrics", "",
        "| Result | GOAT SR | GOAT SPL | Snapshot SR | Distance SR | Snapshot SPL | Distance SPL | Samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, result in report.get("results", {}).items():
        metrics = result.get("metrics", {})
        count = max((item.get("count", 0) for item in metrics.values()), default=0)
        values = [_cell(metrics.get(name, {}).get("mean_percent")) for name in METRIC_LABELS]
        lines.append(f"| {_cell(label)} | {' | '.join(values)} | {count} |")

    lines.extend([
        "", "## Snapshot ID mapping diagnostics", "",
        "| Result | Mapping statuses |",
        "| --- | --- |",
    ])
    for label, result in report.get("results", {}).items():
        statuses = (result.get("snapshot_mapping") or {}).get("statuses", {})
        lines.append(f"| {_cell(label)} | {_counter_cell(statuses)} |")

    lines.extend([
        "", "## Execution and cost", "",
        "| Result | Acceptance ready | Statuses | Errors | Steps | Distance (m) | VLM calls | HTTP attempts | VLM time (s) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, result in report.get("results", {}).items():
        quality = result.get("execution_quality", {})
        operations = result.get("operations", {})
        vlm = operations.get("vlm", {})
        lines.append(
            f"| {_cell(label)} | {_cell(quality.get('acceptance_ready'))} | "
            f"{_counter_cell(quality.get('statuses', {}))} | "
            f"{_counter_cell(quality.get('error_counts', {}))} | "
            f"{_cell(operations.get('steps'))} | {_cell(operations.get('distance_m'))} | "
            f"{_cell(vlm.get('logical_calls'))} | {_cell(vlm.get('http_attempts'))} | "
            f"{_cell(vlm.get('elapsed_seconds'))} |"
        )

    lines.extend([
        "", "## Topology, memory, and verification", "",
        "| Result | Selections | Historical source reuse | Repeated evidence | Completed revisits | Authorization reselections | Incremental assessments | Successful preemptions | Map start → end (P/E/F/X) | Prior-goal evidence at end | Verifications | False confirms | Confirmed mapping unavailable |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for label, result in report.get("results", {}).items():
        navigation = (result.get("routes", {}).get("hypothesis_aware_navigation", {}))
        starts = navigation.get("scene_map_goal_starts") or []
        ends = navigation.get("scene_map_goal_ends") or []
        def map_cell(row):
            return "/".join(str(row.get(key, 0)) for key in ("places", "entities", "frontiers", "edges"))
        growth = f"{map_cell(starts[0]) if starts else '—'} → {map_cell(ends[-1]) if ends else '—'}"
        historical = navigation.get("historical_source_selections")
        if historical is None:
            legacy_revisits = navigation.get("historical_revisit_selections", 0)
            historical = f"— (legacy REVISIT: {legacy_revisits})"
        prior_evidence = (ends[-1].get("prior_goal_evidence_count")
                          if ends else None)
        lines.append(
            f"| {_cell(label)} | {_counter_cell(navigation.get('selections_by_kind', {}))} | "
            f"{_cell(historical)} | {_cell(navigation.get('repeated_evidence_selections', 0))} | "
            f"{_cell(navigation.get('completed_revisits', 0))} | "
            f"{_cell(navigation.get('authorization_reselections', 0))} | "
            f"{_cell(navigation.get('incremental_assessments', 0))} | "
            f"{_cell(navigation.get('successful_incremental_preemptions', 0))} | "
            f"{growth} | "
            f"{_cell(prior_evidence)} | "
            f"{_cell(navigation.get('terminal_verifications', 0))} | "
            f"{_cell(navigation.get('false_confirmation_count_by_snapshot', 0))} | "
            f"{_cell(navigation.get('confirmed_snapshot_mapping_unavailable_count', 0))} |"
        )

    comparisons = report.get("paired_comparisons", [])
    if comparisons:
        lines.extend([
            "", "## Paired comparisons", "",
            "| Candidate vs reference | Metric | Paired | Scenes | Delta (pp) | Scene-block 95% CI (pp) | W/T/L scenes | Fail → success | Success → fail | Missing candidate |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ])
        for comparison in comparisons:
            pair = f"{comparison['candidate']} vs {comparison['reference']}"
            for name, metric in comparison.get("metrics", {}).items():
                lines.append(
                    f"| {_cell(pair)} | {METRIC_LABELS.get(name, name)} | "
                    f"{_cell(metric.get('paired_count'))} | "
                    f"{_cell(metric.get('scene_count'))} | "
                    f"{_cell(metric.get('candidate_minus_reference_percent_points'))} | "
                    f"{_cell(metric.get('scene_block_bootstrap_ci95_percent_points'))} | "
                    f"{_cell(metric.get('scene_wins'))}/{_cell(metric.get('scene_ties'))}/"
                    f"{_cell(metric.get('scene_losses'))} | "
                    f"{_cell(metric.get('fail_to_success'))} | "
                    f"{_cell(metric.get('success_to_fail'))} | "
                    f"{_cell(metric.get('reference_only_count'))} |"
                )
    lines.append("")
    return "\n".join(lines)


def _parse_comparison(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("comparison must use LABEL=RESULT_DIR")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("comparison must use LABEL=RESULT_DIR")
    return label, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--label", default="reference")
    parser.add_argument(
        "--compare", action="append", default=[], type=_parse_comparison,
        metavar="LABEL=RESULT_DIR",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    reference_report, reference_values = summarize_result_dir(args.result_dir)
    report = {"results": {args.label: reference_report}, "paired_comparisons": []}
    for label, result_dir in args.compare:
        candidate_report, candidate_values = summarize_result_dir(result_dir)
        report["results"][label] = candidate_report
        report["paired_comparisons"].append(paired_comparison(
            args.label, reference_values, label, candidate_values
        ))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as handle:
            handle.write(render_markdown(report))
    print(rendered)


if __name__ == "__main__":
    main()
