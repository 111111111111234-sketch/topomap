#!/usr/bin/env python3
"""Replay v3's deterministic category policy over existing v2 JSONL traces.

This is a policy diagnostic, not a counterfactual navigation evaluation: later
observations in the trace still come from the original v2 trajectory.
"""

import argparse
from collections import Counter
import glob
import json
import os


PRIORITY = {
    "confirm_approach": 5,
    "navigate": 4,
    "verify": 3,
    "explore": 2,
    "unknown_explore": 1,
}


def _best(actions):
    if not actions:
        return None
    return max(actions, key=lambda action: (
        PRIORITY.get(action.get("action_type"), 0),
        float(action.get("utility", 0.0)),
        float((action.get("belief") or {}).get("posterior", 0.0)),
        float((action.get("belief") or {}).get("reachability", 0.0)),
        -float(action.get("normalized_path_cost", 0.0)),
        str(action.get("node_id", "")),
    ))


def replay(trace_dir, unknown_threshold=0.75, top_max=0.30, max_unknown=2):
    category_tasks = set()
    selected_v2 = Counter()
    selected_v3 = Counter()
    unknown_reasons = Counter()
    per_goal = {}
    projections = 0
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event") == "goal_switch":
                    if event.get("goal_type") == "category":
                        category_tasks.add(event["subtask_id"])
                    continue
                if (
                    event.get("event") != "belief_projection"
                    or event.get("subtask_id") not in category_tasks
                ):
                    continue
                projections += 1
                hypotheses = event.get("hypotheses") or {}
                goal_key = hypotheses.get("goal_key", "")
                state = per_goal.setdefault(goal_key, {
                    "consecutive": 0, "ids": set(), "top": None,
                })
                ids = set(hypotheses.get("executable_node_ids") or [])
                top = max((
                    float(item.get("posterior", 0.0))
                    for item in hypotheses.get("hypotheses") or []
                ), default=0.0)
                if ids - state["ids"] or (
                    state["top"] is not None and abs(top - state["top"]) >= 0.05
                ):
                    state["consecutive"] = 0
                state["ids"].update(ids)
                state["top"] = top

                actions = list(event.get("actions") or [])
                if actions:
                    selected_v2[actions[0].get("action_type", "none")] += 1
                unknown_ok = (
                    float(hypotheses.get("unknown_probability", 1.0)) >= unknown_threshold
                    and top < top_max and state["consecutive"] < max_unknown
                    and not any(
                        action.get("action_type") == "navigate" for action in actions
                    )
                )
                filtered = [
                    action for action in actions
                    if action.get("action_type") != "unknown_explore" or unknown_ok
                ]
                chosen = _best(filtered)
                if chosen is not None:
                    action_type = chosen.get("action_type", "none")
                    selected_v3[action_type] += 1
                    if action_type == "unknown_explore":
                        state["consecutive"] += 1
                unknown_reasons["eligible" if unknown_ok else "suppressed"] += 1
    return {
        "category_subtasks": len(category_tasks),
        "belief_projections": projections,
        "v2_trace_top_actions": dict(selected_v2),
        "v3_replayed_actions": dict(selected_v3),
        "unknown_policy": dict(unknown_reasons),
        "warning": (
            "Policy-only replay: downstream observations remain from v2 and "
            "cannot predict SR/SPL."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    args = parser.parse_args()
    print(json.dumps(replay(args.trace_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
