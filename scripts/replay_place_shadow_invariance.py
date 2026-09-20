#!/usr/bin/env python3
"""Offline frozen-input replay for Phase-A Place shadow invariance."""

import argparse
import json
from pathlib import Path
import sys

# Keep the documented ``python scripts/...`` entry point working without
# requiring callers to manage PYTHONPATH themselves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.place_topology import replay_place_shadow_steps


def load_records(trace_path: str):
    records = []
    with open(trace_path, "r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "step":
                continue
            record = event.get("place_shadow_replay")
            if record is not None:
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="Phase-A active_topology JSONL trace")
    args = parser.parse_args()
    result = replay_place_shadow_steps(load_records(args.trace))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
