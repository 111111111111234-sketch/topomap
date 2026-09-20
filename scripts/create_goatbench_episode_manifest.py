#!/usr/bin/env python3
"""Create a fixed GOAT-Bench scene/episode manifest from legacy ratio selection."""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.goatbench_manifest import build_episode_manifest, load_episode_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data-dir", required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--start-ratio", type=float, default=0.0)
    parser.add_argument("--end-ratio", type=float, default=1.0)
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = build_episode_manifest(
        args.test_data_dir,
        args.seed,
        args.start_ratio,
        args.end_ratio,
        args.splits,
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for split in args.splits:
        load_episode_manifest(args.output, args.test_data_dir, split)
    print(
        f"Wrote {args.output}: {len(manifest['splits'])} splits, "
        f"{sum(len(items) for items in manifest['splits'].values())} entries"
    )


if __name__ == "__main__":
    main()

