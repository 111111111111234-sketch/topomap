#!/usr/bin/env python3
"""Render the Phase-A Place graph stored in an episode summary JSON."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def render(summary_path: str, output_path: str) -> None:
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    graph = summary.get("place_topology")
    if not graph:
        raise ValueError(f"No place_topology payload in {summary_path}")
    places = {item["place_id"]: item for item in graph.get("places", [])}

    fig, ax = plt.subplots(figsize=(8, 8))
    for edge in graph.get("traversable_edges", []):
        source = places.get(edge["source_place_id"])
        target = places.get(edge["target_place_id"])
        if source is None or target is None:
            continue
        sx, sy = source["position"][:2]
        tx, ty = target["position"][:2]
        valid = edge.get("status") == "valid"
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(sx, sy),
            arrowprops={
                "arrowstyle": "->",
                "color": "#2c7fb8" if valid else "#999999",
                "linestyle": "-" if valid else "--",
                "alpha": 0.8,
            },
        )
    for place_id, place in places.items():
        x, y = place["position"][:2]
        ax.scatter([x], [y], color="#d95f0e", s=45, zorder=3)
        ax.text(x, y, f" {place_id}", fontsize=8, va="bottom")

    for frontier in graph.get("frontiers", []):
        x, y = frontier["frontier_position"][:2]
        ax.scatter([x], [y], marker="x", color="#31a354", s=35)
        approach = frontier.get("approach_place_id")
        if approach in places:
            px, py = places[approach]["position"][:2]
            ax.plot([px, x], [py, y], color="#31a354", linestyle=":", alpha=0.5)

    ax.set_title("HGR Phase-A Place topology (directed valid edges)")
    ax.set_xlabel("TSDF voxel x")
    ax.set_ylabel("TSDF voxel y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", help="*.summary.json produced by a shadow run")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    render(args.summary, args.output)


if __name__ == "__main__":
    main()
