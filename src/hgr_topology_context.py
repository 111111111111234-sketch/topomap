"""Spatial context for the existing HGR selector and Place route executor."""

import json

from src.topology_navigation import select_reachable_approach_place


def build_hgr_topology_context(graph, snapshots, frontiers):
    """Use the same destinations as route execution, without semantic ranking.

    Snapshot routes end at the capture Place, not at the object. Graph distance
    excludes the current-pose connector and HGR's final local observation leg.
    Missing graph routes are unknown connectivity, not semantic negative evidence.
    """
    def route(place):
        path, cost = graph.shortest_path(graph.current_place_id, place) if place else ([], None)
        return {"approach_place_id": place, "route_places": path,
                "graph_distance_m": cost,
                "connectivity": "known_route" if cost is not None else "no_known_route"}

    snapshot_rows = {}
    for image, snapshot in snapshots.items():
        binding = graph.observations.get(f"snapshot:{snapshot.image}")
        snapshot_rows[image] = {
            **route(binding.capture_place_id if binding else None),
            "destination_kind": "historical_capture_place",
        }
    frontier_rows = []
    for frontier in frontiers:
        binding = graph.frontiers.get(str(getattr(frontier, "topo_id", "")))
        place = (select_reachable_approach_place(graph, binding.approach_candidates_m)
                 if binding else None)
        frontier_rows.append({**route(place),
                              "known_terminal_distance_m": (
                                  binding.approach_candidates_m[place] if place else None)})
    return {"current_place_id": graph.current_place_id,
            "snapshots": snapshot_rows, "frontiers": frontier_rows}


def format_hgr_topology_context(context, snapshot_images):
    """Assign labels after HGR prefiltering/image budgets, preserving source IDs."""
    def snapshot_selection_view(row):
        # Object identity must be decided from the question and pixels. Route
        # lengths remain in the runtime context for post-selection execution,
        # but exposing them here caused the VLM to prefer nearer wrong objects.
        if "objects" not in row:
            return {"route_status": row.get("connectivity", "unknown")}
        return {"objects": {
            label: {
                "source_object_id": item.get("object_id"),
                "route_status": item.get("status", "unknown"),
                **({"reason": item["reason"]} if item.get("reason") else {}),
            }
            for label, item in row["objects"].items()
        }}

    snapshot_rows = {
        f"Snapshot {i}": (
            snapshot_selection_view(context["snapshots"][image])
            if context.get("object_approach")
            else context["snapshots"][image]
        )
        for i, image in enumerate(snapshot_images)
        if image in context["snapshots"]
    }
    rows = {"current_place_id": context["current_place_id"],
            "candidates": {
                **snapshot_rows,
                **{f"Frontier {i}": row for i, row in enumerate(context["frontiers"])},
            }}
    if context.get("object_approach"):
        if context.get("shared_route"):
            return (
                "Navigation topology: Snapshot/Object identity is a hard visual-semantic constraint. "
                "First identify the crop that matches the target without considering travel distance; "
                "never replace a better identity match with a nearer object. Snapshot route distances "
                "are intentionally withheld and the certified route is applied only after selection. "
                "route_status unavailable means no certified route, not target absence. Frontier route "
                "data may guide exploration only when no Snapshot identifies the target. Keep the "
                "original Snapshot/Object or Frontier selection and stopping protocol.\n"
                + json.dumps(rows, ensure_ascii=False))
        return (
            "Navigation topology: Snapshot/Object identity is a hard visual-semantic constraint. "
            "First identify the crop that matches the target without considering travel distance; "
            "never replace a better identity match with a nearer object. Snapshot route distances "
            "are intentionally withheld and route planning occurs only after selection. route_status "
            "unavailable is not negative target evidence. Frontier route data may guide exploration "
            "only when no Snapshot identifies the target. Keep the original selection protocol.\n"
            + json.dumps(rows, ensure_ascii=False))
    return (
        "Navigation topology from observed space (candidate labels match the images below). "
        "Use this connectivity and route distance together with HGR visual evidence and "
        "frontier hypotheses to choose where to search next. Prefer useful reachable exploration "
        "when no snapshot identifies the target; distance alone does not establish a match. "
        "Routes use valid Place edges and will guide navigation after selection. "
        "Graph distance excludes the current-pose connector and final local movement. "
        "Snapshot routes lead to historical capture Places, not certified object locations. "
        "no_known_route means the observed graph has no route, not that the target is absent "
        "or physically unreachable; the existing local safety fallback remains available. "
        "Keep the original Snapshot/Object or Frontier answer protocol.\n"
        + json.dumps(rows, ensure_ascii=False)
    )
