"""V2-A: HGR observation terminals and certified costs for anchor execution."""
import numpy as np
from time import perf_counter

from src.geom import get_proper_observe_point
from src.route_guidance import grid_path, free_cell


def resolve_frontier_approach(graph, planner, frontier, current, observe_distance, previous=None, origin=None):
    """Query HGR's known-side frontier terminal without mutating planner state."""
    mask = planner.island & planner.unoccupied & ~planner.occupied
    terminal = np.asarray(frontier.position[:2], float).copy()
    direction = np.asarray(frontier.orientation[:2], float)
    if not np.all(np.isfinite(terminal)) or not np.all(np.isfinite(direction)):
        return {"status": "unavailable", "reason": "invalid_frontier_geometry"}
    # Same finite retreat bound as HGR set_next_navigation_point. No new
    # region timer or semantic rule; zero orientation cannot make progress.
    for _ in range(1001):
        cell = terminal.astype(int)
        if free_cell(mask, cell):
            terminal = cell
            break
        if np.linalg.norm(direction) < 1e-9:
            return {"status": "unavailable", "reason": "no_known_frontier_terminal"}
        terminal -= direction
    else:
        return {"status": "unavailable", "reason": "no_known_frontier_terminal"}
    previous = previous if previous and np.array_equal(previous.get("terminal"), terminal) else None
    result = resolve_terminal_route(graph, mask, current, terminal, observe_distance, origin, previous)
    return {**result, "terminal": terminal.tolist(),
            "frontier_position": np.asarray(frontier.position[:2]).tolist(),
            "look_at": (terminal+direction*3).tolist()}


def resolve_object_approach(graph, planner, objects, object_id, current,
                            observe_distance, previous=None, origin=None):
    if object_id not in objects:
        return {"status": "unavailable", "reason": "object_missing"}
    center = np.asarray(planner.habitat2voxel(objects[object_id]["bbox"].center)[:2])
    mask = np.asarray(planner.island, bool) & np.asarray(planner.unoccupied, bool) & ~np.asarray(planner.occupied, bool)
    terminal = (previous["terminal"] if previous and previous.get("status") == "valid"
                and np.array_equal(previous["look_at"], center)
                and (getattr(graph, "dual_topo_runtime", None) is None or free_cell(mask, previous["terminal"])) else
                get_proper_observe_point(center, mask, np.asarray(current),
                                         dist=observe_distance / graph.voxel_size))
    if terminal is None:
        return {"status": "unavailable", "reason": "no_known_observation_terminal"}
    same_grid_terminal = (previous is not None and previous.get("terminal") is not None
                          and np.array_equal(np.rint(previous.get("terminal")), np.rint(terminal)))
    result = resolve_terminal_route(graph, mask, current, terminal, observe_distance, origin,
                                    previous=previous if same_grid_terminal else None)
    # grid_path and the executor both operate on the rounded voxel. Expose that
    # same terminal so route evidence, logs and execution share one endpoint.
    route_terminal = (result["certified_path"][-1]
                      if result.get("status") == "valid" else np.rint(terminal).tolist())
    return {**result, "object_id": int(object_id), "terminal": route_terminal,
            "look_at": center.tolist()}


def resolve_terminal_route(graph, mask, current, terminal, observe_distance, origin=None, previous=None):
    runtime = getattr(graph, "dual_topo_runtime", None)
    if runtime is not None:
        return runtime.resolve(graph, mask, current, terminal, observe_distance, origin, previous)
    started = perf_counter()
    origin = graph.current_place_id if origin is None else origin
    best = None
    limit = 2 * graph.place_spacing_m + observe_distance
    pinned = previous.get("approach_place_id") if previous and previous.get("status") == "valid" else None
    nodes = [(p,n) for p,n in sorted(graph.nodes.items()) if pinned is None or p == pinned]
    for place, node in nodes:
        if place != origin and np.linalg.norm(node.position - terminal)*graph.voxel_size > limit:
            continue
        route, _ = graph.shortest_path(origin, place)
        if not route:
            continue
        anchors = [graph.edges[(a,b)].target_anchor for a,b in zip(route, route[1:])]
        if any(a is None for a in anchors):
            continue
        points = [np.asarray(current), *anchors, np.asarray(terminal)]
        path, costs = [], []
        for i, (start, end) in enumerate(zip(points, points[1:])):
            segment = grid_path(mask, start, end)
            if segment is None:
                break
            cost = sum(np.linalg.norm(b-a) for a,b in zip(segment,segment[1:]))*graph.voxel_size
            bound = (limit if i == len(points)-2 else
                     graph.edges[(route[i], route[i+1])].path_length_m + 2*graph.place_spacing_m)
            if cost > bound + 1e-6:
                break
            costs.append(float(cost))
            path.extend(np.asarray(segment if not path else segment[1:]).tolist())
        else:
            rank = (sum(costs), len(route), place)
            if best is None or rank < best[0]:
                best = (rank, {"status": "valid", "approach_place_id": place,
                               "route_places": route, "total_distance_m": sum(costs),
                               "segment_distances_m": costs, "certified_path": path,
                               "cost_basis": "current_pose_to_edge_targets_to_terminal"})
    if best:
        result = {**best[1], "route_event": "approach_retained" if pinned else "approach_selected"}
    elif pinned is not None:
        result = resolve_terminal_route(graph, mask, current, terminal, observe_distance, origin)
        result["route_event"] = "approach_invalid_reroute"
    else:
        result = {"status": "unavailable", "reason": "no_certified_approach_route"}
    result["route_query_seconds"] = perf_counter() - started
    return result


def retain_frontier_hop(graph, planner, execution, commitment, current):
    """Check installed hop without moving its anchor or advancing progress."""
    plan = execution.active_plan
    if not commitment or commitment.get("target_kind") != "frontier" or plan is None or not plan.uses_override:
        return False
    if plan.target_id != commitment.get("target_id"):
        return False
    if planner.target_point is None or not np.array_equal(planner.target_point, plan.target_voxel):
        return False
    if not any(str(getattr(f, "topo_id", "")) == commitment["target_id"] for f in planner.frontiers):
        return False
    edge = graph.edges.get((plan.current_place_id, plan.next_place_id))
    if edge is None or edge.status.value != "valid":
        return False
    mask = planner.island & planner.unoccupied & ~planner.occupied
    return grid_path(mask, current, plan.target_voxel) is not None
