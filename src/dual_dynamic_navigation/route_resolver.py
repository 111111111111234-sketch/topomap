"""Real terminals and known-space geometry for hypothesis-aware tasks."""

import numpy as np

from src.geom import get_proper_observe_point
from src.object_approach import resolve_frontier_approach, resolve_terminal_route
from .executor import known_mask
from src.route_guidance import visible


class RouteResolver:
    def certify(self, route, graph, planner, current):
        if route.get("status") != "valid":
            return route
        path = np.asarray(route["certified_path"], float)
        if not np.allclose(path[0], current):
            if not visible(known_mask(planner), current, path[0]):
                return {"status": "unavailable", "reason": "unknown_start_connector"}
            path = np.vstack([current, path])
        transitions, cursor = [], 0
        places = route.get("route_places", [])
        for source, target in zip(places, places[1:]):
            anchor = np.rint(graph.edges[(source, target)].target_anchor)
            indices = np.flatnonzero(np.all(np.isclose(path[cursor:], anchor), axis=1))
            if not len(indices):
                return {"status": "unavailable", "reason": "missing_transition_geometry"}
            cursor += int(indices[0])
            transitions.append((target, cursor))
        return {**route, "certified_path": path.tolist(), "transitions": transitions,
                "total_distance_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                                          * graph.voxel_size)}

    def object_view(self, graph, planner, objects, entity, current, observe_distance, checked=()):
        if entity not in objects:
            return {"status": "unavailable", "reason": "object_missing"}
        center = np.asarray(planner.habitat2voxel(objects[entity]["bbox"].center)[:2], float)
        sample_mask = known_mask(planner).copy()
        x, y = np.ogrid[:sample_mask.shape[0], :sample_mask.shape[1]]
        # Sampling excludes already checked physical views; routing uses the full map.
        for point in checked:
            sample_mask &= np.hypot(x-point[0], y-point[1]) * graph.voxel_size > .5
        radius_voxels = observe_distance / graph.voxel_size
        # A missing known-space observation terminal is an expected candidate
        # rejection, not a runtime error. Avoid the generic geometry helper's
        # error-level log for this preflight case.
        if not np.any(sample_mask & (
                np.hypot(x-center[0], y-center[1]) < radius_voxels)):
            return {"status": "unavailable", "reason": "no_unchecked_known_view"}
        terminal = get_proper_observe_point(center, sample_mask, np.asarray(current),
                                           dist=radius_voxels)
        if terminal is None:
            return {"status": "unavailable", "reason": "no_unchecked_known_view"}
        route = resolve_terminal_route(graph, known_mask(planner), current, terminal,
                                       observe_distance)
        route = self.certify(route, graph, planner, current)
        return {**route, "terminal": (route["certified_path"][-1]
            if route.get("status") == "valid" else np.rint(terminal).tolist()),
            "look_at": center.tolist(), "object_id": int(entity)}

    def frontier(self, graph, planner, frontier, current, observe_distance):
        return self.certify(resolve_frontier_approach(graph, planner, frontier, current,
                            observe_distance), graph, planner, current)
