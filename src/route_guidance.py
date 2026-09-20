"""Certified, known-grid route execution. No simulator pathfinder or semantics."""

import heapq
import math
import copy
from dataclasses import dataclass

import numpy as np


def free_cell(mask, point):
    cell = tuple(np.rint(point).astype(int))
    return (len(cell) == 2 and all(0 <= cell[i] < mask.shape[i] for i in range(2))
            and bool(mask[cell]))


def visible(mask, start, end):
    """Conservative supercover for a straight executed segment, including corners."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    cuts = {0., 1.}
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            continue
        lo, hi = sorted((start[axis], end[axis]))
        for boundary in np.arange(math.ceil(lo - .5), math.floor(hi - .5) + 1) + .5:
            cuts.add(float((boundary - start[axis]) / delta[axis]))
    cuts = sorted(cuts)
    samples = cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
    for t in samples:
        point = start + delta * t
        # Both sides of a grid boundary belong to the conservative footprint.
        axes = [[int(math.floor(x + .5))] if abs(x + .5 - round(x + .5)) > 1e-9
                else [int(round(x + .5)) - 1, int(round(x + .5))] for x in point]
        if any(not free_cell(mask, (x, y)) for x in axes[0] for y in axes[1]):
            return False
    return True


def grid_path(mask, start, end):
    start, end = tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int))
    if not free_cell(mask, start) or not free_cell(mask, end):
        return None
    queue, distances, parent = [(0., start)], {start: 0.}, {}
    while queue:
        _, cell = heapq.heappop(queue)
        if cell == end:
            path = [cell]
            while cell != start:
                cell = parent[cell]
                path.append(cell)
            return np.asarray(path[::-1], float)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nxt = (cell[0] + dx, cell[1] + dy)
            if not free_cell(mask, nxt):
                continue
            if dx and dy and (not free_cell(mask, (cell[0] + dx, cell[1]))
                              or not free_cell(mask, (cell[0], cell[1] + dy))):
                continue
            distance = distances[cell] + math.hypot(dx, dy)
            if distance < distances.get(nxt, float("inf")):
                distances[nxt], parent[nxt] = distance, cell
                heapq.heappush(queue, (distance + math.dist(nxt, end), nxt))
    return None


def path_mask(shape, path):
    mask = np.zeros(shape, bool)
    for point in path:
        mask[tuple(np.rint(point).astype(int))] = True
    # Include the cardinal footprint of certified diagonals, not nearby rooms.
    for a, b in zip(path, path[1:]):
        x, y = np.rint(a).astype(int), np.rint(b).astype(int)
        mask[x[0], y[1]] = mask[y[0], x[1]] = True
    return mask


def structural_route_view(graph, route):
    """Lossless degree-two chain view, retaining directed edge expansion."""
    protected = {route[0], route[-1]}
    for frontier in graph.frontiers.values():
        protected.update(frontier.approach_candidates_m)
    neighbors = {place: set() for place in graph.nodes}
    for (source,target), edge in graph.edges.items():
        if edge.status.value == "valid":
            neighbors[source].add(target)
            neighbors[target].add(source)
    cuts = [i for i, place in enumerate(route) if place in protected or len(neighbors[place]) != 2]
    return [route[a:b+1] for a,b in zip(cuts,cuts[1:])] if len(route)>1 else [route]


@dataclass
class RouteGuidance:
    route: list
    path: np.ndarray
    transitions: list
    corridor: np.ndarray
    terminal: np.ndarray
    cursor: int = 0
    last_segment: object = None
    structural_chains: object = None

    @classmethod
    def build(cls, graph, mask, current, terminal, target_place, origin=None, observe_distance_m=1.5,
              excluded_edges=frozenset()):
        origin = graph.current_place_id if origin is None else origin
        routing = copy.copy(graph)
        routing.edges = {key: value for key, value in graph.edges.items() if key not in excluded_edges}
        route, _ = routing.shortest_path(origin, target_place)
        if not route:
            return None
        points, transitions = [np.rint(current).astype(float)], []
        for source, destination in zip(route, route[1:]):
            edge = graph.edges[(source, destination)]
            if edge.status.value != "valid":
                return None
            # Recover actual paths from current known geometry near this edge.
            anchors = (edge.source_anchor, edge.target_anchor)
            for anchor in anchors:
                if anchor is None:
                    return None
                start = points[-1]
                x, y = np.ogrid[:mask.shape[0], :mask.shape[1]]
                allowance = edge.path_length_m + 2 * graph.place_spacing_m
                local = mask & ((np.hypot(x-start[0], y-start[1])
                                 + np.hypot(x-anchor[0], y-anchor[1])) * graph.voxel_size
                                <= allowance + 1e-6)
                segment = grid_path(local, start, anchor)
                if segment is None:
                    # Only this planning attempt excludes the unavailable edge.
                    # Unknown/blocked local geometry is not a permanent graph edit.
                    return cls.build(graph, mask, current, terminal, target_place, origin,
                                     observe_distance_m, excluded_edges | {(source, destination)})
                points.extend(segment[1:])
            transitions.append((destination, len(points) - 1))
        segment = grid_path(mask, points[-1], terminal)
        if segment is None:
            return None
        if sum(np.linalg.norm(b-a) for a,b in zip(segment,segment[1:])) * graph.voxel_size > 2*graph.place_spacing_m + observe_distance_m:
            return None
        points.extend(segment[1:])
        path = np.asarray(points)
        return cls(route, path, transitions, path_mask(mask.shape, path), np.rint(terminal).astype(float),
                   structural_chains=structural_route_view(graph, route))

    def repair(self, mask, current):
        """Try the same ordered transitions within the certified corridor first."""
        if not np.allclose(current, self.path[self.cursor]):
            return False
        remaining = [(place,index) for place,index in self.transitions if index > self.cursor]
        points, transitions = [np.asarray(current,float)], []
        for place, index in remaining + [(None,len(self.path)-1)]:
            segment = grid_path(mask & self.corridor, points[-1], self.path[index])
            if segment is None:
                return False
            points.extend(segment[1:])
            if place is not None:
                transitions.append((place,len(points)-1))
        self.route = self.remaining_route()
        self.structural_chains = None
        self.path, self.transitions, self.cursor = np.asarray(points), transitions, 0
        self.corridor &= mask
        return True

    def valid(self, graph, mask):
        return (all(graph.edges.get((a,b)) is not None and graph.edges[(a,b)].status.value == "valid"
                    for a,b in zip(self.remaining_route(), self.remaining_route()[1:]))
                and all(free_cell(mask, p) for p in self.path[self.cursor:])
                and all(visible(mask, a, b) for a,b in zip(self.path[self.cursor:],self.path[self.cursor+1:])))

    def remaining_route(self):
        reached = sum(index <= self.cursor for _, index in self.transitions)
        return self.route[reached:]

    def step(self, current, max_distance_m, voxel_size, mask):
        current = np.asarray(current, float)
        if np.linalg.norm(current - self.path[self.cursor]) > 1e-5:
            return None
        best = self.cursor
        # The corridor for this prefix excludes later branches at intersections.
        for index in range(self.cursor + 1, len(self.path)):
            if np.linalg.norm(self.path[index] - current) * voxel_size > max_distance_m:
                break
            prefix = path_mask(mask.shape, self.path[self.cursor:index+1]) & mask
            end = self.path[index]
            delta = end - current
            norm2 = float(delta @ delta)
            portals = [self.path[p] for _,p in self.transitions if self.cursor < p <= index]
            ordered = []
            for portal in portals:
                t = float((portal-current) @ delta) / norm2 if norm2 else 0.
                ordered.append((t,float(np.linalg.norm(portal-(current+np.clip(t,0.,1.)*delta)))))
            respects_portals = (all(-1e-9 <= t <= 1+1e-9 and d <= .5 for t,d in ordered)
                                and all(a[0] <= b[0] for a,b in zip(ordered,ordered[1:])))
            if respects_portals and visible(prefix, current, end):
                best = index
        if best == self.cursor and self.cursor != len(self.path)-1:
            return None
        end = self.path[best].copy()
        self.last_segment = [current.tolist(), end.tolist()]
        self.cursor = best
        return end, best == len(self.path)-1

    def to_trace_dict(self, include_path=False):
        return {"route_places": self.route, "transitions": self.transitions,
                "structural_chains": self.structural_chains,
                "progress_index": self.cursor, "terminal": self.terminal.tolist(),
                "last_segment": self.last_segment,
                **({"certified_path": self.path.tolist()} if include_path else {})}
