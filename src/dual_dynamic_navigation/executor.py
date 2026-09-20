"""Known-space execution adapter shared by the staged HGR backends."""

import hashlib
from dataclasses import dataclass

import numpy as np

from src.route_guidance import RouteGuidance, grid_path, path_mask, visible, free_cell


def known_mask(planner):
    return (np.asarray(planner.island, bool) & np.asarray(planner.unoccupied, bool)
            & ~np.asarray(planner.occupied, bool))


@dataclass
class KnownSpaceExecutor:
    guidance: object = None
    last_audit: object = None
    installed_route: object = None

    def prepare(self, planner, pts):
        """Plan to the installed terminal, never use simulator geometry."""
        mask = known_mask(planner)
        current = np.asarray(planner.habitat2voxel(pts)[:2], float)
        terminal = planner.target_point
        self.guidance = None
        self.last_audit = {"controller": "known_grid", "fallback": False,
                           "start": current.tolist(), "valid": False}
        if terminal is None:
            self.last_audit["reason"] = "no_installed_terminal"
            return False
        path = grid_path(mask, current, terminal)
        if path is None:
            self.last_audit["reason"] = "no_known_path"
            return False
        # Include the actual starting pose rather than teleport to a grid center.
        if not np.allclose(current, path[0]):
            if not visible(mask, current, path[0]):
                self.last_audit["reason"] = "no_known_start_connector"
                return False
            path = np.vstack([current, path])
        self.guidance = RouteGuidance([], path, [], path_mask(mask.shape, path), path[-1])
        self.last_audit = {
            "controller": "known_grid", "fallback": False,
            "geometry_sha256": hashlib.sha256(np.packbits(mask).tobytes()
                + str(mask.shape).encode()).hexdigest(),
            "start": current.tolist(), "terminal": path[-1].tolist(),
            "certified_path": path.tolist(),
            "distance_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                                * planner._voxel_size),
        }
        return True

    def step(self, planner, pts, angle, objects, snapshots, cfg, save_visualization=False,
             certified_route=None):
        # Re-plan on current geometry; only the semantic decision stays installed.
        if certified_route is None:
            if not self.prepare(planner, pts):
                return (None,)
        else:
            current, mask = np.asarray(planner.habitat2voxel(pts)[:2], float), known_mask(planner)
            if self.installed_route is not certified_route:
                path = np.asarray(certified_route["certified_path"], float)
                if not np.allclose(path[0], current):
                    if not visible(mask, current, path[0]):
                        self.last_audit = {"valid": False, "reason": "invalid_route_start"}
                        return (None,)
                    path = np.vstack([current, path])
                self.guidance = RouteGuidance(certified_route.get("route_places", []), path,
                    certified_route.get("transitions", []), path_mask(mask.shape, path), path[-1])
                self.installed_route = certified_route
            guide = self.guidance
            valid = (np.allclose(guide.path[guide.cursor], current) and
                     all(free_cell(mask, point) for point in guide.path[guide.cursor:]) and
                     all(visible(mask, a, b) for a, b in
                         zip(guide.path[guide.cursor:], guide.path[guide.cursor+1:])))
            self.last_audit = {"controller": "certified_route", "fallback": False,
                "valid": False, "terminal": guide.terminal.tolist(), "start": current.tolist(),
                "certified_path": guide.path[guide.cursor:].tolist(),
                "geometry_sha256": hashlib.sha256(np.packbits(mask).tobytes()
                    + str(mask.shape).encode()).hexdigest()}
            if not valid:
                self.last_audit["reason"] = "route_geometry_invalid"
                return (None,)
        result = planner.agent_step(pts, angle, objects, snapshots, None, cfg,
                                   save_visualization=save_visualization,
                                   route_guidance=self.guidance)
        segment = self.guidance.last_segment
        valid = (result[0] is not None and segment is not None
                 and visible(known_mask(planner), *segment)
                 and np.allclose(result[2], segment[-1]))
        self.last_audit.update(actual_segment=segment, valid=bool(valid))
        if result[0] is not None and not valid:
            raise RuntimeError("Known-space execution produced an uncertified movement")
        return result
