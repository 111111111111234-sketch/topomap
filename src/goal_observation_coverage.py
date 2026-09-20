"""Goal-local independent viewing regions and versioned execution feedback."""

import hashlib
import math

import numpy as np


class ObservationCoverage:
    def __init__(self, voxel_size, spacing_m):
        self.voxel_size, self.spacing_m = voxel_size, spacing_m
        self.sources = {}
        self.records = {}
        self.failures = {}
        self.requests = 0

    def bearing(self, position, target):
        direction = np.asarray(position) - np.asarray(target)
        return int(round(math.atan2(direction[1], direction[0]) / (math.pi/4))) % 8

    def observe_source(self, entity, pose, target):
        # Source filenames and frame timestamps do not constitute new evidence.
        view = (*[int(v) for v in np.rint(np.asarray(pose)*self.voxel_size/self.spacing_m)],
                self.bearing(pose, target))
        self.sources.setdefault(entity, set()).add(view)

    def version(self, entity, target, mask, radius):
        lo = np.maximum(0, np.floor(np.asarray(target) - radius).astype(int))
        hi = np.minimum(mask.shape, np.ceil(np.asarray(target) + radius + 1).astype(int))
        patch = np.ascontiguousarray(mask[lo[0]:hi[0],lo[1]:hi[1]])
        digest = hashlib.sha256(patch.tobytes()).hexdigest()
        return (tuple(sorted(self.sources.get(entity, set()))), tuple(map(int,lo)), tuple(map(int,hi)), digest)

    def available(self, entity, position, target, version):
        key = (entity, self.bearing(position, target))
        record = self.records.get(key)
        if record is not None:
            if record["outcome"] == "verification_uncertain":
                # Only an independent source view can reopen an inconclusive
                # semantic check. Free-grid churn is not visibility evidence.
                if not set(version[0]) - set(record["version"][0]):
                    return False
            elif record["outcome"].startswith("verification_"):
                # A rejected view remains contradictory; other directions are
                # still eligible. Never reject the entity or its whole Place.
                return False
            elif record["version"] == version:
                return False
        return self.failures.get(key, {}).get("version") != version

    def finish(self, candidate, outcome, version, step):
        key = (candidate.entity_id, self.bearing(candidate.terminal, candidate.look_at))
        value = {"version": version, "outcome": outcome, "step": step,
                 "position": candidate.terminal, "observation_id": candidate.observation_id,
                 "place_id": candidate.place_id, "intent": candidate.intent}
        if outcome in ("verification_matched", "verification_rejected", "verification_uncertain", "frontier_observed"):
            value["semantic_state"] = {
                "verification_rejected": "contradicted_view",
                "verification_uncertain": "unresolved_view",
                "verification_matched": "matched_view",
                "frontier_observed": "observed_frontier",
            }[outcome]
            self.records[key] = value
        elif outcome != "subtask_budget_ended":
            self.failures[key] = value

    def rebind(self, old, new):
        if old == new:
            return
        self.sources.setdefault(new, set()).update(self.sources.pop(old, set()))
        for table in (self.records, self.failures):
            for key in list(table):
                if key[0] == old:
                    value = table.pop(key)
                    # Preserve coverage through identity merges, including the
                    # union of already known source regions; no automatic retry.
                    version = value["version"]
                    value = {**value, "version": (tuple(sorted(self.sources[new])), *version[1:])}
                    dest = (new, key[1])
                    if dest not in table or value["step"] > table[dest]["step"]:
                        table[dest] = value

    def trace(self):
        return {"checked_regions": [dict(entity_id=k[0], region=k[1], **v) for k,v in self.records.items()],
                "execution_failures": [dict(entity_id=k[0], region=k[1], **v) for k,v in self.failures.items()]}

    def project(self, overlay):
        for (entity, region), record in self.records.items():
            state = overlay.places.get(record["place_id"])
            if state is not None:
                state.checked_actions[(record["intent"], f"{entity}/view:{region}")] = {
                    "outcome": record["outcome"], "step": record["step"], "version": record["version"]}
                state.evidence_ids.add(record["observation_id"])
                if not state.verify_entities and not state.frontier_entities:
                    state.search_state = "checked_views_no_current_action"
