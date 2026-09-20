"""HGR owns semantics; this adapter filters executable source identities."""

import hashlib
import json

import numpy as np


def evidence_digest(scene, image, entity):
    pixels = np.ascontiguousarray(scene.all_observations[image])
    snapshot = scene.snapshots[image]
    boxes = []
    for index in range(len(snapshot.visual_prompt)):
        item = snapshot.visual_prompt[index]
        if int(item.data["obj_id"][0]) == int(entity):
            boxes.append(np.asarray(item.xyxy).tolist())
    digest = hashlib.sha256()
    for box in boxes:
        x1, y1, x2, y2 = np.asarray(box).reshape(-1, 4)[0].astype(int)
        digest.update(np.ascontiguousarray(pixels[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]).tobytes())
    digest.update(json.dumps([str(pixels.dtype), boxes,
        np.asarray(snapshot.obs_point).tolist()], sort_keys=True).encode())
    return digest.hexdigest()


class HGRAdapter:
    def select(self, query, metadata, scene, planner, views, cfg, eligible, frontier_indices,
               task_context=None, purpose=None):
        return query(subtask_metadata=metadata, scene=scene, tsdf_planner=planner,
                     rgb_egocentric_views=views, cfg=cfg,
                     eligible_objects_by_snapshot=eligible,
                     navigation_task_context=task_context,
                     request_purpose=purpose,
                     allowed_frontier_indices=frontier_indices, verbose=True)
