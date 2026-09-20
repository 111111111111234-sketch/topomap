"""Local research checkpoints and immutable decision records.

Checkpoints contain Python objects and must only be loaded from a trusted run
created by this checkout. They are not a general map interchange format.
"""
import copy
import hashlib
import json
import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np


SCENE_MEMORY = ("objects", "object_id_aliases", "object_id_counter", "snapshots",
                "frames", "all_observations", "clustering")

_OPEN3D_PERSISTENT_TAG = "hgr_open3d_geometry_v1"


def _open3d_geometry_state(obj):
    """Return a pickle-safe description for legacy Open3D geometry objects."""
    cls = type(obj)
    if not cls.__module__.startswith("open3d."):
        return None
    if cls.__name__ == "PointCloud":
        state = {
            "points": np.asarray(obj.points).copy(),
            "colors": np.asarray(obj.colors).copy(),
            "normals": np.asarray(obj.normals).copy(),
        }
        if hasattr(obj, "covariances"):
            state["covariances"] = np.asarray(obj.covariances).copy()
        return "PointCloud", state
    if cls.__name__ == "OrientedBoundingBox":
        return "OrientedBoundingBox", {
            "center": np.asarray(obj.center).copy(),
            "rotation": np.asarray(obj.R).copy(),
            "extent": np.asarray(obj.extent).copy(),
            "color": np.asarray(obj.color).copy(),
        }
    if cls.__name__ == "AxisAlignedBoundingBox":
        return "AxisAlignedBoundingBox", {
            "min_bound": np.asarray(obj.min_bound).copy(),
            "max_bound": np.asarray(obj.max_bound).copy(),
            "color": np.asarray(obj.color).copy(),
        }
    return None


class _CheckpointPickler(pickle.Pickler):
    def persistent_id(self, obj):
        state = _open3d_geometry_state(obj)
        return (_OPEN3D_PERSISTENT_TAG, *state) if state is not None else None


class _CheckpointUnpickler(pickle.Unpickler):
    def persistent_load(self, persistent_id):
        if not (isinstance(persistent_id, tuple) and len(persistent_id) == 3
                and persistent_id[0] == _OPEN3D_PERSISTENT_TAG):
            raise pickle.UnpicklingError(f"Unsupported checkpoint persistent id: {persistent_id!r}")
        import open3d as o3d
        _, kind, state = persistent_id
        if kind == "PointCloud":
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(state["points"])
            geometry.colors = o3d.utility.Vector3dVector(state["colors"])
            geometry.normals = o3d.utility.Vector3dVector(state["normals"])
            if "covariances" in state and len(state["covariances"]):
                geometry.covariances = o3d.utility.Matrix3dVector(state["covariances"])
            return geometry
        if kind == "OrientedBoundingBox":
            geometry = o3d.geometry.OrientedBoundingBox(
                state["center"], state["rotation"], state["extent"])
        elif kind == "AxisAlignedBoundingBox":
            geometry = o3d.geometry.AxisAlignedBoundingBox(
                state["min_bound"], state["max_bound"])
        else:
            raise pickle.UnpicklingError(f"Unsupported Open3D checkpoint geometry: {kind}")
        geometry.color = state["color"]
        return geometry


def _atomic_pickle_create(path, payload):
    """Serialize completely before exclusively publishing the checkpoint."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            _CheckpointPickler(handle, protocol=pickle.HIGHEST_PROTOCOL).dump(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link atomically creates the final name and refuses to replace
        # a checkpoint from another run.
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compatibility(cfg):
    # These affect coordinate interpretation, sensing or perception. Feature
    # flags, output paths and model-selection experiments are intentionally out.
    keys = ("tsdf_grid_size", "camera_height", "img_width", "img_height", "hfov",
            "camera_tilt_deg", "class_set", "scene_dataset_config_path")
    return {key: str(cfg.get(key)) for key in keys}


def navigation_compatibility(cfg):
    from omegaconf import OmegaConf
    keys = ("vlm_model", "vlm_base_url", "choose_every_step", "prompt_h", "prompt_w",
            "prefiltering", "top_k_categories", "planner", "hypothesis")
    values = {}
    for key in keys:
        value = cfg.get(key)
        values[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return values


def save_checkpoint(path, scene, planner, place, persistent, pts, angle, step,
                    subtask_index, cfg, scene_id, episode_id, prefix_metrics,
                    navigation=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spatial = copy.copy(place) if place is not None else None
    if spatial is not None:
        spatial.__dict__ = {k: v for k, v in vars(place).items() if k != "dual_topo_runtime"}
    payload = {"version": 3, "scene_id": str(scene_id), "episode_id": str(episode_id),
               "compatibility": compatibility(cfg), "scene_memory": {k: getattr(scene, k) for k in SCENE_MEMORY},
               "planner": planner, "place": spatial, "persistent": persistent,
               "pts": np.asarray(pts), "angle": angle, "global_step": step,
               "subtask_index": subtask_index, "prefix_metrics": prefix_metrics,
               "navigation": navigation,
               "random": random.getstate(), "numpy_random": np.random.get_state(),
               "torch_random": torch.get_rng_state(),
               "cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    if getattr(navigation, "backend_name", None) == "hypothesis_aware_dual_topo":
        payload["version"] = 4
        payload["navigation_contract"] = {
            "backend": str(cfg.get("navigation_backend")),
            "config": dict(navigation.config),
            "policy": navigation_compatibility(cfg),
        }
    _atomic_pickle_create(path, payload)
    return file_sha256(path)


def load_checkpoint(path, cfg, scene_id, episode_id):
    with Path(path).open("rb") as handle:
        payload = _CheckpointUnpickler(handle).load()
    if (payload["version"] not in (1, 2, 3, 4) or payload["scene_id"] != str(scene_id)
            or payload["episode_id"] != str(episode_id)
            or payload["compatibility"] != compatibility(cfg)):
        raise ValueError("Research checkpoint scene/episode/sensor/coordinate mismatch")
    if payload["version"] == 4:
        contract = payload["navigation_contract"]
        navigation_cfg = cfg.get("dual_dynamic_navigation", {})
        from omegaconf import OmegaConf
        if OmegaConf.is_config(navigation_cfg):
            navigation_cfg = OmegaConf.to_container(navigation_cfg, resolve=True)
        if (contract["backend"] != str(cfg.get("navigation_backend")) or
                contract["config"] != dict(navigation_cfg) or
                contract["policy"] != navigation_compatibility(cfg)):
            raise ValueError("Research checkpoint navigation backend/configuration mismatch")
    payload["checkpoint_sha256"] = file_sha256(path)
    payload.setdefault("navigation", None)
    return payload


def restore_random(payload):
    import torch
    random.setstate(payload["random"])
    np.random.set_state(payload["numpy_random"])
    torch.set_rng_state(payload["torch_random"])
    if payload["cuda_random"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_random"])


def record_decision(directory, step, graph, mask, current, approach, candidate):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"decision_{step}.npz"
    record = {"version": 1, "voxel_size": graph.voxel_size, "graph": graph.to_trace_dict(include_graph=True),
              "current": np.asarray(current).tolist(), "approach": approach,
              "candidate": candidate.context() if candidate is not None else None}
    with path.open("xb") as handle:
        np.savez_compressed(handle, known_free=np.asarray(mask, bool),
                            record=np.asarray(json.dumps(record)))


def write_run_fingerprint(directory, cfg, repository):
    directory, repository = Path(directory), Path(repository)
    directory.mkdir(parents=True, exist_ok=True)
    paths = list((repository / "src").rglob("*.py")) + list((repository / "configs").rglob("*.py"))
    paths += list(repository.glob("run_*evaluation.py"))
    hashes = {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(cfg, resolve=True)
    # Record configuration identity, never credentials/configuration values.
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    manifest = cfg.get("episode_manifest")
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if manifest else None
    result = {"source_sha256": hashes, "config_sha256": config_hash, "manifest_sha256": manifest_hash}
    path = directory / "run_fingerprint.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != result:
            raise ValueError(
                f"Existing run fingerprint does not match current run: {path}"
            )
        return existing
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
