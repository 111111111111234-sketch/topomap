"""Deterministic GOAT-Bench episode selection manifests.

Ratio-based scene selection depends on directory enumeration order.  A manifest
captures the exact scene file and episode id so baseline and method runs can be
paired without changing the navigation pipeline.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = 1


def _episode_id(value: Any) -> str:
    return str(value)


def build_episode_manifest(
    test_data_dir: str,
    seed: int,
    start_ratio: float,
    end_ratio: float,
    splits: Iterable[int],
) -> Dict[str, Any]:
    """Capture the legacy ratio selection as explicit scene/episode pairs."""
    scene_files = os.listdir(test_data_dir)
    random.Random(int(seed)).shuffle(scene_files)
    selected = scene_files[
        int(float(start_ratio) * len(scene_files)):
        int(float(end_ratio) * len(scene_files))
    ]
    split_entries: Dict[str, List[Dict[str, Any]]] = {}
    for split in splits:
        split = int(split)
        if split < 1:
            raise ValueError(f"split must be >= 1, got {split}")
        entries = []
        for scene_file in selected:
            scene_path = os.path.join(test_data_dir, scene_file)
            with open(scene_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            episodes = payload.get("episodes") or []
            index = split - 1
            if index >= len(episodes):
                raise ValueError(
                    f"{scene_file} has {len(episodes)} episodes; cannot select split {split}"
                )
            entries.append({
                "scene_file": scene_file,
                "scene_name": os.path.splitext(scene_file)[0],
                "episode_index": index,
                "episode_id": _episode_id(episodes[index]["episode_id"]),
            })
        split_entries[str(split)] = entries
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_dir": os.path.normpath(test_data_dir),
        "seed": int(seed),
        "legacy_selection": {
            "start_ratio": float(start_ratio),
            "end_ratio": float(end_ratio),
        },
        "splits": split_entries,
    }


def load_episode_manifest(
    manifest_path: str,
    test_data_dir: str,
    split: int,
) -> List[Dict[str, Any]]:
    """Load and validate exact entries for one runner split."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GOAT manifest schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    entries = (manifest.get("splits") or {}).get(str(int(split)))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest {manifest_path} has no entries for split {split}")

    validated = []
    seen = set()
    for raw in entries:
        scene_file = str(raw.get("scene_file", ""))
        if not scene_file or os.path.basename(scene_file) != scene_file:
            raise ValueError(f"Invalid scene_file in manifest: {scene_file!r}")
        scene_path = os.path.join(test_data_dir, scene_file)
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(f"Manifest scene file does not exist: {scene_path}")
        with open(scene_path, "r", encoding="utf-8") as handle:
            scene_payload = json.load(handle)
        episodes = scene_payload.get("episodes") or []
        episode_id = _episode_id(raw.get("episode_id"))
        matching = [
            (index, episode) for index, episode in enumerate(episodes)
            if _episode_id(episode.get("episode_id")) == episode_id
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Manifest entry {scene_file}/{episode_id} matched {len(matching)} episodes"
            )
        actual_index, _ = matching[0]
        declared_index = raw.get("episode_index")
        if declared_index is not None and int(declared_index) != actual_index:
            raise ValueError(
                f"Manifest index mismatch for {scene_file}/{episode_id}: "
                f"declared {declared_index}, actual {actual_index}"
            )
        key = (scene_file, episode_id)
        if key in seen:
            raise ValueError(f"Duplicate manifest entry: {scene_file}/{episode_id}")
        seen.add(key)
        validated.append({
            "scene_file": scene_file,
            "scene_name": os.path.splitext(scene_file)[0],
            "episode_index": actual_index,
            "episode_id": episode_id,
        })
    return validated

