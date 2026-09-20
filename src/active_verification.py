"""Training-free active verification utilities for goal belief topology."""

from __future__ import annotations

import json
import logging
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.eval_utils_gpt_goatbench import call_openai_api, encode_tensor2base64
from src.goal_belief import (
    BeliefActionCandidate,
    EvidenceCluster,
    VerificationOption,
    angular_distance,
)


def detect_category_confidence(
    image: np.ndarray, detection_model, class_names: Sequence[str], category: str,
) -> Optional[float]:
    """Return the strongest local detector confidence without semantic GT."""
    target = " ".join(str(category).lower().replace("_", " ").split())
    try:
        result = detection_model.predict(
            np.asarray(image)[..., :3], conf=0.1, verbose=False
        )[0]
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    except Exception as exc:
        logging.warning("Active VERIFY detector failed: %s", exc)
        return None
    matches = [
        float(confidence)
        for confidence, class_id in zip(confidences, class_ids)
        if 0 <= class_id < len(class_names)
        and " ".join(str(class_names[class_id]).lower().replace("_", " ").split()) == target
    ]
    return max(matches) if matches else None


def _within(point: np.ndarray, shape: Tuple[int, int]) -> bool:
    return 0 <= int(point[0]) < shape[0] and 0 <= int(point[1]) < shape[1]


def _line_visibility(
    start: np.ndarray, end: np.ndarray,
    occupied: np.ndarray, unoccupied: np.ndarray,
) -> Optional[float]:
    distance = float(np.linalg.norm(end - start))
    samples = max(2, int(math.ceil(distance * 2.0)))
    coords = np.rint(np.linspace(start, end, samples)).astype(int)
    values = []
    for point in coords[:-1]:  # the hypothesis/frontier endpoint may be unknown
        if not _within(point, occupied.shape) or occupied[tuple(point)]:
            return None
        values.append(1.0 if unoccupied[tuple(point)] else 0.5)
    return float(np.mean(values)) if values else 0.5


def viewpoint_novelty(
    point: np.ndarray, heading: float,
    history: Iterable[EvidenceCluster],
    distance_scale_voxels: float,
    angle_scale_rad: float = math.radians(45.0),
) -> float:
    scores = []
    for cluster in history:
        if cluster.viewpoint is None:
            continue
        distance = float(np.linalg.norm(point - cluster.viewpoint)) / max(distance_scale_voxels, 1e-8)
        angle = angular_distance(heading, cluster.heading)
        angle_score = 1.0 if angle == math.inf else angle / max(angle_scale_rad, 1e-8)
        scores.append(float(np.clip(max(distance, angle_score), 0.0, 1.0)))
    return min(scores) if scores else 1.0


def sample_verification_viewpoint(
    center: np.ndarray,
    current: np.ndarray,
    occupied: np.ndarray,
    unoccupied: np.ndarray,
    island: np.ndarray,
    voxel_size: float,
    path_query: Callable[[np.ndarray, np.ndarray], Tuple[float, object]],
    history: Sequence[EvidenceCluster] = (),
    entropy: float = 1.0,
    radii_m: Sequence[float] = (1.0, 1.5, 2.0),
    samples_per_radius: int = 16,
) -> Optional[VerificationOption]:
    """Choose a reachable, visible and novel TSDF observation point."""
    center = np.asarray(center, dtype=float)[:2]
    current = np.asarray(current, dtype=float)[:2]
    candidates = []
    for radius_m in radii_m:
        radius_voxels = float(radius_m) / float(voxel_size)
        for index in range(int(samples_per_radius)):
            theta = 2.0 * math.pi * index / int(samples_per_radius)
            point = np.rint(center + radius_voxels * np.array([math.cos(theta), math.sin(theta)])).astype(int)
            if not _within(point, occupied.shape):
                continue
            if occupied[tuple(point)] or not unoccupied[tuple(point)] or not island[tuple(point)]:
                continue
            visibility = _line_visibility(point, center, occupied, unoccupied)
            if visibility is None:
                continue
            try:
                path_cost, path_points = path_query(current, point)
            except Exception:
                continue
            if path_points is None or not np.isfinite(path_cost):
                continue
            heading = math.atan2(center[1] - point[1], center[0] - point[0])
            novelty = viewpoint_novelty(
                point, heading, history,
                distance_scale_voxels=0.75 / float(voxel_size),
            )
            score = float(entropy) * visibility * novelty / (1.0 + max(0.0, float(path_cost)))
            candidates.append((score, float(path_cost), point, visibility, novelty))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    _, raw_path, point, visibility, novelty = candidates[0]
    finite_costs = [item[1] for item in candidates]
    min_cost, max_cost = min(finite_costs), max(finite_costs)
    normalized = ((raw_path - min_cost) / (max_cost - min_cost)
                  if max_cost > min_cost else 0.0)
    return VerificationOption(
        target_point=np.asarray(point, dtype=int),
        look_at_point=np.asarray(center, dtype=float),
        visibility=float(visibility),
        novelty=float(novelty),
        path_cost=float(normalized),
    )


def _extract_json(response: Optional[str]) -> Optional[dict]:
    if not response:
        return None
    try:
        start, end = response.find("{"), response.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        return json.loads(response[start:end])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def query_active_verify(
    image: np.ndarray,
    category: str,
    model_name: str,
    extra_body=None,
) -> Optional[dict]:
    """Request one structured target-presence probability without retries."""
    system = (
        "You verify whether a target object is visibly present in one indoor RGB image. "
        "Return only JSON with keys target_present_probability (0 to 1) and reason."
    )
    prompt = (
        f"Target category: {category}. Judge visible presence, not whether the room might contain it."
    )
    response = call_openai_api(
        system,
        [(prompt, encode_tensor2base64(np.asarray(image)[..., :3].astype(np.uint8)))],
        model_name=model_name,
        extra_body=extra_body,
        purpose="active_verify",
        max_tries=1,
    )
    payload = _extract_json(response)
    if payload is None:
        return None
    try:
        probability = float(payload["target_present_probability"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        return None
    return {
        "target_present_probability": probability,
        "reason": str(payload.get("reason", "")),
    }


def query_action_tiebreak(
    actions: Sequence[BeliefActionCandidate],
    images: Sequence[np.ndarray],
    category: str,
    model_name: str,
    extra_body=None,
) -> Optional[int]:
    if not actions or len(actions) != len(images):
        return None
    system = (
        "Choose between two navigation actions using their images and belief statistics. "
        "Return only JSON: {\"choice_index\": 0 or 1, \"reason\": \"...\"}."
    )
    contents = []
    for index, (action, image) in enumerate(zip(actions, images)):
        text = (
            f"Action {index}: type={action.action_type.value}, target={category}, "
            f"posterior={action.belief.posterior:.3f}, "
            f"reachability={action.belief.reachability:.3f}, "
            f"utility={action.utility:.3f}, evidence={action.belief.evidence_summary}."
        )
        contents.append((text, encode_tensor2base64(np.asarray(image)[..., :3].astype(np.uint8))))
    response = call_openai_api(
        system, contents,
        model_name=model_name,
        extra_body=extra_body,
        purpose="goat_explorer",
    )
    payload = _extract_json(response)
    try:
        choice = int(payload["choice_index"])
    except (TypeError, KeyError, ValueError):
        return None
    return choice if 0 <= choice < len(actions) else None
