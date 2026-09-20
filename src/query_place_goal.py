"""Shared visual selector/verifier for all Phase-C goal input types."""

import copy
import json
import math
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf

from src.eval_utils_gpt_goatbench import (
    call_openai_api, encode_tensor2base64, format_question,
    strip_thinking_content,
)
from src.place_visual_cache import PlaceVisualCache


def parse_goal_response(response, allowed_ids=None):
    if not isinstance(response, str):
        return None
    text = strip_thinking_content(response).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict):
        return None
    if allowed_ids is not None:
        score = result.get("support")
        if (not isinstance(result.get("candidate_id"), str)
                or result.get("candidate_id") not in allowed_ids or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 1):
            return None
    elif result.get("verdict") not in ("matched", "rejected", "uncertain"):
        return None
    return result


def _request(prompt, content, cfg, purpose):
    extra = cfg.get("vlm_extra_body", None)
    if OmegaConf.is_config(extra):
        extra = OmegaConf.to_container(extra, resolve=True)
    return call_openai_api(prompt, content, model_name=cfg.vlm_model,
                           extra_body=extra, purpose=purpose)


def _goal_content(metadata):
    question, image_goal = format_question(metadata)
    return [(f"Goal: {question}\n", image_goal)] if image_goal else [(f"Goal: {question}\n",)]


def validated_request(prompt, content, cfg, purpose, allowed_ids=None):
    """Retry malformed semantic output with a bounded shared parser policy.

    Transport retries are already owned by call_openai_api; do not multiply
    those retries when the transport returns None.
    """
    limit = 3 if cfg.get("place_goal_navigation", {}).get("evidence_coverage", False) else 1
    failures = []
    request_content = list(content)
    for attempt in range(limit):
        raw = _request(prompt, request_content, cfg, purpose)
        result = parse_goal_response(raw, allowed_ids)
        if result is not None:
            return result, {"request_attempts": attempt + 1, "parse_failures": failures}
        stage = "request" if raw is None else "parse"
        failures.append({"stage": stage, "response_excerpt": raw[:1000] if isinstance(raw,str) else None})
        if raw is None:
            break
        correction = ('The previous response failed validation. Return a JSON object with '
                      'verdict equal to matched, rejected, or uncertain.')
        if allowed_ids is not None:
            correction = ('The previous response failed validation. Choose candidate_id ONLY from '
                          'this current executable list: ' + json.dumps(sorted(allowed_ids)) +
                          '. Historical actions are not selectable. support must be a number in [0,1]. '
                          'Reconsider Explore if the current evidence does not identify the goal. '
                          'Return only candidate_id, support, reason as JSON.')
        request_content = list(content) + [(correction,)]
    return None, {"request_attempts": len(failures), "parse_failures": failures,
                  "failure_stage": stage, "action": "error"}


def selection_payload(value, ids, historical=False):
    """Keep persistent action IDs out of the model's executable namespace."""
    if isinstance(value, dict):
        return {selection_payload(k, ids, historical): selection_payload(v, ids, historical)
                for k, v in value.items() if not (historical and k == "candidate_id")}
    if isinstance(value, (list, tuple)):
        return [selection_payload(v, ids, historical) for v in value]
    if isinstance(value, str):
        if value in ids and not historical:
            return ids[value]
        if value.startswith(("verify:", "explore:")):
            return "historical_action_not_selectable"
    return value


def candidate_visual(candidate, scene, planner, cache=None):
    encode = cache.encode if cache is not None else encode_tensor2base64
    if candidate.intent == "Explore":
        choice = planner.frontiers[candidate.frontier_index]
        if str(getattr(choice, "topo_id", "")) != candidate.entity_id or choice.feature is None:
            return None
        return choice, [(candidate.candidate_id, encode(np.asarray(choice.feature)))], None
    snapshot = next((s for s in scene.snapshots.values() if s.image == candidate.source_image), None)
    rgb = scene.all_observations.get(candidate.source_image)
    if snapshot is None or rgb is None or candidate.object_id not in scene.objects:
        return None
    prompt = getattr(snapshot, "visual_prompt", None)
    if prompt is None:
        return None
    bbox = next((prompt[i].xyxy[0] for i in range(len(prompt))
                 if int(prompt[i].data["obj_id"][0]) == candidate.object_id), None)
    if bbox is None:
        return None
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    h, w = rgb.shape[:2]
    crop = rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if not crop.size:
        return None
    encoded_crop = encode(crop)
    choice = copy.copy(snapshot)
    choice.cluster = [candidate.object_id]
    choice.obs_point = np.asarray(snapshot.obs_point).copy()
    return choice, [
        (candidate.candidate_id + " context", encode(rgb)),
        (candidate.candidate_id + " object", encoded_crop),
    ], encoded_crop


def select_place_goal(state, metadata, scene, planner, views, cfg):
    started = perf_counter()
    if state.evidence_coverage and state.active is not None:
        return None, {"reason": "active_intent", "action": "continue", "model_called": False}
    if state.event_driven_selection and not state.goal_topology.decision_required() and not state.evidence_coverage:
        return None, {
            "reason": "no_goal_topology_change",
            "model_called": False,
            "goal_topology_revision": state.goal_topology.revision,
            "timings": {"visual_prepare_seconds": 0.0},
        }
    choices, tiles, crops, rows = {}, [], {}, []
    phase_cfg = cfg.get("place_goal_navigation", {})
    enabled = bool(phase_cfg.get("cache_visuals", True))
    cache = getattr(state, "_visual_cache", None)
    if cache is None or cache.enabled != enabled:
        cache = state._visual_cache = PlaceVisualCache(enabled=enabled)
    hits_before, misses_before = cache.hits, cache.misses
    for candidate in state.candidates:
        visual = candidate_visual(candidate, scene, planner, cache=cache)
        if visual is None:
            state.exclusions.append({"candidate_id": candidate.candidate_id,
                                     "reason": "missing_visual_source_or_object_crop"})
            continue
        choice, images, crop = visual
        choices[candidate.candidate_id] = choice
        crops[candidate.candidate_id] = crop
        tiles.extend(images)
        rows.append(candidate.to_trace_dict())
    if not choices:
        return None, {"reason": "no_executable_visual_candidate",
                      "action": "no_action", "model_called": False,
                      "timings": {"visual_prepare_seconds": perf_counter() - started}}
    cached = getattr(state, "_selected_response", None)
    if (state.evidence_coverage and cached is not None
            and cached[0] == state.goal_topology.revision and cached[1]["candidate_id"] in choices):
        response = cached[1]
        item = next(c for c in state.candidates if c.candidate_id == response["candidate_id"])
        return (item, choices[item.candidate_id], crops[item.candidate_id], response["support"]), {
            "action": "selected", "candidate_id": item.candidate_id,
            "reason": "unchanged_goal_state_reuse", "model_called": False,
            "presented_ids": list(choices), "timings": {"visual_prepare_seconds": perf_counter()-started}}
    content = _goal_content(metadata)
    if views:
        content.append(("Current observations (not proof of target identity).", cache.sheet([
            (f"current:{i}", cache.encode(np.asarray(v))) for i, v in enumerate(views)
        ])))
    decision_state = {"goal_id": state.goal_id,
                      "reason_for_selection": state.selection_reason,
                      "candidates": rows, "recent_feedback": state.feedback[-8:]}
    if state.topological_actions:
        decision_state["goal_topology"] = state.goal_topology.to_trace_dict()
    request_ids = {key: f"choice_{i}" for i, key in enumerate(choices)} if state.evidence_coverage else {key:key for key in choices}
    if state.evidence_coverage:
        decision_state["recent_feedback"] = selection_payload(state.feedback[-8:], {}, historical=True)
        decision_state = selection_payload(decision_state, request_ids)
        tiles = [(next((alias + label[len(key):] for key, alias in request_ids.items()
                        if label == key or label.startswith(key + " ")), label), encoded)
                 for label, encoded in tiles]
    content.append((
        "Choose one intent. Explore observes a real frontier. Verify goes to a safe object "
        "observation point and requires a fresh visual check; historical matches do not finish the task. "
        "Compare target evidence with the known route costs in meters. null support means unscored, "
        "not absent. Supports are model scores, not calibrated probabilities. "
        "Do not infer reachability from images. All listed candidates passed geometry checks.\n"
        + json.dumps(decision_state, ensure_ascii=False),
    ))
    # Identical to R1: context + crop per Verify, all candidates, 12 tiles/page.
    for start in range(0, len(tiles), 12):
        content.append(("Labeled candidate evidence:", cache.sheet(tiles[start:start + 12])))
    prompt = ('Select a navigation intent for the goal using the labeled visual evidence and route costs. '
              'Return only JSON: {"candidate_id":"exact listed ID","support":0.0,"reason":"brief explanation"}. '
              'Support in [0,1] describes visual goal match for Verify or exploration relevance for Explore.')
    request_start = perf_counter()
    response, request_info = validated_request(prompt, content, cfg, "place_goal_select", set(request_ids.values()))
    if response is not None:
        reverse_ids = {v:k for k,v in request_ids.items()}
        response = {**response, "candidate_id": reverse_ids[response["candidate_id"]]}
    request_end = perf_counter()
    diagnostic = {
        **request_info,
        "presented_ids": list(choices), "image_parts": sum(len(item) > 1 for item in content),
        "request_id_map": request_ids,
        "model_called": True,
        **({"evidence_versions": state.coverage_versions} if state.evidence_coverage else {}),
        "visual_tiles": len(tiles), "cache_hits": cache.hits - hits_before,
        "cache_misses": cache.misses - misses_before,
        "timings": {"visual_prepare_seconds": request_start - started,
                    "api_seconds": request_end - request_start,
                    "parse_seconds": perf_counter() - request_end},
    }
    if response is None:
        return None, {**diagnostic, "reason": "model_request_or_selection_parse_failed"}
    state.goal_topology.mark_decision()
    state._selected_response = (state.goal_topology.revision, response)
    candidate = next(c for c in state.candidates if c.candidate_id == response["candidate_id"])
    reactivation = None
    if state.evidence_coverage:
        region = state.coverage.bearing(candidate.terminal, candidate.look_at)
        previous = state.coverage.records.get((candidate.entity_id, region))
        current = state.coverage_versions.get(candidate.entity_id)
        if previous is not None and current is not None and previous.get("version") is not None:
            old = previous["version"]
            reactivation = {"previous_outcome": previous["outcome"],
                            "sources_changed": old[0] != current[0],
                            "local_bounds_changed": old[1:3] != current[1:3],
                            "local_mask_changed": old[3] != current[3]}
    return (candidate, choices[candidate.candidate_id], crops[candidate.candidate_id], response["support"]), {
        **diagnostic, "reason": response.get("reason", "vlm_selection"),
        "action": "selected",
        "candidate_id": candidate.candidate_id,
        "coverage_reactivation": reactivation,
    }


def verify_place_goal(metadata, current_rgb, selected_crop, cfg):
    started = perf_counter()
    if selected_crop is None:
        return "uncertain", {"reason": "missing_selected_evidence"}
    content = _goal_content(metadata)
    content.extend([
        ("Previously selected object evidence (historical; not proof of current visibility):", selected_crop),
        ("Fresh observation at the reached safe observation point:", encode_tensor2base64(current_rgb)),
    ])
    prompt = (
        'Verify the navigation goal using the fresh observation. Return only JSON '
        '{"verdict":"matched|rejected|uncertain","reason":"brief visual evidence"}. '
        'matched requires that the selected object is clearly visible NOW and matches the goal, '
        'including instance/environment details for image or description goals. '
        'A historic match or arrival alone is insufficient. rejected requires visible contradictory '
        'evidence about this object. If not visible, occluded, ambiguous, or uncertain, use uncertain. '
        'Never conclude the entire place lacks the goal.'
    )
    request_start = perf_counter()
    result, request_info = validated_request(prompt, content, cfg, "place_goal_verify")
    request_end = perf_counter()
    timings = {"visual_prepare_seconds": request_start - started,
               "api_seconds": request_end - request_start}
    if result is None:
        verdict = "request_failed" if cfg.get("place_goal_navigation", {}).get("evidence_coverage", False) else "uncertain"
        return verdict, {**request_info, "reason": "model_request_or_verification_parse_failed",
                             "timings": timings}
    return result["verdict"], {**result, **request_info, "timings": timings}
