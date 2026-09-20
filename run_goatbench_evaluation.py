import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"  # Allow loading ultralytics models

import argparse
from pathlib import Path
from omegaconf import OmegaConf
import random
import numpy as np
import torch

# Monkey-patch torch.load to use weights_only=False by default for compatibility
_original_torch_load = torch.load
def _patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=weights_only, **kwargs)
torch.load = _patched_torch_load
import math
import time
import json
import base64
import logging
from dataclasses import replace
import matplotlib.pyplot as plt
from PIL import Image

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene_goatbench import Scene
from src.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from src.goatbench_utils import prepare_goatbench_navigation_goals
from src.query_vlm_goatbench import query_vlm_for_response
from src.place_goal_navigation import PlaceGoalNavigation, invalidate_blocked_hop
from src.route_guidance import RouteGuidance, free_cell, grid_path
from src.hgr_dual_topo import DualTopoRuntime
from src.dual_dynamic_navigation import (
    BaselineDualDynamicNavigator,
    DualDynamicNavigator,
)
from src.dual_dynamic_navigation.config import validate_baseline_backend
from src.dual_dynamic_navigation.executor import KnownSpaceExecutor, known_mask
from src.dual_dynamic_navigation.unified_navigator import HypothesisAwareNavigator
from src.dual_dynamic_navigation.task_planner import ActionKind, NoAction, RequestError
from src.dual_dynamic_navigation.feedback import verify_entity
from src.subtask_execution import ExecutionStatus, SubtaskExecution
from src.hgr_timing import StageTimer, active_timer
from src.hgr_experiment_state import (SCENE_MEMORY, load_checkpoint, save_checkpoint,
    restore_random, record_decision, write_run_fingerprint)
from src.query_place_goal import select_place_goal, verify_place_goal
from src.query_vlm_hypothesis import infer_room_type_from_objects
from src.semantic_critic import SemanticCritic
from src.logger_goatbench import Logger
from src.goatbench_manifest import load_episode_manifest
from src.vlm_runtime import configure_vlm_runtime
from src.vlm_runtime import get_vlm_telemetry
from src.metric_goal_execution import (
    DirectFirstRouteMode,
    build_goal_conditioned_topology_overlay,
    build_metric_action_view,
    create_target_commitment,
    evaluate_revisit_intervention_gate,
    evaluate_target_commitment,
    plan_direct_first_revisit,
    revisit_gate_authorizes_selected_action,
)
from src.lightweight_goal_confidence import LightweightGoalConfidence
from src.dynamic_goal_topology import DynamicGoalTopology
from src.dual_layer_topology import (
    begin_topo_explore_commitment,
    choose_dual_layer_topology_action,
    refresh_topo_explore_commitment,
)
from src.region_topology import (
    build_region_topology_decision,
    select_local_explore_for_snapshot,
)
from src.place_topology import PlaceTopology
from src.topology_navigation import (
    PlaceRouteExecution,
    SnapshotSourceCommitment,
    PlaceRouteMode,
    plan_place_route,
    select_reachable_approach_place,
)
from src.persistent_belief_topology import (
    ActiveTopologyTraceWriter,
    GoalTopoActionType,
    GoalContext,
    PersistentBeliefTopology,
    TopoNodeType,
    VerificationEvidence,
    active_topology_goal_type_is_listed,
    restrict_active_topology_policy,
    resolve_active_topology_policy,
)
from src.goal_belief import (
    BeliefActionType,
    EvidenceSource,
    ExecutableNode,
    GoalBeliefMemory,
    ROOM_OBJECT_ASSOCIATIONS,
    clip_cosine_similarity,
    calibrated_clip_probability,
    category_compatibility_score,
    evaluate_snapshot_acceptance,
    normalize_goal_key,
    room_support_for_category,
)
from src.active_verification import (
    detect_category_confidence,
    query_action_tiebreak,
    query_active_verify,
    sample_verification_viewpoint,
)
from src.hierarchical_navigator import (
    HierarchicalNavigator,
    IntentKind,
    NavigationEvent,
    NavigationEventType,
    NavigatorCommandType,
    VerificationVerdict,
    query_terminal_verification,
)


def _vlm_telemetry_delta(after, before):
    """Compute a JSON-safe per-subtask delta from cumulative VLM telemetry."""
    scalar_keys = (
        "logical_calls", "http_attempts", "successes", "failures",
        "elapsed_seconds",
    )
    delta = {
        key: after.get(key, 0) - before.get(key, 0)
        for key in scalar_keys
    }
    purposes = set(after.get("by_purpose", {})) | set(before.get("by_purpose", {}))
    delta["by_purpose"] = {}
    for purpose in sorted(purposes):
        after_bucket = after.get("by_purpose", {}).get(purpose, {})
        before_bucket = before.get("by_purpose", {}).get(purpose, {})
        bucket = {
            key: after_bucket.get(key, 0) - before_bucket.get(key, 0)
            for key in scalar_keys
        }
        if any(value != 0 for value in bucket.values()):
            delta["by_purpose"][purpose] = bucket
    return delta


def _apply_diagnostic_limits(goal_types, goals, max_subtasks=None,
                             step_budget=None, max_steps_per_subtask=None):
    """Apply explicit short-run caps without changing normal evaluation defaults."""
    goal_types, goals = list(goal_types), list(goals)
    if max_subtasks is not None:
        goal_types = goal_types[:int(max_subtasks)]
        goals = goals[:int(max_subtasks)]
    if step_budget is not None and max_steps_per_subtask is not None:
        step_budget = min(int(step_budget), int(max_steps_per_subtask))
    return goal_types, goals, step_budget


def _choice_diagnostic(choice, frontiers):
    if isinstance(choice, SnapShot):
        return {
            "type": "snapshot",
            "image": choice.image,
            "object_ids": [int(item) for item in choice.cluster],
        }
    if isinstance(choice, Frontier):
        return {
            "type": "frontier",
            "source_index": next(
                (index for index, frontier in enumerate(frontiers) if frontier is choice),
                None,
            ),
            "topo_id": getattr(choice, "topo_id", None),
        }
    return {"type": "none"}


def _canonical_object_id(object_id, aliases):
    """Resolve scene merge aliases for evaluation diagnostics only."""
    current, seen = object_id, set()
    while current is not None and str(current) not in seen:
        seen.add(str(current))
        if current in aliases:
            current = aliases[current]
        elif str(current) in aliases:
            current = aliases[str(current)]
        else:
            break
    return None if current is None else str(current)


def _place_shadow_navigation_signature(
    choice,
    frontiers,
    selected_goal_action,
    selected_belief_action,
    planner_target,
    planner_look_at,
    target_override,
    look_at_override,
    target_arrived,
):
    """JSON-safe frozen output used by the Phase-A offline replay audit."""
    return {
        "choice": _choice_diagnostic(choice, frontiers),
        "selected_goal_topo_id": (
            getattr(selected_goal_action, "topo_node_id", None)
        ),
        "selected_goal_action_type": (
            getattr(getattr(selected_goal_action, "action_type", None),
                    "value", None)
        ),
        "selected_belief_node_id": (
            getattr(selected_belief_action, "node_id", None)
        ),
        "selected_belief_action_type": (
            getattr(getattr(selected_belief_action, "action_type", None),
                    "value", None)
        ),
        "planner_target": (
            None if planner_target is None
            else np.asarray(planner_target, dtype=float).tolist()
        ),
        "planner_look_at": (
            None if planner_look_at is None
            else np.asarray(planner_look_at, dtype=float).tolist()
        ),
        "target_override": (
            None if target_override is None
            else np.asarray(target_override, dtype=float).tolist()
        ),
        "look_at_override": (
            None if look_at_override is None
            else np.asarray(look_at_override, dtype=float).tolist()
        ),
        "target_arrived": bool(target_arrived),
    }


def _clip_image_embedding(image, clip_model, clip_preprocess):
    """Return a normalized CPU CLIP embedding for an RGB array/PIL image."""
    try:
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)[..., :3].astype(np.uint8))
        device = next(clip_model.parameters()).device
        tensor = clip_preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = clip_model.encode_image(tensor)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feature[0].detach().cpu().numpy()
    except Exception as exc:
        logging.warning("ActiveTopo CLIP image embedding failed: %s", exc)
        return None


def _clip_text_embedding(text, clip_model, clip_tokenizer):
    try:
        device = next(clip_model.parameters()).device
        tokens = clip_tokenizer([str(text)]).to(device)
        with torch.no_grad():
            feature = clip_model.encode_text(tokens)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feature[0].detach().cpu().numpy()
    except Exception as exc:
        logging.warning("ActiveTopo CLIP text embedding failed: %s", exc)
        return None


def _make_goal_context(
    subtask_id, metadata, clip_model, clip_preprocess, clip_tokenizer,
    include_goal_embedding=True,
):
    goal_type = str(metadata["task_type"])
    category = str(metadata.get("class") or "")
    text = metadata.get("question") or category
    image_embedding = None
    text_embedding = None
    if include_goal_embedding and goal_type == "image" and metadata.get("image"):
        try:
            with Image.open(metadata["image"]) as image:
                image_embedding = _clip_image_embedding(image, clip_model, clip_preprocess)
        except Exception as exc:
            logging.warning("ActiveTopo goal image load failed: %s", exc)
    elif include_goal_embedding:
        text_embedding = _clip_text_embedding(text, clip_model, clip_tokenizer)
    return GoalContext(
        subtask_id=subtask_id,
        goal_type="category" if goal_type == "object" else goal_type,
        category=category,
        description=text if goal_type == "description" else None,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
    )


def _load_config(config_path):
    """Load one config with optional relative ``extends`` inheritance."""
    config = OmegaConf.load(config_path)
    base_name = config.get("extends", None)
    if base_name is None:
        return config
    override = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    del override["extends"]
    base_path = str(base_name)
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(config_path), base_path)
    if os.path.abspath(base_path) == os.path.abspath(config_path):
        raise ValueError(f"Config cannot extend itself: {config_path}")
    base = _load_config(base_path)
    return OmegaConf.merge(base, override)


def _normalized_label(value):
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _numeric_confidence(value) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        return float(np.max(array)) if array.size else 0.0
    except (TypeError, ValueError):
        return 0.0


def _object_stability_score(obj, target_label: str) -> float:
    compatibility = category_compatibility_score(
        target_label, obj.get("class_name", "")
    )
    confidence = _numeric_confidence(obj.get("conf", 0.0))
    detections = max(0, int(obj.get("num_detections", 0)))
    return float(compatibility * confidence * min(detections / 3.0, 1.0))


def _semantic_distribution_fingerprint(semantic_dist) -> str:
    if semantic_dist is None:
        return "none"
    pairs = zip(
        getattr(semantic_dist, "categories", []),
        getattr(semantic_dist, "probabilities", []),
    )
    return "|".join(
        f"{_normalized_label(category)}:{float(probability):.3f}"
        for category, probability in pairs
    ) or "empty"


def _evaluate_v3_snapshot(
    scene, target_label, object_id, posterior, gate_cfg, confirmed=False,
):
    if object_id is None or object_id not in scene.objects:
        return evaluate_snapshot_acceptance(
            target_label=target_label, detected_label="",
            detector_confidence=0.0, detection_count=0,
            posterior=float(posterior), candidate_score=0.0,
            best_candidate_score=0.0, confirmed=confirmed,
        )
    obj = scene.objects[object_id]
    scores = [
        _object_stability_score(candidate, target_label)
        for candidate in scene.objects.values()
        if category_compatibility_score(
            target_label, candidate.get("class_name", "")
        ) >= float(gate_cfg.get("min_compatibility", 0.90))
    ]
    score = _object_stability_score(obj, target_label)
    return evaluate_snapshot_acceptance(
        target_label=target_label,
        detected_label=obj.get("class_name", ""),
        detector_confidence=_numeric_confidence(obj.get("conf", 0.0)),
        detection_count=int(obj.get("num_detections", 0)),
        posterior=float(posterior),
        candidate_score=score,
        best_candidate_score=max(scores, default=score),
        confirmed=confirmed,
        min_compatibility=float(gate_cfg.get("min_compatibility", 0.90)),
        min_detector_confidence=float(
            gate_cfg.get("min_detector_confidence", 0.50)
        ),
        min_detections=int(gate_cfg.get("min_detections", 2)),
        min_posterior=float(gate_cfg.get("min_posterior", 0.35)),
        alias_min_detections=int(gate_cfg.get("alias_min_detections", 3)),
        alias_min_posterior=float(gate_cfg.get("alias_min_posterior", 0.85)),
        best_candidate_margin=float(gate_cfg.get("best_candidate_margin", 0.05)),
    )


def _select_v3_frontier_fallback(
    executable_nodes, frontiers, belief_actions=None,
):
    mapped_actions = [
        action for action in (belief_actions or [])
        if action.source_kind == "frontier"
        and action.action_type in (
            BeliefActionType.EXPLORE, BeliefActionType.UNKNOWN_EXPLORE,
        )
        and action.source_index is not None
    ]
    if mapped_actions:
        selected = max(
            mapped_actions,
            key=lambda action: (
                action.utility, action.belief.posterior,
                action.belief.reachability, -action.normalized_path_cost,
                action.node_id,
            ),
        )
        index = int(selected.source_index)
        if 0 <= index < len(frontiers):
            return frontiers[index]
    reachable = [
        item for item in executable_nodes
        if item.source_kind == "frontier" and item.reachable
    ]
    if reachable:
        fact = max(
            reachable,
            key=lambda item: (
                item.map_information_gain - 0.1 * item.path_cost - 0.2 * item.revisit
            ),
        )
        index = int(fact.source_index)
        return frontiers[index] if 0 <= index < len(frontiers) else None
    if frontiers:
        return max(
            frontiers,
            key=lambda item: float(np.sum(getattr(item, "region", 0))),
        )
    return None


def _record_goal_evidence(
    memory, trace, topology, evidence, scene_id, episode_id, subtask_id,
):
    node = topology.nodes.get(evidence.node_id)
    if node is None:
        return False
    if memory.fast_evidence_path and memory.contains_evidence(
        evidence, count_fast_skip=True,
    ):
        return False
    before = memory.belief(
        evidence.node_id, evidence.goal_key,
        node.map_confidence, node.accessibility,
    )
    added = memory.add_evidence(evidence)
    if not added:
        return False
    after = memory.belief(
        evidence.node_id, evidence.goal_key,
        node.map_confidence, node.accessibility,
    )
    trace.write({
        "event": "goal_evidence",
        "scene_id": scene_id,
        "episode_id": episode_id,
        "subtask_id": subtask_id,
        "step": evidence.step,
        "evidence": evidence,
        "posterior_before": before.posterior,
        "posterior_after": after.posterior,
        "entropy_before": before.entropy,
        "entropy_after": after.entropy,
        "conflict": after.conflict,
        "fusion": memory.evidence_trace(evidence.node_id, evidence.goal_key),
    })
    return True


def _record_clip_evidence_v2(
    memory, trace, topology, *, evidence_id, node_id, goal_key,
    raw_cosine, step, viewpoint, heading, observation_id, reason,
    scene_id, episode_id, subtask_id,
):
    replay, event = memory.observe_clip(
        evidence_id=evidence_id,
        node_id=node_id,
        goal_key=goal_key,
        raw_cosine=raw_cosine,
        step=step,
        viewpoint=viewpoint,
        heading=heading,
        observation_id=observation_id,
        reason=reason,
    )
    calibration_stage = event.get("event", "unknown")
    event_payload = {key: value for key, value in event.items() if key != "event"}
    trace.write({
        "event": "clip_calibration",
        "calibration_stage": calibration_stage,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "subtask_id": subtask_id,
        "step": step,
        "node_id": node_id,
        **event_payload,
    })
    added = 0
    for evidence in replay:
        added += int(_record_goal_evidence(
            memory, trace, topology, evidence,
            scene_id, episode_id, subtask_id,
        ))
    return added


def _single_object_snapshot(snapshot, object_id, confidence):
    return SnapShot(
        image=snapshot.image,
        color=snapshot.color,
        obs_point=np.asarray(snapshot.obs_point).copy(),
        full_obj_list={object_id: confidence},
        cluster=[object_id],
    )


def _snapshot_choice_mapping_available(choice, scene) -> bool:
    """Return whether a committed one-object snapshot is still executable."""
    if type(choice) != SnapShot:
        return False
    image_available = any(
        getattr(snapshot, "image", None) == choice.image
        for snapshot in scene.snapshots.values()
    )
    return bool(
        image_available
        and choice.cluster
        and all(object_id in scene.objects for object_id in choice.cluster)
    )


def _snapshot_for_image(scene, snapshot_image):
    """Re-resolve a Snapshot by stable image identity, never by stored object."""
    image = str(snapshot_image)
    return next(
        (
            snapshot for snapshot in scene.snapshots.values()
            if getattr(snapshot, "image", None) == image
        ),
        None,
    )


def _direct_first_selection_mode(
    snapshot_image,
    current_step_frame_images,
    resolved_action_type=None,
    anchor_reobserve_snapshot=None,
):
    """Classify a selected Snapshot without changing HGR selection.

    The one-shot ``post_anchor_original_hgr`` state is the regression guard
    against repeatedly selecting an already reached capture anchor.
    """
    snapshot_image = str(snapshot_image)
    if (
        anchor_reobserve_snapshot is not None
        and snapshot_image == str(anchor_reobserve_snapshot)
    ):
        return "post_anchor_original_hgr"
    if resolved_action_type == GoalTopoActionType.REVISIT:
        return "revisit"
    if (
        resolved_action_type is None
        and snapshot_image not in set(current_step_frame_images or [])
    ):
        return "revisit"
    return "direct_or_current"


def _best_current_object_snapshot(
    scene, frame_images, target_label: str, min_detections: int = 1,
):
    """Return the strongest exact-category object seen in the current step.

    Route commitments are useful for revisiting a particular historical view,
    but a GOAT ``object`` goal accepts any instance of the requested category.
    Current-frame detections therefore take precedence without consulting
    semantic ground truth. The returned one-object snapshot still uses HGR's
    normal Snapshot navigation and success checks.
    """
    target = _normalized_label(target_label)
    if not target:
        return None
    best = None
    for image_name in frame_images:
        frame = scene.frames.get(image_name)
        if frame is None:
            continue
        for object_id, frame_confidence in frame.full_obj_list.items():
            obj = scene.objects.get(object_id)
            if obj is None or _normalized_label(obj.get("class_name")) != target:
                continue
            if int(obj.get("num_detections", 0)) < int(min_detections):
                continue
            confidence = _numeric_confidence(frame_confidence)
            rank = (confidence, _object_stability_score(obj, target))
            if best is None or rank > best[0]:
                best = (rank, frame, object_id, confidence)
    if best is None:
        return None
    _, frame, object_id, confidence = best
    return _single_object_snapshot(frame, object_id, confidence)


def main(
    cfg,
    start_ratio=0.0,
    end_ratio=1.0,
    split=1,
    episode_manifest=None,
):
    # Keep all GOAT VLM calls (explorer, hypothesis predictor, semantic critic)
    # on the same configured provider/model.
    validate_baseline_backend(cfg)
    configure_vlm_runtime(cfg)
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    manifest_path = episode_manifest or cfg.get("episode_manifest", None)
    manifest_entries = None
    manifest_episode_ids = {}
    if manifest_path:
        manifest_path = str(manifest_path)
        manifest_entries = load_episode_manifest(
            manifest_path, str(cfg.test_data_dir), split
        )
        scene_data_list = list(dict.fromkeys(
            entry["scene_file"] for entry in manifest_entries
        ))
        for entry in manifest_entries:
            manifest_episode_ids.setdefault(entry["scene_file"], []).append(
                entry["episode_id"]
            )
        num_episode = len(manifest_entries)
        logging.info(
            "Using fixed episode manifest: %s (split %s, %s entries)",
            manifest_path, split, num_episode,
        )
    else:
        scene_data_list = os.listdir(cfg.test_data_dir)
        num_scene = len(scene_data_list)
        random.shuffle(scene_data_list)
        # Legacy ratio selection is retained for configs without a manifest.
        scene_data_list = scene_data_list[
            int(start_ratio * num_scene) : int(end_ratio * num_scene)
        ]
        num_episode = 0
        for scene_data_file in scene_data_list:
            with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
                num_episode += len(json.load(f)["episodes"])
    logging.info(
        f"Total number of episodes in selected files: {num_episode}; "
        f"Selected runnable episodes: "
        f"{len(manifest_entries) if manifest_entries is not None else len(scene_data_list)}"
    )
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )
    if cfg.get("record_run_fingerprint", False):
        write_run_fingerprint(os.path.join(cfg.output_dir, "provenance", f"split_{split}"),
                              cfg, os.path.dirname(os.path.abspath(__file__)))

    for scene_data_file in scene_data_list:
        # load goatbench data
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )

        # Select the exact manifest episodes when configured.  The legacy
        # split-index behavior remains unchanged for other configurations.
        if manifest_entries is not None:
            episode_by_id = {
                str(episode["episode_id"]): episode
                for episode in scene_data["episodes"]
            }
            scene_data["episodes"] = [
                episode_by_id[episode_id]
                for episode_id in manifest_episode_ids[scene_data_file]
            ]
        else:
            scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
        total_episodes = len(scene_data["episodes"])

        all_navigation_goals = scene_data[
            "goals"
        ]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data["episodes"]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            all_subtask_goal_types, all_subtask_goals = (
                prepare_goatbench_navigation_goals(
                    scene_name=scene_name,
                    episode=episode,
                    all_navigation_goals=all_navigation_goals,
                )
            )
            all_subtask_goal_types, all_subtask_goals, _ = _apply_diagnostic_limits(
                all_subtask_goal_types,
                all_subtask_goals,
                cfg.get("diagnostic_max_subtasks"),
            )

            # check whether this episode has been processed
            finished_subtask_ids = list(logger.success_by_snapshot.keys())
            finished_episode_subtask = [
                subtask_id
                for subtask_id in finished_subtask_ids
                if subtask_id.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load scene
            try:
                del scene
            except:
                pass
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            _, _, num_step = _apply_diagnostic_limits(
                (), (), step_budget=num_step,
                max_steps_per_subtask=cfg.get(
                    "diagnostic_max_steps_per_subtask"
                ),
            )
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
                hypothesis_graph_cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True),
            )

            # Phase II: Initialize Semantic Critic for hypothesis verification
            if cfg.hypothesis.get("enable_hypothesis_refinement", True):
                semantic_critic = SemanticCritic(
                    cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True),
                    hypothesis_graph=tsdf_planner.hypothesis_graph,
                )
                semantic_critic.set_clip_model(clip_model, clip_preprocess)
            else:
                semantic_critic = None

            active_topology_cfg = cfg.get("active_topology", None)
            active_topology_enabled = bool(
                active_topology_cfg is not None
                and active_topology_cfg.get("enabled", False)
            )
            if active_topology_enabled:
                active_topology_mode = str(active_topology_cfg.get("mode", "full"))
                # Validate once before any paid VLM request is made.
                resolve_active_topology_policy(active_topology_mode, "category")
                persistent_topology = PersistentBeliefTopology(
                    OmegaConf.to_container(active_topology_cfg, resolve=True),
                    voxel_size=cfg.tsdf_grid_size,
                )
                topology_trace = ActiveTopologyTraceWriter(
                    str(cfg.output_dir), f"{scene_id}_ep_{episode_id}"
                )
                place_topology_cfg = active_topology_cfg.get(
                    "place_topology", {}
                )
                place_topology_enabled = bool(
                    place_topology_cfg.get("enabled", False)
                )
                if place_topology_enabled:
                    place_topology_stage = str(
                        place_topology_cfg.get("stage", "shadow")
                    ).strip().lower()
                    if place_topology_stage not in (
                        "shadow", "route_only"
                    ):
                        raise ValueError(
                            "PlaceTopology stage must be shadow or route_only"
                        )
                    if (
                        place_topology_stage == "route_only"
                        and active_topology_mode != "stable_only"
                    ):
                        raise ValueError(
                            "Place route-only requires stable_only so HGR "
                            "target selection remains unchanged"
                        )
                    place_topology = PlaceTopology(
                        place_spacing_m=float(
                            place_topology_cfg.get("place_spacing_m", 1.5)
                        ),
                        voxel_size=cfg.tsdf_grid_size,
                        stage=place_topology_stage,
                    )
                else:
                    place_topology_stage = None
                    place_topology = None
                place_route_only_enabled = bool(
                    place_topology_enabled
                    and place_topology_stage == "route_only"
                )
                place_route_stats = {
                    "plans": 0,
                    "same_place": 0,
                    "next_hop": 0,
                    "unmapped_target": 0,
                    "missing_current_place": 0,
                    "disconnected": 0,
                    "commitment_resumes": 0,
                    "source_mapping_failures": 0,
                    "waypoint_fallbacks": 0,
                    "edges_invalidated_by_geometry": 0,
                }
                active_embedding_cache = {}
                active_seen_snapshots = set()
                episode_vlm_start = get_vlm_telemetry()
                belief_v3_enabled = active_topology_mode == "belief_category_v3"
                simple_memory_enabled = active_topology_mode == "simple_memory"
                goal_topo_map_enabled = active_topology_mode == "goal_topo_map"
                goal_topo_object_live_priority_configured = bool(
                    goal_topo_map_enabled
                    and active_topology_cfg.get("object_live_priority", False)
                )
                goal_topo_unified_task_view_configured = bool(
                    goal_topo_map_enabled
                    and active_topology_cfg.get("unified_task_view", False)
                )
                goal_topo_behavior_goal_types = active_topology_cfg.get(
                    "behavior_goal_types", None
                )
                goal_topo_projection_only_goal_types = active_topology_cfg.get(
                    "projection_only_goal_types", []
                )
                metric_execution_cfg = active_topology_cfg.get(
                    "metric_execution", {}
                )
                metric_action_shadow_enabled = bool(
                    metric_execution_cfg.get("enabled", False)
                    and metric_execution_cfg.get("shadow_only", False)
                )
                if metric_action_shadow_enabled:
                    shadow_goal_types = {
                        str(item).strip().lower()
                        for item in metric_execution_cfg.get("goal_types", [])
                    }
                    required_shadow_goal_types = {
                        "object", "description", "image"
                    }
                    if shadow_goal_types != required_shadow_goal_types:
                        raise ValueError(
                            "V7.1-A shadow mode requires exactly object, "
                            "description, and image goal types"
                        )
                    if not goal_topo_map_enabled:
                        raise ValueError(
                            "V7.1-A shadow mode requires goal_topo_map mode"
                        )
                direct_first_cfg = active_topology_cfg.get(
                    "direct_first_execution", {}
                )
                metric_direct_first_enabled = bool(
                    direct_first_cfg.get("enabled", False)
                )
                if metric_direct_first_enabled:
                    direct_first_goal_types = {
                        str(item).strip().lower()
                        for item in direct_first_cfg.get("goal_types", [])
                    }
                    if direct_first_goal_types != {
                        "object", "description", "image"
                    }:
                        raise ValueError(
                            "V7.1-B direct-first requires exactly object, "
                            "description, and image goal types"
                        )
                    if not goal_topo_map_enabled:
                        raise ValueError(
                            "V7.1-B direct-first requires goal_topo_map mode"
                        )
                revisit_gate_cfg = active_topology_cfg.get("revisit_gate", {})
                revisit_gate_enabled = bool(revisit_gate_cfg.get("enabled", False))
                if revisit_gate_enabled:
                    gate_goal_types = {
                        str(item).strip().lower()
                        for item in revisit_gate_cfg.get("goal_types", [])
                    }
                    if gate_goal_types != {"object", "description", "image"}:
                        raise ValueError(
                            "V7.1-C revisit gate requires exactly object, "
                            "description, and image goal types"
                        )
                    if not metric_action_shadow_enabled:
                        raise ValueError(
                            "V7.1-C revisit gate requires V7.1-A metric "
                            "action projection"
                        )
                    for key in (
                        "min_relevance_margin",
                        "max_budget_fraction",
                        "max_revisit_to_explore_ratio",
                    ):
                        value = revisit_gate_cfg.get(key)
                        if value is None or not np.isfinite(float(value)):
                            raise ValueError(
                                f"V7.1-C revisit gate requires finite {key}"
                            )
                target_commitment_cfg = active_topology_cfg.get(
                    "target_commitment", {}
                )
                target_commitment_enabled = bool(
                    target_commitment_cfg.get("enabled", False)
                )
                if target_commitment_enabled:
                    if not (metric_direct_first_enabled and revisit_gate_enabled):
                        raise ValueError(
                            "V7.1-D target commitment requires V7.1-B "
                            "direct-first and V7.1-C revisit gate"
                        )
                    for key in (
                        "switch_relevance_margin", "tie_margin",
                        "min_switch_saving_m", "min_switch_saving_ratio",
                        "min_progress_m", "max_no_progress_segments",
                    ):
                        value = target_commitment_cfg.get(key)
                        if value is None or not np.isfinite(float(value)):
                            raise ValueError(
                                f"V7.1-D target commitment requires finite {key}"
                            )
                lightweight_confidence_cfg = active_topology_cfg.get(
                    "lightweight_confidence", {}
                )
                lightweight_confidence_enabled = bool(
                    lightweight_confidence_cfg.get("enabled", False)
                )
                if lightweight_confidence_enabled:
                    lightweight_goal_types = {
                        str(item).strip().lower()
                        for item in lightweight_confidence_cfg.get(
                            "goal_types", []
                        )
                    }
                    if lightweight_goal_types != {
                        "object", "description", "image"
                    }:
                        raise ValueError(
                            "Final V7 lightweight confidence requires exactly "
                            "object, description, and image goal types"
                        )
                    if not revisit_gate_enabled:
                        raise ValueError(
                            "Final V7 lightweight confidence requires the "
                            "frozen conservative reuse gate"
                        )
                    if target_commitment_enabled:
                        raise ValueError(
                            "Final V7 lightweight confidence does not combine "
                            "the failed V7.1-D commitment policy"
                        )
                    lightweight_goal_memory = LightweightGoalConfidence(
                        OmegaConf.to_container(
                            lightweight_confidence_cfg, resolve=True
                        )
                    )
                else:
                    lightweight_goal_memory = None
                dynamic_goal_topology_cfg = active_topology_cfg.get(
                    "dynamic_goal_topology", {}
                )
                dynamic_goal_topology_enabled = bool(
                    dynamic_goal_topology_cfg.get("enabled", False)
                )
                if dynamic_goal_topology_enabled:
                    dynamic_goal_types = {
                        str(item).strip().lower()
                        for item in dynamic_goal_topology_cfg.get(
                            "goal_types", []
                        )
                    }
                    if dynamic_goal_types != {
                        "object", "description", "image"
                    }:
                        raise ValueError(
                            "V7 DynamicGoalTopology requires exactly object, "
                            "description, and image goal types"
                        )
                    if not revisit_gate_enabled:
                        raise ValueError(
                            "V7 DynamicGoalTopology requires the frozen "
                            "C-strict revisit gate"
                        )
                    if lightweight_confidence_enabled:
                        raise ValueError(
                            "DynamicGoalTopology and the failed lightweight "
                            "confidence ablation are mutually exclusive"
                        )
                    if target_commitment_enabled:
                        raise ValueError(
                            "DynamicGoalTopology preserves the C-strict "
                            "executor and does not combine V7.1-D"
                        )
                    for key in (
                        "prior_strength", "positive_relevance",
                        "coverage_radius_m", "negative_evidence_weight",
                        "min_negative_views",
                        "min_independent_step_gap",
                        "min_independent_distance_voxels",
                        "freshness_half_life_steps", "one_hop_discount",
                    ):
                        value = dynamic_goal_topology_cfg.get(key)
                        if value is None or not np.isfinite(float(value)):
                            raise ValueError(
                                "V7 DynamicGoalTopology requires finite "
                                f"{key}"
                            )
                    dynamic_goal_topology = DynamicGoalTopology(
                        OmegaConf.to_container(
                            dynamic_goal_topology_cfg, resolve=True
                        )
                    )
                else:
                    dynamic_goal_topology = None
                dual_layer_planner_cfg = active_topology_cfg.get(
                    "dual_layer_planner", {}
                )
                dual_layer_planner_enabled = bool(
                    dual_layer_planner_cfg.get("enabled", False)
                )
                if dual_layer_planner_enabled:
                    dual_layer_goal_types = {
                        str(item).strip().lower()
                        for item in dual_layer_planner_cfg.get(
                            "goal_types", []
                        )
                    }
                    if dual_layer_goal_types != {
                        "object", "description", "image"
                    }:
                        raise ValueError(
                            "DualLayerTopology requires exactly object, "
                            "description, and image goal types"
                        )
                    if not dynamic_goal_topology_enabled:
                        raise ValueError(
                            "DualLayerTopology requires DynamicGoalTopology"
                        )
                    if dynamic_goal_topology.shadow_only:
                        raise ValueError(
                            "DualLayerTopology requires active dynamic goal "
                            "relevance (shadow_only=false)"
                        )
                    if not (
                        goal_topo_unified_task_view_configured
                        and metric_direct_first_enabled
                        and revisit_gate_enabled
                    ):
                        raise ValueError(
                            "DualLayerTopology requires unified GoalTopo, "
                            "direct-first execution, and the revisit gate"
                        )
                    if target_commitment_enabled:
                        raise ValueError(
                            "DualLayerTopology does not combine the failed "
                            "target-commitment ablation"
                        )
                dual_layer_explore_commitment_cfg = (
                    dual_layer_planner_cfg.get("explore_commitment", {})
                )
                dual_layer_explore_commitment_enabled = bool(
                    dual_layer_explore_commitment_cfg.get("enabled", False)
                )
                if dual_layer_explore_commitment_enabled:
                    if not dual_layer_planner_enabled:
                        raise ValueError(
                            "DualLayer explore commitment requires the "
                            "dual-layer planner"
                        )
                    for key in (
                        "min_progress_voxels", "max_no_progress_steps",
                    ):
                        value = dual_layer_explore_commitment_cfg.get(key)
                        if value is None or not np.isfinite(float(value)):
                            raise ValueError(
                                "DualLayer explore commitment requires finite "
                                f"{key}"
                            )
                    if float(
                        dual_layer_explore_commitment_cfg[
                            "min_progress_voxels"
                        ]
                    ) < 0.0 or int(
                        dual_layer_explore_commitment_cfg[
                            "max_no_progress_steps"
                        ]
                    ) < 1:
                        raise ValueError(
                            "DualLayer explore commitment requires nonnegative "
                            "min progress and a positive no-progress budget"
                        )
                dual_layer_geometric_rebind_cfg = (
                    dual_layer_explore_commitment_cfg.get(
                        "geometric_rebind", {}
                    )
                )
                dual_layer_geometric_rebind_enabled = bool(
                    dual_layer_geometric_rebind_cfg.get("enabled", False)
                )
                if dual_layer_geometric_rebind_enabled:
                    if not dual_layer_explore_commitment_enabled:
                        raise ValueError(
                            "DualLayer geometric rebind requires explore "
                            "commitment"
                        )
                    radius = dual_layer_geometric_rebind_cfg.get(
                        "max_anchor_distance_voxels"
                    )
                    if radius is None or not np.isfinite(float(radius)) or float(radius) < 0.0:
                        raise ValueError(
                            "DualLayer geometric rebind requires a finite "
                            "nonnegative max_anchor_distance_voxels"
                        )
                dual_layer_explore_shortlist_cfg = (
                    dual_layer_planner_cfg.get("explore_shortlist", {})
                )
                dual_layer_explore_shortlist_enabled = bool(
                    dual_layer_explore_shortlist_cfg.get("enabled", False)
                )
                if (
                    dual_layer_explore_shortlist_enabled
                    and not dual_layer_planner_enabled
                ):
                    raise ValueError(
                        "DualLayer explore shortlist requires the dual-layer "
                        "planner"
                    )
                dual_layer_region_shortlist_cfg = (
                    dual_layer_planner_cfg.get("region_shortlist", {})
                )
                dual_layer_region_shortlist_enabled = bool(
                    dual_layer_region_shortlist_cfg.get("enabled", False)
                )
                dual_layer_region_snapshot_evidence_enabled = bool(
                    dual_layer_region_shortlist_cfg.get(
                        "snapshot_evidence", {}
                    ).get("enabled", False)
                )
                dual_layer_region_frontier_only_enabled = bool(
                    dual_layer_region_shortlist_cfg.get(
                        "frontier_only", {}
                    ).get("enabled", False)
                )
                if (
                    dual_layer_region_frontier_only_enabled
                    and not dual_layer_region_shortlist_enabled
                ):
                    raise ValueError(
                        "DualLayer frontier-only mode requires the region shortlist"
                    )
                if (
                    dual_layer_region_snapshot_evidence_enabled
                    and not dual_layer_region_shortlist_enabled
                ):
                    raise ValueError(
                        "DualLayer snapshot evidence requires the region shortlist"
                    )
                if dual_layer_region_shortlist_enabled:
                    if not dual_layer_planner_enabled:
                        raise ValueError(
                            "DualLayer region shortlist requires the dual-layer "
                            "planner"
                        )
                    max_regions = dual_layer_region_shortlist_cfg.get(
                        "max_regions"
                    )
                    if (
                        max_regions is None
                        or not np.isfinite(float(max_regions))
                        or int(max_regions) < 1
                    ):
                        raise ValueError(
                            "DualLayer region shortlist requires a positive "
                            "max_regions"
                        )
                goal_overlay_cfg = active_topology_cfg.get(
                    "goal_conditioned_overlay", {}
                )
                goal_overlay_enabled = bool(
                    goal_overlay_cfg.get("enabled", False)
                )
                if goal_overlay_enabled:
                    overlay_goal_types = {
                        str(item).strip().lower()
                        for item in goal_overlay_cfg.get("goal_types", [])
                    }
                    if overlay_goal_types != {
                        "object", "description", "image"
                    }:
                        raise ValueError(
                            "Final V7 goal-conditioned overlay requires "
                            "exactly object, description, and image goal types"
                        )
                    if not revisit_gate_enabled:
                        raise ValueError(
                            "Final V7 goal-conditioned overlay requires the "
                            "frozen C-strict conservative gate"
                        )
                    if (
                        lightweight_confidence_enabled
                        or dynamic_goal_topology_enabled
                    ):
                        raise ValueError(
                            "Final V7 goal-conditioned overlay does not "
                            "combine failed dynamic confidence ablations"
                        )
                    if target_commitment_enabled:
                        raise ValueError(
                            "Final V7 goal-conditioned overlay preserves "
                            "C-strict and does not combine failed commitment"
                        )
                belief_v2_enabled = active_topology_mode in (
                    "belief_category_v2", "belief_category_v3",
                )
                if active_topology_mode in (
                    "belief_category", "belief_category_v2", "belief_category_v3",
                ):
                    belief_memory = GoalBeliefMemory(
                        OmegaConf.to_container(
                            active_topology_cfg.get("belief", {}), resolve=True
                        )
                    )
                else:
                    belief_memory = None
            else:
                active_topology_mode = None
                persistent_topology = None
                topology_trace = None
                active_embedding_cache = None
                active_seen_snapshots = None
                belief_memory = None
                belief_v2_enabled = False
                belief_v3_enabled = False
                simple_memory_enabled = False
                goal_topo_map_enabled = False
                goal_topo_object_live_priority_configured = False
                goal_topo_unified_task_view_configured = False
                goal_topo_behavior_goal_types = None
                goal_topo_projection_only_goal_types = []
                metric_execution_cfg = {}
                metric_action_shadow_enabled = False
                direct_first_cfg = {}
                metric_direct_first_enabled = False
                revisit_gate_cfg = {}
                revisit_gate_enabled = False
                target_commitment_cfg = {}
                target_commitment_enabled = False
                lightweight_confidence_cfg = {}
                lightweight_confidence_enabled = False
                lightweight_goal_memory = None
                dynamic_goal_topology_cfg = {}
                dynamic_goal_topology_enabled = False
                dynamic_goal_topology = None
                dual_layer_planner_cfg = {}
                dual_layer_planner_enabled = False
                dual_layer_explore_commitment_cfg = {}
                dual_layer_explore_commitment_enabled = False
                dual_layer_geometric_rebind_cfg = {}
                dual_layer_geometric_rebind_enabled = False
                dual_layer_explore_shortlist_cfg = {}
                dual_layer_region_shortlist_cfg = {}
                dual_layer_region_shortlist_enabled = False
                dual_layer_region_snapshot_evidence_enabled = False
                dual_layer_region_frontier_only_enabled = False
                dual_layer_explore_shortlist_enabled = False
                goal_overlay_cfg = {}
                goal_overlay_enabled = False
                place_topology_cfg = {}
                place_topology_enabled = False
                place_topology_stage = None
                place_topology = None
                place_route_only_enabled = False
                place_route_stats = {}

            phase_c_cfg = cfg.get("place_goal_navigation", {})
            phase_c_enabled = bool(phase_c_cfg.get("enabled", False))
            hgr_topology_fusion_enabled = bool(cfg.get("hgr_topology_fusion", {}).get("enabled", False))
            object_approach_enabled = bool(cfg.get("hgr_topology_fusion", {}).get("object_approach", False))
            if hgr_topology_fusion_enabled and (phase_c_enabled or not place_route_only_enabled):
                raise ValueError("HGR topology fusion requires Place routing and the original HGR selector (Phase C disabled)")
            guidance_enabled = bool(phase_c_cfg.get("route_guidance", False))
            if guidance_enabled and not phase_c_enabled:
                raise ValueError("route_guidance requires Phase C")
            if phase_c_enabled and not place_route_only_enabled:
                raise ValueError("Phase C requires the stable_only Place route execution configuration")
            dual_v2_cfg = cfg.get("hgr_dual_topo", {})
            legacy_dual_v2_enabled = bool(dual_v2_cfg.get("enabled", False))
            navigation_backend = str(cfg.get("navigation_backend", "legacy"))
            unified_enabled = navigation_backend in (
                "hgr_dual_topo_memory", "hgr_dual_topo_intent",
                "hgr_dual_topo_verified", "hgr_dual_topo_adaptive")
            if not unified_enabled and navigation_backend not in ("legacy", "dual_dynamic", "dual_dynamic_shadow", "dual_dynamic_route_only", "dual_dynamic_known_space"):
                raise ValueError(f"Unknown navigation_backend: {navigation_backend}")
            baseline_dual_dynamic_enabled = unified_enabled or navigation_backend in (
                "dual_dynamic", "dual_dynamic_shadow", "dual_dynamic_route_only", "dual_dynamic_known_space"
            )
            baseline_dual_dynamic_cfg = cfg.get("dual_dynamic_navigation", {})
            baseline_dual_dynamic_stage = str(
                baseline_dual_dynamic_cfg.get("stage", "shadow")
            ).strip().lower()
            legacy_hierarchical_cfg = cfg.get("hierarchical_navigation", {})
            if baseline_dual_dynamic_enabled:
                if bool(legacy_hierarchical_cfg.get("enabled", False)):
                    raise ValueError(
                        "navigation_backend=dual_dynamic is mutually exclusive with hierarchical_navigation"
                    )
                if legacy_dual_v2_enabled or phase_c_enabled or hgr_topology_fusion_enabled:
                    raise ValueError(
                        "baseline-preserving dual_dynamic cannot combine with V2-D, Phase C, or topology fusion"
                    )
                if baseline_dual_dynamic_stage not in (
                    "shadow", "route_only", "known_space", "memory", "intent",
                    "verified", "adaptive"
                ):
                    raise ValueError(
                        "dual_dynamic_navigation.stage must be shadow or route_only"
                    )
                expected_place_stage = ("route_only" if unified_enabled else
                                        "shadow" if baseline_dual_dynamic_stage == "known_space"
                                        else baseline_dual_dynamic_stage)
                if not (
                    active_topology_enabled
                    and active_topology_mode == "stable_only"
                    and place_topology_enabled
                    and place_topology_stage == expected_place_stage
                ):
                    raise ValueError(
                        "baseline-preserving dual_dynamic requires stable_only Place topology with the same stage"
                    )
                object_approach_enabled = bool(
                    baseline_dual_dynamic_stage == "route_only"
                    and baseline_dual_dynamic_cfg.get("object_approach", True)
                )
                if unified_enabled:
                    # Geometry is reused through adapters, not legacy Phase-B lifecycle.
                    place_route_only_enabled = False
                    object_approach_enabled = False
            elif object_approach_enabled and not hgr_topology_fusion_enabled:
                raise ValueError("object_approach requires HGR topology fusion")
            hierarchical_cfg = legacy_hierarchical_cfg
            hierarchical_enabled = bool(hierarchical_cfg.get("enabled", False))
            hierarchical_shadow_only = bool(
                hierarchical_enabled and hierarchical_cfg.get("shadow_only", False)
            )
            hierarchical_active = bool(hierarchical_enabled and not hierarchical_shadow_only)
            if hierarchical_enabled:
                if legacy_dual_v2_enabled and hierarchical_active:
                    raise ValueError(
                        "hierarchical_navigation is mutually exclusive with legacy hgr_dual_topo"
                    )
                if phase_c_enabled or not object_approach_enabled:
                    raise ValueError(
                        "hierarchical_navigation requires object approach and Phase C disabled"
                    )
                if not (
                    active_topology_enabled
                    and active_topology_mode == "stable_only"
                    and place_route_only_enabled
                ):
                    raise ValueError(
                        "hierarchical_navigation requires stable_only Place route topology"
                    )
                if hierarchical_shadow_only and not legacy_dual_v2_enabled:
                    raise ValueError(
                        "hierarchical shadow mode requires the V2-D execution baseline"
                    )
                if hierarchical_active:
                    # Reuse V2's geometry/cache/source implementation internally;
                    # the legacy selector and lifecycle remain disabled by config.
                    dual_v2_cfg = {
                        "enabled": True,
                        "goal_state": True,
                        "goal_context": False,
                        "cascade_correction": True,
                        "persistent_intent": True,
                        "cache_paths": bool(hierarchical_cfg.get("cache_paths", True)),
                        "record_decisions": bool(hierarchical_cfg.get("record_decisions", False)),
                        "local_same_place": True,
                    }
            dual_v2_enabled = bool(legacy_dual_v2_enabled or hierarchical_active)
            if dual_v2_enabled and (not object_approach_enabled or phase_c_enabled):
                raise ValueError("HGR V2 requires fusion object approach and Phase C disabled")
            if dual_v2_cfg.get("persistent_intent", False) and not dual_v2_cfg.get("goal_state", False):
                raise ValueError("persistent_intent requires goal_state")
            if dual_v2_cfg.get("cascade_correction", False) and not dual_v2_cfg.get("goal_state", False):
                raise ValueError("cascade_correction requires goal_state")
            dual_v2 = DualTopoRuntime(dual_v2_cfg) if dual_v2_enabled else None
            hierarchical_navigator = None
            if hierarchical_enabled:
                hierarchical_navigator = HierarchicalNavigator(
                    OmegaConf.to_container(hierarchical_cfg, resolve=True)
                )
            baseline_dual_dynamic = (
                (HypothesisAwareNavigator if unified_enabled else BaselineDualDynamicNavigator)(
                    OmegaConf.to_container(
                        baseline_dual_dynamic_cfg, resolve=True
                    )
                )
                if baseline_dual_dynamic_enabled else None
            )
            known_space_execution = bool(baseline_dual_dynamic_enabled
                and baseline_dual_dynamic_stage in (
                    "known_space", "route_only", "memory", "intent", "verified",
                    "adaptive"
                ))
            known_executor = (baseline_dual_dynamic.executor if unified_enabled else
                              KnownSpaceExecutor() if known_space_execution else None)
            if unified_enabled:
                tsdf_planner.hypothesis_graph.preserve_independent_evidence = True
            if dual_v2_enabled:
                place_topology.dual_topo_runtime = dual_v2
            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            logging.info(f"\n\nScene {scene_id} initialization successful!")

            # run questions in the scene
            global_step = -1
            memory_cfg = cfg.get("controlled_memory", {})
            memory_mode = memory_cfg.get("mode", "off")
            if memory_mode != "off" and cfg.clear_up_memory_every_subtask:
                raise ValueError("Controlled memory experiments require persistent episode memory")
            if memory_mode == "export" and (not place_topology_enabled or place_topology_stage != "shadow"):
                raise ValueError("Export shared memory using the original selector and a shadow Place graph")
            if memory_mode == "export" and not 0 < int(memory_cfg.get("prefix_subtasks", 1)) < len(all_subtask_goals):
                raise ValueError("Checkpoint prefix must leave at least one evaluation subtask")
            if memory_mode not in ("off", "export", "cold", "warm"):
                raise ValueError("controlled_memory.mode must be off/export/cold/warm")
            memory_start = 0
            checkpoint_path = os.path.join(str(memory_cfg.get("directory", "")), f"{scene_id}_ep_{episode_id}.pkl")
            if memory_mode in ("warm", "cold"):
                checkpoint = load_checkpoint(checkpoint_path, cfg, scene_id, episode_id)
                memory_start = checkpoint["subtask_index"]
                global_step = checkpoint["global_step"]
                pts, angle = checkpoint["pts"].copy(), checkpoint["angle"]
                if memory_mode == "warm" or memory_cfg.get("preserve_hgr_memory", False):
                    for attribute in SCENE_MEMORY:
                        setattr(scene, attribute, checkpoint["scene_memory"][attribute])
                    tsdf_planner = checkpoint["planner"]
                    if active_topology_enabled and checkpoint["persistent"] is not None:
                        persistent_topology = checkpoint["persistent"]
                    if place_topology_enabled and memory_mode == "warm":
                        if checkpoint["place"] is None:
                            raise ValueError("Warm checkpoint must include a shadow Place graph")
                        place_topology = checkpoint["place"]
                        place_topology.stage = place_topology_stage
                        if dual_v2_enabled:
                            place_topology.dual_topo_runtime = dual_v2
                    if (baseline_dual_dynamic_enabled
                            and checkpoint.get("navigation") is not None):
                        restored_navigation = checkpoint["navigation"]
                        expected_type = HypothesisAwareNavigator if unified_enabled else BaselineDualDynamicNavigator
                        if type(restored_navigation) is not expected_type:
                            raise ValueError("Original-HGR stages cannot restore a legacy semantic navigation checkpoint")
                        baseline_dual_dynamic = restored_navigation.validate_restore(
                            OmegaConf.to_container(baseline_dual_dynamic_cfg, resolve=True)
                        )
                        if unified_enabled:
                            known_executor = baseline_dual_dynamic.executor
                else:
                    tsdf_planner = TSDFPlanner(vol_bnds=tsdf_bnds, voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height, floor_height_offset=0, pts_init=pts,
                        init_clearance=cfg.init_clearance * 2, save_visualization=cfg.save_visualization,
                        hypothesis_graph_cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True))
                tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                if semantic_critic is not None:
                    semantic_critic.hypothesis_graph = tsdf_planner.hypothesis_graph
                restore_random(checkpoint)
                with open(os.path.join(cfg.output_dir, f"memory_{scene_id}_ep_{episode_id}.json"), "w") as handle:
                    json.dump({"mode": memory_mode, "checkpoint": checkpoint_path,
                        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                        "first_subtask": memory_start, "start_position": pts.tolist(),
                        "start_angle": float(angle),
                        "preserve_hgr_memory": bool(memory_cfg.get("preserve_hgr_memory", False)),
                        "single_subtask": bool(memory_cfg.get("single_subtask", False)),
                        "prefix_metrics": checkpoint["prefix_metrics"]}, handle, indent=2)
            for subtask_idx, (goal_type, subtask_goal) in enumerate(
                zip(all_subtask_goal_types, all_subtask_goals)
            ):
                if subtask_idx < memory_start:
                    continue
                if memory_mode == "export" and subtask_idx == int(memory_cfg.get("prefix_subtasks", 1)):
                    save_checkpoint(checkpoint_path, scene, tsdf_planner, place_topology,
                        persistent_topology if active_topology_enabled else None, pts, angle, global_step,
                        subtask_idx, cfg, scene_id, episode_id,
                        {k: v for k, v in logger.subtask_metrics.items() if k.startswith(f"{scene_id}_{episode_id}_")},
                        navigation=(
                            baseline_dual_dynamic
                            if baseline_dual_dynamic_enabled else None
                        ))
                    break
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
                logging.info(
                    f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
                )

                subtask_metadata = logger.init_subtask(
                    subtask_id=subtask_id,
                    goal_type=goal_type,
                    subtask_goal=subtask_goal,
                    pts=pts,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                )
                subtask_vlm_start = get_vlm_telemetry()
                stage_timer = StageTimer()
                stage_timer_token = active_timer.set(stage_timer)
                subtask_frame_start = len(scene.frames)

                if active_topology_enabled:
                    active_topology_policy = resolve_active_topology_policy(
                        active_topology_mode, goal_type
                    )
                    if goal_topo_map_enabled:
                        active_topology_policy = restrict_active_topology_policy(
                            active_topology_policy,
                            goal_type,
                            goal_topo_behavior_goal_types,
                        )
                    goal_topo_task_behavior_enabled = bool(
                        goal_topo_map_enabled
                        and active_topology_policy.use_active_view
                    )
                    goal_topo_projection_only = bool(
                        goal_topo_task_behavior_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            goal_topo_projection_only_goal_types,
                        )
                    )
                    goal_topo_route_enabled = bool(
                        goal_topo_task_behavior_enabled
                        and not goal_topo_projection_only
                    )
                    goal_topo_object_live_priority = bool(
                        goal_topo_route_enabled
                        and goal_topo_object_live_priority_configured
                    )
                    goal_topo_unified_task_view = bool(
                        goal_topo_route_enabled
                        and goal_topo_unified_task_view_configured
                    )
                    metric_direct_first_task_enabled = bool(
                        metric_direct_first_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            direct_first_cfg.get("goal_types", []),
                        )
                    )
                    revisit_gate_task_enabled = bool(
                        revisit_gate_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            revisit_gate_cfg.get("goal_types", []),
                        )
                    )
                    target_commitment_task_enabled = bool(
                        target_commitment_enabled
                        and revisit_gate_task_enabled
                    )
                    lightweight_confidence_task_enabled = bool(
                        lightweight_confidence_enabled
                        and revisit_gate_task_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            lightweight_confidence_cfg.get("goal_types", []),
                        )
                    )
                    dynamic_goal_topology_task_enabled = bool(
                        dynamic_goal_topology_enabled
                        and revisit_gate_task_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            dynamic_goal_topology_cfg.get("goal_types", []),
                        )
                    )
                    dual_layer_planner_task_enabled = bool(
                        dual_layer_planner_enabled
                        and dynamic_goal_topology_task_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            dual_layer_planner_cfg.get("goal_types", []),
                        )
                    )
                    dual_layer_explore_commitment_task_enabled = bool(
                        dual_layer_explore_commitment_enabled
                        and dual_layer_planner_task_enabled
                    )
                    dual_layer_explore_shortlist_task_enabled = bool(
                        dual_layer_explore_shortlist_enabled
                        and dual_layer_planner_task_enabled
                    )
                    goal_overlay_task_enabled = bool(
                        goal_overlay_enabled
                        and revisit_gate_task_enabled
                        and active_topology_goal_type_is_listed(
                            goal_type,
                            goal_overlay_cfg.get("goal_types", []),
                        )
                    )
                    goal_context = _make_goal_context(
                        subtask_id,
                        subtask_metadata,
                        clip_model,
                        clip_preprocess,
                        clip_tokenizer,
                        include_goal_embedding=active_topology_policy.use_goal_relevance,
                    )
                    if active_topology_policy.use_belief_planner:
                        # Belief likelihoods are category-conditioned; retain
                        # the question embedding used by every pre-existing
                        # ActiveTopo mode unchanged.
                        goal_context = replace(
                            goal_context,
                            text_embedding=_clip_text_embedding(
                                goal_context.category, clip_model, clip_tokenizer
                            ),
                        )
                    persistent_topology.set_goal(goal_context)
                    if dynamic_goal_topology_task_enabled:
                        dynamic_goal_topology.begin_goal(
                            goal_context.subtask_id
                        )
                    if belief_v3_enabled and active_topology_policy.use_belief_planner:
                        belief_memory.reset_unknown_policy(
                            normalize_goal_key(goal_context.category), "goal_switch"
                        )
                    dual_layer_region_shortlist_task_enabled = bool(
                        dual_layer_region_shortlist_enabled
                        and dual_layer_planner_task_enabled
                    )
                    dual_layer_region_snapshot_evidence_task_enabled = bool(
                        dual_layer_region_snapshot_evidence_enabled
                        and dual_layer_region_shortlist_task_enabled
                    )
                    dual_layer_region_frontier_only_task_enabled = bool(
                        dual_layer_region_frontier_only_enabled
                        and dual_layer_region_shortlist_task_enabled
                    )
                    topology_trace.write({
                        "event": "goal_switch",
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "subtask_id": subtask_id,
                        "goal_type": goal_context.goal_type,
                        "category": goal_context.category,
                        "mode": active_topology_policy.mode.value,
                        "active_view_enabled": active_topology_policy.use_active_view,
                        "goal_relevance_enabled": active_topology_policy.use_goal_relevance,
                        "topology_behavior_enabled": goal_topo_task_behavior_enabled,
                        "topology_projection_only": goal_topo_projection_only,
                        "topology_route_enabled": goal_topo_route_enabled,
                        "direct_first_execution_enabled": (
                            metric_direct_first_task_enabled
                        ),
                        "target_commitment_enabled": (
                            target_commitment_task_enabled
                        ),
                        "lightweight_confidence_enabled": (
                            lightweight_confidence_task_enabled
                        ),
                        "dynamic_goal_topology_enabled": (
                            dynamic_goal_topology_task_enabled
                        ),
                        "dynamic_goal_topology_shadow_only": (
                            bool(dynamic_goal_topology.shadow_only)
                            if dynamic_goal_topology_task_enabled else None
                        ),
                        "dual_layer_planner_enabled": (
                            dual_layer_planner_task_enabled
                        ),
                        "dual_layer_explore_commitment_enabled": (
                            dual_layer_explore_commitment_task_enabled
                        ),
                        "dual_layer_geometric_rebind_enabled": bool(
                            dual_layer_explore_commitment_task_enabled
                            and dual_layer_geometric_rebind_enabled
                        ),
                        "dual_layer_explore_shortlist_enabled": (
                            dual_layer_explore_shortlist_task_enabled
                        ),
                        "goal_conditioned_overlay_enabled": (
                            goal_overlay_task_enabled
                        ),
                        "dual_layer_region_shortlist_enabled": (
                            dual_layer_region_shortlist_task_enabled
                        ),
                        "dual_layer_region_snapshot_evidence_enabled": (
                            dual_layer_region_snapshot_evidence_task_enabled
                        ),
                        "dual_layer_region_frontier_only_enabled": (
                            dual_layer_region_frontier_only_task_enabled
                        ),
                    })
                else:
                    goal_context = None
                    active_topology_policy = None
                    goal_topo_task_behavior_enabled = False
                    goal_topo_projection_only = False
                    goal_topo_route_enabled = False
                    goal_topo_object_live_priority = False
                    goal_topo_unified_task_view = False
                    metric_direct_first_task_enabled = False
                    revisit_gate_task_enabled = False
                    target_commitment_task_enabled = False
                    lightweight_confidence_task_enabled = False
                    dynamic_goal_topology_task_enabled = False
                    dual_layer_region_shortlist_task_enabled = False
                    dual_layer_region_snapshot_evidence_task_enabled = False
                    dual_layer_region_frontier_only_task_enabled = False
                    dual_layer_planner_task_enabled = False
                    dual_layer_explore_commitment_task_enabled = False
                    dual_layer_explore_shortlist_task_enabled = False
                    goal_overlay_task_enabled = False

                # Only this flag authorizes changes to candidate presentation,
                # navigation failure handling, recovery, or SemanticCritic
                # side effects. Stable-only and category-only non-category
                # subtasks therefore retain the original HGR behavior.
                active_topology_behavior_enabled = bool(
                    active_topology_enabled
                    and (
                        active_topology_policy.use_active_view
                        or active_topology_policy.use_belief_planner
                    )
                )
                belief_planner_enabled = bool(
                    active_topology_enabled
                    and active_topology_policy.use_belief_planner
                )

                # mapping from the obj id in habitat to the id assigned by concept graph
                # this mapping/alignment is done by heuristic matching between object masks
                goal_obj_ids_mapping = {
                    obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
                }

                # run steps
                task_success = False
                subtask_execution = SubtaskExecution()
                critic_feature_errors_start = (
                    0 if semantic_critic is None else int(
                        semantic_critic.stats.get("feature_residual_errors", 0)
                    )
                )
                cnt_step = -1
                n_filtered_snapshots = 0
                target_obj_ids_estimate = []

                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                tsdf_planner.look_at_point = None
                max_point_choice = None
                decision_source = None
                active_topo_route = None
                committed_topo_route = None
                target_commitment = None
                selected_belief_action = None
                selected_goal_topo_action = None
                goal_topo_task_view = None
                dual_explore_commitment = None
                place_route_commitment = None
                place_route_snapshot_source = None
                place_route_execution = PlaceRouteExecution()
                phase_c = (
                    PlaceGoalNavigation(
                        subtask_id, cfg.tsdf_grid_size, cfg.planner.final_observe_distance,
                        float(phase_c_cfg.get("view_spacing_m", .5)),
                        bool(phase_c_cfg.get("cache_paths", True)),
                        bool(phase_c_cfg.get("topological_actions", False)),
                        bool(phase_c_cfg.get("event_driven_selection", False)),
                        bool(phase_c_cfg.get("evidence_coverage", False)),
                    ) if phase_c_enabled else None
                )
                phase_c_selected_crop = None
                route_guidance = None
                hierarchical_verification_target = None
                hierarchical_seen_stable_objects = {
                    str(object_id)
                    for object_id, obj in scene.objects.items()
                    if int(obj.get("num_detections", 0)) >= int(cfg.min_detection)
                }
                # After reaching one historical capture anchor, force exactly
                # one re-observation cycle through original HGR if it selects
                # the same Snapshot again. This prevents zero-motion anchor
                # loops without adding a V7.1-C intervention gate.
                direct_first_anchor_reobserve_snapshot = None

                if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
                    scene.clear_up_detections()
                    tsdf_planner = TSDFPlanner(
                        vol_bnds=tsdf_bnds,
                        voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height,
                        floor_height_offset=0,
                        pts_init=pts,
                        init_clearance=cfg.init_clearance * 2,
                        save_visualization=cfg.save_visualization,
                        hypothesis_graph_cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True),
                    )
                    # Re-create semantic critic for new hypothesis graph
                    if cfg.hypothesis.get("enable_hypothesis_refinement", True):
                        semantic_critic = SemanticCritic(
                            cfg=OmegaConf.to_container(cfg.hypothesis, resolve=True),
                            hypothesis_graph=tsdf_planner.hypothesis_graph,
                        )
                        semantic_critic.set_clip_model(clip_model, clip_preprocess)

                if dual_v2_enabled:
                    dual_v2.begin_goal(subtask_id)
                    tsdf_planner.hypothesis_graph.preserve_independent_evidence = bool(dual_v2_cfg.get("cascade_correction", False))
                if hierarchical_enabled:
                    hierarchical_navigator.begin_goal(subtask_id)
                    topology_trace.write({
                        "event": "hierarchical_goal_started",
                        "subtask_id": subtask_id,
                        "goal_type": goal_type,
                        "navigation_backend": navigation_backend,
                        "navigator": hierarchical_navigator.to_trace_dict(),
                    })
                if baseline_dual_dynamic_enabled:
                    baseline_dual_dynamic.begin_goal(subtask_id)
                    topology_trace.write({
                        "event": "dual_dynamic_goal_started",
                        "subtask_id": subtask_id,
                        "goal_type": goal_type,
                        "navigator": baseline_dual_dynamic.to_trace_dict(),
                    })
                while cnt_step < num_step - 1:
                    stage_timer.switch("mapping")
                    cnt_step += 1
                    global_step += 1
                    v2_arrival_frontier = None
                    hierarchical_terminal_arrived = False
                    if dual_v2_enabled:
                        dual_v2.cache.context = (str(subtask_id), global_step, None)
                    topology_route_arrived_this_step = False
                    navigation_target_override = None
                    navigation_look_at_override = None
                    topology_intervention_allowed = True
                    metric_direct_first_step_enabled = (
                        metric_direct_first_task_enabled
                    )
                    dual_layer_decision = None
                    place_route_plan = None
                    place_replay_observations = []
                    place_replay_frontiers = []
                    place_verified_paths = {}
                    place_known_connections_added = 0
                    logging.info(
                        f"\n== step: {cnt_step}, global step: {global_step} =="
                    )

                    if active_topology_enabled:
                        persistent_topology.decay_confidence(global_step)
                        current_voxel = tsdf_planner.habitat2voxel(pts)[:2]
                        visited_node_id = persistent_topology.upsert_visited(
                            current_voxel,
                            global_step,
                            evidence_ref=f"pose:{global_step}",
                        )
                        place_assignment = None
                        if place_topology_enabled:
                            if (
                                getattr(tsdf_planner, "island", None)
                                is not None
                                and place_topology.nodes
                            ):
                                nearby_places = (
                                    place_topology.local_connection_targets(
                                        current_voxel
                                    )
                                )
                                place_verified_paths = (
                                    PlaceTopology.known_grid_path_lengths_m(
                                        known_mask(tsdf_planner) if known_space_execution else tsdf_planner.island,
                                        current_voxel,
                                        nearby_places,
                                        cfg.tsdf_grid_size,
                                    )
                                )
                            place_assignment = place_topology.observe_pose(
                                current_voxel,
                                global_step,
                                observation_id=f"pose:{global_step}",
                                verified_paths_m=place_verified_paths,
                            )
                            place_known_connections_added = (
                                place_topology.add_known_free_connections(
                                    place_verified_paths,
                                    global_step,
                                    f"pose:{global_step}",
                                )
                            )

                    # (1) Observe the surroundings, update the scene graph and occupancy map
                    stage_timer.switch("perception")
                    # Determine the viewing angles for the current step
                    if cnt_step == 0:
                        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_2
                    else:
                        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_1
                    all_angles = [
                        angle + angle_increment * (i - total_views // 2)
                        for i in range(total_views)
                    ]
                    # Let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    rgb_egocentric_views = []
                    current_step_frame_images = []
                    all_added_obj_ids = (
                        []
                    )  # Record all the objects that are newly added in this step
                    for view_idx, ang in enumerate(all_angles):
                        # For each view
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        with torch.no_grad():
                            # Concept graph pipeline update
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                                scene.update_scene_graph(
                                    image_rgb=rgb[..., :3],
                                    depth=depth,
                                    intrinsics=cam_intr,
                                    cam_pos=cam_pose,
                                    pts=pts,
                                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                                    img_path=obs_file_name,
                                    frame_idx=cnt_step * total_views + view_idx,
                                    semantic_obs=semantic_obs,
                                    gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                                )
                            )
                            scene.all_observations[obs_file_name] = rgb
                            if obs_file_name in scene.frames:
                                current_step_frame_images.append(obs_file_name)
                            rgb_egocentric_views.append(
                                resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                            )
                            if cfg.save_visualization:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name),
                                    annotated_rgb,
                                )
                            else:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name), rgb
                                )
                            # update the mapping of hm3d object id to our detected object id
                            for (
                                gt_goal_id,
                                det_goal_id,
                            ) in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            all_added_obj_ids += added_obj_ids

                        # Clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=cnt_step * total_views + view_idx,
                            pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                        )

                        # Update depth map, occupancy map
                        tsdf_planner.integrate(
                            color_im=rgb,
                            depth_im=depth,
                            cam_intr=cam_intr,
                            cam_pose=pose_habitat_to_tsdf(cam_pose),
                            obs_weight=1.0,
                            margin_h=int(cfg.margin_h_ratio * img_height),
                            margin_w=int(cfg.margin_w_ratio * img_width),
                            explored_depth=cfg.explored_depth,
                        )
                    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")

                    # (2) Update Memory Snapshots with hierarchical clustering
                    stage_timer.switch("memory_update")
                    # Choose all the newly added objects as well as the objects nearby as the cluster targets
                    all_added_obj_ids = [
                        obj_id
                        for obj_id in all_added_obj_ids
                        if obj_id in scene.objects
                    ]
                    fresh_detected_obj_ids = set(all_added_obj_ids)
                    for obj_id, obj in scene.objects.items():
                        if (
                            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                            < cfg.scene_graph.obj_include_dist + 0.5
                        ):
                            all_added_obj_ids.append(obj_id)
                    scene.update_snapshots(
                        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
                    )
                    logging.info(
                        f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
                    )

                    # Update observation history for hypothesis prediction
                    belief_snapshot_sources = {}
                    confirmation_snapshot_node_id = None
                    for snap_key, snapshot in scene.snapshots.items():
                        obj_names = [
                            scene.objects[oid]["class_name"]
                            for oid in snapshot.cluster
                            if oid in scene.objects
                        ]
                        room_type = infer_room_type_from_objects(obj_names)
                        tsdf_planner.update_observation_history(snapshot, room_type)
                        if active_topology_enabled:
                            snapshot_embedding = (
                                None if simple_memory_enabled
                                else active_embedding_cache.get(snapshot.image)
                            )
                            if (
                                not simple_memory_enabled
                                and snapshot_embedding is None
                                and snapshot.image in scene.all_observations
                            ):
                                snapshot_embedding = _clip_image_embedding(
                                    scene.all_observations[snapshot.image],
                                    clip_model,
                                    clip_preprocess,
                                )
                                active_embedding_cache[snapshot.image] = snapshot_embedding
                            is_new_snapshot = snapshot.image not in active_seen_snapshots
                            observed_topo_id = persistent_topology.upsert_observed(
                                snapshot=snapshot,
                                position=snapshot.obs_point[:2],
                                categories=obj_names,
                                step=global_step,
                                anchor_visited_id=visited_node_id,
                                evidence_ref=f"snapshot:{snapshot.image}",
                                visual_embedding=snapshot_embedding,
                                reanchor_existing_evidence=(
                                    not goal_topo_map_enabled
                                ),
                            )
                            if place_topology_enabled and is_new_snapshot:
                                place_topology.bind_observation(
                                    observation_id=f"snapshot:{snapshot.image}",
                                    observed_from_place_id=(
                                        place_assignment.place_id
                                    ),
                                    capture_pose=snapshot.obs_point[:2],
                                    step=global_step,
                                )
                                place_replay_observations.append({
                                    "observation_id": (
                                        f"snapshot:{snapshot.image}"
                                    ),
                                    "capture_pose": np.asarray(
                                        snapshot.obs_point[:2], dtype=float
                                    ).tolist(),
                                    "expected_place_id": (
                                        place_assignment.place_id
                                    ),
                                })
                            if belief_planner_enabled:
                                # Every current snapshot may support active
                                # verification.  Only an exact detector match
                                # below upgrades it to an executable NAVIGATE
                                # target with a concrete object mapping.
                                belief_snapshot_sources.setdefault(
                                    observed_topo_id, (snapshot, None, 0.0)
                                )
                            active_seen_snapshots.add(snapshot.image)
                            if (active_topology_behavior_enabled and
                                    active_topology_policy.use_goal_relevance and
                                    is_new_snapshot and
                                    persistent_topology.node_relevance(
                                        observed_topo_id, goal_context
                                    ) >= 0.55):
                                persistent_topology.recovery.record_progress()

                            if belief_planner_enabled:
                                goal_key = normalize_goal_key(goal_context.category)
                                observation_id = f"snapshot:{snapshot.image}"
                                clip_probability = calibrated_clip_probability(
                                    snapshot_embedding, goal_context.text_embedding,
                                    floor=float(active_topology_cfg.get("relevance", {}).get("clip_floor", 0.10)),
                                    ceiling=float(active_topology_cfg.get("relevance", {}).get("clip_ceiling", 0.35)),
                                )
                                raw_clip_cosine = clip_cosine_similarity(
                                    snapshot_embedding, goal_context.text_embedding
                                )
                                if belief_v2_enabled and raw_clip_cosine is not None:
                                    _record_clip_evidence_v2(
                                        belief_memory, topology_trace, persistent_topology,
                                        evidence_id=f"{observation_id}:clip:{goal_key}",
                                        node_id=observed_topo_id, goal_key=goal_key,
                                        raw_cosine=raw_clip_cosine, step=global_step,
                                        viewpoint=snapshot.obs_point[:2], heading=None,
                                        observation_id=observation_id,
                                        reason="snapshot CLIP/category similarity",
                                        scene_id=scene_id, episode_id=episode_id,
                                        subtask_id=subtask_id,
                                    )
                                elif clip_probability is not None:
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{observation_id}:clip:{goal_key}", observed_topo_id,
                                            goal_key, EvidenceSource.CLIP_GOAL,
                                            clip_probability, global_step,
                                            viewpoint=snapshot.obs_point[:2],
                                            observation_id=observation_id,
                                            reason="snapshot CLIP/category similarity",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )

                                target_label = _normalized_label(goal_context.category)
                                matching_objects = [
                                    oid for oid in snapshot.cluster
                                    if oid in scene.objects and
                                    (
                                        category_compatibility_score(
                                            target_label,
                                            scene.objects[oid]["class_name"],
                                        ) >= 0.90
                                        if belief_v3_enabled
                                        else _normalized_label(
                                            scene.objects[oid]["class_name"]
                                        ) == target_label
                                    )
                                ]
                                if matching_objects:
                                    object_id = max(
                                        matching_objects,
                                        key=(
                                            (lambda oid: _object_stability_score(
                                                scene.objects[oid], target_label
                                            ))
                                            if belief_v3_enabled
                                            else (lambda oid: _numeric_confidence(
                                                scene.objects[oid].get("conf", 0.0)
                                            ))
                                        ),
                                    )
                                    detector_confidence = _numeric_confidence(
                                        scene.objects[object_id].get("conf", 0.0)
                                    )
                                    snapshot_rejected = False
                                    if belief_v3_enabled:
                                        snapshot_rejected, _ = belief_memory.is_snapshot_rejected(
                                            goal_key, object_id, global_step,
                                            compatibility=category_compatibility_score(
                                                target_label,
                                                scene.objects[object_id]["class_name"],
                                            ),
                                            detector_confidence=detector_confidence,
                                            detection_count=int(
                                                scene.objects[object_id].get(
                                                    "num_detections", 0
                                                )
                                            ),
                                        )
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{observation_id}:detector:{goal_key}", observed_topo_id,
                                            goal_key, EvidenceSource.DETECTOR,
                                            0.5 + 0.5 * detector_confidence * (
                                                category_compatibility_score(
                                                    target_label,
                                                    scene.objects[object_id]["class_name"],
                                                ) if belief_v3_enabled else 1.0
                                            ), global_step,
                                            viewpoint=snapshot.obs_point[:2],
                                            observation_id=observation_id,
                                            reason=f"detected {target_label} with confidence {detector_confidence:.3f}",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )
                                    existing_source = belief_snapshot_sources.get(
                                        observed_topo_id
                                    )
                                    if (
                                        not snapshot_rejected
                                        and (
                                        not belief_v3_enabled
                                        or existing_source is None
                                        or existing_source[1] is None
                                        or _object_stability_score(
                                            scene.objects[object_id], target_label
                                        ) > _object_stability_score(
                                            scene.objects[existing_source[1]], target_label
                                        )
                                        )
                                    ):
                                        belief_snapshot_sources[observed_topo_id] = (
                                            snapshot,
                                            object_id,
                                            detector_confidence,
                                        )

                                inferred_room = infer_room_type_from_objects(obj_names)
                                normalized_room = _normalized_label(inferred_room)
                                if target_label in ROOM_OBJECT_ASSOCIATIONS.get(normalized_room, set()):
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{observation_id}:context:{goal_key}", observed_topo_id,
                                            goal_key, EvidenceSource.CONTEXT_PRIOR,
                                            0.80, global_step,
                                            viewpoint=snapshot.obs_point[:2],
                                            observation_id=observation_id,
                                            reason=f"target/room association: {target_label}/{normalized_room}",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )

                    if belief_v2_enabled and belief_planner_enabled:
                        confirmation_cfg = active_topology_cfg.get(
                            "confirmation", {}
                        )
                        confirmation = belief_memory.active_confirmation(
                            normalize_goal_key(goal_context.category), global_step
                        )
                        if confirmation is not None:
                            confirmed_node = persistent_topology.nodes.get(
                                confirmation.node_id
                            )
                            if confirmed_node is not None:
                                confirmed_belief = belief_memory.belief(
                                    confirmation.node_id,
                                    confirmation.goal_key,
                                    confirmed_node.map_confidence,
                                    confirmed_node.accessibility,
                                )
                                if confirmed_belief.posterior < 0.30:
                                    belief_memory.resolve_confirmation(
                                        confirmation.goal_key, "posterior_reversed"
                                    )
                                    confirmation = None
                            if confirmation is not None:
                                eligible_snapshots = []
                                for topo_id, source_info in belief_snapshot_sources.items():
                                    snapshot, object_id, detector_confidence = source_info
                                    snapshot_node = persistent_topology.nodes.get(topo_id)
                                    compatibility = (
                                        category_compatibility_score(
                                            goal_context.category,
                                            scene.objects[object_id].get("class_name", ""),
                                        )
                                        if object_id is not None and object_id in scene.objects
                                        else 0.0
                                    )
                                    if (
                                        object_id is not None
                                        and (
                                            not belief_v3_enabled
                                            or compatibility >= 0.90
                                        )
                                        and detector_confidence >= float(
                                            confirmation_cfg.get(
                                                "detector_min_confidence", 0.50
                                            )
                                        )
                                        and snapshot_node is not None
                                        and confirmed_node is not None
                                        and persistent_topology._distance_m(
                                            snapshot_node.position,
                                            confirmed_node.position,
                                        ) <= float(
                                            confirmation_cfg.get(
                                                "snapshot_max_distance_m", 2.5
                                            )
                                        )
                                    ):
                                        eligible_snapshots.append((
                                            persistent_topology._distance_m(
                                                snapshot_node.position,
                                                confirmed_node.position,
                                            ),
                                            topo_id,
                                        ))
                                if eligible_snapshots:
                                    confirmation_snapshot_node_id = min(
                                        eligible_snapshots
                                    )[1]
                                if belief_v3_enabled:
                                    belief_memory.record_confirmation_observation(
                                        confirmation.goal_key,
                                        global_step,
                                        snapshot_found=bool(eligible_snapshots),
                                    )
                                elif eligible_snapshots:
                                    confirmation.awaiting_observation = False
                                elif (
                                    confirmation.awaiting_observation
                                    and confirmation.approach_count
                                    >= confirmation.max_approaches
                                ):
                                    belief_memory.resolve_confirmation(
                                        confirmation.goal_key,
                                        "no_snapshot_after_approach_budget",
                                    )

                    # (3) Update the Frontier Snapshots
                    stage_timer.switch("mapping")
                    update_success = tsdf_planner.update_frontier_map(
                        pts=pts,
                        cfg=cfg.planner,
                        scene=scene,
                        cnt_step=cnt_step,
                        save_frontier_image=cfg.save_visualization,
                        eps_frontier_dir=eps_frontier_dir,
                        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                        defer_hypothesis_prediction=bool(
                            belief_v2_enabled and belief_planner_enabled
                        ),
                        preserve_hypothesis_node_ids=(
                            dual_v2.active_hypothesis_refs()
                            if dual_v2_enabled else None
                        ),
                    )
                    if not update_success:
                        logging.info("Warning! Update frontier map failed!")

                    active_topo_view = None
                    belief_action_candidates = []
                    belief_hypothesis_set = None
                    executable_nodes = []
                    executable_target_positions = {}
                    confirmation_priority_action = None
                    snapshot_gate_events = []
                    unknown_policy_decision = None
                    if active_topology_enabled and len(tsdf_planner.frontiers) > 0:
                        frontier_embeddings = {}
                        if not simple_memory_enabled:
                            for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                                cache_key = (
                                    f"frontier:{subtask_id}:{frontier.image}"
                                    if belief_planner_enabled
                                    else f"frontier:{frontier.image}"
                                )
                                embedding = active_embedding_cache.get(cache_key)
                                if embedding is None and frontier.feature is not None:
                                    embedding = _clip_image_embedding(
                                        frontier.feature, clip_model, clip_preprocess
                                    )
                                    active_embedding_cache[cache_key] = embedding
                                frontier_embeddings[frontier_index] = embedding
                        persistent_topology.match_frontiers(
                            tsdf_planner.frontiers,
                            global_step,
                            visual_embeddings=frontier_embeddings,
                        )
                        if place_topology_enabled:
                            frontier_targets = {
                                persistent_topology.frontier_source_map[index]:
                                frontier.position
                                for index, frontier in enumerate(
                                    tsdf_planner.frontiers
                                )
                                if index in persistent_topology.frontier_source_map
                            }
                            known_frontier_paths = (
                                PlaceTopology.known_grid_path_lengths_m(
                                    known_mask(tsdf_planner) if known_space_execution else tsdf_planner.island,
                                    current_voxel,
                                    frontier_targets,
                                    cfg.tsdf_grid_size,
                                )
                            )
                            for frontier_index, frontier in enumerate(
                                tsdf_planner.frontiers
                            ):
                                frontier_topo_id = (
                                    persistent_topology.frontier_source_map.get(
                                        frontier_index
                                    )
                                )
                                if frontier_topo_id is not None:
                                    current_path_m = (
                                        known_frontier_paths.get(
                                            frontier_topo_id
                                        )
                                    )
                                    verified_approach_m = (
                                        None
                                        if current_path_m is None
                                        else current_path_m
                                        + place_topology.current_place_offset_m
                                    )
                                    place_topology.observe_frontier(
                                        frontier_topo_id,
                                        frontier.position,
                                        place_assignment.place_id,
                                        global_step,
                                        source_frontier_index=frontier_index,
                                        verified_approach_path_m=(
                                            verified_approach_m
                                        ),
                                    )
                                    place_replay_frontiers.append({
                                        "frontier_id": frontier_topo_id,
                                        "frontier_position": np.asarray(
                                            frontier.position, dtype=float
                                        )[:2].tolist(),
                                        "source_frontier_index": (
                                            frontier_index
                                        ),
                                        "expected_place_id": (
                                            place_assignment.place_id
                                        ),
                                        "verified_approach_path_m": (
                                            verified_approach_m
                                        ),
                                    })
                        if belief_v2_enabled and belief_planner_enabled:
                            semantic_cache_cfg = active_topology_cfg.get(
                                "semantic_cache", {}
                            )
                            cached_semantics = {}
                            semantic_cache_events = []
                            for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                                topo_id = persistent_topology.frontier_source_map.get(
                                    frontier_index
                                )
                                if topo_id is None:
                                    continue
                                cached, reason = persistent_topology.lookup_frontier_semantic_cache(
                                    topo_id,
                                    frontier_embeddings.get(frontier_index),
                                    frontier.position,
                                    min_visual_cosine=float(
                                        semantic_cache_cfg.get("min_visual_cosine", 0.85)
                                    ),
                                    max_displacement_m=float(
                                        semantic_cache_cfg.get("max_displacement_m", 1.0)
                                    ),
                                )
                                if cached is not None:
                                    cached_semantics[frontier.frontier_id] = cached
                                semantic_cache_events.append({
                                    "frontier_index": frontier_index,
                                    "frontier_id": frontier.frontier_id,
                                    "topo_id": topo_id,
                                    "result": reason,
                                })
                            prediction_sources = tsdf_planner.finalize_hypothesis_node_semantics(
                                scene, pts, cached_semantics=cached_semantics,
                                preserve_node_ids=(dual_v2.active_hypothesis_refs()
                                                   if dual_v2_enabled else None),
                            )
                            for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                                topo_id = persistent_topology.frontier_source_map.get(
                                    frontier_index
                                )
                                if topo_id is None:
                                    continue
                                source = prediction_sources.get(
                                    frontier.frontier_id, "heuristic"
                                )
                                node = persistent_topology.nodes.get(topo_id)
                                if node is not None and frontier.semantic_dist is not None:
                                    node.semantic_dist = frontier.semantic_dist
                                if source == "vlm":
                                    persistent_topology.store_frontier_semantic_cache(
                                        topo_id,
                                        frontier.semantic_dist,
                                        frontier_embeddings.get(frontier_index),
                                        frontier.position,
                                        global_step,
                                        source,
                                    )
                            topology_trace.write({
                                "event": "frontier_semantic_cache",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "lookups": semantic_cache_events,
                                "prediction_sources": prediction_sources,
                            })
                        if belief_planner_enabled:
                            target_label = _normalized_label(goal_context.category)
                            goal_key = normalize_goal_key(target_label)
                            nearby_names = []
                            for obj in scene.objects.values():
                                if "bbox" not in obj or not hasattr(obj["bbox"], "center"):
                                    continue
                                if np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]]) <= 3.5:
                                    nearby_names.append(obj.get("class_name", ""))
                            nearby_room = _normalized_label(
                                infer_room_type_from_objects(nearby_names)
                            )
                            context_support = float(
                                target_label in ROOM_OBJECT_ASSOCIATIONS.get(nearby_room, set())
                            )
                            for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                                topo_id = persistent_topology.frontier_source_map.get(frontier_index)
                                if topo_id is None:
                                    continue
                                observation_id = (
                                    f"frontier:{global_step}:{frontier.image}"
                                    if belief_v2_enabled
                                    else f"frontier:{subtask_id}:{frontier.image}"
                                )
                                clip_probability = calibrated_clip_probability(
                                    frontier_embeddings.get(frontier_index),
                                    goal_context.text_embedding,
                                    floor=float(active_topology_cfg.get("relevance", {}).get("clip_floor", 0.10)),
                                    ceiling=float(active_topology_cfg.get("relevance", {}).get("clip_ceiling", 0.35)),
                                )
                                frontier_heading = math.atan2(
                                    frontier.position[1] - current_voxel[1],
                                    frontier.position[0] - current_voxel[0],
                                )
                                raw_clip_cosine = clip_cosine_similarity(
                                    frontier_embeddings.get(frontier_index),
                                    goal_context.text_embedding,
                                )
                                if belief_v2_enabled and raw_clip_cosine is not None:
                                    _record_clip_evidence_v2(
                                        belief_memory, topology_trace, persistent_topology,
                                        evidence_id=f"{observation_id}:clip:{goal_key}",
                                        node_id=topo_id, goal_key=goal_key,
                                        raw_cosine=raw_clip_cosine, step=global_step,
                                        viewpoint=current_voxel, heading=frontier_heading,
                                        observation_id=observation_id,
                                        reason="frontier CLIP/category similarity",
                                        scene_id=scene_id, episode_id=episode_id,
                                        subtask_id=subtask_id,
                                    )
                                elif clip_probability is not None:
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{observation_id}:clip:{goal_key}", topo_id,
                                            goal_key, EvidenceSource.CLIP_GOAL,
                                            clip_probability, global_step,
                                            viewpoint=current_voxel,
                                            heading=frontier_heading,
                                            observation_id=observation_id,
                                            reason="frontier CLIP/category similarity",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )
                                room_support = room_support_for_category(
                                    frontier.semantic_dist, target_label
                                )
                                if room_support > 0.0:
                                    room_observation_id = (
                                        f"frontier-semantic:{topo_id}:"
                                        f"{_semantic_distribution_fingerprint(frontier.semantic_dist)}"
                                        if belief_v3_enabled else observation_id
                                    )
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{room_observation_id}:room:{goal_key}", topo_id,
                                            goal_key, EvidenceSource.VLM_ROOM_PRIOR,
                                            0.5 + 0.45 * room_support, global_step,
                                            viewpoint=current_voxel,
                                            heading=math.atan2(
                                                frontier.position[1] - current_voxel[1],
                                                frontier.position[0] - current_voxel[0],
                                            ),
                                            observation_id=room_observation_id,
                                            reason=f"room/object support {room_support:.3f}",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )
                                if context_support > 0.0:
                                    context_observation_id = (
                                        f"frontier-context:{topo_id}:{nearby_room}"
                                        if belief_v3_enabled else observation_id
                                    )
                                    _record_goal_evidence(
                                        belief_memory, topology_trace, persistent_topology,
                                        belief_memory.make_evidence(
                                            f"{context_observation_id}:context:{goal_key}", topo_id,
                                            goal_key, EvidenceSource.CONTEXT_PRIOR,
                                            0.5 + 0.30 * context_support, global_step,
                                            viewpoint=current_voxel,
                                            heading=math.atan2(
                                                frontier.position[1] - current_voxel[1],
                                                frontier.position[0] - current_voxel[0],
                                            ),
                                            observation_id=context_observation_id,
                                            reason=f"nearby room/object context {nearby_room}",
                                        ),
                                        scene_id, episode_id, subtask_id,
                                    )
                        for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                            topo_id = persistent_topology.frontier_source_map.get(frontier_index)
                            if topo_id and frontier.hypothesis_node_id:
                                dependency_confidence = 0.5
                                if frontier.semantic_dist is not None:
                                    _, top_probs = frontier.semantic_dist.get_top_k(1)
                                    if len(top_probs):
                                        dependency_confidence = float(top_probs[0])
                                persistent_topology.bind_hypothesis(
                                    topo_id,
                                    frontier.hypothesis_node_id,
                                    dependency_confidence,
                                )
                        if active_topology_policy.use_active_view:
                            current_voxel = tsdf_planner.habitat2voxel(pts)[:2]
                            path_costs = {}
                            unreachable_indices = set()
                            if goal_topo_map_enabled:
                                active_topo_view = persistent_topology.build_goal_topology_view(
                                    goal_context,
                                    tsdf_planner.frontiers,
                                    global_step,
                                )
                                topology_trace.write({
                                    "event": "goal_topology_projection",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "context": active_topo_view.topology_context,
                                    "included_node_ids": active_topo_view.included_node_ids,
                                    "evidence_node_ids": active_topo_view.evidence_node_ids,
                                    "connector_node_ids": active_topo_view.connector_node_ids,
                                })
                            else:
                                for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                                    try:
                                        distance, path_points = tsdf_planner.get_distance(
                                            current_voxel,
                                            frontier.position,
                                            height=floor_height,
                                            pathfinder=scene.pathfinder,
                                        )
                                        path_costs[frontier_index] = float(distance)
                                        if path_points is None:
                                            unreachable_indices.add(frontier_index)
                                    except Exception as exc:
                                        path_costs[frontier_index] = float(
                                            np.linalg.norm(current_voxel - frontier.position)
                                            * cfg.tsdf_grid_size
                                        )
                                        logging.warning("ActiveTopo path query fallback: %s", exc)
                            if simple_memory_enabled:
                                active_topo_view = persistent_topology.build_simple_memory_view(
                                    goal_context,
                                    tsdf_planner.frontiers,
                                    global_step,
                                    unreachable_indices=unreachable_indices,
                                )
                            elif not goal_topo_map_enabled:
                                active_topo_view = persistent_topology.build_active_view(
                                    goal_context,
                                    tsdf_planner.frontiers,
                                    path_costs,
                                    global_step,
                                    unreachable_indices=unreachable_indices,
                                    use_goal_relevance=active_topology_policy.use_goal_relevance,
                                )
                            if len(active_topo_view.candidates) == 0:
                                logging.warning(
                                    "ActiveTopo recovery exhausted; using original HGR frontier fallback"
                                )
                                topology_trace.write({
                                    "event": "recovery_exhausted",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                })
                                active_topo_view = None

                    if belief_planner_enabled:
                        goal_key = normalize_goal_key(goal_context.category)
                        raw_facts = []
                        frontier_areas = [
                            float(np.sum(getattr(frontier, "region", 0)))
                            for frontier in tsdf_planner.frontiers
                        ]
                        max_frontier_area = max(frontier_areas, default=0.0)
                        for frontier_index, frontier in enumerate(tsdf_planner.frontiers):
                            topo_id = persistent_topology.frontier_source_map.get(frontier_index)
                            node = persistent_topology.nodes.get(topo_id)
                            if node is None:
                                continue
                            try:
                                raw_cost, path_points = tsdf_planner.get_distance(
                                    current_voxel, frontier.position,
                                    height=floor_height, pathfinder=scene.pathfinder,
                                )
                                reachable = path_points is not None and np.isfinite(raw_cost)
                            except Exception:
                                raw_cost, reachable = float("inf"), False
                            raw_facts.append({
                                "node": node,
                                "source_kind": "frontier",
                                "source_index": frontier_index,
                                "target_position": np.asarray(frontier.position)[:2],
                                "raw_cost": float(raw_cost),
                                "reachable": reachable,
                                "revisit": min(node.visit_count / 3.0, 1.0),
                                "map_information_gain": (
                                    frontier_areas[frontier_index] / max_frontier_area
                                    if max_frontier_area > 0 else 0.0
                                ),
                            })
                        for topo_id, (snapshot, object_id, detector_confidence) in belief_snapshot_sources.items():
                            node = persistent_topology.nodes.get(topo_id)
                            if node is None:
                                continue
                            try:
                                raw_cost, path_points = tsdf_planner.get_distance(
                                    current_voxel, snapshot.obs_point[:2],
                                    height=floor_height, pathfinder=scene.pathfinder,
                                )
                                reachable = path_points is not None and np.isfinite(raw_cost)
                            except Exception:
                                raw_cost, reachable = float("inf"), False
                            raw_facts.append({
                                "node": node,
                                "source_kind": (
                                    "snapshot" if object_id is not None
                                    else "snapshot_context"
                                ),
                                "source_index": topo_id,
                                "target_position": (
                                    tsdf_planner.habitat2voxel(
                                        scene.objects[object_id]["bbox"].center
                                    )[:2]
                                    if object_id is not None and object_id in scene.objects
                                    else np.asarray(snapshot.obs_point)[:2]
                                ),
                                "raw_cost": float(raw_cost),
                                "reachable": reachable,
                                "revisit": min(node.visit_count / 3.0, 1.0),
                                "map_information_gain": 0.0,
                            })
                        finite_costs = [
                            item["raw_cost"] for item in raw_facts
                            if item["reachable"] and np.isfinite(item["raw_cost"])
                        ]
                        min_cost = min(finite_costs, default=0.0)
                        max_cost = max(finite_costs, default=min_cost)
                        executable_nodes = []
                        executable_target_positions = {}
                        for item in raw_facts:
                            raw_cost = item["raw_cost"]
                            norm_cost = (
                                (raw_cost - min_cost) / (max_cost - min_cost)
                                if item["reachable"] and max_cost > min_cost else 0.0
                            )
                            executable_nodes.append(ExecutableNode(
                                node_id=item["node"].node_id,
                                source_kind=item["source_kind"],
                                source_index=item["source_index"],
                                structural_confidence=item["node"].map_confidence,
                                reachability=item["node"].accessibility,
                                path_cost=float(norm_cost),
                                revisit=item["revisit"],
                                map_information_gain=item["map_information_gain"],
                                reachable=item["reachable"],
                            ))
                            executable_target_positions[item["node"].node_id] = (
                                np.asarray(item["target_position"], dtype=float)[:2]
                            )
                        belief_hypothesis_set = belief_memory.build_hypotheses(
                            goal_key, executable_nodes, global_step
                        )
                        verification_cfg = active_topology_cfg.get("verification", {})
                        verification_options = {}
                        for belief in belief_hypothesis_set.hypotheses:
                            if not (
                                float(verification_cfg.get("posterior_low", 0.30))
                                <= belief.posterior
                                <= float(verification_cfg.get("posterior_high", 0.70))
                            ):
                                continue
                            node = persistent_topology.nodes.get(belief.node_id)
                            if node is None:
                                continue
                            option = sample_verification_viewpoint(
                                center=executable_target_positions.get(
                                    belief.node_id, node.position
                                ),
                                current=current_voxel,
                                occupied=tsdf_planner.occupied,
                                unoccupied=tsdf_planner.unoccupied,
                                island=tsdf_planner.island,
                                voxel_size=cfg.tsdf_grid_size,
                                path_query=lambda start, end: tsdf_planner.get_distance(
                                    start, end, height=floor_height,
                                    pathfinder=scene.pathfinder,
                                ),
                                history=belief_memory.clusters(belief.node_id, goal_key),
                                entropy=belief.entropy,
                                radii_m=list(verification_cfg.get("radii_m", [1.0, 1.5, 2.0])),
                                samples_per_radius=int(
                                    verification_cfg.get("samples_per_radius", 16)
                                ),
                            )
                            if option is not None:
                                verification_options[belief.node_id] = option
                        belief_action_candidates = belief_memory.build_actions(
                            belief_hypothesis_set,
                            executable_nodes,
                            verification_options=verification_options,
                            posterior_low=float(verification_cfg.get("posterior_low", 0.30)),
                            posterior_high=float(verification_cfg.get("posterior_high", 0.70)),
                            navigate_threshold=float(verification_cfg.get("navigate_threshold", 0.75)),
                            step=global_step,
                            max_verify_attempts=int(
                                verification_cfg.get("max_attempts_per_node_goal", 2)
                            ),
                            verify_cooldown_steps=int(
                                verification_cfg.get("cooldown_steps", 5)
                            ),
                            include_unknown=not belief_v3_enabled,
                        )
                        if belief_v2_enabled:
                            confirmation = belief_memory.active_confirmation(
                                goal_key, global_step
                            )
                            if confirmation is not None:
                                confirmed_node = persistent_topology.nodes.get(
                                    confirmation.node_id
                                )
                                confirmed_belief = None
                                if confirmed_node is not None:
                                    confirmed_belief = belief_memory.belief(
                                        confirmation.node_id,
                                        goal_key,
                                        confirmed_node.map_confidence,
                                        confirmed_node.accessibility,
                                    )
                                if (
                                    confirmed_belief is not None
                                    and confirmed_belief.posterior >= float(
                                        active_topology_cfg.get(
                                            "confirmation", {}
                                        ).get("posterior_threshold", 0.75)
                                    )
                                    and confirmation_snapshot_node_id is not None
                                ):
                                    snapshot_fact = next((
                                        item for item in executable_nodes
                                        if item.node_id == confirmation_snapshot_node_id
                                        and item.source_kind == "snapshot"
                                        and item.reachable
                                    ), None)
                                    if snapshot_fact is not None:
                                        confirmation_priority_action = (
                                            belief_memory.build_confirmation_snapshot_action(
                                                confirmation,
                                                snapshot_fact,
                                                confirmed_belief,
                                                navigate_threshold=float(
                                                    active_topology_cfg.get(
                                                        "confirmation", {}
                                                    ).get(
                                                        "posterior_threshold", 0.75
                                                    )
                                                ),
                                            )
                                        )
                                if confirmation_priority_action is None:
                                    confirmation_priority_action = (
                                        belief_memory.build_confirmation_action(
                                            confirmation, executable_nodes
                                        )
                                    )
                                if confirmation_priority_action is not None:
                                    belief_action_candidates.insert(
                                        0, confirmation_priority_action
                                    )
                        if belief_v3_enabled:
                            gate_cfg = active_topology_cfg.get("snapshot_gate", {})
                            gated_actions = []
                            for action in belief_action_candidates:
                                if action.action_type != BeliefActionType.NAVIGATE:
                                    gated_actions.append(action)
                                    continue
                                source_info = belief_snapshot_sources.get(action.node_id)
                                object_id = source_info[1] if source_info is not None else None
                                decision = _evaluate_v3_snapshot(
                                    scene,
                                    goal_context.category,
                                    object_id,
                                    action.belief.posterior,
                                    gate_cfg,
                                    confirmed=(
                                        action.confirmation_source_node_id is not None
                                    ),
                                )
                                belief_memory.record_snapshot_gate(decision)
                                snapshot_gate_events.append({
                                    "source": "belief_action",
                                    "node_id": action.node_id,
                                    "object_id": object_id,
                                    "decision": decision,
                                })
                                if decision.accepted:
                                    gated_actions.append(action)
                                else:
                                    belief_memory.reject_snapshot(
                                        goal_key, object_id, global_step, decision,
                                        duration_steps=int(
                                            gate_cfg.get("rejection_ttl_steps", 5)
                                        ),
                                    )
                                    if action is confirmation_priority_action:
                                        confirmation_priority_action = None
                            belief_action_candidates = gated_actions
                            unknown_cfg = active_topology_cfg.get("belief", {})
                            unknown_action, unknown_policy_decision = (
                                belief_memory.build_controlled_unknown_action(
                                    belief_hypothesis_set,
                                    executable_nodes,
                                    belief_action_candidates,
                                    step=global_step,
                                    active_confirmation=(
                                        belief_memory.active_confirmation(
                                            goal_key, global_step
                                        ) is not None
                                    ),
                                    unknown_threshold=float(
                                        unknown_cfg.get(
                                            "unknown_action_threshold", 0.75
                                        )
                                    ),
                                    top_posterior_max=float(
                                        unknown_cfg.get(
                                            "unknown_top_posterior_max", 0.30
                                        )
                                    ),
                                    max_consecutive_actions=int(
                                        unknown_cfg.get(
                                            "max_consecutive_unknown_actions", 2
                                        )
                                    ),
                                    reset_posterior_delta=float(
                                        unknown_cfg.get(
                                            "unknown_reset_posterior_delta", 0.05
                                        )
                                    ),
                                )
                            )
                            if unknown_action is not None:
                                belief_action_candidates.append(unknown_action)
                        topology_trace.write({
                            "event": "belief_projection",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "goal_key": goal_key,
                            "hypotheses": belief_hypothesis_set,
                            "actions": belief_action_candidates,
                            "snapshot_gate_events": snapshot_gate_events,
                            "unknown_policy": unknown_policy_decision,
                        })

                    if metric_action_shadow_enabled:
                        shadow_direct_snapshot = None
                        if subtask_metadata["task_type"] == "object":
                            shadow_direct_choice = _best_current_object_snapshot(
                                scene,
                                current_step_frame_images,
                                subtask_metadata.get("class", ""),
                                min_detections=cfg.min_detection,
                            )
                            if shadow_direct_choice is not None:
                                shadow_direct_node_id = (
                                    persistent_topology.observed_node_for_snapshot(
                                        shadow_direct_choice.image
                                    )
                                )
                                if shadow_direct_node_id is not None:
                                    shadow_direct_snapshot = (
                                        shadow_direct_node_id,
                                        shadow_direct_choice.image,
                                        shadow_direct_choice.cluster[0],
                                    )
                        shadow_task_view = persistent_topology.build_goal_task_view(
                            goal_context,
                            tsdf_planner.frontiers,
                            [
                                snapshot.image
                                for snapshot in scene.snapshots.values()
                            ],
                            global_step,
                            direct_snapshot=shadow_direct_snapshot,
                            current_snapshot_images=current_step_frame_images,
                            record_state=False,
                        )
                        metric_action_view = build_metric_action_view(
                            shadow_task_view,
                            persistent_topology,
                            tsdf_planner.frontiers,
                            tsdf_planner,
                            scene.pathfinder,
                            current_voxel,
                            floor_height,
                            remaining_steps=max(0, num_step - cnt_step),
                            planner_step_m=float(
                                metric_execution_cfg.get(
                                    "planner_step_m",
                                    max(
                                        cfg.planner.max_dist_from_cur_phase_1,
                                        cfg.planner.max_dist_from_cur_phase_2,
                                    ),
                                )
                            ),
                        )
                        confidence_trace = None
                        if lightweight_confidence_task_enabled:
                            metric_action_view, confidence_trace = (
                                lightweight_goal_memory.update_and_adjust(
                                    metric_action_view,
                                    persistent_topology,
                                    current_voxel,
                                    global_step,
                                )
                            )
                            topology_trace.write({
                                "event": "lightweight_goal_confidence",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "behavior_applied": True,
                                "view": confidence_trace,
                            })
                        dynamic_goal_view = None
                        if dynamic_goal_topology_task_enabled:
                            metric_action_view, dynamic_goal_view = (
                                dynamic_goal_topology.update_and_project(
                                    metric_action_view,
                                    persistent_topology,
                                    current_voxel,
                                    global_step,
                                )
                            )
                            topology_trace.write({
                                "event": "dynamic_goal_topology_view",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "behavior_applied": bool(
                                    not dynamic_goal_topology.shadow_only
                                ),
                                "view": dynamic_goal_view.to_trace_dict(),
                            })
                        goal_conditioned_overlay = None
                        if goal_overlay_task_enabled:
                            goal_conditioned_overlay = (
                                build_goal_conditioned_topology_overlay(
                                    metric_action_view
                                )
                            )
                            topology_trace.write({
                                "event": "goal_conditioned_topology_overlay",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "behavior_applied": True,
                                "view": (
                                    goal_conditioned_overlay.to_trace_dict()
                                ),
                            })
                        topology_trace.write({
                            "event": "metric_action_shadow_view",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "behavior_applied": bool(
                                lightweight_confidence_task_enabled
                                or (
                                    dynamic_goal_topology_task_enabled
                                    and not dynamic_goal_topology.shadow_only
                                )
                                or goal_overlay_task_enabled
                            ),
                            "view": metric_action_view.to_trace_dict(),
                        })
                        if revisit_gate_task_enabled:
                            revisit_gate_decision = (
                                evaluate_revisit_intervention_gate(
                                    goal_conditioned_overlay
                                    if goal_conditioned_overlay is not None
                                    else metric_action_view,
                                    min_relevance_margin=float(
                                        revisit_gate_cfg[
                                            "min_relevance_margin"
                                        ]
                                    ),
                                    max_budget_fraction=float(
                                        revisit_gate_cfg[
                                            "max_budget_fraction"
                                        ]
                                    ),
                                    max_revisit_to_explore_ratio=float(
                                        revisit_gate_cfg[
                                            "max_revisit_to_explore_ratio"
                                        ]
                                    ),
                                    planner_step_m=float(
                                        metric_execution_cfg.get(
                                            "planner_step_m", 1.0
                                        )
                                    ),
                                )
                            )
                            topology_trace.write({
                                "event": "revisit_intervention_gate",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "goal_type": goal_context.goal_type,
                                "decision": (
                                    revisit_gate_decision.to_trace_dict()
                                ),
                            })
                            if not revisit_gate_decision.allowed:
                                # This is the V7.1-C boundary: do not alter the
                                # HGR frontier ordering/prompt or invoke the
                                # V7.1-B historical route executor this step.
                                topology_intervention_allowed = False
                                metric_direct_first_step_enabled = False
                                active_topo_view = None
                        if dual_layer_planner_task_enabled:
                            dual_layer_decision = (
                                choose_dual_layer_topology_action(
                                    metric_action_view,
                                    dynamic_goal_view,
                                    revisit_gate_decision,
                                    min_direct_margin=float(
                                        revisit_gate_cfg[
                                            "min_relevance_margin"
                                        ]
                                    ),
                                )
                            )
                            topology_trace.write({
                                "event": "dual_layer_topology_decision",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "decision": (
                                    dual_layer_decision.to_trace_dict()
                                ),
                            })
                        if (
                            dual_layer_explore_commitment_task_enabled
                            and dual_explore_commitment is not None
                        ):
                            frontier_positions = {
                                index: frontier.position
                                for index, frontier in enumerate(
                                    tsdf_planner.frontiers
                                )
                            }
                            refreshed_commitment, release_reason = (
                                refresh_topo_explore_commitment(
                                    dual_explore_commitment,
                                    metric_action_view,
                                    current_voxel,
                                    frontier_positions,
                                    global_step,
                                    min_progress_voxels=float(
                                        dual_layer_explore_commitment_cfg[
                                            "min_progress_voxels"
                                        ]
                                    ),
                                    max_no_progress_steps=int(
                                        dual_layer_explore_commitment_cfg[
                                            "max_no_progress_steps"
                                        ]
                                    ),
                                    geometric_rebind_radius_voxels=(
                                        float(
                                            dual_layer_geometric_rebind_cfg[
                                                "max_anchor_distance_voxels"
                                            ]
                                        )
                                        if dual_layer_geometric_rebind_enabled
                                        else 0.0
                                    ),
                                    live_target_position=(
                                        tsdf_planner.max_point.position
                                        if type(tsdf_planner.max_point)
                                        == Frontier else None
                                    ),
                                )
                            )
                            if refreshed_commitment is None:
                                topology_trace.write({
                                    "event": "dual_layer_explore_commitment_released",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": (
                                        dual_explore_commitment.to_trace_dict()
                                    ),
                                    "reason": release_reason,
                                })
                                dual_explore_commitment = None
                            else:
                                dual_explore_commitment = refreshed_commitment
                                if release_reason is not None:
                                    topology_trace.write({
                                        "event": (
                                            "dual_layer_explore_commitment_rebound"
                                            if release_reason == "geometric_rebind"
                                            else "dual_layer_explore_commitment_anchor_kept"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "commitment": (
                                            dual_explore_commitment.to_trace_dict()
                                        ),
                                        "reason": release_reason,
                                    })

                    # (4) Choose the next navigation point by querying the VLM
                    stage_timer.switch("candidate_preparation")
                    if baseline_dual_dynamic_enabled:
                        map_delta = baseline_dual_dynamic.sync(
                            place_topology,
                            scene,
                            tsdf_planner.frontiers,
                            f"{subtask_id}:{global_step}",
                            min_detection=cfg.min_detection,
                        )
                        topology_trace.write({
                            "event": "dual_dynamic_map_delta",
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "stage": baseline_dual_dynamic_stage,
                            "delta": map_delta,
                        })
                    if hierarchical_active:
                        current_stable_objects = {
                            str(object_id)
                            for object_id, obj in scene.objects.items()
                            if int(obj.get("num_detections", 0)) >= int(cfg.min_detection)
                        }
                        newly_stable = current_stable_objects - hierarchical_seen_stable_objects
                        hierarchical_seen_stable_objects |= current_stable_objects
                        if subtask_metadata["task_type"] == "object":
                            target_label = _normalized_label(subtask_metadata["class"])
                            newly_relevant = {
                                object_id for object_id in newly_stable
                                if int(object_id) in scene.objects
                                and _normalized_label(scene.objects[int(object_id)]["class_name"])
                                == target_label
                            }
                        else:
                            # Description/image identity cannot be decided by a
                            # class-name heuristic. A newly stable visual object
                            # is the minimal evidence-change event; the VLM gate
                            # still decides whether it is relevant.
                            newly_relevant = newly_stable
                        if newly_relevant and hierarchical_navigator.active_intent is not None:
                            command = hierarchical_navigator.on_event(NavigationEvent(
                                NavigationEventType.EVIDENCE_CHANGED,
                                global_step,
                                "new_stable_object_evidence",
                            ))
                            dual_v2.release("new_stable_object_evidence")
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            topology_trace.write({
                                "event": "hierarchical_evidence_changed",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "new_object_ids": sorted(newly_relevant),
                                "command": command.to_trace_dict(),
                            })
                    if dual_v2_enabled and dual_v2_cfg.get("goal_state", False):
                        source_events = dual_v2.synchronize(scene, tsdf_planner)
                        if source_events:
                            topology_trace.write({"event": "v2_source_events", "subtask_id": subtask_id,
                                "step": global_step, "changes": source_events})
                        observation_id = f"{subtask_id}:{global_step}"
                        dual_v2.goal.observation_coverage[observation_id] = {
                            "place_id": place_topology.current_place_id,
                            "pose": np.asarray(pts).tolist(), "angle": float(angle),
                            "meaning": "observed_pose_only"}
                    if dual_v2_enabled and dual_v2.goal.active_intent is not None:
                        live_choice = dual_v2.refresh_source(scene, tsdf_planner)
                        if live_choice is None:
                            topology_trace.write({"event": "v2_feedback", "subtask_id": subtask_id,
                                "step": global_step, "feedback": dual_v2.release("source_mapping_invalid")})
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            if hierarchical_active:
                                hierarchical_navigator.on_event(NavigationEvent(
                                    NavigationEventType.SOURCE_INVALIDATED,
                                    global_step,
                                    "source_mapping_invalid",
                                ))
                        elif tsdf_planner.target_point is not None:
                            max_point_choice = tsdf_planner.max_point = live_choice
                            # A moved object changes the physical terminal, not its semantic identity.
                            if type(live_choice) == SnapShot:
                                center = tsdf_planner.habitat2voxel(scene.objects[live_choice.cluster[0]]["bbox"].center)[:2]
                                if (not np.array_equal(center, dual_v2.selected_approach["look_at"])
                                        or not free_cell(tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                                         dual_v2.selected_approach["terminal"])):
                                    from src.object_approach import resolve_object_approach
                                    refreshed = resolve_object_approach(place_topology, tsdf_planner, scene.objects,
                                        live_choice.cluster[0], tsdf_planner.habitat2voxel(pts)[:2],
                                        cfg.planner.final_observe_distance, dual_v2.selected_approach)
                                    if refreshed["status"] == "valid":
                                        dual_v2.install(live_choice, refreshed, tsdf_planner.island.shape, retain_intent=True)
                                        tsdf_planner.target_point = np.asarray(refreshed["terminal"])
                                        tsdf_planner.look_at_point = np.asarray(refreshed["look_at"])
                                    else:
                                        dual_v2.release("object_terminal_unavailable")
                                        if hierarchical_active:
                                            hierarchical_navigator.on_event(NavigationEvent(
                                                NavigationEventType.SOURCE_INVALIDATED,
                                                global_step,
                                                "object_terminal_unavailable",
                                            ))
                                        tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            elif (not np.array_equal(live_choice.position[:2], dual_v2.selected_approach.get("frontier_position"))
                                  or not free_cell(tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                                   dual_v2.selected_approach["terminal"])):
                                from src.object_approach import resolve_frontier_approach
                                refreshed = resolve_frontier_approach(place_topology, tsdf_planner, live_choice,
                                    tsdf_planner.habitat2voxel(pts)[:2], cfg.planner.final_observe_distance,
                                    previous=dual_v2.selected_approach)
                                if refreshed["status"] == "valid":
                                    dual_v2.install(live_choice, refreshed, tsdf_planner.island.shape, retain_intent=True)
                                    tsdf_planner.target_point = np.asarray(refreshed["terminal"])
                                    tsdf_planner.look_at_point = np.asarray(refreshed["look_at"])
                                else:
                                    dual_v2.release("frontier_terminal_unavailable")
                                    if hierarchical_active:
                                        hierarchical_navigator.on_event(NavigationEvent(
                                            NavigationEventType.SOURCE_INVALIDATED,
                                            global_step,
                                            "frontier_terminal_unavailable",
                                        ))
                                    tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                    if phase_c_enabled and phase_c.active is not None and tsdf_planner.target_point is not None:
                        live_choice, source_info = phase_c.resume(scene, tsdf_planner)
                        known_path = {"active": 0.} if guidance_enabled else PlaceTopology.known_grid_path_lengths_m(
                            tsdf_planner.island, tsdf_planner.habitat2voxel(pts)[:2],
                            {"active": tsdf_planner.target_point}, cfg.tsdf_grid_size,
                        )
                        if live_choice is None or ("active" not in known_path and not guidance_enabled):
                            reason = "source_mapping_invalid" if live_choice is None else "known_route_became_unreachable"
                            invalidated = invalidate_blocked_hop(
                                place_topology, place_route_execution.active_plan,
                                tsdf_planner.occupied, global_step,
                            ) if live_choice is not None else False
                            topology_trace.write({
                                "event": "phase_c_execution_feedback", "subtask_id": subtask_id,
                                "step": global_step, "source_resolution": source_info,
                                "feedback": phase_c.finish(reason, global_step),
                                "edge_invalidated": invalidated,
                            })
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None
                            tsdf_planner.look_at_point = None
                            place_route_commitment = None
                            place_route_snapshot_source = None
                            place_route_execution.reset()
                        else:
                            # Perception may merge an object during a local segment.
                            tsdf_planner.max_point = max_point_choice = live_choice
                    if (unified_enabled and baseline_dual_dynamic.active_intent is not None):
                        preempted = baseline_dual_dynamic.assess_incremental(
                            query_vlm_for_response, subtask_metadata, scene, tsdf_planner,
                            place_topology, rgb_egocentric_views, cfg, pts, fresh_detected_obj_ids,
                            lambda: scene.get_observation(pts, angle=angle)[0]["color_sensor"], verify_entity,
                            remaining_steps=max(0, num_step-cnt_step))
                        if preempted:
                            topology_trace.write({"event": "hypothesis_aware_preemption",
                                "step": global_step, "navigator": baseline_dual_dynamic.to_trace_dict()})
                        live_choice = baseline_dual_dynamic.continue_choice(scene, tsdf_planner,
                            place_topology, pts, cfg)
                        if live_choice is None:
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                        else:
                            max_point_choice = tsdf_planner.max_point = live_choice
                    if cfg.choose_every_step:
                        retain_v2_hop = bool(unified_enabled and baseline_dual_dynamic.persistent
                                             and baseline_dual_dynamic.active_intent is not None)
                        if dual_v2_enabled and dual_v2.execution is not None:
                            # Preserve R2's cross-Place continuation without
                            # enabling the separate persistent-frontier policy.
                            retain_v2_hop = len(dual_v2.execution.remaining_route()) > 1
                        if object_approach_enabled and not dual_v2_enabled:
                            from src.object_approach import retain_frontier_hop
                            hop_check_started = time.perf_counter()
                            retain_v2_hop = retain_frontier_hop(
                                place_topology, tsdf_planner, place_route_execution,
                                place_route_commitment, tsdf_planner.habitat2voxel(pts)[:2])
                            topology_trace.write({"event": "v2_hop_validation", "subtask_id": subtask_id,
                                "step": global_step, "retained": retain_v2_hop,
                                "seconds": time.perf_counter() - hop_check_started})
                        # if we choose to query vlm every step, we clear the target point every step
                        if (
                            tsdf_planner.max_point is not None
                            and type(tsdf_planner.max_point) == Frontier
                            and not retain_v2_hop
                            and not (dual_v2_enabled and dual_v2_cfg.get("persistent_intent", False)
                                     and dual_v2.goal.active_intent is not None)
                            and not (phase_c_enabled and phase_c.active is not None)
                            and not (
                                selected_belief_action is not None
                                and selected_belief_action.action_type in (
                                    BeliefActionType.VERIFY,
                                    BeliefActionType.CONFIRM_APPROACH,
                                )
                            )
                            and not (
                                dual_layer_explore_commitment_task_enabled
                                and dual_explore_commitment is not None
                            )
                        ):
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None
                            tsdf_planner.look_at_point = None

                    # use the most common id in the mapped ids as the detected target object id
                    if dual_v2_enabled and tsdf_planner.target_point is None and dual_v2.goal.active_intent is not None:
                        dual_v2.release("original_selection_schedule")
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(
                            max(set(det_ids), key=det_ids.count)
                        )

                    if (
                        tsdf_planner.max_point is None
                        and tsdf_planner.target_point is None
                    ):
                        selected_belief_action = None
                        selected_goal_topo_action = None
                        decision_source = "original_hgr"
                        route_commitment_resumed = False
                        place_route_commitment_resumed = False
                        object_live_choice_selected = False
                        dual_layer_choice_selected = False
                        current_object_choice = None
                        if phase_c_enabled and phase_c.active is not None:
                            committed_choice, source_resolution = phase_c.resume(scene, tsdf_planner)
                            topology_trace.write({
                                "event": "phase_c_source_resolution", "subtask_id": subtask_id,
                                "step": global_step, "resolution": source_resolution,
                                "resolved": committed_choice is not None,
                            })
                            if committed_choice is not None:
                                max_point_choice = committed_choice
                                route_commitment_resumed = True
                                place_route_commitment_resumed = True
                                decision_source = "phase_c_intent_resume"
                            else:
                                phase_c.finish("source_mapping_invalid", global_step)
                                phase_c_selected_crop = None
                                place_route_commitment = None
                                place_route_snapshot_source = None
                                place_route_execution.reset()
                        if (
                            place_route_only_enabled
                            and not phase_c_enabled
                            and place_route_commitment is not None
                        ):
                            if (
                                place_route_commitment["target_kind"]
                                == "snapshot"
                            ):
                                committed_choice, source_resolution = place_route_snapshot_source.resolve(
                                    scene.objects,
                                    scene.object_id_aliases,
                                    scene.snapshots,
                                    scene.frames,
                                )
                                topology_trace.write({
                                    "event": "place_route_source_resolution",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "resolution": source_resolution,
                                    "resolved": committed_choice is not None,
                                })
                            else:
                                source_resolution = {"reason": "frontier_source_disappeared"}
                                committed_choice = next((
                                    frontier for frontier
                                    in tsdf_planner.frontiers
                                    if str(getattr(frontier, "topo_id", ""))
                                    == place_route_commitment["target_id"]
                                ), None)
                            if committed_choice is not None:
                                max_point_choice = committed_choice
                                route_commitment_resumed = True
                                place_route_commitment_resumed = True
                                decision_source = "place_route_commitment"
                                place_route_stats[
                                    "commitment_resumes"
                                ] += 1
                                topology_trace.write({
                                    "event": "place_route_commitment_resumed",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": place_route_commitment,
                                })
                            else:
                                topology_trace.write({
                                    "event": "place_route_commitment_released",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": place_route_commitment,
                                    "reason": "hgr_source_mapping_disappeared",
                                    "source_resolution": source_resolution,
                                })
                                place_route_stats[
                                    "source_mapping_failures"
                                ] += 1
                                place_route_commitment = None
                                place_route_snapshot_source = None
                                place_route_execution.reset()
                        if (
                            goal_topo_object_live_priority
                            and subtask_metadata["task_type"] == "object"
                        ):
                            current_object_choice = _best_current_object_snapshot(
                                scene,
                                current_step_frame_images,
                                subtask_metadata.get("class", ""),
                                min_detections=cfg.min_detection,
                            )
                        if goal_topo_unified_task_view:
                            direct_snapshot_spec = None
                            if current_object_choice is not None:
                                direct_topo_id = persistent_topology.upsert_observed(
                                    snapshot=current_object_choice,
                                    position=current_object_choice.obs_point[:2],
                                    categories=[subtask_metadata.get("class", "")],
                                    step=global_step,
                                    anchor_visited_id=visited_node_id,
                                    evidence_ref=(
                                        f"snapshot:{current_object_choice.image}"
                                    ),
                                    reanchor_existing_evidence=False,
                                )
                                direct_snapshot_spec = (
                                    direct_topo_id,
                                    current_object_choice.image,
                                    current_object_choice.cluster[0],
                                )
                            goal_topo_task_view = (
                                persistent_topology.build_goal_task_view(
                                    goal_context,
                                    tsdf_planner.frontiers,
                                    [
                                        snapshot.image
                                        for snapshot in scene.snapshots.values()
                                    ],
                                    global_step,
                                    direct_snapshot=direct_snapshot_spec,
                                    current_snapshot_images=(
                                        current_step_frame_images
                                    ),
                                )
                            )
                            topology_trace.write({
                                "event": "goal_topology_task_projection",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "actions": goal_topo_task_view.actions,
                            })
                        if current_object_choice is not None:
                            preempted_commitment = (
                                committed_topo_route is not None
                                or target_commitment is not None
                            )
                            if (
                                target_commitment is not None
                                and bool(
                                    target_commitment_cfg.get(
                                        "preempt_on_live_object", True
                                    )
                                )
                            ):
                                topology_trace.write({
                                    "event": "target_commitment_preempted",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": target_commitment.to_trace_dict(),
                                    "reason": "current_exact_object_detected",
                                })
                                target_commitment = None
                            if committed_topo_route is not None:
                                topology_trace.write({
                                    "event": "topology_route_commitment_cancelled",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "waypoints_completed": committed_topo_route[
                                        "waypoints_completed"
                                    ],
                                    "reason": "current_object_detected",
                                })
                                persistent_topology.stats.setdefault(
                                    "topology_route_commitments_cancelled", 0
                                )
                                persistent_topology.stats[
                                    "topology_route_commitments_cancelled"
                                ] += 1
                                persistent_topology.stats.setdefault(
                                    "topology_object_live_preemptions", 0
                                )
                                persistent_topology.stats[
                                    "topology_object_live_preemptions"
                                ] += 1
                                committed_topo_route = None
                            max_point_choice = current_object_choice
                            object_live_choice_selected = True
                            if goal_topo_unified_task_view:
                                selected_goal_topo_action = (
                                    goal_topo_task_view.resolve_snapshot(
                                        current_object_choice.image,
                                        current_object_choice.cluster[0],
                                    )
                                )
                            decision_source = "object_live_detection"
                            n_filtered_snapshots = 0
                            persistent_topology.stats.setdefault(
                                "topology_object_live_selections", 0
                            )
                            persistent_topology.stats[
                                "topology_object_live_selections"
                            ] += 1
                            topology_trace.write({
                                "event": "topology_object_live_target_selected",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "snapshot_image": current_object_choice.image,
                                "object_id": current_object_choice.cluster[0],
                                "preempted_commitment": preempted_commitment,
                            })
                        if (
                            target_commitment_task_enabled
                            and target_commitment is not None
                            and not object_live_choice_selected
                        ):
                            if not topology_intervention_allowed:
                                topology_trace.write({
                                    "event": "target_commitment_cancelled",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": target_commitment.to_trace_dict(),
                                    "reason": "gate_rejected_after_reobservation",
                                })
                                target_commitment = None
                            else:
                                commitment_decision, updated_commitment = (
                                    evaluate_target_commitment(
                                        target_commitment,
                                        metric_action_view,
                                        global_step,
                                        switch_relevance_margin=float(
                                            target_commitment_cfg[
                                                "switch_relevance_margin"
                                            ]
                                        ),
                                        tie_margin=float(
                                            target_commitment_cfg["tie_margin"]
                                        ),
                                        min_switch_saving_m=float(
                                            target_commitment_cfg[
                                                "min_switch_saving_m"
                                            ]
                                        ),
                                        min_switch_saving_ratio=float(
                                            target_commitment_cfg[
                                                "min_switch_saving_ratio"
                                            ]
                                        ),
                                        min_progress_m=float(
                                            target_commitment_cfg["min_progress_m"]
                                        ),
                                        max_no_progress_segments=int(
                                            target_commitment_cfg[
                                                "max_no_progress_segments"
                                            ]
                                        ),
                                    )
                                )
                                event = {
                                    "keep": "target_commitment_kept",
                                    "switch": "target_commitment_switched",
                                    "cancel": "target_commitment_cancelled",
                                }[commitment_decision.action]
                                topology_trace.write({
                                    "event": event,
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": target_commitment.to_trace_dict(),
                                    "decision": commitment_decision.to_trace_dict(),
                                })
                                if commitment_decision.action == "keep":
                                    # Only resolve an executable snapshot at the
                                    # point of use.  The commitment itself owns
                                    # only stable identity and capture anchor.
                                    committed_choice = _snapshot_for_image(
                                        scene, target_commitment.snapshot_image
                                    )
                                    if committed_choice is None:
                                        topology_trace.write({
                                            "event": "target_commitment_cancelled",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "commitment": target_commitment.to_trace_dict(),
                                            "reason": "source_identity_unresolvable",
                                        })
                                        target_commitment = None
                                    else:
                                        target_commitment = updated_commitment
                                        max_point_choice = committed_choice
                                        selected_goal_topo_action = (
                                            goal_topo_task_view.resolve_snapshot(
                                                committed_choice.image,
                                                committed_choice.cluster[0]
                                                if committed_choice.cluster else None,
                                            )
                                            if goal_topo_unified_task_view else None
                                        )
                                        route_commitment_resumed = True
                                        decision_source = "target_commitment"
                                        n_filtered_snapshots = 0
                                else:
                                    # Switch directly to the metric challenger;
                                    # it was computed from this exact fresh
                                    # observation and therefore needs no extra
                                    # VLM request.  REVISIT challengers will
                                    # enter the same direct-first executor
                                    # below, while DIRECT/EXPLORE use HGR's
                                    # ordinary target setter.
                                    challenger = max(
                                        (
                                            action for action in metric_action_view.actions
                                            if action.reachable
                                            and action.effective_cost_m is not None
                                            and not (
                                                action.action_type == "revisit"
                                                and action.topo_node_id
                                                == target_commitment.topo_node_id
                                                and action.snapshot_image
                                                == target_commitment.snapshot_image
                                            )
                                        ),
                                        key=lambda action: action.relevance,
                                        default=None,
                                    )
                                    target_commitment = None
                                    if challenger is not None:
                                        if (
                                            challenger.source_kind == "frontier"
                                            and challenger.source_frontier_index is not None
                                            and 0 <= challenger.source_frontier_index
                                            < len(tsdf_planner.frontiers)
                                        ):
                                            max_point_choice = tsdf_planner.frontiers[
                                                challenger.source_frontier_index
                                            ]
                                            selected_goal_topo_action = (
                                                goal_topo_task_view.resolve_frontier(
                                                    challenger.source_frontier_index
                                                )
                                                if goal_topo_unified_task_view else None
                                            )
                                            route_commitment_resumed = True
                                            decision_source = "target_commitment_switch"
                                        elif challenger.snapshot_image:
                                            challenger_choice = _snapshot_for_image(
                                                scene, challenger.snapshot_image
                                            )
                                            if challenger_choice is not None:
                                                max_point_choice = challenger_choice
                                                selected_goal_topo_action = (
                                                    goal_topo_task_view.resolve_snapshot(
                                                        challenger_choice.image,
                                                        challenger.object_id,
                                                    )
                                                    if goal_topo_unified_task_view else None
                                                )
                                                route_commitment_resumed = True
                                                decision_source = "target_commitment_switch"
                        if (
                            goal_topo_route_enabled
                            and topology_intervention_allowed
                            and committed_topo_route is not None
                            and not object_live_choice_selected
                            and not target_commitment_task_enabled
                            and not lightweight_confidence_task_enabled
                        ):
                            committed_choice = committed_topo_route[
                                "snapshot_choice"
                            ]
                            if _snapshot_choice_mapping_available(
                                committed_choice, scene
                            ):
                                max_point_choice = committed_choice
                                if goal_topo_unified_task_view:
                                    selected_goal_topo_action = (
                                        goal_topo_task_view.resolve_snapshot(
                                            committed_choice.image,
                                            committed_choice.cluster[0]
                                            if committed_choice.cluster else None,
                                        )
                                    )
                                route_commitment_resumed = True
                                decision_source = "topology_route_commitment"
                                n_filtered_snapshots = 0
                                persistent_topology.stats.setdefault(
                                    "topology_route_commitment_resumes", 0
                                )
                                persistent_topology.stats[
                                    "topology_route_commitment_resumes"
                                ] += 1
                                topology_trace.write({
                                    "event": "topology_route_commitment_resumed",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "waypoints_completed": committed_topo_route[
                                        "waypoints_completed"
                                    ],
                                })
                            else:
                                topology_trace.write({
                                    "event": "topology_route_commitment_cancelled",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "reason": "snapshot_mapping_disappeared",
                                })
                                persistent_topology.stats.setdefault(
                                    "topology_route_commitments_cancelled", 0
                                )
                                persistent_topology.stats[
                                    "topology_route_commitments_cancelled"
                                ] += 1
                                committed_topo_route = None
                        if (
                            not route_commitment_resumed
                            and not object_live_choice_selected
                            and belief_planner_enabled
                            and belief_action_candidates
                        ):
                            if confirmation_priority_action is not None:
                                direct_action = confirmation_priority_action
                                tie_actions = []
                                decision_source = "confirmation_priority"
                            elif (
                                belief_v3_enabled
                                and not bool(
                                    active_topology_cfg.get("belief", {}).get(
                                        "use_vlm_tiebreak", False
                                    )
                                )
                            ):
                                direct_action = belief_memory.choose_deterministic_action(
                                    belief_action_candidates
                                )
                                tie_actions = []
                                decision_source = "category_deterministic"
                            else:
                                direct_action, tie_actions = (
                                    belief_memory.choose_actions_for_tiebreak(
                                        belief_action_candidates,
                                        decision_margin=float(
                                            active_topology_cfg.get("belief", {}).get(
                                                "decision_margin", 0.15
                                            )
                                        ),
                                    )
                                )
                            if direct_action is not None:
                                selected_belief_action = direct_action
                                if decision_source not in (
                                    "confirmation_priority", "category_deterministic",
                                ):
                                    decision_source = "planner_direct"
                            else:
                                tie_images = []
                                for action in tie_actions:
                                    if action.source_kind == "frontier":
                                        source = tsdf_planner.frontiers[int(action.source_index)]
                                        image = source.feature
                                    else:
                                        source_info = belief_snapshot_sources.get(action.node_id)
                                        image = (
                                            scene.all_observations.get(source_info[0].image)
                                            if source_info is not None else None
                                        )
                                    if image is None:
                                        image = np.zeros((cfg.prompt_h, cfg.prompt_w, 3), dtype=np.uint8)
                                    tie_images.append(np.asarray(image)[..., :3])
                                extra_body_cfg = cfg.get("vlm_extra_body", None)
                                extra_body = (
                                    OmegaConf.to_container(extra_body_cfg, resolve=True)
                                    if extra_body_cfg is not None else None
                                )
                                tie_index = query_action_tiebreak(
                                    tie_actions,
                                    tie_images,
                                    goal_context.category,
                                    model_name=cfg.get("vlm_model", "openai/gpt-4o"),
                                    extra_body=extra_body,
                                )
                                if tie_index is None:
                                    selected_belief_action = tie_actions[0]
                                    decision_source = "vlm_tiebreak_fallback_top1"
                                else:
                                    selected_belief_action = tie_actions[tie_index]
                                    belief_memory.stats["vlm_tiebreak_choices"] += 1
                                    decision_source = "vlm_tiebreak"

                            if selected_belief_action.source_kind == "frontier":
                                max_point_choice = tsdf_planner.frontiers[
                                    int(selected_belief_action.source_index)
                                ]
                            else:
                                source_info = belief_snapshot_sources.get(
                                    selected_belief_action.node_id
                                )
                                if source_info is None:
                                    logging.warning(
                                        "Belief action %s lost its executable snapshot mapping; "
                                        "falling back to original HGR",
                                        selected_belief_action.node_id,
                                    )
                                    selected_belief_action = None
                                else:
                                    if source_info[1] is None:
                                        max_point_choice = source_info[0]
                                    else:
                                        max_point_choice = _single_object_snapshot(
                                            *source_info
                                        )
                            if selected_belief_action is not None:
                                if (
                                    belief_v2_enabled
                                    and selected_belief_action.confirmation_source_node_id
                                    is not None
                                    and selected_belief_action.action_type
                                    == BeliefActionType.NAVIGATE
                                ):
                                    resolved = belief_memory.resolve_confirmation(
                                        normalize_goal_key(goal_context.category),
                                        "snapshot_selected",
                                    )
                                belief_memory.record_action_selection(
                                    selected_belief_action
                                )
                            n_filtered_snapshots = 0
                            topology_trace.write({
                                "event": "belief_decision",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "decision_source": decision_source,
                                "selected_action": selected_belief_action,
                                "tie_actions": tie_actions,
                            })

                        if (
                            selected_belief_action is None
                            and not route_commitment_resumed
                            and not object_live_choice_selected
                            and dual_layer_region_shortlist_task_enabled
                            and dual_layer_decision is not None
                            and dual_layer_decision.action is not None
                            and dual_layer_decision.action.action_type
                            == GoalTopoActionType.EXPLORE.value
                        ):
                            region_decision = build_region_topology_decision(
                                metric_action_view, dynamic_goal_view,
                                persistent_topology,
                                max_regions=int(dual_layer_region_shortlist_cfg[
                                    "max_regions"
                                ]),
                                require_frontier=(
                                    dual_layer_region_frontier_only_task_enabled
                                ),
                            )
                            topology_trace.write({
                                "event": "dual_layer_region_decision",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "decision": region_decision.to_trace_dict(),
                            })
                            if (
                                region_decision.snapshot_images
                                or region_decision.frontier_indices
                            ):
                                region_response = query_vlm_for_response(
                                    subtask_metadata=subtask_metadata,
                                    scene=scene,
                                    tsdf_planner=tsdf_planner,
                                    rgb_egocentric_views=rgb_egocentric_views,
                                    cfg=cfg,
                                    active_topo_view=None,
                                    allowed_snapshot_images=(
                                        region_decision.snapshot_images
                                    ),
                                    allowed_frontier_indices=(
                                        region_decision.frontier_indices
                                    ),
                                    frontier_only=(
                                        dual_layer_region_frontier_only_task_enabled
                                    ),
                                    verbose=True,
                                )
                                if region_response is not None:
                                    max_point_choice, n_filtered_snapshots = (
                                        region_response
                                    )
                                    selected_region_action = None
                                    evidence_snapshot = None
                                    if type(max_point_choice) == Frontier:
                                        selected_source_index = next(
                                            (
                                                index for index, frontier
                                                in enumerate(tsdf_planner.frontiers)
                                                if frontier is max_point_choice
                                            ), None,
                                        )
                                        selected_region_action = next(
                                            (
                                                action for action
                                                in region_decision.actions
                                                if action.source_frontier_index
                                                == selected_source_index
                                            ), None,
                                        )
                                    elif (
                                        type(max_point_choice) == SnapShot
                                        and dual_layer_region_snapshot_evidence_task_enabled
                                    ):
                                        evidence_snapshot = max_point_choice.image
                                        selected_region_action = (
                                            select_local_explore_for_snapshot(
                                                region_decision,
                                                persistent_topology,
                                                evidence_snapshot,
                                            )
                                        )
                                        if selected_region_action is not None:
                                            frontier_index = (
                                                selected_region_action
                                                .source_frontier_index
                                            )
                                            if (
                                                frontier_index is None
                                                or frontier_index < 0
                                                or frontier_index >= len(
                                                    tsdf_planner.frontiers
                                                )
                                            ):
                                                selected_region_action = None
                                            else:
                                                max_point_choice = (
                                                    tsdf_planner.frontiers[
                                                        frontier_index
                                                    ]
                                                )
                                    if selected_region_action is not None:
                                        dual_layer_choice_selected = True
                                        decision_source = (
                                            "dual_layer_region_snapshot_evidence"
                                            if evidence_snapshot is not None
                                            else "dual_layer_region_topology"
                                        )
                                        topology_trace.write({
                                            "event": "dual_layer_region_local_selected",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "region_ids": region_decision.region_ids,
                                            "selected_topo_id": (
                                                selected_region_action.topo_node_id
                                            ),
                                            "evidence_snapshot": evidence_snapshot,
                                        })
                                        if dual_layer_explore_commitment_task_enabled:
                                            dual_explore_commitment = (
                                                begin_topo_explore_commitment(
                                                    selected_region_action,
                                                    current_voxel,
                                                    max_point_choice.position,
                                                    global_step,
                                                )
                                            )
                                            topology_trace.write({
                                                "event": "dual_layer_explore_commitment_created",
                                                "scene_id": scene_id,
                                                "episode_id": episode_id,
                                                "subtask_id": subtask_id,
                                                "step": global_step,
                                                "commitment": (
                                                    dual_explore_commitment.to_trace_dict()
                                                ),
                                                "source": (
                                                    "vlm_region_snapshot_evidence"
                                                    if evidence_snapshot is not None
                                                    else "vlm_region_local"
                                                ),
                                            })
                            if not dual_layer_choice_selected:
                                topology_trace.write({
                                    "event": "dual_layer_prompt_fallback",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "reason": "region_query_no_local_frontier",
                                })
                                dual_layer_decision = None

                        if (
                            selected_belief_action is None
                            and not route_commitment_resumed
                            and not object_live_choice_selected
                            and dual_layer_planner_task_enabled
                            and not dual_layer_choice_selected
                            and dual_layer_decision is not None
                            and dual_layer_decision.action is not None
                        ):
                            dual_action = dual_layer_decision.action
                            if (
                                dual_action.action_type
                                == GoalTopoActionType.EXPLORE.value
                                and dual_action.source_frontier_index is not None
                                and 0 <= dual_action.source_frontier_index
                                < len(tsdf_planner.frontiers)
                            ):
                                if dual_layer_explore_shortlist_task_enabled:
                                    shortlist_actions = [dual_action]
                                    if (
                                        dual_layer_decision.fallback_explore
                                        is not None
                                        and dual_layer_decision.fallback_explore
                                        .source_frontier_index is not None
                                    ):
                                        shortlist_actions.append(
                                            dual_layer_decision.fallback_explore
                                        )
                                    shortlist_indices = list(dict.fromkeys(
                                        int(action.source_frontier_index)
                                        for action in shortlist_actions
                                    ))
                                    shortlist_response = query_vlm_for_response(
                                        subtask_metadata=subtask_metadata,
                                        scene=scene,
                                        tsdf_planner=tsdf_planner,
                                        rgb_egocentric_views=(
                                            rgb_egocentric_views
                                        ),
                                        cfg=cfg,
                                        active_topo_view=None,
                                        allowed_snapshot_images=[],
                                        allowed_frontier_indices=shortlist_indices,
                                        verbose=True,
                                    )
                                    if (
                                        shortlist_response is not None
                                        and type(shortlist_response[0]) == Frontier
                                    ):
                                        (
                                            max_point_choice,
                                            n_filtered_snapshots,
                                        ) = shortlist_response
                                        selected_source_index = next(
                                            (
                                                index for index, frontier
                                                in enumerate(tsdf_planner.frontiers)
                                                if frontier is max_point_choice
                                            ),
                                            None,
                                        )
                                        selected_shortlist_action = next(
                                            (
                                                action for action in shortlist_actions
                                                if action.source_frontier_index
                                                == selected_source_index
                                            ),
                                            None,
                                        )
                                        if selected_shortlist_action is not None:
                                            dual_layer_choice_selected = True
                                            decision_source = (
                                                "dual_layer_explore_shortlist"
                                            )
                                            topology_trace.write({
                                                "event": (
                                                    "dual_layer_explore_shortlist_selected"
                                                ),
                                                "scene_id": scene_id,
                                                "episode_id": episode_id,
                                                "subtask_id": subtask_id,
                                                "step": global_step,
                                                "shortlist_topo_ids": [
                                                    action.topo_node_id
                                                    for action in shortlist_actions
                                                ],
                                                "selected_topo_id": (
                                                    selected_shortlist_action.topo_node_id
                                                ),
                                            })
                                            if (
                                                dual_layer_explore_commitment_task_enabled
                                            ):
                                                dual_explore_commitment = (
                                                    begin_topo_explore_commitment(
                                                        selected_shortlist_action,
                                                        current_voxel,
                                                        max_point_choice.position,
                                                        global_step,
                                                    )
                                                )
                                                topology_trace.write({
                                                    "event": "dual_layer_explore_commitment_created",
                                                    "scene_id": scene_id,
                                                    "episode_id": episode_id,
                                                    "subtask_id": subtask_id,
                                                    "step": global_step,
                                                    "commitment": (
                                                        dual_explore_commitment.to_trace_dict()
                                                    ),
                                                    "source": "vlm_shortlist",
                                                })
                                    else:
                                        topology_trace.write({
                                            "event": "dual_layer_prompt_fallback",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "reason": "explore_shortlist_query_failed",
                                            "selected_topo_id": (
                                                dual_action.topo_node_id
                                            ),
                                        })
                                else:
                                    max_point_choice = tsdf_planner.frontiers[
                                        dual_action.source_frontier_index
                                    ]
                                    n_filtered_snapshots = 0
                                    dual_layer_choice_selected = True
                                    decision_source = "dual_layer_explore"
                                    if dual_layer_explore_commitment_task_enabled:
                                        dual_explore_commitment = (
                                            begin_topo_explore_commitment(
                                                dual_action,
                                                current_voxel,
                                                max_point_choice.position,
                                                global_step,
                                            )
                                        )
                                        topology_trace.write({
                                            "event": "dual_layer_explore_commitment_created",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "commitment": (
                                                dual_explore_commitment.to_trace_dict()
                                            ),
                                        })
                            elif dual_action.snapshot_image:
                                fallback_indices = []
                                if (
                                    dual_layer_decision.fallback_explore
                                    is not None
                                    and dual_layer_decision.fallback_explore
                                    .source_frontier_index is not None
                                ):
                                    fallback_indices.append(
                                        dual_layer_decision.fallback_explore
                                        .source_frontier_index
                                    )
                                restricted_response = query_vlm_for_response(
                                    subtask_metadata=subtask_metadata,
                                    scene=scene,
                                    tsdf_planner=tsdf_planner,
                                    rgb_egocentric_views=(
                                        rgb_egocentric_views
                                    ),
                                    cfg=cfg,
                                    active_topo_view=None,
                                    allowed_snapshot_images=[
                                        dual_action.snapshot_image
                                    ],
                                    allowed_frontier_indices=(
                                        fallback_indices
                                    ),
                                    verbose=True,
                                )
                                if restricted_response is not None:
                                    (
                                        max_point_choice,
                                        n_filtered_snapshots,
                                    ) = restricted_response
                                    dual_layer_choice_selected = True
                                    decision_source = (
                                        "dual_layer_snapshot_shortlist"
                                        if isinstance(
                                            max_point_choice, SnapShot
                                        )
                                        else "dual_layer_frontier_fallback"
                                    )
                                else:
                                    topology_trace.write({
                                        "event": (
                                            "dual_layer_prompt_fallback"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "reason": (
                                            "restricted_query_failed"
                                        ),
                                        "selected_topo_id": (
                                            dual_action.topo_node_id
                                        ),
                                    })

                        if (
                            selected_belief_action is None
                            and not route_commitment_resumed
                            and not object_live_choice_selected
                            and not dual_layer_choice_selected
                        ):
                            if phase_c_enabled:
                                phase_c.build_candidates(
                                    place_topology, scene, tsdf_planner,
                                    tsdf_planner.habitat2voxel(pts)[:2],
                                )
                                selection, selection_diagnostic = select_place_goal(
                                    phase_c, subtask_metadata, scene, tsdf_planner,
                                    rgb_egocentric_views, cfg,
                                )
                                topology_trace.write({
                                    "event": "phase_c_decision", "subtask_id": subtask_id,
                                    "step": global_step, "view": phase_c.trace(),
                                    "selection": selection_diagnostic,
                                })
                                if selection is None:
                                    phase_c.selection_reason = selection_diagnostic["reason"]
                                    if selection_diagnostic.get("action") == "continue":
                                        continue
                                    selection_action = selection_diagnostic.get("action")
                                    subtask_execution.terminate(
                                        ExecutionStatus.ERROR
                                        if selection_action == "error"
                                        else ExecutionStatus.EXHAUSTED,
                                        "phase_c_selection_failed"
                                        if selection_action == "error"
                                        else "no_executable_candidate",
                                        {"selection": selection_diagnostic},
                                        "selection" if selection_action == "error" else None,
                                    )
                                    break
                                candidate, choice, phase_c_selected_crop, support = selection
                                phase_c.select(candidate, choice, support, global_step)
                                vlm_response = (choice, len({c.source_image for c in phase_c.candidates if c.intent == "Verify"}))
                                decision_source = "phase_c_goal_layer"
                            else:
                                # Fusion augments the original selector with the
                                # same Place routes consumed by execution below.
                                if unified_enabled:
                                    vlm_response = baseline_dual_dynamic.select(
                                        query_vlm_for_response, subtask_metadata, scene, tsdf_planner,
                                        place_topology, rgb_egocentric_views, cfg, pts,
                                        fresh_detected_obj_ids, remaining_steps=max(0, num_step-cnt_step),
                                    )
                                    topology_trace.write({"event": "hypothesis_aware_selection",
                                        "subtask_id": subtask_id, "step": global_step,
                                        "navigator": baseline_dual_dynamic.to_trace_dict()})
                                else:
                                    vlm_response = query_vlm_for_response(
                                        subtask_metadata=subtask_metadata,
                                        scene=scene,
                                        tsdf_planner=tsdf_planner,
                                        rgb_egocentric_views=rgb_egocentric_views,
                                        cfg=cfg,
                                        active_topo_view=active_topo_view,
                                        place_topology=place_topology if hgr_topology_fusion_enabled else None,
                                        navigation_position=tsdf_planner.habitat2voxel(pts)[:2],
                                        excluded_object_ids=(
                                            belief_memory.rejected_object_ids(
                                                normalize_goal_key(goal_context.category),
                                                global_step,
                                            )
                                            if belief_v3_enabled else None
                                        ),
                                        hierarchical_navigator=hierarchical_navigator,
                                        verbose=True,
                                    )
                            if vlm_response is None:
                                if hierarchical_enabled:
                                    topology_trace.write({
                                        "event": "hierarchical_selection_failed",
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "decision": (
                                            None if hierarchical_navigator.pending_decision is None
                                            else hierarchical_navigator.pending_decision.to_trace_dict()
                                        ),
                                    })
                                selection_result = (
                                    baseline_dual_dynamic.last_selection
                                    if unified_enabled else None
                                )
                                selection_reason = getattr(
                                    selection_result, "reason", "unclassified_selection_failure"
                                )
                                if isinstance(selection_result, NoAction):
                                    subtask_execution.terminate(
                                        ExecutionStatus.EXHAUSTED,
                                        "no_executable_candidate",
                                        {"selection_reason": selection_reason},
                                    )
                                else:
                                    subtask_execution.terminate(
                                        ExecutionStatus.ERROR,
                                        "semantic_selection_failed",
                                        {"selection_reason": selection_reason},
                                        "selection",
                                    )
                                logging.info(
                                    "Subtask %s selection ended: %s",
                                    subtask_id,
                                    selection_reason,
                                )
                                break
                            max_point_choice, n_filtered_snapshots = vlm_response
                            if hierarchical_enabled:
                                topology_trace.write({
                                    "event": "hierarchical_semantic_decision",
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "navigation_backend": navigation_backend,
                                    "shadow_only": hierarchical_shadow_only,
                                    "decision": (
                                        None
                                        if hierarchical_navigator.pending_decision is None
                                        else hierarchical_navigator.pending_decision.to_trace_dict()
                                    ),
                                    "semantic_digest": hierarchical_navigator.pending_semantic_digest,
                                    "semantic_protocol": dict(
                                        hierarchical_navigator.last_semantic_protocol
                                    ),
                                })

                            if (
                                belief_v3_enabled
                                and belief_planner_enabled
                                and type(max_point_choice) == SnapShot
                            ):
                                object_id = (
                                    max_point_choice.cluster[0]
                                    if max_point_choice.cluster else None
                                )
                                mapped_topo_id = next((
                                    topo_id for topo_id, source_info
                                    in belief_snapshot_sources.items()
                                    if source_info[1] == object_id
                                ), None)
                                mapped_node = persistent_topology.nodes.get(
                                    mapped_topo_id
                                )
                                fallback_posterior = belief_memory.prior
                                if mapped_node is not None:
                                    fallback_posterior = belief_memory.belief(
                                        mapped_topo_id,
                                        normalize_goal_key(goal_context.category),
                                        mapped_node.map_confidence,
                                        mapped_node.accessibility,
                                    ).posterior
                                gate_decision = _evaluate_v3_snapshot(
                                    scene,
                                    goal_context.category,
                                    object_id,
                                    fallback_posterior,
                                    active_topology_cfg.get("snapshot_gate", {}),
                                )
                                belief_memory.record_snapshot_gate(gate_decision)
                                topology_trace.write({
                                    "event": "snapshot_gate",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "source": "original_hgr_fallback",
                                    "object_id": object_id,
                                    "topo_id": mapped_topo_id,
                                    "decision": gate_decision,
                                })
                                if not gate_decision.accepted:
                                    belief_memory.reject_snapshot(
                                        normalize_goal_key(goal_context.category),
                                        object_id,
                                        global_step,
                                        gate_decision,
                                        duration_steps=int(
                                            active_topology_cfg.get(
                                                "snapshot_gate", {}
                                            ).get("rejection_ttl_steps", 5)
                                        ),
                                    )
                                    fallback_frontier = _select_v3_frontier_fallback(
                                        executable_nodes,
                                        tsdf_planner.frontiers,
                                        belief_action_candidates,
                                    )
                                    if fallback_frontier is not None:
                                        max_point_choice = fallback_frontier
                                        decision_source = "snapshot_gate_frontier_fallback"
                                    else:
                                        logging.warning(
                                            "BeliefTopo v3 rejected snapshot but has no frontier; "
                                            "continuing without terminal selection"
                                        )
                                        tsdf_planner.max_point = None
                                        tsdf_planner.target_point = None
                                        tsdf_planner.look_at_point = None
                                        continue

                        if (
                            goal_topo_unified_task_view
                            and selected_goal_topo_action is None
                        ):
                            if type(max_point_choice) == SnapShot:
                                selected_goal_topo_action = (
                                    goal_topo_task_view.resolve_snapshot(
                                        max_point_choice.image,
                                        max_point_choice.cluster[0]
                                        if max_point_choice.cluster else None,
                                    )
                                )
                            elif type(max_point_choice) == Frontier:
                                source_frontier_index = next((
                                    index for index, frontier
                                    in enumerate(tsdf_planner.frontiers)
                                    if frontier is max_point_choice
                                ), None)
                                if source_frontier_index is not None:
                                    selected_goal_topo_action = (
                                        goal_topo_task_view.resolve_frontier(
                                            source_frontier_index
                                        )
                                    )
                            if selected_goal_topo_action is None:
                                topology_trace.write({
                                    "event": "goal_topology_action_unresolved",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "choice_type": type(max_point_choice).__name__,
                                })
                                logging.warning(
                                    "Unified GoalTopo could not resolve the HGR "
                                    "choice; rebuilding on the next step"
                                )
                                tsdf_planner.max_point = None
                                tsdf_planner.target_point = None
                                tsdf_planner.look_at_point = None
                                continue

                        if (
                            goal_topo_unified_task_view
                            and selected_goal_topo_action is not None
                        ):
                            persistent_topology.stats.setdefault(
                                "goal_task_actions_selected", 0
                            )
                            persistent_topology.stats[
                                "goal_task_actions_selected"
                            ] += 1
                            topology_trace.write({
                                "event": "goal_topology_action_selected",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "decision_source": decision_source,
                                "action": selected_goal_topo_action,
                            })

                        # The pre-selection gate evaluates one concrete best
                        # historical node.  Do not let a different Snapshot
                        # chosen later by the VLM inherit that node's approval.
                        # The VLM choice itself is preserved; only the V7
                        # route override falls back to original HGR.
                        bind_selected_revisit = bool(
                            revisit_gate_task_enabled
                            and revisit_gate_cfg.get(
                                "bind_selected_action", False
                            )
                        )
                        if (
                            bind_selected_revisit
                            and selected_goal_topo_action is not None
                            and selected_goal_topo_action.action_type
                            == GoalTopoActionType.REVISIT
                        ):
                            selected_revisit_authorized = (
                                revisit_gate_authorizes_selected_action(
                                    revisit_gate_decision,
                                    selected_goal_topo_action,
                                )
                            )
                            topology_trace.write({
                                "event": "selected_revisit_authorization",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "selected_topo_id": (
                                    selected_goal_topo_action.topo_node_id
                                ),
                                "approved_topo_id": (
                                    revisit_gate_decision.best_revisit_topo_id
                                ),
                                "allowed": selected_revisit_authorized,
                                "reason": (
                                    "approved_revisit_selected"
                                    if selected_revisit_authorized
                                    else "selected_revisit_not_approved"
                                ),
                            })
                            if not selected_revisit_authorized:
                                topology_intervention_allowed = False
                                metric_direct_first_step_enabled = False
                                if committed_topo_route is not None:
                                    topology_trace.write({
                                        "event": (
                                            "topology_route_commitment_cancelled"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "snapshot_image": committed_topo_route[
                                            "snapshot_image"
                                        ],
                                        "target_node_id": committed_topo_route[
                                            "target_node_id"
                                        ],
                                        "waypoints_completed": (
                                            committed_topo_route[
                                                "waypoints_completed"
                                            ]
                                        ),
                                        "reason": (
                                            "selected_revisit_not_approved"
                                        ),
                                    })
                                    persistent_topology.stats.setdefault(
                                        "topology_route_commitments_cancelled",
                                        0,
                                    )
                                    persistent_topology.stats[
                                        "topology_route_commitments_cancelled"
                                    ] += 1
                                    committed_topo_route = None

                        selected_topo_id = None
                        selected_path_cost = None
                        selection_made_progress = False
                        direct_first_selection_mode = None
                        if (
                            metric_direct_first_step_enabled
                            and type(max_point_choice) == SnapShot
                        ):
                            direct_first_selection_mode = (
                                _direct_first_selection_mode(
                                    max_point_choice.image,
                                    current_step_frame_images,
                                    resolved_action_type=(
                                        selected_goal_topo_action.action_type
                                        if selected_goal_topo_action is not None
                                        else None
                                    ),
                                    anchor_reobserve_snapshot=(
                                        direct_first_anchor_reobserve_snapshot
                                    ),
                                )
                            )
                        direct_first_anchor_reobserve_fallback = bool(
                            direct_first_selection_mode
                            == "post_anchor_original_hgr"
                        )
                        if (
                            direct_first_anchor_reobserve_snapshot is not None
                            and not direct_first_anchor_reobserve_fallback
                        ):
                            direct_first_anchor_reobserve_snapshot = None
                        direct_first_revisit_selected = bool(
                            direct_first_selection_mode == "revisit"
                        )
                        if (
                            direct_first_revisit_selected
                            and active_topo_route is None
                        ):
                            observed_topo_id = (
                                committed_topo_route["target_node_id"]
                                if committed_topo_route is not None
                                else persistent_topology.observed_node_for_snapshot(
                                    max_point_choice.image
                                )
                            )
                            direct_first_plan = plan_direct_first_revisit(
                                max_point_choice.image,
                                persistent_topology,
                                tsdf_planner,
                                scene.pathfinder,
                                current_voxel,
                                floor_height,
                                observed_topo_id=observed_topo_id,
                                arrival_tolerance_m=float(
                                    direct_first_cfg.get(
                                        "anchor_arrival_tolerance_m", 0.15
                                    )
                                ),
                            )
                            topology_trace.write({
                                "event": "direct_first_route_planned",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "goal_type": goal_context.goal_type,
                                "behavior_applied": bool(
                                    direct_first_plan.uses_override
                                ),
                                "plan": direct_first_plan.to_trace_dict(),
                            })
                            if (
                                direct_first_plan.route_mode
                                == DirectFirstRouteMode.ANCHOR_REACHED.value
                            ):
                                if target_commitment is not None:
                                    topology_trace.write({
                                        "event": "target_commitment_completed",
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "commitment": target_commitment.to_trace_dict(),
                                        "reason": "capture_anchor_arrived",
                                        "success_inferred": False,
                                    })
                                    target_commitment = None
                                if committed_topo_route is not None:
                                    topology_trace.write({
                                        "event": (
                                            "topology_route_commitment_completed"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "snapshot_image": committed_topo_route[
                                            "snapshot_image"
                                        ],
                                        "target_node_id": committed_topo_route[
                                            "target_node_id"
                                        ],
                                        "waypoints_completed": (
                                            committed_topo_route[
                                                "waypoints_completed"
                                            ]
                                        ),
                                        "reason": "capture_anchor_reached",
                                    })
                                    persistent_topology.stats.setdefault(
                                        "topology_route_commitments_completed", 0
                                    )
                                    persistent_topology.stats[
                                        "topology_route_commitments_completed"
                                    ] += 1
                                    committed_topo_route = None
                                topology_trace.write({
                                    "event": "direct_first_anchor_reached",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "goal_type": goal_context.goal_type,
                                    "snapshot_image": max_point_choice.image,
                                    "capture_anchor_id": (
                                        direct_first_plan.capture_anchor_id
                                    ),
                                    "success_inferred": False,
                                    "reason": "reobserve_before_targeting",
                                })
                                tsdf_planner.max_point = None
                                tsdf_planner.target_point = None
                                tsdf_planner.look_at_point = None
                                direct_first_anchor_reobserve_snapshot = (
                                    max_point_choice.image
                                )
                                max_point_choice = None
                                selected_goal_topo_action = None
                                continue
                            if direct_first_plan.uses_override:
                                if (
                                    target_commitment_task_enabled
                                    and target_commitment is None
                                ):
                                    commitment_action = next(
                                        (
                                            action for action in metric_action_view.actions
                                            if action.action_type == "revisit"
                                            and action.topo_node_id == observed_topo_id
                                            and action.snapshot_image == max_point_choice.image
                                        ),
                                        None,
                                    )
                                    if commitment_action is not None:
                                        target_commitment = create_target_commitment(
                                            commitment_action,
                                            goal_context.subtask_id,
                                            global_step,
                                        )
                                        topology_trace.write({
                                            "event": "target_commitment_created",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "commitment": target_commitment.to_trace_dict(),
                                            "reason": "gate_accepted_revisit_selected",
                                        })
                                elif (
                                    committed_topo_route is None
                                    and goal_topo_route_enabled
                                    and topology_intervention_allowed
                                    and not lightweight_confidence_task_enabled
                                ):
                                    committed_topo_route = {
                                        "target_node_id": observed_topo_id,
                                        "snapshot_image": max_point_choice.image,
                                        "snapshot_choice": max_point_choice,
                                        "created_step": global_step,
                                        "waypoints_completed": 0,
                                    }
                                    persistent_topology.stats.setdefault(
                                        "topology_route_commitments_created", 0
                                    )
                                    persistent_topology.stats[
                                        "topology_route_commitments_created"
                                    ] += 1
                                    topology_trace.write({
                                        "event": (
                                            "topology_route_commitment_created"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "snapshot_image": max_point_choice.image,
                                        "target_node_id": observed_topo_id,
                                        "executor": "direct_first",
                                    })
                                navigation_target_override = (
                                    direct_first_plan.target_voxel.copy()
                                )
                                navigation_look_at_override = (
                                    None
                                    if direct_first_plan.look_at_voxel is None
                                    else direct_first_plan.look_at_voxel.copy()
                                )
                                selected_topo_id = observed_topo_id
                                active_topo_route = {
                                    "target_node_id": observed_topo_id,
                                    "snapshot_image": max_point_choice.image,
                                    "route": direct_first_plan.route,
                                    "waypoint_id": (
                                        direct_first_plan.next_waypoint_id
                                    ),
                                    "capture_anchor_id": (
                                        direct_first_plan.capture_anchor_id
                                    ),
                                    "route_mode": direct_first_plan.route_mode,
                                    "executor": "direct_first",
                                }
                        if (
                            goal_topo_route_enabled
                            and topology_intervention_allowed
                            and type(max_point_choice) == SnapShot
                            and active_topo_route is None
                            and not direct_first_revisit_selected
                            and not direct_first_anchor_reobserve_fallback
                            and not lightweight_confidence_task_enabled
                            and (
                                selected_goal_topo_action is not None
                                and selected_goal_topo_action.action_type
                                == GoalTopoActionType.REVISIT
                                if goal_topo_unified_task_view
                                else not object_live_choice_selected
                            )
                        ):
                            observed_topo_id = (
                                committed_topo_route["target_node_id"]
                                if committed_topo_route is not None
                                else persistent_topology.observed_node_for_snapshot(
                                    max_point_choice.image
                                )
                            )
                            if observed_topo_id is not None:
                                route, waypoint_id = (
                                    persistent_topology.plan_safe_route_to_observed(
                                        observed_topo_id
                                    )
                                )
                                if waypoint_id is not None:
                                    if committed_topo_route is None:
                                        committed_topo_route = {
                                            "target_node_id": observed_topo_id,
                                            "snapshot_image": max_point_choice.image,
                                            "snapshot_choice": max_point_choice,
                                            "created_step": global_step,
                                            "waypoints_completed": 0,
                                        }
                                        persistent_topology.stats.setdefault(
                                            "topology_route_commitments_created", 0
                                        )
                                        persistent_topology.stats[
                                            "topology_route_commitments_created"
                                        ] += 1
                                        topology_trace.write({
                                            "event": "topology_route_commitment_created",
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "snapshot_image": max_point_choice.image,
                                            "target_node_id": observed_topo_id,
                                        })
                                    waypoint = persistent_topology.nodes[waypoint_id]
                                    observed_node = persistent_topology.nodes[
                                        observed_topo_id
                                    ]
                                    capture_anchor_id = route[-2]
                                    capture_anchor_node = persistent_topology.nodes[
                                        capture_anchor_id
                                    ]
                                    navigation_target_override = waypoint.position.copy()
                                    navigation_look_at_override = observed_node.position.copy()
                                    selected_topo_id = observed_topo_id
                                    active_topo_route = {
                                        "target_node_id": observed_topo_id,
                                        "snapshot_image": max_point_choice.image,
                                        "route": route,
                                        "waypoint_id": waypoint_id,
                                    }
                                    topology_route_length_m = (
                                        persistent_topology.route_length_m(route[:-1])
                                    )
                                    try:
                                        direct_distance_m, direct_path_points = (
                                            tsdf_planner.get_direct_geodesic_distance(
                                                current_voxel,
                                                capture_anchor_node.position,
                                                height=floor_height,
                                                pathfinder=scene.pathfinder,
                                            )
                                        )
                                        direct_distance_m = float(direct_distance_m)
                                        direct_path_reachable = bool(
                                            direct_path_points is not None
                                            and np.isfinite(direct_distance_m)
                                        )
                                    except Exception as exc:
                                        direct_distance_m = float("nan")
                                        direct_path_reachable = False
                                        logging.warning(
                                            "GoalTopo route diagnostic path query failed: %s",
                                            exc,
                                        )
                                    route_detour_ratio = None
                                    if (
                                        direct_path_reachable
                                        and direct_distance_m > 0.0
                                        and np.isfinite(topology_route_length_m)
                                    ):
                                        route_detour_ratio = (
                                            topology_route_length_m
                                            / direct_distance_m
                                        )
                                    topology_trace.write({
                                        "event": "topology_route_selected",
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "snapshot_image": max_point_choice.image,
                                        "target_node_id": observed_topo_id,
                                        "capture_anchor_id": capture_anchor_id,
                                        "route": route,
                                        "next_waypoint_id": waypoint_id,
                                        "next_waypoint_position": waypoint.position,
                                        "committed": True,
                                        "topology_route_length_m": topology_route_length_m,
                                        "direct_geodesic_distance_m": (
                                            direct_distance_m
                                            if np.isfinite(direct_distance_m) else None
                                        ),
                                        "direct_path_reachable": direct_path_reachable,
                                        "route_detour_ratio": route_detour_ratio,
                                    })
                                elif committed_topo_route is not None:
                                    if route:
                                        completion_reason = "origin_anchor_reached"
                                        persistent_topology.stats.setdefault(
                                            "topology_route_commitments_completed", 0
                                        )
                                        persistent_topology.stats[
                                            "topology_route_commitments_completed"
                                        ] += 1
                                        event_name = (
                                            "topology_route_commitment_completed"
                                        )
                                    else:
                                        completion_reason = (
                                            "graph_route_unavailable"
                                        )
                                        persistent_topology.stats.setdefault(
                                            "topology_route_commitments_cancelled", 0
                                        )
                                        persistent_topology.stats[
                                            "topology_route_commitments_cancelled"
                                        ] += 1
                                        event_name = (
                                            "topology_route_commitment_cancelled"
                                        )
                                    topology_trace.write({
                                        "event": event_name,
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "snapshot_image": committed_topo_route[
                                            "snapshot_image"
                                        ],
                                        "target_node_id": committed_topo_route[
                                            "target_node_id"
                                        ],
                                        "waypoints_completed": committed_topo_route[
                                            "waypoints_completed"
                                        ],
                                        "reason": completion_reason,
                                    })
                                    committed_topo_route = None
                        if direct_first_anchor_reobserve_fallback:
                            topology_trace.write({
                                "event": (
                                    "direct_first_original_hgr_after_anchor"
                                ),
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "goal_type": goal_context.goal_type,
                                "snapshot_image": max_point_choice.image,
                                "reason": "post_anchor_reobservation",
                            })
                            direct_first_anchor_reobserve_snapshot = None
                        if active_topology_behavior_enabled and (
                            type(max_point_choice) == Frontier
                            or selected_belief_action is not None
                            or selected_goal_topo_action is not None
                            or active_topo_route is not None
                        ):
                            if selected_topo_id is None:
                                selected_topo_id = (
                                    selected_goal_topo_action.topo_node_id
                                    if selected_goal_topo_action is not None
                                    else selected_belief_action.node_id
                                    if selected_belief_action is not None
                                    else getattr(max_point_choice, "topo_id", None)
                                )
                            if selected_belief_action is not None:
                                selected_path_cost = (
                                    selected_belief_action.normalized_path_cost
                                )
                            if selected_topo_id is not None and active_topo_view is not None:
                                selected_candidate = next(
                                    (item for item in active_topo_view.candidates
                                     if item.topo_node_id == selected_topo_id),
                                    None,
                                )
                                if selected_candidate is not None:
                                    selected_path_cost = selected_candidate.raw_path_cost
                            if selected_topo_id is not None:
                                selection_made_progress = persistent_topology.record_selection(
                                    selected_topo_id, selected_path_cost, global_step
                                )

                        # Phase B preserves the HGR-selected source and only
                        # replaces a cross-Place route with its next valid hop.
                        v2_approach = None
                        if place_route_only_enabled and not dual_v2_enabled:
                            v2_approach = None
                            if not place_route_commitment_resumed:
                                place_route_execution.reset()
                            target_kind = None
                            target_id = None
                            target_place_id = None
                            terminal_position = None
                            if type(max_point_choice) == SnapShot:
                                target_kind = "snapshot"
                                target_id = str(max_point_choice.image)
                                observation_binding = (
                                    place_topology.observations.get(
                                        f"snapshot:{target_id}"
                                    )
                                )
                                if observation_binding is not None:
                                    target_place_id = (
                                        observation_binding.capture_place_id
                                    )
                                    terminal_position = (
                                        observation_binding.capture_pose
                                    )
                            elif type(max_point_choice) == Frontier:
                                target_kind = "frontier"
                                target_id = str(
                                    getattr(max_point_choice, "topo_id", "")
                                )
                                frontier_binding = (
                                    place_topology.frontiers.get(target_id)
                                )
                                if frontier_binding is not None:
                                    target_place_id = (
                                        select_reachable_approach_place(
                                            place_topology,
                                            frontier_binding.approach_candidates_m,
                                        )
                                    )
                                    terminal_position = (
                                        frontier_binding.frontier_position
                                    )
                            if phase_c_enabled and phase_c.active is not None:
                                target_place_id = phase_c.active.place_id
                                terminal_position = phase_c.active.terminal
                            if object_approach_enabled:
                                from src.object_approach import resolve_object_approach, resolve_terminal_route
                                approach_query_started = time.perf_counter()
                                previous = ((place_route_commitment or {}).get("object_approach")
                                            if place_route_commitment_resumed else
                                            getattr(max_point_choice, "object_approach", None))
                                origin = place_route_execution.reached_place_id
                                current = tsdf_planner.habitat2voxel(pts)[:2]
                                if type(max_point_choice) == SnapShot and len(max_point_choice.cluster) == 1:
                                    v2_approach = resolve_object_approach(
                                        place_topology, tsdf_planner, scene.objects, max_point_choice.cluster[0],
                                        current, cfg.planner.final_observe_distance, previous, origin)
                                elif type(max_point_choice) == Frontier:
                                    v2_approach = resolve_terminal_route(
                                        place_topology, tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                        current, max_point_choice.position[:2], cfg.planner.final_observe_distance, origin,
                                        previous=(previous if previous and np.array_equal(
                                            np.rint(previous.get("terminal")),
                                            np.rint(max_point_choice.position[:2])) else None))
                                    route_terminal = (v2_approach["certified_path"][-1]
                                        if v2_approach.get("status") == "valid"
                                        else np.rint(max_point_choice.position[:2]).tolist())
                                    v2_approach.update(terminal=route_terminal,
                                        look_at=(np.asarray(max_point_choice.position[:2]) + np.asarray(max_point_choice.orientation[:2])*3).tolist())
                                if v2_approach and v2_approach["status"] == "valid":
                                    target_place_id = v2_approach["approach_place_id"]
                                    terminal_position = v2_approach["terminal"]
                                else:
                                    # Explicit legacy local fallback, never a capture-Place detour.
                                    target_place_id = None
                                topology_trace.write({"event": "v2_object_approach", "subtask_id": subtask_id,
                                    "step": global_step, "approach": v2_approach,
                                    "geometry_query_seconds": time.perf_counter() - approach_query_started,
                                    "fallback": not v2_approach or v2_approach["status"] != "valid"})
                                if (cfg.get("hgr_topology_fusion", {}).get("record_decisions", False)
                                        and v2_approach and v2_approach["status"] == "valid"):
                                    record_decision(os.path.join(cfg.output_dir, "decision_records", subtask_id),
                                        global_step, place_topology,
                                        tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                        current, v2_approach, None)
                            if target_kind is not None:
                                if place_route_commitment_resumed and not phase_c_enabled and not object_approach_enabled:
                                    # Preserve the committed approach as well as
                                    # the semantic source across edge arrivals.
                                    target_place_id = place_route_commitment[
                                        "target_place_id"
                                    ]
                                place_route_plan = plan_place_route(
                                    place_topology,
                                    target_place_id,
                                    target_kind,
                                    target_id,
                                    terminal_position,
                                    reached_place_id=(
                                        place_route_execution.reached_place_id
                                    ),
                                )
                                place_route_stats["plans"] += 1
                                place_route_stats.setdefault(
                                    place_route_plan.route_mode, 0
                                )
                                place_route_stats[
                                    place_route_plan.route_mode
                                ] += 1
                                topology_trace.write({
                                    "event": "place_route_planned",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "hgr_choice_preserved": not phase_c_enabled,
                                    "hgr_topology_fusion": hgr_topology_fusion_enabled,
                                    "execution_mode": "route_guidance" if guidance_enabled else "anchor_waypoints",
                                    "phase_c_intent": phase_c.active.to_trace_dict() if phase_c_enabled else None,
                                    "observed_place_id": place_topology.current_place_id,
                                    "execution_reached_place_id": (
                                        place_route_execution.reached_place_id
                                    ),
                                    "commitment_resumed": (
                                        place_route_commitment_resumed
                                    ),
                                    "plan": place_route_plan.to_trace_dict(),
                                })
                                if phase_c_enabled and not place_route_plan.route_valid:
                                    phase_c.finish("graph_route_invalid", global_step)
                                    place_route_commitment = None
                                    place_route_snapshot_source = None
                                    place_route_execution.reset()
                                    continue
                                if place_route_plan.uses_override:
                                    navigation_target_override = (
                                        place_route_plan.target_voxel.copy()
                                    )
                                    navigation_look_at_override = (
                                        place_route_plan.look_at_voxel.copy()
                                        if place_route_plan.look_at_voxel
                                        is not None else None
                                    )
                                    if place_route_commitment is None:
                                        place_route_snapshot_source = (
                                            SnapshotSourceCommitment(max_point_choice)
                                            if target_kind == "snapshot" else None
                                        )
                                        place_route_commitment = {
                                            "object_approach": v2_approach,
                                            "target_kind": target_kind,
                                            "target_id": target_id,
                                            "object_ids": (
                                                list(max_point_choice.cluster)
                                                if target_kind == "snapshot" else []
                                            ),
                                            "target_place_id": target_place_id,
                                            "created_step": global_step,
                                        }
                                        topology_trace.write({
                                            "event": (
                                                "place_route_commitment_created"
                                            ),
                                            "scene_id": scene_id,
                                            "episode_id": episode_id,
                                            "subtask_id": subtask_id,
                                            "step": global_step,
                                            "commitment": (
                                                place_route_commitment
                                            ),
                                        })
                                elif (
                                    place_route_plan.route_mode
                                    == PlaceRouteMode.SAME_PLACE.value
                                    and place_route_commitment is not None
                                ):
                                    topology_trace.write({
                                        "event": (
                                            "place_route_commitment_completed"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "commitment": (
                                            place_route_commitment
                                        ),
                                        "reason": "approach_place_reached",
                                    })
                                    place_route_commitment = None
                                    place_route_snapshot_source = None
                                elif (
                                    not place_route_plan.route_valid
                                    and place_route_commitment is not None
                                ):
                                    topology_trace.write({
                                        "event": (
                                            "place_route_commitment_released"
                                        ),
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "commitment": (
                                            place_route_commitment
                                        ),
                                        "reason": (
                                            place_route_plan.reason
                                        ),
                                    })
                                    place_route_commitment = None
                                    place_route_snapshot_source = None

                        if baseline_dual_dynamic_enabled:
                            selection_record = baseline_dual_dynamic.observe_hgr_selection(
                                max_point_choice, v2_approach, global_step
                            )
                            topology_trace.write({
                                "event": "dual_dynamic_hgr_selection_observed",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "stage": baseline_dual_dynamic_stage,
                                "hgr_choice_preserved": not unified_enabled,
                                    "hgr_stop_preserved": baseline_dual_dynamic_stage not in (
                                        "verified", "adaptive"
                                    ),
                                "selection": selection_record.to_trace_dict(),
                            })

                        if dual_v2_enabled:
                            from src.object_approach import resolve_object_approach, resolve_terminal_route
                            current = tsdf_planner.habitat2voxel(pts)[:2]
                            previous = getattr(max_point_choice, "object_approach", None)
                            if dual_v2.reusable(previous, place_topology,
                                    tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied, current):
                                v2_approach = previous
                            elif type(max_point_choice) == SnapShot and len(max_point_choice.cluster) == 1:
                                v2_approach = resolve_object_approach(place_topology, tsdf_planner, scene.objects,
                                    max_point_choice.cluster[0], current, cfg.planner.final_observe_distance, previous)
                            elif type(max_point_choice) == Frontier:
                                from src.object_approach import resolve_frontier_approach
                                v2_approach = resolve_frontier_approach(place_topology, tsdf_planner,
                                    max_point_choice, current, cfg.planner.final_observe_distance, previous=previous)
                            topology_trace.write({"event": "v2_route_selected", "subtask_id": subtask_id,
                                "step": global_step, "approach": v2_approach,
                                "selector_route_id": previous.get("route_id") if previous else None,
                                "route_changed_after_selection": bool(previous and v2_approach and
                                    previous.get("route_id") != v2_approach.get("route_id"))})
                            if not v2_approach or v2_approach["status"] != "valid":
                                dual_v2.release("no_certified_route")
                                if hierarchical_active:
                                    hierarchical_navigator.release("no_certified_route")
                                tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                                continue
                            hierarchical_intent = None
                            if hierarchical_active:
                                decision = getattr(max_point_choice, "hierarchical_decision", None)
                                if decision is None:
                                    raise RuntimeError("Hierarchical choice lost its navigation decision")
                                hierarchical_intent = hierarchical_navigator.install(
                                    decision, terminal=v2_approach["terminal"],
                                )
                            dual_v2.install(
                                max_point_choice, v2_approach, tsdf_planner.island.shape,
                                external_intent=hierarchical_intent,
                            )
                            if hierarchical_active:
                                if dual_v2.goal.active_intent.intent_id != hierarchical_intent.intent_id:
                                    raise RuntimeError("Geometry adapter lost the authoritative intent ID")
                                topology_trace.write({
                                    "event": "hierarchical_intent_installed",
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "intent": hierarchical_intent.to_trace_dict(),
                                    "decision": decision.to_trace_dict(),
                                    "semantic_assessments": {
                                        key: value.to_trace_dict()
                                        for key, value in hierarchical_navigator.last_assessments.items()
                                    },
                                })
                            topology_trace.write({"event": "v2_intent_installed", "subtask_id": subtask_id,
                                "step": global_step, "intent": dual_v2.goal.active_intent,
                                "candidate": dual_v2.goal.candidates[dual_v2.goal.active_intent.candidate_id].context()})
                            if dual_v2_cfg.get("record_decisions", False):
                                record_decision(os.path.join(cfg.output_dir, "decision_records", subtask_id),
                                    global_step, place_topology,
                                    tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                    current, v2_approach, dual_v2.goal.candidates[dual_v2.goal.active_intent.candidate_id])
                            navigation_target_override = np.asarray(v2_approach["terminal"])
                            navigation_look_at_override = np.asarray(v2_approach["look_at"])
                        if not dual_v2_enabled and object_approach_enabled and v2_approach and v2_approach["status"] == "valid":
                            if place_route_commitment is not None:
                                place_route_commitment["object_approach"] = v2_approach
                                place_route_commitment["target_place_id"] = v2_approach["approach_place_id"]
                            if not place_route_plan.uses_override:
                                navigation_target_override = np.asarray(v2_approach["terminal"])
                                navigation_look_at_override = np.asarray(v2_approach["look_at"])
                        if phase_c_enabled and (guidance_enabled or not place_route_plan.uses_override):
                            navigation_target_override = np.asarray(phase_c.active.terminal)
                            navigation_look_at_override = np.asarray(phase_c.active.look_at)

                        if phase_c_enabled and not guidance_enabled:
                            segment = PlaceTopology.known_grid_path_lengths_m(
                                tsdf_planner.island, tsdf_planner.habitat2voxel(pts)[:2],
                                {"next": navigation_target_override}, cfg.tsdf_grid_size,
                            )
                            if "next" not in segment:
                                invalidated = invalidate_blocked_hop(
                                    place_topology, place_route_plan, tsdf_planner.occupied, global_step,
                                )
                                topology_trace.write({
                                    "event": "phase_c_execution_feedback", "subtask_id": subtask_id,
                                    "step": global_step,
                                    "feedback": phase_c.finish("local_segment_unreachable", global_step),
                                    "edge_invalidated": invalidated,
                                })
                                place_route_commitment = None
                                place_route_snapshot_source = None
                                place_route_execution.reset()
                                continue

                        # set the vlm choice as the navigation target
                        if unified_enabled:
                            update_success = baseline_dual_dynamic.setup(tsdf_planner, pts, cfg)
                        else:
                            update_success = tsdf_planner.set_next_navigation_point(
                                choice=max_point_choice,
                                pts=pts,
                                objects=scene.objects,
                                cfg=cfg.planner,
                                pathfinder=scene.pathfinder,
                                known_space_only=known_space_execution,
                                target_point_override=(
                                    navigation_target_override
                                    if navigation_target_override is not None
                                    else selected_belief_action.verification.target_point
                                    if selected_belief_action is not None
                                    and selected_belief_action.action_type == BeliefActionType.VERIFY
                                    and selected_belief_action.verification is not None
                                    else None
                                ),
                                look_at_point=(
                                    navigation_look_at_override
                                    if navigation_look_at_override is not None
                                    else selected_belief_action.verification.look_at_point
                                    if selected_belief_action is not None
                                    and selected_belief_action.action_type == BeliefActionType.VERIFY
                                    and selected_belief_action.verification is not None
                                    else None
                                ),
                            )
                        if phase_c_enabled and not update_success:
                            invalidated = invalidate_blocked_hop(
                                place_topology, place_route_plan,
                                tsdf_planner.occupied, global_step,
                            )
                            topology_trace.write({
                                "event": "phase_c_execution_feedback", "subtask_id": subtask_id,
                                "step": global_step,
                                "feedback": phase_c.finish("local_path_setup_failed", global_step),
                                "edge_invalidated": invalidated,
                            })
                            # Do not reinterpret one controller failure as permanent
                            # graph disconnection, or fall back to an unchecked stop.
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None
                            tsdf_planner.look_at_point = None
                            place_route_commitment = None
                            place_route_snapshot_source = None
                            place_route_execution.reset()
                            continue
                        if dual_v2_enabled and not update_success:
                            topology_trace.write({"event": "v2_feedback", "subtask_id": subtask_id,
                                "step": global_step, "feedback": dual_v2.release("local_setup_failed")})
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue
                        if update_success and place_route_only_enabled and not guidance_enabled and not dual_v2_enabled:
                            place_route_execution.start(place_route_plan)
                        if (
                            not update_success
                            and place_route_plan is not None
                            and place_route_plan.uses_override
                        ):
                            invalidated_edge = False
                            next_cell = np.rint(
                                place_route_plan.target_voxel
                            ).astype(int)
                            if (
                                tsdf_planner.check_within_bnds(next_cell)
                                and tsdf_planner.occupied[
                                    tuple(next_cell)
                                ]
                                and place_route_plan.current_place_id
                                is not None
                                and place_route_plan.next_place_id
                                is not None
                            ):
                                invalidated_edge = (
                                    place_topology.invalidate_edge(
                                        place_route_plan.current_place_id,
                                        place_route_plan.next_place_id,
                                        global_step,
                                    )
                                )
                                if invalidated_edge:
                                    place_route_stats[
                                        "edges_invalidated_by_geometry"
                                    ] += 1
                            topology_trace.write({
                                "event": "place_route_waypoint_fallback",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "plan": place_route_plan.to_trace_dict(),
                                "reason": "waypoint_invalid_in_current_tsdf",
                                "edge_invalidated": invalidated_edge,
                                "fallback": "same_hgr_target_terminal_segment",
                            })
                            place_route_stats["waypoint_fallbacks"] += 1
                            place_route_execution.start(None)
                            place_route_commitment = None
                            place_route_snapshot_source = None
                            navigation_target_override = None
                            navigation_look_at_override = None
                            update_success = (
                                tsdf_planner.set_next_navigation_point(
                                    choice=max_point_choice,
                                    pts=pts,
                                    objects=scene.objects,
                                    cfg=cfg.planner,
                                    pathfinder=scene.pathfinder,
                                    known_space_only=known_space_execution,
                                )
                            )
                        # A stored VISITED pose is normally a safe waypoint, but
                        # the current TSDF may have reclassified that exact voxel.
                        # In that case retain HGR's original snapshot navigation
                        # instead of turning a stale waypoint into a hard failure.
                        if (
                            not update_success
                            and active_topo_route is not None
                            and type(max_point_choice) == SnapShot
                        ):
                            topology_trace.write({
                                "event": "topology_route_fallback",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                **active_topo_route,
                                "reason": "waypoint_invalid_in_current_tsdf",
                            })
                            persistent_topology.stats.setdefault(
                                "topology_route_fallbacks", 0
                            )
                            persistent_topology.stats[
                                "topology_route_fallbacks"
                            ] += 1
                            if committed_topo_route is not None:
                                topology_trace.write({
                                    "event": "topology_route_commitment_cancelled",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "waypoints_completed": committed_topo_route[
                                        "waypoints_completed"
                                    ],
                                    "reason": "waypoint_invalid_in_current_tsdf",
                                })
                                persistent_topology.stats.setdefault(
                                    "topology_route_commitments_cancelled", 0
                                )
                                persistent_topology.stats[
                                    "topology_route_commitments_cancelled"
                                ] += 1
                                committed_topo_route = None
                            active_topo_route = None
                            navigation_target_override = None
                            navigation_look_at_override = None
                            update_success = tsdf_planner.set_next_navigation_point(
                                choice=max_point_choice,
                                pts=pts,
                                objects=scene.objects,
                                cfg=cfg.planner,
                                pathfinder=scene.pathfinder,
                                known_space_only=known_space_execution,
                            )
                        if not update_success:
                            if active_topology_behavior_enabled and (
                                type(max_point_choice) == Frontier
                                or selected_belief_action is not None
                                or selected_goal_topo_action is not None
                                or active_topo_route is not None
                            ):
                                if selected_topo_id is not None:
                                    persistent_topology.record_navigation_result(
                                        selected_topo_id,
                                        False,
                                        global_step,
                                        reason="set_next_navigation_point_failed",
                                    )
                                if dual_explore_commitment is not None:
                                    topology_trace.write({
                                        "event": "dual_layer_explore_commitment_released",
                                        "scene_id": scene_id,
                                        "episode_id": episode_id,
                                        "subtask_id": subtask_id,
                                        "step": global_step,
                                        "commitment": (
                                            dual_explore_commitment.to_trace_dict()
                                        ),
                                        "reason": "set_next_navigation_point_failed",
                                    })
                                    dual_explore_commitment = None
                                if (
                                    belief_v2_enabled
                                    and selected_belief_action is not None
                                    and selected_belief_action.action_type
                                    == BeliefActionType.CONFIRM_APPROACH
                                ):
                                    belief_memory.record_confirmation_failure(
                                        normalize_goal_key(goal_context.category)
                                    )
                                tsdf_planner.max_point = None
                                tsdf_planner.target_point = None
                                tsdf_planner.look_at_point = None
                                active_topo_route = None
                                topology_trace.write({
                                    "event": "navigation_failure",
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "topo_id": selected_topo_id,
                                    "reason": "set_next_navigation_point_failed",
                                })
                                continue
                            logging.info(
                                f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!"
                            )
                            subtask_execution.terminate(
                                ExecutionStatus.ERROR,
                                "navigation_setup_failed",
                                error_category="navigation_setup",
                            )
                            break

                    # (5) Agent navigate to the target point for one step
                    stage_timer.switch("route_execution")
                    if dual_v2_enabled:
                        if hierarchical_active and hierarchical_verification_target is not None:
                            route_event = "verification_viewpoint_retained"
                        else:
                            route_event = dual_v2.ensure_route(place_topology,
                                tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                tsdf_planner.habitat2voxel(pts)[:2], cfg.planner.final_observe_distance)
                        effective_execution_mode = (
                            "local_tsdf"
                            if hierarchical_active and (
                                dual_v2.uses_local_controller
                                or hierarchical_verification_target is not None
                            )
                            else (
                                hierarchical_navigator.active_intent.execution_mode.value
                                if hierarchical_active and hierarchical_navigator.active_intent is not None
                                else "certified_route"
                            )
                        )
                        topology_trace.write({"event": "v2_execution", "subtask_id": subtask_id,
                            "step": global_step, "reason": route_event,
                            "execution_mode": effective_execution_mode,
                            "intent_id": dual_v2.goal.active_intent.intent_id if dual_v2.goal.active_intent else None,
                            "route_id": dual_v2.goal.active_intent.route_id if dual_v2.goal.active_intent else None,
                            "guidance": dual_v2.execution.to_trace_dict(include_path=(
                                route_event != "retained" or dual_v2.execution.route_id != dual_v2.logged_route_id))
                                if dual_v2.execution else None})
                        if dual_v2.execution is not None:
                            dual_v2.logged_route_id = dual_v2.execution.route_id
                        if route_event in ("missing_route", "route_unavailable"):
                            dual_v2.release(route_event)
                            if hierarchical_active:
                                hierarchical_navigator.on_event(NavigationEvent(
                                    NavigationEventType.EDGE_INVALIDATED,
                                    global_step,
                                    route_event,
                                ))
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue
                        route_guidance = (
                            None
                            if hierarchical_active and (
                                dual_v2.uses_local_controller
                                or hierarchical_verification_target is not None
                            )
                            else dual_v2.execution
                        )
                    if guidance_enabled:
                        if route_guidance is None or getattr(route_guidance, "intent", None) is not phase_c.active:
                            route_guidance = None
                        mask = tsdf_planner.island & ~tsdf_planner.occupied
                        if route_guidance is not None and not route_guidance.valid(place_topology, mask):
                            repaired = route_guidance.repair(mask, tsdf_planner.habitat2voxel(pts)[:2])
                            topology_trace.write({"event": "route_guidance_local_repair", "subtask_id": subtask_id,
                                "step": global_step, "repaired": repaired,
                                "guidance": route_guidance.to_trace_dict(include_path=True) if repaired else None})
                        if route_guidance is None or not route_guidance.valid(place_topology, mask):
                            origin = route_guidance.remaining_route()[0] if route_guidance is not None else None
                            route_guidance = RouteGuidance.build(
                                place_topology, mask, tsdf_planner.habitat2voxel(pts)[:2],
                                phase_c.active.terminal, phase_c.active.place_id, origin=origin,
                                observe_distance_m=cfg.planner.final_observe_distance,
                            )
                            if route_guidance is None:
                                topology_trace.write({"event": "phase_c_execution_feedback",
                                    "subtask_id": subtask_id, "step": global_step,
                                    "feedback": phase_c.finish("corridor_unavailable", global_step)})
                                tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                                place_route_commitment = place_route_snapshot_source = None
                                place_route_execution.reset()
                                continue
                            route_guidance.intent = phase_c.active
                            topology_trace.write({"event": "route_guidance_installed", "subtask_id": subtask_id,
                                "step": global_step, "guidance": route_guidance.to_trace_dict(include_path=True)})
                        place_route_execution.reset()
                    if known_space_execution:
                        return_values = known_executor.step(
                            tsdf_planner, pts, angle, scene.objects, scene.snapshots,
                            cfg.planner, save_visualization=cfg.save_visualization,
                            certified_route=(baseline_dual_dynamic.active_intent.task.route
                                if unified_enabled and baseline_dual_dynamic.active_intent is not None else None),
                        )
                        topology_trace.write({
                            "event": "known_space_motion", "subtask_id": subtask_id,
                            "step": global_step, "stage": baseline_dual_dynamic_stage,
                            "audit": known_executor.last_audit,
                            "blocked": return_values[0] is None,
                        })
                    else:
                        return_values = tsdf_planner.agent_step(
                            pts=pts,
                            angle=angle,
                            objects=scene.objects,
                            snapshots=scene.snapshots,
                            pathfinder=scene.pathfinder,
                            cfg=cfg.planner,
                            path_points=None,
                            save_visualization=cfg.save_visualization,
                            route_guidance=route_guidance if guidance_enabled or dual_v2_enabled else None,
                        )
                    if guidance_enabled:
                        topology_trace.write({"event": "route_guidance_step", "subtask_id": subtask_id,
                            "step": global_step, "guidance": route_guidance.to_trace_dict()})
                    if return_values[0] is None:
                        if unified_enabled:
                            intent = baseline_dual_dynamic.active_intent
                            if baseline_dual_dynamic.recover(tsdf_planner, place_topology, pts,
                                                           cfg.planner.final_observe_distance):
                                topology_trace.write({"event": "hypothesis_aware_recovery",
                                    "step": global_step, "intent_id": intent.intent_id,
                                    "attempt": intent.recovery_attempts})
                                continue
                            baseline_dual_dynamic.release("recovery_exhausted", tsdf_planner.hypothesis_graph)
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue
                        if dual_v2_enabled:
                            dual_v2.execution.pending_cursor = None
                            if not dual_v2.recovery_attempted:
                                dual_v2.recovery_attempted = True
                                origin = dual_v2.execution.remaining_route()[0]
                                recovered = dual_v2.resolve(place_topology,
                                    tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied,
                                    tsdf_planner.habitat2voxel(pts)[:2], dual_v2.execution.terminal,
                                    cfg.planner.final_observe_distance, origin=origin,
                                    previous=dual_v2.selected_approach)
                                if recovered["status"] == "valid":
                                    recovered["look_at"] = dual_v2.selected_approach["look_at"]
                                    dual_v2.install(max_point_choice, recovered, tsdf_planner.island.shape, retain_intent=True)
                                    dual_v2.recovery_attempted = True
                                    topology_trace.write({"event": "v2_feedback", "subtask_id": subtask_id,
                                        "step": global_step, "reason": "same_target_execution_recovery"})
                                    continue
                            topology_trace.write({"event": "v2_feedback", "subtask_id": subtask_id,
                                "step": global_step, "feedback": dual_v2.release("local_execution_failed")})
                            if hierarchical_active:
                                hierarchical_navigator.on_event(NavigationEvent(
                                    NavigationEventType.LOCAL_BLOCKED,
                                    global_step,
                                    "local_execution_failed",
                                ))
                                hierarchical_navigator.release("runtime_recovery_exhausted")
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue
                        if phase_c_enabled:
                            topology_trace.write({
                                "event": "phase_c_execution_feedback", "subtask_id": subtask_id,
                                "step": global_step,
                                "feedback": phase_c.finish("local_execution_failed", global_step),
                            })
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None
                            tsdf_planner.look_at_point = None
                            place_route_commitment = None
                            place_route_snapshot_source = None
                            place_route_execution.reset()
                            continue
                        if place_route_execution.active_plan is not None:
                            topology_trace.write({
                                "event": "place_route_execution_failure",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "reason": "agent_step_failed",
                                "plan": place_route_execution.active_plan.to_trace_dict(),
                            })
                        if active_topology_behavior_enabled and (
                            type(max_point_choice) == Frontier
                            or selected_belief_action is not None
                            or selected_goal_topo_action is not None
                            or active_topo_route is not None
                        ):
                            failed_topo_id = (
                                active_topo_route.get("target_node_id")
                                if active_topo_route is not None
                                else selected_goal_topo_action.topo_node_id
                                if selected_goal_topo_action is not None
                                else selected_belief_action.node_id
                                if selected_belief_action is not None
                                else getattr(max_point_choice, "topo_id", None)
                            )
                            if failed_topo_id is not None:
                                persistent_topology.record_navigation_result(
                                    failed_topo_id, False, global_step,
                                    reason="agent_step_failed",
                                )
                            if dual_explore_commitment is not None:
                                topology_trace.write({
                                    "event": "dual_layer_explore_commitment_released",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "commitment": (
                                        dual_explore_commitment.to_trace_dict()
                                    ),
                                    "reason": "agent_step_failed",
                                })
                                dual_explore_commitment = None
                            if (
                                belief_v2_enabled
                                and selected_belief_action is not None
                                and selected_belief_action.action_type
                                == BeliefActionType.CONFIRM_APPROACH
                            ):
                                belief_memory.record_confirmation_failure(
                                    normalize_goal_key(goal_context.category)
                                )
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None
                            tsdf_planner.look_at_point = None
                            if committed_topo_route is not None:
                                topology_trace.write({
                                    "event": "topology_route_commitment_cancelled",
                                    "scene_id": scene_id,
                                    "episode_id": episode_id,
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "waypoints_completed": committed_topo_route[
                                        "waypoints_completed"
                                    ],
                                    "reason": "agent_step_failed",
                                })
                                persistent_topology.stats.setdefault(
                                    "topology_route_commitments_cancelled", 0
                                )
                                persistent_topology.stats[
                                    "topology_route_commitments_cancelled"
                                ] += 1
                                committed_topo_route = None
                            active_topo_route = None
                            topology_trace.write({
                                "event": "navigation_failure",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "topo_id": failed_topo_id,
                                "reason": "agent_step_failed",
                            })
                            continue
                        logging.info(
                            f"Subtask id {subtask_id} invalid: agent_step failed!"
                        )
                        subtask_execution.terminate(
                            ExecutionStatus.ERROR,
                            "agent_step_failed",
                            error_category="agent_step",
                        )
                        break

                    # update agent's position and rotation
                    stage_timer.switch("feedback")
                    pts, angle, pts_voxel, fig, _, target_arrived = return_values
                    if unified_enabled and baseline_dual_dynamic.active_intent is not None:
                        segment = (known_executor.last_audit or {}).get("actual_segment")
                        moved = bool(segment is not None and not np.allclose(
                            np.asarray(segment[0]), np.asarray(segment[1])
                        ))
                        if baseline_dual_dynamic.record_motion_progress(
                                tsdf_planner, place_topology, pts,
                                moved=moved, arrived=target_arrived):
                            intent = baseline_dual_dynamic.active_intent
                            recovered = baseline_dual_dynamic.recover(
                                tsdf_planner, place_topology, pts,
                                cfg.planner.final_observe_distance,
                            )
                            topology_trace.write({
                                "event": "hypothesis_aware_stall_recovery",
                                "subtask_id": subtask_id, "step": global_step,
                                "intent_id": intent.intent_id,
                                "recovered": bool(recovered),
                                "attempt": intent.recovery_attempts,
                            })
                            if not recovered:
                                baseline_dual_dynamic.release(
                                    "stall_recovery_exhausted",
                                    tsdf_planner.hypothesis_graph,
                                )
                                tsdf_planner.max_point = None
                                tsdf_planner.target_point = None
                                tsdf_planner.look_at_point = None
                    if dual_v2_enabled:
                        remaining_places_before = tuple(dual_v2.execution.remaining_route())
                        acknowledged = (
                            True
                            if hierarchical_active and hierarchical_verification_target is not None
                            else dual_v2.acknowledge_motion(
                                tsdf_planner.habitat2voxel(pts)[:2],
                                target_arrived=target_arrived,
                            )
                        )
                        remaining_places_after = tuple(dual_v2.execution.remaining_route())
                        if acknowledged:
                            dual_v2.recovery_attempted = False
                        target_arrived = bool(target_arrived and acknowledged)
                        if (
                            hierarchical_active
                            and acknowledged
                            and not target_arrived
                            and len(remaining_places_after) < len(remaining_places_before)
                        ):
                            command = hierarchical_navigator.on_event(NavigationEvent(
                                NavigationEventType.WAYPOINT_ARRIVED,
                                global_step,
                                "certified_transition_crossed",
                            ))
                            topology_trace.write({
                                "event": "hierarchical_waypoint_arrived",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "remaining_places": list(remaining_places_after),
                                "command": command.to_trace_dict(),
                                "semantic_request_triggered": False,
                            })
                        effective_execution_mode = (
                            "local_tsdf"
                            if hierarchical_active and (
                                dual_v2.uses_local_controller
                                or hierarchical_verification_target is not None
                            )
                            else (
                                hierarchical_navigator.active_intent.execution_mode.value
                                if hierarchical_active and hierarchical_navigator.active_intent is not None
                                else "certified_route"
                            )
                        )
                        topology_trace.write({"event": "v2_motion_acknowledged", "subtask_id": subtask_id,
                            "step": global_step, "acknowledged": acknowledged, "terminal_arrived": target_arrived,
                            "execution_mode": effective_execution_mode,
                            "intent_id": dual_v2.goal.active_intent.intent_id,
                            "route_id": dual_v2.goal.active_intent.route_id,
                            "guidance": dual_v2.execution.to_trace_dict()})
                        if target_arrived:
                            if hierarchical_active:
                                hierarchical_terminal_arrived = True
                                dual_v2.goal.active_intent.status = "verification_pending"
                                hierarchical_navigator.on_event(NavigationEvent(
                                    NavigationEventType.TERMINAL_ARRIVED,
                                    global_step,
                                    "tsdf_terminal_arrived",
                                ))
                            elif dual_v2.goal.active_intent.kind == "EXPLORE":
                                v2_arrival_frontier = dual_v2.choice
                                dual_v2.goal.active_intent.status = "verification_pending"
                            else:
                                topology_trace.write({"event": "v2_intent_finished", "subtask_id": subtask_id,
                                    "step": global_step, "feedback": dual_v2.release("terminal_arrived")})
                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(
                        f"Current position: {pts}, {logger.subtask_explore_dist:.3f}"
                    )

                    completed_place_plan = (
                        place_route_execution.consume_waypoint_arrival(target_arrived)
                    )
                    if completed_place_plan is not None:
                        topology_route_arrived_this_step = True
                        # Suppress semantic arrival for BOTH Snapshot and Frontier.
                        # Final Place arrival still requires the terminal segment.
                        target_arrived = False
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None
                        tsdf_planner.look_at_point = None
                        place_route_stats.setdefault("waypoint_arrivals", 0)
                        place_route_stats["waypoint_arrivals"] += 1
                        topology_trace.write({
                            "event": "place_route_waypoint_arrived",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "plan": completed_place_plan.to_trace_dict(),
                            "semantic_target_arrived": False,
                            "next_action": "resume_same_target_route_or_terminal",
                        })

                    if target_arrived and active_topo_route is not None:
                        topology_route_arrived_this_step = True
                        if committed_topo_route is not None:
                            committed_topo_route["waypoints_completed"] += 1
                        persistent_topology.stats.setdefault(
                            "topology_waypoint_arrivals", 0
                        )
                        persistent_topology.stats["topology_waypoint_arrivals"] += 1
                        topology_trace.write({
                            "event": "topology_waypoint_arrived",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            **active_topo_route,
                        })
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None
                        tsdf_planner.look_at_point = None
                        active_topo_route = None

                    if (
                        target_arrived
                        and dual_explore_commitment is not None
                        and type(max_point_choice) == Frontier
                    ):
                        topology_trace.write({
                            "event": "dual_layer_explore_commitment_completed",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "commitment": (
                                dual_explore_commitment.to_trace_dict()
                            ),
                            "reason": "frontier_arrived",
                        })
                        dual_explore_commitment = None

                    if active_topology_behavior_enabled and (
                        type(max_point_choice) == Frontier
                        or selected_belief_action is not None
                        or (
                            selected_goal_topo_action is not None
                            and not topology_route_arrived_this_step
                        )
                    ):
                        arrived_topo_id = (
                            selected_goal_topo_action.topo_node_id
                            if selected_goal_topo_action is not None
                            else selected_belief_action.node_id
                            if selected_belief_action is not None
                            else getattr(max_point_choice, "topo_id", None)
                        )
                        if target_arrived and arrived_topo_id is not None:
                            persistent_topology.record_navigation_result(
                                arrived_topo_id,
                                True,
                                global_step,
                                reason="target_arrived",
                                consume=(
                                    selected_belief_action is None
                                    or selected_belief_action.action_type not in (
                                        BeliefActionType.VERIFY,
                                        BeliefActionType.CONFIRM_APPROACH,
                                    )
                                ),
                            )
                            if (
                                belief_v2_enabled
                                and selected_belief_action is not None
                                and selected_belief_action.action_type
                                == BeliefActionType.CONFIRM_APPROACH
                            ):
                                belief_memory.record_confirmation_approach(
                                    normalize_goal_key(goal_context.category),
                                    global_step,
                                    wait_for_observation=belief_v3_enabled,
                                )
                        if active_topology_policy.use_active_view:
                            no_progress_budget = int(
                                active_topology_cfg.get("active_view", {}).get(
                                    "no_progress_steps", 12
                                )
                            )
                            if persistent_topology.recovery.record_step(
                                bool(target_arrived or selection_made_progress),
                                no_progress_budget,
                            ):
                                persistent_topology.stats["recovery_escalations"][
                                    persistent_topology.recovery.level
                                ] += 1

                    # Belief-mode VERIFY writes only local semantic evidence.  It
                    # never invokes SemanticCritic or declares GOAT success.
                    if (
                        belief_planner_enabled
                        and selected_belief_action is not None
                        and selected_belief_action.action_type == BeliefActionType.VERIFY
                        and target_arrived
                    ):
                        verify_node_id = selected_belief_action.node_id
                        verify_goal_key = normalize_goal_key(goal_context.category)
                        verify_node = persistent_topology.nodes.get(verify_node_id)
                        belief_memory.record_verify_attempt(
                            verify_node_id, verify_goal_key, global_step
                        )
                        before_verify = belief_memory.belief(
                            verify_node_id,
                            verify_goal_key,
                            verify_node.map_confidence,
                            verify_node.accessibility,
                        )
                        verify_obs, verify_pose = scene.get_observation(pts, angle=angle)
                        verify_rgb = verify_obs["color_sensor"]
                        verify_observation_id = (
                            f"active_verify:{subtask_id}:{global_step}:{verify_node_id}"
                        )
                        verify_viewpoint = tsdf_planner.habitat2voxel(pts)[:2]
                        detector_confidence = detect_category_confidence(
                            verify_rgb,
                            scene.detection_model,
                            scene.obj_classes.get_classes_arr(),
                            goal_context.category,
                        )
                        if detector_confidence is not None:
                            _record_goal_evidence(
                                belief_memory, topology_trace, persistent_topology,
                                belief_memory.make_evidence(
                                    f"{verify_observation_id}:detector",
                                    verify_node_id,
                                    verify_goal_key,
                                    EvidenceSource.DETECTOR,
                                    0.5 + 0.5 * detector_confidence,
                                    global_step,
                                    viewpoint=verify_viewpoint,
                                    heading=angle,
                                    observation_id=verify_observation_id,
                                    reason=(
                                        "active verification local detector "
                                        f"confidence {detector_confidence:.3f}"
                                    ),
                                ),
                                scene_id, episode_id, subtask_id,
                            )
                        verify_embedding = _clip_image_embedding(
                            verify_rgb, clip_model, clip_preprocess
                        )
                        verify_clip_probability = calibrated_clip_probability(
                            verify_embedding,
                            goal_context.text_embedding,
                            floor=float(active_topology_cfg.get("relevance", {}).get("clip_floor", 0.10)),
                            ceiling=float(active_topology_cfg.get("relevance", {}).get("clip_ceiling", 0.35)),
                        )
                        verify_raw_clip_cosine = clip_cosine_similarity(
                            verify_embedding, goal_context.text_embedding,
                        )
                        if belief_v2_enabled and verify_raw_clip_cosine is not None:
                            _record_clip_evidence_v2(
                                belief_memory, topology_trace, persistent_topology,
                                evidence_id=f"{verify_observation_id}:clip",
                                node_id=verify_node_id, goal_key=verify_goal_key,
                                raw_cosine=verify_raw_clip_cosine, step=global_step,
                                viewpoint=verify_viewpoint, heading=angle,
                                observation_id=verify_observation_id,
                                reason="active verification local CLIP",
                                scene_id=scene_id, episode_id=episode_id,
                                subtask_id=subtask_id,
                            )
                        elif verify_clip_probability is not None:
                            _record_goal_evidence(
                                belief_memory, topology_trace, persistent_topology,
                                belief_memory.make_evidence(
                                    f"{verify_observation_id}:clip",
                                    verify_node_id,
                                    verify_goal_key,
                                    EvidenceSource.CLIP_GOAL,
                                    verify_clip_probability,
                                    global_step,
                                    viewpoint=verify_viewpoint,
                                    heading=angle,
                                    observation_id=verify_observation_id,
                                    reason="active verification local CLIP",
                                ),
                                scene_id, episode_id, subtask_id,
                            )
                        after_local = belief_memory.belief(
                            verify_node_id,
                            verify_goal_key,
                            verify_node.map_confidence,
                            verify_node.accessibility,
                        )
                        verification_cfg = active_topology_cfg.get("verification", {})
                        verify_vlm_result = None
                        verify_vlm_attempted = False
                        if (
                            float(verification_cfg.get("posterior_low", 0.30))
                            <= after_local.posterior
                            <= float(verification_cfg.get("posterior_high", 0.70))
                        ):
                            verify_vlm_attempted = True
                            extra_body_cfg = cfg.get("vlm_extra_body", None)
                            extra_body = (
                                OmegaConf.to_container(extra_body_cfg, resolve=True)
                                if extra_body_cfg is not None else None
                            )
                            belief_memory.stats["verify_vlm_calls"] += 1
                            verify_vlm_result = query_active_verify(
                                verify_rgb,
                                goal_context.category,
                                model_name=cfg.get("vlm_model", "openai/gpt-4o"),
                                extra_body=extra_body,
                            )
                            if verify_vlm_result is not None:
                                _record_goal_evidence(
                                    belief_memory, topology_trace, persistent_topology,
                                    belief_memory.make_evidence(
                                        f"{verify_observation_id}:vlm",
                                        verify_node_id,
                                        verify_goal_key,
                                        EvidenceSource.ACTIVE_VERIFY_VLM,
                                        verify_vlm_result["target_present_probability"],
                                        global_step,
                                        viewpoint=verify_viewpoint,
                                        heading=angle,
                                        observation_id=verify_observation_id,
                                        reason=verify_vlm_result["reason"],
                                    ),
                                    scene_id, episode_id, subtask_id,
                                )
                        after_verify = belief_memory.belief(
                            verify_node_id,
                            verify_goal_key,
                            verify_node.map_confidence,
                            verify_node.accessibility,
                        )
                        confirmation_created = False
                        confirmation_transition_reason = "v2_disabled"
                        if belief_v2_enabled:
                            confirmation_cfg = active_topology_cfg.get(
                                "confirmation", {}
                            )
                            detector_positive_threshold = float(
                                confirmation_cfg.get(
                                    "detector_min_confidence", 0.50
                                )
                            )
                            vlm_positive_threshold = float(
                                confirmation_cfg.get(
                                    "direct_positive_probability", 0.75
                                )
                            )
                            confirmation_posterior_threshold = float(
                                confirmation_cfg.get(
                                    "posterior_threshold", 0.75
                                )
                            )
                            direct_positive = (
                                detector_confidence is not None
                                and detector_confidence >= detector_positive_threshold
                            ) or (
                                verify_vlm_result is not None
                                and float(verify_vlm_result.get(
                                    "target_present_probability", 0.0
                                )) >= vlm_positive_threshold
                            )
                            positive_source = (
                                "active_verify_vlm"
                                if verify_vlm_result is not None
                                and float(verify_vlm_result.get(
                                    "target_present_probability", 0.0
                                )) >= vlm_positive_threshold
                                else "detector"
                            )
                            has_current_frontier_mapping = (
                                verify_node is not None
                                and verify_node.node_type == TopoNodeType.FRONTIER
                                and verify_node_id in
                                persistent_topology.frontier_source_map.values()
                            )
                            if (
                                direct_positive
                                and after_verify.posterior
                                >= confirmation_posterior_threshold
                                and has_current_frontier_mapping
                            ):
                                confirmation = belief_memory.create_confirmation(
                                    verify_node_id,
                                    verify_goal_key,
                                    global_step,
                                    after_verify.posterior,
                                    duration_steps=int(
                                        confirmation_cfg.get("max_steps", 5)
                                    ),
                                    max_approaches=int(
                                        confirmation_cfg.get("max_approaches", 2)
                                    ),
                                    positive_source=positive_source,
                                    reuse_existing=belief_v3_enabled,
                                )
                                confirmation_created = True
                                confirmation_transition_reason = "created"
                            elif not direct_positive:
                                confirmation_transition_reason = "no_direct_positive_evidence"
                            elif after_verify.posterior < confirmation_posterior_threshold:
                                confirmation_transition_reason = "posterior_below_threshold"
                            else:
                                confirmation_transition_reason = "no_current_frontier_mapping"
                        topology_trace.write({
                            "event": "active_verify_result",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "node_id": verify_node_id,
                            "viewpoint": verify_viewpoint,
                            "look_at_point": (
                                selected_belief_action.verification.look_at_point
                                if selected_belief_action.verification is not None else None
                            ),
                            "posterior_before": before_verify.posterior,
                            "posterior_after_local": after_local.posterior,
                            "posterior_after": after_verify.posterior,
                            "entropy_reduction": (
                                before_verify.entropy - after_verify.entropy
                            ),
                            "detector_confidence": detector_confidence,
                            "clip_probability": verify_clip_probability,
                            "vlm_attempted": verify_vlm_attempted,
                            "vlm_parsed": verify_vlm_result is not None,
                            "vlm_result": verify_vlm_result,
                            "confirmation_created": confirmation_created,
                            "confirmation_transition_reason": confirmation_transition_reason,
                        })
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None
                        tsdf_planner.look_at_point = None

                    # Unified terminal verification. Arrival is geometric only;
                    # semantic confirmation is required before a target intent
                    # may stop the GOAT subtask.
                    if hierarchical_active and hierarchical_terminal_arrived:
                        arrived_via_reobservation = hierarchical_verification_target is not None
                        hierarchical_verification_target = None
                        fresh_obs, _ = scene.get_observation(pts, angle=angle)
                        fresh_rgb = np.asarray(fresh_obs["color_sensor"])[..., :3]
                        goal_image_base64 = None
                        if subtask_metadata["task_type"] == "image" and subtask_metadata.get("image"):
                            try:
                                with open(subtask_metadata["image"], "rb") as image_handle:
                                    goal_image_base64 = base64.b64encode(image_handle.read()).decode("utf-8")
                            except OSError:
                                goal_image_base64 = None
                        attempt = (
                            1 if hierarchical_navigator.active_intent is None
                            else hierarchical_navigator.active_intent.verification_attempts + 1
                        )
                        hierarchical_navigator.stats["terminal_verification_requests"] += 1
                        candidate_image = getattr(
                            dual_v2.choice, "hierarchical_candidate_crop", None
                        )
                        candidate_label = getattr(
                            dual_v2.choice, "hierarchical_candidate_label", None
                        )
                        require_candidate_identity = intent_kind in (
                            IntentKind.TARGET_APPROACH,
                            IntentKind.EVIDENCE_REVISIT,
                        )
                        verification_result = query_terminal_verification(
                            fresh_rgb,
                            subtask_metadata["task_type"],
                            subtask_metadata["question"],
                            subtask_metadata["class"],
                            cfg.get("vlm_model", "openai/gpt-4o"),
                            extra_body=(
                                OmegaConf.to_container(cfg.get("vlm_extra_body"), resolve=True)
                                if cfg.get("vlm_extra_body", None) is not None else None
                            ),
                            goal_image_base64=goal_image_base64,
                            attempt=attempt,
                            decision_confidence_threshold=float(
                                hierarchical_cfg.get("verification_min_confidence", 0.70)
                            ),
                            candidate_image=candidate_image,
                            candidate_label=candidate_label,
                            require_candidate_identity=require_candidate_identity,
                        )
                        intent_before_verification = hierarchical_navigator.active_intent
                        intent_kind = (
                            None if intent_before_verification is None
                            else intent_before_verification.kind
                        )
                        evidence_id = f"terminal:{subtask_id}:{global_step}:{attempt}"
                        command = hierarchical_navigator.record_verification(
                            verification_result,
                            global_step,
                            evidence_digest=hierarchical_navigator.pending_semantic_digest,
                        )
                        topology_trace.write({
                            "event": "hierarchical_terminal_verification",
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "fresh_observation_id": evidence_id,
                            "arrived_via_reobservation": arrived_via_reobservation,
                            "candidate_identity_required": require_candidate_identity,
                            "candidate_reference_available": candidate_image is not None,
                            "candidate_label": candidate_label,
                            "result": verification_result.to_trace_dict(),
                            "command": command.to_trace_dict(),
                        })

                        if (
                            intent_kind == IntentKind.FRONTIER_EXPLORE
                            and getattr(dual_v2.choice, "hypothesis_node_id", None) is not None
                            and semantic_critic is not None
                        ):
                            from src.query_vlm_hypothesis import verify_fresh_intent_arrival
                            hypothesis_result = verify_fresh_intent_arrival(
                                dual_v2.choice,
                                scene,
                                tsdf_planner,
                                semantic_critic,
                                pts,
                                angle,
                            )
                            topology_trace.write({
                                "event": "hierarchical_hypothesis_verification",
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "result": hypothesis_result,
                            })
                            if hypothesis_result.get("is_falsified", False):
                                dual_v2.revoke(
                                    hypothesis_result,
                                    f"hierarchical-hypothesis:{evidence_id}",
                                )

                        if verification_result.verdict == VerificationVerdict.CONFIRMED:
                            runtime_intent = dual_v2.goal.active_intent
                            if runtime_intent is not None:
                                runtime_intent.observed_evidence_refs.add(evidence_id)
                                candidate = dual_v2.goal.candidates.get(runtime_intent.candidate_id)
                                if candidate is not None:
                                    candidate.observed_evidence_refs.add(evidence_id)
                                    dual_v2.goal._refresh_intent_evidence(candidate)
                            for hypothesis_id in (
                                () if intent_before_verification is None
                                else intent_before_verification.hypothesis_refs
                            ):
                                node = tsdf_planner.hypothesis_graph.nodes.get(hypothesis_id)
                                if node is not None and evidence_id not in node.independent_observation_refs:
                                    node.independent_observation_refs.append(evidence_id)
                            if command.command == NavigatorCommandType.STOP:
                                plt.imsave(
                                    os.path.join(logger.subtask_object_observe_dir, "target.png"),
                                    fresh_rgb,
                                )
                                dual_v2.release("terminal_semantics_confirmed")
                                task_success = True
                                subtask_execution.terminate(
                                    ExecutionStatus.COMPLETED,
                                    "verified_confirmation",
                                )
                                break
                            # A confirmed historical observation becomes new
                            # evidence; it still needs a real object approach.
                            dual_v2.release("evidence_revisit_confirmed")
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue

                        if verification_result.verdict == VerificationVerdict.REJECTED:
                            hypothesis_refs = (
                                () if intent_before_verification is None
                                else intent_before_verification.hypothesis_refs
                            )
                            # A target-absence verdict does not by itself
                            # falsify a Frontier's room hypothesis. Frontier
                            # hypotheses are handled by the fresh critic above.
                            for hypothesis_id in (
                                () if intent_kind == IntentKind.FRONTIER_EXPLORE
                                else hypothesis_refs
                            ):
                                node = tsdf_planner.hypothesis_graph.nodes.get(hypothesis_id)
                                if node is None:
                                    continue
                                affected = tsdf_planner.hypothesis_graph.cascade_affected_ids(hypothesis_id)
                                node.mark_falsified(
                                    verification_result.reason or "terminal semantic rejection",
                                    1.0 - float(verification_result.confidence or 0.0),
                                )
                                tsdf_planner.hypothesis_graph.hypothesis_nodes.discard(hypothesis_id)
                                tsdf_planner.hypothesis_graph.falsified_nodes.add(hypothesis_id)
                                tsdf_planner.hypothesis_graph.cascade_delete(hypothesis_id)
                                dual_v2.revoke({
                                    "hypothesis_node_id": hypothesis_id,
                                    "cascade_deleted_nodes": affected,
                                }, f"hierarchical-reject:{evidence_id}:{hypothesis_id}")
                            dual_v2.release("terminal_semantics_rejected")
                            tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                            continue

                        if (
                            verification_result.verdict == VerificationVerdict.UNCERTAIN
                            and command.command == NavigatorCommandType.VERIFY
                            and dual_v2.execution is not None
                        ):
                            traversable = tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied
                            current_voxel = tsdf_planner.habitat2voxel(pts)[:2]

                            def verification_path_query(start, end):
                                path = grid_path(traversable, start, end)
                                if path is None:
                                    return float("inf"), None
                                cost = float(
                                    np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                                    * cfg.tsdf_grid_size
                                )
                                return cost, path

                            option = sample_verification_viewpoint(
                                center=np.asarray(dual_v2.selected_approach.get("look_at", dual_v2.execution.terminal)),
                                current=current_voxel,
                                occupied=tsdf_planner.occupied,
                                unoccupied=tsdf_planner.unoccupied,
                                island=tsdf_planner.island,
                                voxel_size=cfg.tsdf_grid_size,
                                path_query=verification_path_query,
                                entropy=1.0,
                            )
                            if option is not None and not np.array_equal(
                                np.rint(option.target_point), np.rint(current_voxel)
                            ):
                                hierarchical_verification_target = np.asarray(option.target_point)
                                tsdf_planner.target_point = hierarchical_verification_target.copy()
                                tsdf_planner.look_at_point = np.asarray(option.look_at_point)
                                topology_trace.write({
                                    "event": "hierarchical_verification_viewpoint_installed",
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "target": hierarchical_verification_target.tolist(),
                                    "look_at": np.asarray(option.look_at_point).tolist(),
                                })
                                continue
                        # Unresolved uncertainty and transport/parser errors are
                        # not negative evidence and never authorize a false stop.
                        hierarchical_navigator.release("terminal_verification_unresolved")
                        dual_v2.release("terminal_verification_unresolved")
                        tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                        continue

                    if unified_enabled and target_arrived and type(max_point_choice) == SnapShot:
                        intent = baseline_dual_dynamic.active_intent
                        if intent is None:
                            raise RuntimeError("Object arrival lost its authoritative intent")
                        if baseline_dual_dynamic.requires_terminal_verification(intent.task):
                            fresh_obs, _ = scene.get_observation(pts, angle=angle)
                            verdict, detail = verify_entity(subtask_metadata, fresh_obs["color_sensor"],
                                                            max_point_choice, cfg)
                            if verdict == "error":
                                subtask_execution.record_error("verification_request")
                            action = baseline_dual_dynamic.feedback(verdict, intent.intent_id,
                                f"{intent.intent_id}:verify:{intent.verification_views+1}", tsdf_planner)
                            topology_trace.write({"event": "hypothesis_aware_goal_verification",
                                "scene_id": scene_id, "episode_id": episode_id,
                                "subtask_id": subtask_id, "step": global_step,
                                "intent_id": intent.intent_id,
                                "entity_id": intent.task.entity_id, "verdict": verdict,
                                "action": action, "detail": detail})
                            if action == "new_view":
                                route = baseline_dual_dynamic.resolver.object_view(place_topology,
                                    tsdf_planner, scene.objects, max_point_choice.cluster[0],
                                    tsdf_planner.habitat2voxel(pts)[:2], cfg.planner.final_observe_distance,
                                    baseline_dual_dynamic.checked_views.get(intent.task.entity_id, ()))
                                if route.get("status") == "valid":
                                    intent.task.route = route
                                    if baseline_dual_dynamic.setup(tsdf_planner, pts, cfg):
                                        continue
                                baseline_dual_dynamic.release("no_unchecked_view", tsdf_planner.hypothesis_graph)
                            if action != "stop":
                                tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                                continue
                        elif intent.task.kind == ActionKind.REVISIT:
                            completed_revisit = baseline_dual_dynamic.promote_revisit(
                                intent.intent_id
                            )
                            if completed_revisit is None:
                                raise RuntimeError(
                                    "Object arrival could not promote its authoritative revisit"
                                )
                            topology_trace.write({
                                "event": "hypothesis_aware_revisit_promoted",
                                "scene_id": scene_id,
                                "episode_id": episode_id,
                                "subtask_id": subtask_id,
                                "step": global_step,
                                "revisit": completed_revisit,
                            })
                            # Fall through to the original HGR snapshot-arrival
                            # completion block with the same source-bound intent.

                    # (5.5) Phase II: Verify hypothesis node on frontier arrival
                    if (
                        semantic_critic is not None
                        and not (
                            selected_belief_action is not None
                            and selected_belief_action.action_type in (
                                BeliefActionType.VERIFY,
                                BeliefActionType.CONFIRM_APPROACH,
                            )
                        )
                        and ((dual_v2_enabled and v2_arrival_frontier is not None
                              and getattr(v2_arrival_frontier, "hypothesis_node_id", None) is not None)
                             or (not dual_v2_enabled and type(max_point_choice) == Frontier
                                 and target_arrived and getattr(max_point_choice, "hypothesis_node_id", None) is not None))
                    ):
                        detected_obj_names = [
                            scene.objects[oid]["class_name"]
                            for oid in all_added_obj_ids
                            if oid in scene.objects
                        ]
                        from src.query_vlm_hypothesis import verify_hypothesis_node_arrival
                        if dual_v2_enabled:
                            from src.query_vlm_hypothesis import verify_fresh_intent_arrival
                            v2_space_before = place_topology.to_trace_dict(include_graph=True)
                            verification_result = verify_fresh_intent_arrival(
                                v2_arrival_frontier, scene, tsdf_planner, semantic_critic, pts, angle)
                            topology_trace.write({"event": "v2_hypothesis_verification", "subtask_id": subtask_id,
                                "step": global_step, "result": verification_result})
                        else:
                            verification_result = verify_hypothesis_node_arrival(
                                frontier=max_point_choice,
                                scene=scene,
                                tsdf_planner=tsdf_planner,
                                semantic_critic=semantic_critic,
                                actual_rgb=rgb,
                                actual_depth=depth,
                                detected_objects=detected_obj_names,
                                apply_graph_update=(
                                    not active_topology_behavior_enabled
                                    or simple_memory_enabled
                                    or goal_topo_map_enabled
                                    or active_topology_cfg.get("correction_mode", "soft") == "hard"
                                ),
                            )
                        logging.info(f"[Phase II] Verification: {verification_result}")
                        if (dual_v2_enabled and dual_v2_cfg.get("cascade_correction", False)
                                and verification_result.get("is_falsified", False)):
                            feedback = dual_v2.revoke(verification_result, f"revoke:{subtask_id}:{global_step}")
                            feedback["spatial_facts_preserved"] = (
                                v2_space_before == place_topology.to_trace_dict(include_graph=True))
                            topology_trace.write({"event": "v2_hypothesis_retraction", "subtask_id": subtask_id,
                                "step": global_step, "feedback": feedback})
                            affected = set(feedback["affected_hypotheses"])
                            for frontier in tsdf_planner.frontiers:
                                if str(getattr(frontier, "hypothesis_node_id", "")) in affected:
                                    frontier.hypothesis_node_id = None
                                    frontier.semantic_dist = None
                            if feedback["intent_cancelled"]:
                                tsdf_planner.max_point = tsdf_planner.target_point = tsdf_planner.look_at_point = None
                        if (
                            active_topology_behavior_enabled
                            and not simple_memory_enabled
                            and not goal_topo_map_enabled
                        ):
                            verified_topo_id = getattr(max_point_choice, "topo_id", None)
                            if verified_topo_id is not None and not verification_result.get("skipped", False):
                                residual = float(verification_result.get("semantic_residual", 0.5))
                                evidence = VerificationEvidence(
                                    node_id=verified_topo_id,
                                    is_positive=not verification_result.get("is_falsified", False),
                                    strength=float(np.clip(abs(residual - 0.5) * 2.0, 0.1, 1.0)),
                                    residual=residual,
                                    reason=verification_result.get("explanation", ""),
                                    evidence_ref=f"verification:{subtask_id}:{global_step}:{verified_topo_id}",
                                )
                                correction_mode = active_topology_cfg.get("correction_mode", "soft")
                                pruned_ids = persistent_topology.apply_verification(
                                    verified_topo_id,
                                    evidence,
                                    global_step,
                                    correction_mode=correction_mode,
                                    hypothesis_graph=(
                                        tsdf_planner.hypothesis_graph
                                        if correction_mode == "soft" else None
                                    ),
                                )
                                if (evidence.is_positive and correction_mode == "soft"):
                                    hypothesis_node_id = verification_result.get("hypothesis_node_id")
                                    hypothesis_node = tsdf_planner.hypothesis_graph.nodes.get(hypothesis_node_id)
                                    if hypothesis_node is not None and hypothesis_node.semantic_dist is not None:
                                        predicted_class = hypothesis_node.semantic_dist.categories[
                                            int(np.argmax(hypothesis_node.semantic_dist.probabilities))
                                        ]
                                        semantic_critic._apply_verification(
                                            hypothesis_node, predicted_class, residual
                                        )
                                topology_trace.write({
                                    "event": "verification",
                                    "subtask_id": subtask_id,
                                    "step": global_step,
                                    "topo_id": verified_topo_id,
                                    "evidence": evidence,
                                    "pruned_ids": pruned_ids,
                                })
                        if verification_result.get("is_falsified", False):
                            if cfg.hypothesis.get("enable_cascade_deletion", True):
                                tsdf_planner.hypothesis_graph.prune_falsified_nodes()
                            elif cfg.hypothesis.get("enable_local_delete_only", False):
                                # Local delete only: remove just the falsified node, no cascade
                                node_id = verification_result.get("hypothesis_node_id")
                                if node_id and node_id in tsdf_planner.hypothesis_graph.nodes:
                                    tsdf_planner.hypothesis_graph._remove_node(node_id)
                            logging.info(
                                f"[Phase II] Hypothesis graph stats: {tsdf_planner.get_hypothesis_graph_statistics()}"
                            )

                    if dual_v2_enabled and v2_arrival_frontier is not None:
                        topology_trace.write({"event": "v2_intent_finished", "subtask_id": subtask_id,
                            "step": global_step, "feedback": dual_v2.release("terminal_arrived")})

                    # sanity check about objects, scene graph, snapshots, ...
                    scene.sanity_check(cfg=cfg)

                    if active_topology_enabled:
                        current_view = persistent_topology.current_view
                        place_shadow_replay = None
                        if (
                            place_topology_enabled
                            and place_topology_stage == "shadow"
                        ):
                            frozen_navigation_decision = (
                                _place_shadow_navigation_signature(
                                    max_point_choice,
                                    tsdf_planner.frontiers,
                                    selected_goal_topo_action,
                                    selected_belief_action,
                                    tsdf_planner.target_point,
                                    tsdf_planner.look_at_point,
                                    navigation_target_override,
                                    navigation_look_at_override,
                                    target_arrived,
                                )
                            )
                            shadow_navigation_decision = (
                                place_topology.shadow_navigation_passthrough(
                                    frozen_navigation_decision
                                )
                            )
                            if (
                                shadow_navigation_decision
                                != frozen_navigation_decision
                            ):
                                raise AssertionError(
                                    "Place shadow changed a frozen navigation "
                                    "decision"
                                )
                            place_shadow_replay = {
                                "stage": place_topology_stage,
                                "voxel_size": float(cfg.tsdf_grid_size),
                                "place_spacing_m": (
                                    place_topology.place_spacing_m
                                ),
                                "pose": {
                                    "position": np.asarray(
                                        current_voxel, dtype=float
                                    )[:2].tolist(),
                                    "step": global_step,
                                    "observation_id": f"pose:{global_step}",
                                    "verified_paths_m": dict(sorted(
                                        place_verified_paths.items()
                                    )),
                                    "expected_assignment": (
                                        place_assignment.to_trace_dict()
                                    ),
                                },
                                "known_connections_added": (
                                    place_known_connections_added
                                ),
                                "observations": place_replay_observations,
                                "frontiers": place_replay_frontiers,
                                "frozen_navigation_decision": (
                                    frozen_navigation_decision
                                ),
                                "shadow_navigation_decision": (
                                    shadow_navigation_decision
                                ),
                                "graph_statistics": (
                                    place_topology.get_statistics()
                                ),
                                "graph_audit": place_topology.audit(),
                            }
                        topology_trace.write({
                            "event": "step",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "step": global_step,
                            "goal_type": goal_context.goal_type,
                            "mode": active_topology_policy.mode.value,
                            "active_view_enabled": active_topology_policy.use_active_view,
                            "recovery_level": persistent_topology.recovery.level,
                            "candidates": (
                                current_view.all_candidates if current_view is not None else []
                            ),
                            "selected_topo_id": (
                                selected_goal_topo_action.topo_node_id
                                if selected_goal_topo_action is not None
                                else selected_belief_action.node_id
                                if selected_belief_action is not None
                                else (
                                    getattr(max_point_choice, "topo_id", None)
                                    if type(max_point_choice) == Frontier else None
                                )
                            ),
                            "goal_topology_actions": (
                                goal_topo_task_view.actions
                                if goal_topo_task_view is not None else []
                            ),
                            "selected_goal_topo_action": selected_goal_topo_action,
                            "belief_hypotheses": belief_hypothesis_set,
                            "belief_actions": belief_action_candidates,
                            "selected_belief_action": selected_belief_action,
                            "target_arrived": bool(target_arrived),
                            "topology_route_arrived": bool(
                                topology_route_arrived_this_step
                            ),
                            "topology_route_commitment": (
                                None
                                if committed_topo_route is None
                                else {
                                    "snapshot_image": committed_topo_route[
                                        "snapshot_image"
                                    ],
                                    "target_node_id": committed_topo_route[
                                        "target_node_id"
                                    ],
                                    "created_step": committed_topo_route[
                                        "created_step"
                                    ],
                                    "waypoints_completed": committed_topo_route[
                                        "waypoints_completed"
                                    ],
                                }
                            ),
                            "confirmation_events": (
                                belief_memory.pop_confirmation_events()
                                if belief_v2_enabled else []
                            ),
                            "place_topology_shadow": (
                                place_topology.to_trace_dict(
                                    include_graph=False
                                )
                                if place_topology_enabled else None
                            ),
                            "place_assignment": (
                                place_assignment.to_trace_dict()
                                if place_topology_enabled else None
                            ),
                            "place_shadow_replay": place_shadow_replay,
                            "place_route_plan": (
                                place_route_plan.to_trace_dict()
                                if place_route_plan is not None else None
                            ),
                            "place_route_commitment": (
                                place_route_commitment
                                if place_route_only_enabled else None
                            ),
                        })

                    if cfg.save_visualization:
                        # save the top-down visualization
                        logger.save_topdown_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            subtask_metadata=subtask_metadata,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                            fig=fig,
                        )
                        # save the visualization of vlm's choice at each step
                        logger.save_frontier_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            tsdf_planner=tsdf_planner,
                            max_point_choice=max_point_choice,
                            global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
                        )

                    if unified_enabled and target_arrived and type(max_point_choice) == Frontier:
                        baseline_dual_dynamic.release("frontier_observation_complete", tsdf_planner.hypothesis_graph)

                    # (6) Check if the agent has arrived at the target to finish the question
                    phase_c_verified = False
                    if phase_c_enabled and target_arrived and not topology_route_arrived_this_step:
                        if phase_c.active is not None:
                            current_position = tsdf_planner.habitat2voxel(pts)[:2]
                            look_at = phase_c.active.look_at
                            verification_geometry = {
                                "candidate": phase_c.active.to_trace_dict(),
                                "actual_position": np.asarray(current_position).tolist(),
                                "actual_yaw": float(angle),
                                "terminal_error_m": float(np.linalg.norm(
                                    np.asarray(current_position) - phase_c.active.terminal) * cfg.tsdf_grid_size),
                            }
                            verification = None
                            fresh_obs, _ = scene.get_observation(pts, angle=angle)
                            verification_image = os.path.join(
                                logger.subtask_object_observe_dir, f"phase_c_observation_{global_step}.png"
                            )
                            plt.imsave(verification_image, fresh_obs["color_sensor"])
                            if phase_c.active.intent == "Verify":
                                verdict, verification = verify_place_goal(
                                    subtask_metadata, fresh_obs["color_sensor"], phase_c_selected_crop, cfg,
                                )
                                phase_c_verified = verdict == "matched"
                                outcome = "verification_" + verdict
                            else:
                                outcome = "frontier_observed"
                            feedback = phase_c.finish(outcome, global_step, current_position, look_at)
                            topology_trace.write({
                                "event": "phase_c_execution_feedback", "subtask_id": subtask_id,
                                "step": global_step, "feedback": feedback,
                                "verification": verification,
                                "verification_geometry": verification_geometry,
                                "observation_id": f"phase_c:{subtask_id}:{global_step}",
                                "observation_image": verification_image,
                            })
                            phase_c_selected_crop = None
                            place_route_commitment = None
                            place_route_snapshot_source = None
                            place_route_execution.reset()
                            if verification is not None and verification.get("action") == "error":
                                subtask_execution.terminate(
                                    ExecutionStatus.ERROR,
                                    "verification_request_failed",
                                    error_category="verification",
                                )
                                break  # Request exhaustion is an explicit task error, not evidence against other objects.
                            if not phase_c_verified:
                                tsdf_planner.max_point = None
                                tsdf_planner.target_point = None
                                tsdf_planner.look_at_point = None
                    if (
                        type(max_point_choice) == SnapShot
                        and (not phase_c_enabled or phase_c_verified)
                        and target_arrived
                        and not topology_route_arrived_this_step
                        and not (
                            selected_belief_action is not None
                            and selected_belief_action.action_type
                            == BeliefActionType.VERIFY
                        )
                    ):
                        # when the target is a snapshot, and the agent arrives at the target
                        # we consider the subtask is finished, take an observation and save the chosen target snapshot
                        obs, _ = scene.get_observation(pts, angle=angle)
                        rgb = obs["color_sensor"]
                        plt.imsave(
                            os.path.join(
                                logger.subtask_object_observe_dir, f"target.png"
                            ),
                            rgb,
                        )

                        snapshot_filename = max_point_choice.image.split(".")[0]
                        snapshot_source = os.path.join(eps_snapshot_dir, max_point_choice.image)
                        snapshot_destination = os.path.join(logger.subtask_object_observe_dir, f"snapshot_{snapshot_filename}.png")
                        if os.path.isfile(snapshot_source):
                            import shutil
                            shutil.copyfile(snapshot_source, snapshot_destination)
                        elif max_point_choice.image in scene.all_observations:
                            # A warm checkpoint retains pixels, not the export
                            # run's disposable visualization directory.
                            plt.imsave(snapshot_destination, scene.all_observations[max_point_choice.image])
                        else:
                            logging.warning("Selected snapshot pixels unavailable: %s", max_point_choice.image)

                        task_success = True
                        subtask_execution.terminate(
                            ExecutionStatus.COMPLETED,
                            "verified_confirmation"
                            if unified_enabled
                            and baseline_dual_dynamic.requires_terminal_verification(
                                baseline_dual_dynamic.active_intent.task
                                if baseline_dual_dynamic.active_intent is not None else None
                            )
                            else "snapshot_terminal_arrived",
                        )
                        break

                if semantic_critic is not None:
                    subtask_execution.record_error(
                        "feature_residual",
                        int(semantic_critic.stats.get("feature_residual_errors", 0))
                        - critic_feature_errors_start,
                    )
                execution_outcome = subtask_execution.finalize(task_success)
                dual_dynamic_diagnostics = (
                    baseline_dual_dynamic.diagnostics()
                    if baseline_dual_dynamic_enabled else None
                )

                if phase_c_enabled:
                    if phase_c.active is not None:
                        phase_c.finish("subtask_budget_ended", global_step)
                    topology_trace.write({
                        "event": "phase_c_subtask_summary", "subtask_id": subtask_id,
                        "step": global_step, "task_success": task_success,
                        "view": phase_c.trace(),
                    })
                if place_route_commitment is not None:
                    topology_trace.write({
                        "event": "place_route_commitment_released",
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "subtask_id": subtask_id,
                        "step": global_step,
                        "commitment": place_route_commitment,
                        "reason": "subtask_ended_before_terminal_segment",
                    })
                if committed_topo_route is not None:
                    topology_trace.write({
                        "event": "topology_route_commitment_cancelled",
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "subtask_id": subtask_id,
                        "step": global_step,
                        "snapshot_image": committed_topo_route[
                            "snapshot_image"
                        ],
                        "target_node_id": committed_topo_route[
                            "target_node_id"
                        ],
                        "waypoints_completed": committed_topo_route[
                            "waypoints_completed"
                        ],
                        "reason": "subtask_end",
                    })
                    persistent_topology.stats.setdefault(
                        "topology_route_commitments_cancelled", 0
                    )
                    persistent_topology.stats[
                        "topology_route_commitments_cancelled"
                    ] += 1
                    committed_topo_route = None

                if hierarchical_enabled:
                    if hierarchical_active and not task_success:
                        hierarchical_navigator.on_event(NavigationEvent(
                            NavigationEventType.BUDGET_ENDED,
                            global_step,
                            "subtask_ended",
                        ))
                    topology_trace.write({
                        "event": "hierarchical_goal_end",
                        "subtask_id": subtask_id,
                        "step": global_step,
                        "task_success": bool(task_success),
                        "navigator": hierarchical_navigator.to_trace_dict(),
                    })
                if baseline_dual_dynamic_enabled:
                    if unified_enabled:
                        baseline_dual_dynamic.release("subtask_end", tsdf_planner.hypothesis_graph)
                    topology_trace.write({
                        "event": "dual_dynamic_goal_end",
                        "subtask_id": subtask_id,
                        "step": global_step,
                        "task_success": bool(task_success),
                        "execution_outcome": execution_outcome,
                        "navigator": baseline_dual_dynamic.to_trace_dict(),
                    })
                if dual_v2_enabled:
                    topology_trace.write({"event": "v2_goal_end", "subtask_id": subtask_id,
                        "step": global_step, "feedback": dual_v2.release("subtask_end"),
                        "observed_view_count": len(dual_v2.goal.observation_coverage),
                        "revoked_hypotheses": sorted(dual_v2.goal.revoked_hypotheses),
                        "cache_hits": dual_v2.cache.hits-dual_v2.cache_start[0],
                        "cache_misses": dual_v2.cache.misses-dual_v2.cache_start[1],
                        "cache_hits_episode": dual_v2.cache.hits, "cache_misses_episode": dual_v2.cache.misses})
                    topology_trace.write({"event": "v2_cache_statistics", "subtask_id": subtask_id,
                        "step": global_step, "statistics": dual_v2.cache.statistics()})
                # get some statistics
                selected_object_ids = (
                    list(max_point_choice.cluster)
                    if isinstance(max_point_choice, SnapShot) else []
                )
                canonical_target_ids = sorted({
                    resolved for obj_id in target_obj_ids_estimate
                    if (resolved := _canonical_object_id(
                        obj_id, scene.object_id_aliases
                    )) is not None
                })
                canonical_selected_ids = sorted({
                    resolved for obj_id in selected_object_ids
                    if (resolved := _canonical_object_id(
                        obj_id, scene.object_id_aliases
                    )) is not None
                })
                snapshot_id_match = bool(
                    set(canonical_target_ids) & set(canonical_selected_ids)
                )
                if task_success and snapshot_id_match:
                    success_by_snapshot = True
                    logging.info(
                        f"Success: {target_obj_ids_estimate} in chosen snapshot {max_point_choice.image}!"
                    )
                else:
                    success_by_snapshot = False
                    logging.info(
                        f"Fail: {target_obj_ids_estimate} not in chosen snapshot!"
                    )
                snapshot_mapping_diagnostics = {
                    "gt_object_ids": [str(item) for item in subtask_metadata["goal_obj_ids"]],
                    "mapping_votes": {
                        str(gt_id): [str(item) for item in mapped_ids]
                        for gt_id, mapped_ids in goal_obj_ids_mapping.items()
                    },
                    "raw_target_ids": [str(item) for item in target_obj_ids_estimate],
                    "canonical_target_ids": canonical_target_ids,
                    "raw_selected_ids": [str(item) for item in selected_object_ids],
                    "canonical_selected_ids": canonical_selected_ids,
                    "aliases": {
                        str(old): None if new is None else str(new)
                        for old, new in scene.object_id_aliases.items()
                    },
                    "status": (
                        "mapping_unavailable" if not canonical_target_ids
                        else "matched" if snapshot_id_match
                        else "selected_id_mismatch"
                    ),
                }
                # calculate the distance to the nearest view point
                agent_subtask_distance = calc_agent_subtask_distance(
                    pts, subtask_metadata["viewpoints"], scene.pathfinder
                )
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance = True
                    logging.info(
                        f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!"
                    )
                else:
                    success_by_distance = False
                    logging.info(
                        f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!"
                    )

                if belief_v3_enabled and belief_memory is not None:
                    current_goal_key = normalize_goal_key(goal_context.category)
                    state = belief_memory.confirmations.get(current_goal_key)
                    if state is not None and state.active:
                        belief_memory.resolve_confirmation(
                            current_goal_key, "subtask_end"
                        )
                    pending_confirmation_events = (
                        belief_memory.pop_confirmation_events()
                    )
                    if pending_confirmation_events:
                        topology_trace.write({
                            "event": "subtask_confirmation_events",
                            "scene_id": scene_id,
                            "episode_id": episode_id,
                            "subtask_id": subtask_id,
                            "events": pending_confirmation_events,
                        })

                subtask_vlm_end = get_vlm_telemetry()
                final_selected_topo_id = (
                    selected_goal_topo_action.topo_node_id
                    if selected_goal_topo_action is not None
                    else selected_belief_action.node_id
                    if selected_belief_action is not None
                    else getattr(max_point_choice, "topo_id", None)
                )
                timing_result = stage_timer.snapshot()
                active_timer.reset(stage_timer_token)
                logger.log_subtask_result(
                    timing=timing_result,
                    success_by_snapshot=success_by_snapshot,
                    success_by_distance=success_by_distance,
                    subtask_id=subtask_id,
                    gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
                    goal_type=goal_type,
                    n_filtered_snapshots=n_filtered_snapshots,
                    n_total_snapshots=len(scene.snapshots),
                    n_total_frames=len(scene.frames),
                    agent_subtask_distance=agent_subtask_distance,
                    subtask_steps=max(cnt_step + 1, 0),
                    subtask_frames=len(scene.frames) - subtask_frame_start,
                    vlm_telemetry=_vlm_telemetry_delta(
                        subtask_vlm_end, subtask_vlm_start
                    ),
                    final_choice=_choice_diagnostic(
                        max_point_choice, tsdf_planner.frontiers
                    ),
                    snapshot_mapping_diagnostics=snapshot_mapping_diagnostics,
                    selected_topo_id=final_selected_topo_id,
                    navigation_diagnostics=(
                        {
                            "navigation_backend": navigation_backend,
                            "decision_source": decision_source,
                            "hierarchical": (
                                hierarchical_navigator.to_trace_dict()
                                if hierarchical_enabled else None
                            ),
                            "dual_dynamic": (
                                dual_dynamic_diagnostics
                            ),
                            "place_route_stats": (
                                dict(place_route_stats)
                                if place_route_only_enabled else None
                            ),
                        }
                    ),
                    execution_outcome=execution_outcome,
                )
                if dual_v2_enabled:
                    topology_trace.write({"event": "v2_goal_diagnostic", "subtask_id": subtask_id,
                        "step": global_step, "goal_type": goal_type,
                        "choice": _choice_diagnostic(max_point_choice, tsdf_planner.frontiers),
                        "target_arrived": bool(target_arrived), "task_success": bool(task_success),
                        "success_by_snapshot": bool(success_by_snapshot),
                        "success_by_distance": bool(success_by_distance),
                        "final_distance_m": float(agent_subtask_distance),
                        "meaning": "original HGR stopping protocol; arrival alone does not establish a match"})

                logging.info(f"Scene graph of question {subtask_id}:")
                logging.info(f"Question: {subtask_metadata['question']}")
                logging.info(f"Task type: {subtask_metadata['task_type']}")
                logging.info(f"Answer: {subtask_metadata['class']}")
                scene.print_scene_graph()

                if memory_mode in ("cold", "warm") and memory_cfg.get("single_subtask", False):
                    break

                if not cfg.save_visualization:
                    # clear up the stored images to save memory
                    os.system(
                        f"rm -r {os.path.join(str(cfg.output_dir), f'{subtask_id}')}"
                    )

            # save the results at the end of each episode
            logger.save_results()

            if active_topology_enabled:
                if belief_v2_enabled and belief_memory is not None:
                    for confirmation_goal_key, state in list(
                        belief_memory.confirmations.items()
                    ):
                        if state.active:
                            belief_memory.resolve_confirmation(
                                confirmation_goal_key, "episode_end"
                            )
                    topology_trace.write({
                        "event": "episode_confirmation_events",
                        "scene_id": scene_id,
                        "episode_id": episode_id,
                        "events": belief_memory.pop_confirmation_events(),
                    })
                episode_vlm_end = get_vlm_telemetry()
                topology_trace.save_summary({
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "topology": persistent_topology.get_statistics(),
                    "goal_belief": (
                        belief_memory.get_statistics()
                        if belief_memory is not None else None
                    ),
                    "dynamic_goal_topology": (
                        dynamic_goal_topology.get_statistics()
                        if dynamic_goal_topology is not None else None
                    ),
                    "place_topology": (
                        place_topology.to_trace_dict(include_graph=True)
                        if place_topology_enabled else None
                    ),
                    "place_route_only": (
                        dict(place_route_stats)
                        if place_route_only_enabled else None
                    ),
                    "vlm_telemetry_start": episode_vlm_start,
                    "vlm_telemetry_end": episode_vlm_end,
                })

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    completion_dir = Path(cfg.output_dir) / "provenance" / f"split_{split}"
    completion_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = completion_dir / "run_fingerprint.json"
    fingerprint = None
    if fingerprint_path.is_file():
        with fingerprint_path.open("r", encoding="utf-8") as handle:
            fingerprint = json.load(handle)
    completion = {
        "schema_version": 1,
        "split": int(split),
        "completed_at_unix": time.time(),
        "subtask_count": len(logger.success_by_snapshot),
        "orchestration_repeat": cfg.get("orchestration_repeat"),
        "orchestration_config_sha256": cfg.get(
            "orchestration_config_sha256"
        ),
        "fingerprint": fingerprint,
    }
    marker_tmp = completion_dir / "completed.json.tmp"
    with marker_tmp.open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2)
        handle.write("\n")
    os.replace(marker_tmp, completion_dir / "completed.json")

    logging.info(f"All scenes finish")


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument("--run_name", help="unique output directory name for a manual run", default=None)
    parser.add_argument("--record_run_fingerprint", action="store_true")
    parser.add_argument("--orchestration-config-sha256", default=None)
    parser.add_argument("--orchestration-repeat", type=int, default=None)
    parser.add_argument("--max-subtasks", type=int, default=None)
    parser.add_argument("--max-steps-per-subtask", type=int, default=None)
    parser.add_argument(
        "--episode_manifest",
        help="optional fixed scene/episode manifest; overrides ratio selection",
        default=None,
        type=str,
    )
    args = parser.parse_args()
    cfg = _load_config(args.cfg_file)
    if args.run_name is not None:
        if Path(args.run_name).name != args.run_name or args.run_name in (".", "..", ""):
            raise ValueError("run_name must be a single directory name")
        cfg.exp_name = args.run_name
    if args.episode_manifest:
        cfg.episode_manifest = args.episode_manifest
    if args.record_run_fingerprint:
        cfg.record_run_fingerprint = True
    if args.orchestration_config_sha256 is not None:
        cfg.orchestration_config_sha256 = args.orchestration_config_sha256
    if args.orchestration_repeat is not None:
        cfg.orchestration_repeat = args.orchestration_repeat
    if args.max_subtasks is not None:
        if args.max_subtasks < 1:
            raise ValueError("max-subtasks must be positive")
        cfg.diagnostic_max_subtasks = args.max_subtasks
    if args.max_steps_per_subtask is not None:
        if args.max_steps_per_subtask < 1:
            raise ValueError("max-steps-per-subtask must be positive")
        cfg.diagnostic_max_steps_per_subtask = args.max_steps_per_subtask
    OmegaConf.resolve(cfg)
    validate_baseline_backend(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    logging.info(f"VLM model: {cfg.get('vlm_model', 'openai/gpt-4o')}")
    main(
        cfg,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        split=args.split,
        episode_manifest=args.episode_manifest,
    )
