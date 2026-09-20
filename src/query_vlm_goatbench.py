import logging
from time import perf_counter
from typing import Tuple, Optional, Union
import random
import numpy as np

from src.eval_utils_gpt_goatbench import explore_step
from src.utils import resize_image
from src.tsdf_planner import TSDFPlanner, SnapShot, Frontier
from src.scene_goatbench import Scene
from src.hgr_topology_context import build_hgr_topology_context



def _infer_target_category(question: str):
    """Simple heuristic to guess target room category from question text."""
    q = question.lower()
    room_keywords = {
        "bedroom": ["bed", "sleep", "bedroom", "pillow", "nightstand"],
        "bathroom": ["bath", "shower", "toilet", "bathroom", "sink"],
        "kitchen": ["cook", "kitchen", "stove", "fridge", "refrigerator", "oven"],
        "living_room": ["living", "sofa", "couch", "tv", "television"],
        "dining_room": ["dining", "eat", "table"],
        "office": ["desk", "office", "computer", "monitor"],
    }
    for room, keywords in room_keywords.items():
        if any(kw in q for kw in keywords):
            return room
    return None


def query_vlm_for_response(
    subtask_metadata: dict,
    scene: Scene,
    tsdf_planner: TSDFPlanner,
    rgb_egocentric_views: list,
    cfg,
    active_topo_view=None,
    excluded_object_ids=None,
    allowed_snapshot_images=None,
    allowed_frontier_indices=None,
    frontier_only: bool = False,
    verbose: bool = False,
    place_topology=None,
    navigation_position=None,
    hierarchical_navigator=None,
    eligible_objects_by_snapshot=None,
    navigation_task_context=None,
    request_purpose=None,
) -> Optional[Tuple[Union[SnapShot, Frontier], int]]:
    # prepare input for vlm
    runtime = getattr(place_topology, "dual_topo_runtime", None) if place_topology is not None else None
    if runtime is not None:
        goal, step, decision = runtime.cache.context
        runtime.cache.context = (goal, step, (decision or 0) + 1)
    step_dict = {}
    step_dict["frontier_only"] = bool(frontier_only)
    if navigation_task_context is not None:
        step_dict["navigation_task_context"] = navigation_task_context
    if request_purpose is not None:
        step_dict["request_purpose"] = request_purpose

    # prepare snapshots
    object_id_to_name = {
        obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()
    }
    step_dict["obj_map"] = object_id_to_name

    step_dict["snapshot_objects"] = {}
    step_dict["snapshot_imgs"] = {}
    step_dict["snapshot_source_indices"] = []
    step_dict["use_full_obj_list"] = cfg.use_full_obj_list
    excluded_object_ids = {str(item) for item in (excluded_object_ids or set())}
    allowed_snapshot_images = (
        None
        if allowed_snapshot_images is None
        else {str(item) for item in allowed_snapshot_images}
    )
    for source_index, (rgb_id, snapshot) in enumerate(scene.snapshots.items()):
        if (
            allowed_snapshot_images is not None
            and str(rgb_id) not in allowed_snapshot_images
            and str(snapshot.image) not in allowed_snapshot_images
        ):
            continue
        if any(str(obj_id) in excluded_object_ids for obj_id in snapshot.cluster):
            continue
        visible_cluster = [
            obj_id for obj_id in snapshot.cluster
            if str(obj_id) not in excluded_object_ids
        ]
        if eligible_objects_by_snapshot is not None:
            allowed = set(eligible_objects_by_snapshot.get(str(rgb_id), ()))
            visible_cluster = [obj_id for obj_id in visible_cluster if obj_id in allowed]
            if not visible_cluster:
                continue
        resized_rgb = resize_image(
            scene.all_observations[rgb_id], cfg.prompt_h, cfg.prompt_w
        )

        step_dict["snapshot_objects"][rgb_id] = visible_cluster
        step_dict["snapshot_source_indices"].append(source_index)
        step_dict["snapshot_imgs"][rgb_id] = {
            "full_img": resized_rgb,
            "object_crop": [],
        }

        # crop the snapshot to contain only the objects in the snapshot
        if cfg.use_full_obj_list:
            selected_bbox_idx = [
                idx
                for idx in range(len(snapshot.visual_prompt))
                if snapshot.visual_prompt[idx].data["obj_id"][0]
                in snapshot.full_obj_list.keys()
                and (eligible_objects_by_snapshot is None or
                     snapshot.visual_prompt[idx].data["obj_id"][0] in visible_cluster)
                and str(snapshot.visual_prompt[idx].data["obj_id"][0])
                not in excluded_object_ids
            ]
        else:
            selected_bbox_idx = [
                idx
                for idx in range(len(snapshot.visual_prompt))
                if snapshot.visual_prompt[idx].data["obj_id"][0] in visible_cluster
            ]
        selected_bbox = snapshot.visual_prompt[selected_bbox_idx].xyxy.copy()
        selected_obj_ids = [
            snapshot.visual_prompt[idx].data["obj_id"][0] for idx in selected_bbox_idx
        ]

        # scale the bbox
        H, W = scene.all_observations[rgb_id].shape[:2]
        scale_h, scale_w = cfg.prompt_h / H, cfg.prompt_w / W
        selected_bbox[:, 0] *= scale_w
        selected_bbox[:, 2] *= scale_w
        selected_bbox[:, 1] *= scale_h
        selected_bbox[:, 3] *= scale_h
        selected_bbox = selected_bbox.astype(int)

        # get the image crop for each object
        for obj_id, bbox in zip(selected_obj_ids, selected_bbox):
            step_dict["snapshot_imgs"][rgb_id]["object_crop"].append(
                {
                    "obj_class": object_id_to_name[obj_id],
                    "obj_id": obj_id,
                    "crop": resized_rgb[bbox[1] : bbox[3], bbox[0] : bbox[2]],
                }
            )

    # ActiveTopo presents a goal-conditioned, utility-sorted frontier subset.
    # Keep the projected->source mapping explicit so a VLM index can never be
    # interpreted against the original list by accident.
    if active_topo_view is None:
        frontier_source_indices = list(range(len(tsdf_planner.frontiers)))
        frontier_candidates = tsdf_planner.frontiers
        active_candidate_by_source = {}
    else:
        frontier_source_indices = [
            candidate.source_frontier_index
            for candidate in active_topo_view.candidates
        ]
        frontier_candidates = [
            tsdf_planner.frontiers[index] for index in frontier_source_indices
        ]
        active_candidate_by_source = {
            candidate.source_frontier_index: candidate
            for candidate in active_topo_view.candidates
        }
    if allowed_frontier_indices is not None:
        allowed_frontier_indices = {
            int(index) for index in allowed_frontier_indices
        }
        filtered = [
            (source_index, frontier)
            for source_index, frontier
            in zip(frontier_source_indices, frontier_candidates)
            if source_index in allowed_frontier_indices
        ]
        frontier_source_indices = [item[0] for item in filtered]
        frontier_candidates = [item[1] for item in filtered]
    if active_topo_view is not None and active_topo_view.topology_context:
        step_dict["goal_topology_context"] = active_topo_view.topology_context

    # prepare frontier with hypothesis semantics
    step_dict["frontier_imgs"] = [
        frontier.feature for frontier in frontier_candidates
    ]
    step_dict["frontier_stable_ids"] = [
        str(getattr(frontier, "topo_id", getattr(frontier, "frontier_id", index)))
        for index, frontier in enumerate(frontier_candidates)
    ]
    if place_topology is not None:
        step_dict["hgr_topology_context"] = build_hgr_topology_context(
            place_topology, scene.snapshots, frontier_candidates,
        )
        logging.info("HGR topology selection context: %s", step_dict["hgr_topology_context"])
        if cfg.get("hgr_topology_fusion", {}).get("object_approach", False):
            approach_build_started = perf_counter()
            from src.object_approach import resolve_object_approach, resolve_terminal_route
            routes = {}
            for image, entry in step_dict["snapshot_imgs"].items():
                for crop in entry["object_crop"]:
                    obj_id = crop["obj_id"]
                    if obj_id not in routes:
                        routes[obj_id] = resolve_object_approach(
                            place_topology, tsdf_planner, scene.objects, obj_id,
                            navigation_position, cfg.planner.final_observe_distance)
                step_dict["hgr_topology_context"]["snapshots"][image] = {
                    "objects": [routes[c["obj_id"]] for c in entry["object_crop"]]}
            mask = tsdf_planner.island & tsdf_planner.unoccupied & ~tsdf_planner.occupied
            step_dict["hgr_topology_context"]["object_approach"] = True
            runtime = getattr(place_topology, "dual_topo_runtime", None)
            if runtime is not None:
                from src.object_approach import resolve_frontier_approach
                step_dict["hgr_topology_context"]["frontiers"] = [resolve_frontier_approach(
                    place_topology, tsdf_planner, f, navigation_position,
                    cfg.planner.final_observe_distance) for f in frontier_candidates]
                step_dict["hgr_topology_context"]["shared_route"] = True
            else:
                step_dict["hgr_topology_context"]["frontiers"] = [resolve_terminal_route(
                    place_topology, mask, navigation_position, f.position[:2],
                    cfg.planner.final_observe_distance) for f in frontier_candidates]
            if runtime is not None and runtime.config.get("goal_state", False):
                from src.hgr_dual_topo import Candidate
                # All entries retain source IDs; crop labels are assigned later.
                for image, entry in step_dict["snapshot_imgs"].items():
                    for crop in entry["object_crop"]:
                        entity = str(crop["obj_id"])
                        candidate = runtime.goal.register(Candidate(
                            f"TARGET_APPROACH:{entity}", entity, "TARGET_APPROACH",
                            routes[crop["obj_id"]].get("approach_place_id"),
                            {f"snapshot:{image}:object:{entity}"}))
                        if runtime.config.get("goal_context", False):
                            routes[crop["obj_id"]]["goal_state"] = candidate.context()
                for frontier, row in zip(frontier_candidates, step_dict["hgr_topology_context"]["frontiers"]):
                    candidate = runtime.candidate(frontier, row)
                    if runtime.config.get("goal_context", False):
                        row["goal_state"] = candidate.context()
                step_dict["hgr_topology_context"]["shared_route"] = True
            logging.info("V2 candidate geometry: objects=%d frontiers=%d build_seconds=%.6f route_query_seconds=%.6f",
                         len(routes), len(frontier_candidates), perf_counter()-approach_build_started,
                         sum(r.get("route_query_seconds",0.) for r in routes.values()) +
                         sum(r.get("route_query_seconds",0.) for r in step_dict["hgr_topology_context"]["frontiers"]))
    step_dict["frontier_semantic_predictions"] = []
    for source_index, frontier in zip(frontier_source_indices, frontier_candidates):
        runtime = getattr(place_topology, "dual_topo_runtime", None) if place_topology is not None else None
        if (runtime is not None and runtime.config.get("goal_state", False)
                and str(getattr(frontier, "hypothesis_node_id", "")) in runtime.goal.revoked_hypotheses):
            frontier.semantic_dist = None
            frontier.hypothesis_node_id = None
        if hasattr(frontier, "semantic_dist") and frontier.semantic_dist is not None:
            top_cats, top_probs = frontier.semantic_dist.get_top_k(3)
            prediction = {
                "predicted_categories": top_cats,
                "probabilities": [float(p) for p in top_probs],
                "entropy": float(frontier.semantic_dist.entropy),
            }
        else:
            prediction = None
        candidate = active_candidate_by_source.get(source_index)
        if candidate is not None and active_topo_view.include_prompt_metadata:
            if prediction is None:
                prediction = {
                    "predicted_categories": [], "probabilities": [], "entropy": 0.0
                }
            active_metadata = {
                "topo_id": candidate.topo_node_id,
                "state": candidate.state.value,
                "confidence": candidate.confidence,
                "accessibility": candidate.accessibility,
                "utility": candidate.utility,
            }
            if active_topo_view.uses_goal_relevance:
                active_metadata["relevance"] = candidate.relevance
            prediction["active_topology"] = active_metadata
        step_dict["frontier_semantic_predictions"].append(prediction)

    # Sort frontiers by exploration score if hypothesis refinement is enabled
    if cfg.get("hypothesis", {}).get("enable_hypothesis_refinement", False):
        target_category = _infer_target_category(subtask_metadata["question"])
        preserve_original_scoring = (
            active_topo_view is None
            or not active_topo_view.include_prompt_metadata
        )
        if preserve_original_scoring and target_category and len(tsdf_planner.frontiers) > 0:
            scored = tsdf_planner.rank_frontiers_by_exploration_score(target_category, None)
            score_by_source = dict(scored)
            step_dict["frontier_scores"] = {
                projected_index: score_by_source[source_index]
                for projected_index, source_index in enumerate(frontier_source_indices)
                if source_index in score_by_source
            }

    # prepare egocentric views
    if cfg.egocentric_views:
        step_dict["egocentric_views"] = rgb_egocentric_views
        step_dict["use_egocentric_views"] = True

    # prepare other metadata
    step_dict["question"] = subtask_metadata["question"]
    step_dict["task_type"] = subtask_metadata["task_type"]
    step_dict["class"] = subtask_metadata["class"]
    step_dict["image"] = subtask_metadata["image"]

    # Query either the legacy hard-choice selector or the route-blind semantic
    # candidate layer. The latter lets the navigator apply geometry only after
    # semantic identity gating.
    hierarchical_decision = None
    if hierarchical_navigator is not None:
        from dataclasses import replace
        from src.eval_utils_gpt_goatbench import assess_navigation_step
        from src.hierarchical_navigator import SemanticCandidate

        if place_topology is None or not step_dict.get("hgr_topology_context", {}).get("object_approach"):
            raise ValueError("Hierarchical navigation requires Place topology and object approach routes")
        (
            prompt_assessments,
            snapshot_id_mapping,
            snapshot_crop_mapping,
            n_filtered_snapshots,
            semantic_digest,
        ) = assess_navigation_step(step_dict, cfg, hierarchical_navigator, verbose=verbose)
        candidates = []
        prompt_by_candidate = {}
        snapshot_keys = list(scene.snapshots.keys())
        exposed_images = list(snapshot_crop_mapping.keys())
        for prompt_snapshot_index, image in enumerate(exposed_images):
            source_snapshot_index = snapshot_id_mapping[prompt_snapshot_index]
            if not 0 <= source_snapshot_index < len(snapshot_keys):
                continue
            snapshot_key = image
            if snapshot_key != snapshot_keys[source_snapshot_index]:
                # Refuse a stale prompt/source permutation instead of silently
                # scoring one image and executing another.
                continue
            crop_rows = step_dict["snapshot_imgs"].get(snapshot_key, {}).get("object_crop", [])
            for prompt_object_index, source_crop_index in enumerate(snapshot_crop_mapping[image]):
                if not 0 <= source_crop_index < len(crop_rows):
                    continue
                object_id = crop_rows[source_crop_index]["obj_id"]
                approach = routes.get(object_id, {"status": "unavailable"})
                prompt_id = f"Snapshot {prompt_snapshot_index}, Object {prompt_object_index}"
                stable_id = f"TARGET_APPROACH:{object_id}:snapshot:{snapshot_key}"
                candidate = SemanticCandidate(
                    candidate_id=stable_id,
                    source_kind="snapshot",
                    entity_id=str(object_id),
                    place_id=approach.get("approach_place_id"),
                    route_id=approach.get("route_id"),
                    route_distance_m=approach.get("total_distance_m"),
                    route_reachable=approach.get("status") == "valid",
                    snapshot_image=str(snapshot_key),
                    object_id=object_id,
                    current_observation=False,
                    evidence_refs=(f"snapshot:{snapshot_key}:object:{object_id}",),
                )
                candidates.append(candidate)
                prompt_by_candidate[stable_id] = prompt_id
        frontier_rows = step_dict["hgr_topology_context"].get("frontiers", [])
        exposed_frontier_count = sum(
            key.startswith("Frontier ") for key in prompt_assessments
        )
        for prompt_frontier_index in range(min(exposed_frontier_count, len(frontier_source_indices))):
            source_index = frontier_source_indices[prompt_frontier_index]
            frontier = tsdf_planner.frontiers[source_index]
            approach = frontier_rows[prompt_frontier_index] if prompt_frontier_index < len(frontier_rows) else {"status": "unavailable"}
            prompt_id = f"Frontier {prompt_frontier_index}"
            entity_id = str(getattr(frontier, "topo_id", getattr(frontier, "frontier_id", source_index)))
            stable_id = f"EXPLORE:{entity_id}"
            entropy = 0.0
            if getattr(frontier, "semantic_dist", None) is not None:
                entropy = float(frontier.semantic_dist.entropy)
            candidate = SemanticCandidate(
                candidate_id=stable_id,
                source_kind="frontier",
                entity_id=entity_id,
                place_id=approach.get("approach_place_id"),
                route_id=approach.get("route_id"),
                route_distance_m=approach.get("total_distance_m"),
                route_reachable=approach.get("status") == "valid",
                source_index=source_index,
                information_gain=entropy / (1.0 + max(0.0, entropy)),
                hypothesis_refs=((str(frontier.hypothesis_node_id),) if getattr(frontier, "hypothesis_node_id", None) else ()),
            )
            candidates.append(candidate)
            prompt_by_candidate[stable_id] = prompt_id
        stable_assessments = {}
        for candidate in candidates:
            prompt_id = prompt_by_candidate[candidate.candidate_id]
            assessment = prompt_assessments.get(prompt_id)
            if assessment is not None:
                stable_assessments[candidate.candidate_id] = replace(
                    assessment, candidate_id=candidate.candidate_id,
                )
        if semantic_digest in hierarchical_navigator.suppressed_evidence:
            # A transport/parser failure at terminal must not immediately loop
            # back to the same object on unchanged evidence. Spatial/frontier
            # candidates stay available, and any evidence revision creates a
            # different digest that naturally releases this suppression.
            from src.hierarchical_navigator import SemanticConfidence
            stable_assessments = {
                candidate_id: (
                    replace(value, confidence=SemanticConfidence.NO_MATCH,
                            evidence="temporarily_suppressed_after_verification_error")
                    if next(item for item in candidates if item.candidate_id == candidate_id).source_kind == "snapshot"
                    else value
                )
                for candidate_id, value in stable_assessments.items()
            }
        hierarchical_decision = hierarchical_navigator.decide(
            candidates, stable_assessments, place_topology.current_place_id,
        )
        hierarchical_navigator.pending_decision = hierarchical_decision
        hierarchical_navigator.pending_semantic_digest = semantic_digest
        hierarchical_navigator.last_assessments = stable_assessments
        logging.info("Hierarchical navigation decision: %s", hierarchical_decision.to_trace_dict())
        if hierarchical_navigator.shadow_only:
            (
                outputs,
                snapshot_id_mapping,
                snapshot_crop_mapping,
                reason,
                n_filtered_snapshots,
            ) = explore_step(step_dict, cfg, verbose=verbose)
        else:
            outputs = (
                None if hierarchical_decision.candidate is None
                else prompt_by_candidate[hierarchical_decision.candidate.candidate_id].lower()
            )
            reason = hierarchical_decision.reason
    else:
        (
            outputs,
            snapshot_id_mapping,
            snapshot_crop_mapping,
            reason,
            n_filtered_snapshots,
        ) = explore_step(step_dict, cfg, verbose=verbose)
    if outputs is None:
        logging.error(f"explore_step failed and returned None")
        return None
    logging.info(f"Response: [{outputs}]\nReason: [{reason}]")

    # parse returned results
    try:
        target_type, target_index = outputs.split(",")[0].strip().split(" ")
        logging.info(f"Prediction: {target_type}, {target_index}")
    except:
        logging.info(f"Wrong output format, failed!")
        return None

    allowed_target_types = ["frontier"] if frontier_only else ["snapshot", "frontier"]
    if target_type not in allowed_target_types:
        logging.info(f"Wrong target type: {target_type}, failed!")
        return None

    if target_type == "snapshot":
        if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
            logging.info(
                f"Target index can not match real objects: {target_index}, failed!"
            )
            return None
        target_index = snapshot_id_mapping[int(target_index)]
        logging.info(f"The index of target snapshot {target_index}")

        # get the target snapshot
        if target_index < 0 or target_index >= len(scene.snapshots):
            logging.info(
                f"Predicted snapshot target index out of range: {target_index}, failed!"
            )
            return None

        pred_target_snapshot = list(scene.snapshots.values())[target_index]
        pred_target_snapshot_id = list(scene.snapshots.keys())[target_index]
        logging.info(f"Next choice: Snapshot of {pred_target_snapshot.image}")

        if (
            pred_target_snapshot_id not in snapshot_crop_mapping
            or pred_target_snapshot_id not in step_dict["snapshot_imgs"]
        ):
            logging.warning(
                "Snapshot mapping became stale for %s; ignoring this VLM choice",
                pred_target_snapshot_id,
            )
            return None

        # get the object choice
        object_choice, object_choice_id = outputs.split(",")[1].strip().split(" ")
        if object_choice != "object":
            logging.info(f"Invalid object choice: {object_choice}, failed!")
            return None
        object_choice_id = int(object_choice_id)
        if object_choice_id < 0 or object_choice_id >= len(
            snapshot_crop_mapping.get(pred_target_snapshot_id, [])
        ):
            logging.info(f"Object choice out of range: {object_choice_id}, failed!")
            return None
        object_choice_id = snapshot_crop_mapping[pred_target_snapshot_id][
            object_choice_id
        ]
        pred_target_obj_id = step_dict["snapshot_imgs"][pred_target_snapshot_id][
            "object_crop"
        ][object_choice_id]["obj_id"]
        logging.info(
            f"Next choice Object: {pred_target_obj_id}, {scene.objects[pred_target_obj_id]['class_name']}"
        )

        # instantiate a snapshot object that contains only the predicted object as the navigation target
        max_point_choice = SnapShot(
            image=pred_target_snapshot.image,
            color=(random.random(), random.random(), random.random()),
            obs_point=np.asarray(pred_target_snapshot.obs_point).copy(),
            full_obj_list={
                pred_target_obj_id: scene.objects[pred_target_obj_id]["conf"]
            },
            cluster=[pred_target_obj_id],
        )

        # Preserve the exact visual evidence that passed the semantic gate so
        # terminal verification can confirm this entity rather than any object
        # of the target class that happens to appear elsewhere in the room.
        selected_source_crop = object_choice_id
        max_point_choice.hierarchical_candidate_crop = np.asarray(
            step_dict["snapshot_imgs"][pred_target_snapshot_id]["object_crop"][
                selected_source_crop
            ]["crop"]
        ).copy()
        max_point_choice.hierarchical_candidate_label = str(
            scene.objects[pred_target_obj_id]["class_name"]
        )
        max_point_choice.hierarchical_candidate_source_image = str(
            pred_target_snapshot_id
        )

        if place_topology is not None and cfg.get("hgr_topology_fusion", {}).get("object_approach", False):
            max_point_choice.object_approach = routes[pred_target_obj_id]
        if hierarchical_decision is not None and not hierarchical_navigator.shadow_only:
            max_point_choice.hierarchical_decision = hierarchical_decision
        return max_point_choice, n_filtered_snapshots
    else:  # target_type == "frontier"
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(frontier_source_indices):
            logging.info(
                f"Predicted frontier target index out of range: {target_index}, failed!"
            )
            return None
        source_frontier_index = frontier_source_indices[target_index]
        target_point = tsdf_planner.frontiers[source_frontier_index].position
        logging.info(f"Next choice: Frontier at {target_point}")
        if active_topo_view is not None:
            logging.info(
                "ActiveTopo frontier mapping: projected %d -> source %d (%s)",
                target_index,
                source_frontier_index,
                active_topo_view.candidates[target_index].topo_node_id,
            )
        pred_target_frontier = tsdf_planner.frontiers[source_frontier_index]
        if place_topology is not None and cfg.get("hgr_topology_fusion", {}).get("object_approach", False):
            pred_target_frontier.object_approach = {
                **step_dict["hgr_topology_context"]["frontiers"][target_index],
                "terminal": np.asarray(pred_target_frontier.position[:2]).tolist(),
                "look_at": (np.asarray(pred_target_frontier.position[:2]) + np.asarray(pred_target_frontier.orientation[:2])*3).tolist()}
            if getattr(place_topology, "dual_topo_runtime", None) is not None:
                pred_target_frontier.object_approach = dict(step_dict["hgr_topology_context"]["frontiers"][target_index])

        if hierarchical_decision is not None and not hierarchical_navigator.shadow_only:
            pred_target_frontier.hierarchical_decision = hierarchical_decision

        return pred_target_frontier, n_filtered_snapshots
