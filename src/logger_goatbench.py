import os
import json
import pickle
from collections import defaultdict
import logging
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image
from typing import Union

import habitat_sim

from src.scene_goatbench import Scene
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.subtask_execution import summarize_execution_quality


def _load_visualization_image(path, fallback=None):
    """Load a run artifact, falling back to the in-memory candidate image."""
    if os.path.exists(path):
        return matplotlib.image.imread(path)
    if fallback is None:
        logging.warning("Visualization image unavailable: %s", path)
        return None
    if hasattr(fallback, "detach"):
        fallback = fallback.detach().cpu().numpy()
    image = np.asarray(fallback)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    logging.info("Using in-memory image for restored visualization artifact: %s", path)
    return image


def compute_spl(success: bool, shortest_distance: float, traversed_distance: float) -> float:
    """Compute a finite SPL value, conservatively scoring undefined paths as zero."""
    return compute_spl_diagnostic(success, shortest_distance, traversed_distance)[0]


def compute_spl_diagnostic(
    success: bool, shortest_distance: float, traversed_distance: float
):
    """Return a finite SPL and a status explaining invalid or zero values."""
    if not success:
        return 0.0, "failure"
    shortest_distance = float(shortest_distance)
    traversed_distance = float(traversed_distance)
    if not np.isfinite(shortest_distance) or shortest_distance < 0.0:
        return 0.0, "invalid_shortest_distance"
    if not np.isfinite(traversed_distance) or traversed_distance < 0.0:
        return 0.0, "invalid_traversed_distance"
    denominator = max(shortest_distance, traversed_distance)
    if denominator == 0.0:
        return 1.0, "zero_length_success"
    return shortest_distance / denominator, "valid"


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    return value


def _sanitize_metric_mapping(values):
    sanitized = {}
    invalid_ids = []
    for key, value in values.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = np.nan
        if not np.isfinite(numeric):
            numeric = 0.0
            invalid_ids.append(str(key))
        sanitized[key] = numeric
    return sanitized, invalid_ids


class Logger:
    def __init__(
        self,
        output_dir,
        start_ratio,
        end_ratio,
        split,
        voxel_size,  # used for calculating the moving distance
    ):
        self.output_dir = output_dir
        self.voxel_size = voxel_size

        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_snapshot = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.success_by_snapshot = {}  # subtask_id -> success
        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_distance = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_distance_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.success_by_distance = {}  # subtask id -> success
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_snapshot = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"spl_by_snapshot_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.spl_by_snapshot = {}  # subtask id -> spl
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_distance = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"spl_by_distance_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            self.spl_by_distance = {}  # subtask id -> spl
        if os.path.exists(
            os.path.join(
                output_dir, f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.success_by_task = pickle.load(
                open(
                    os.path.join(
                        output_dir,
                        f"success_by_task_{start_ratio}_{end_ratio}_{split}.pkl",
                    ),
                    "rb",
                )
            )
        else:
            # success_by_task = {}  # task type -> success
            self.success_by_task = defaultdict(list)
        if os.path.exists(
            os.path.join(
                output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
            )
        ):
            self.spl_by_task = pickle.load(
                open(
                    os.path.join(
                        output_dir, f"spl_by_task_{start_ratio}_{end_ratio}_{split}.pkl"
                    ),
                    "rb",
                )
            )
        else:
            # spl_by_task = {}  # task type -> spl
            self.spl_by_task = defaultdict(list)
        if os.path.exists(
            os.path.join(
                output_dir,
                f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json",
            )
        ):
            with open(
                os.path.join(
                    output_dir,
                    f"n_filtered_snapshots_{start_ratio}_{end_ratio}_{split}.json",
                ),
                "r",
            ) as f:
                self.n_filtered_snapshots_list = json.load(f)
        else:
            self.n_filtered_snapshots_list = {}
        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir,
                    f"n_total_snapshots_{start_ratio}_{end_ratio}_{split}.json",
                ),
                "r",
            ) as f:
                self.n_total_snapshots_list = json.load(f)
        else:
            self.n_total_snapshots_list = {}
        if os.path.exists(
            os.path.join(
                output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"
            )
        ):
            with open(
                os.path.join(
                    output_dir, f"n_total_frames_{start_ratio}_{end_ratio}_{split}.json"
                ),
                "r",
            ) as f:
                self.n_total_frames_list = json.load(f)
        else:
            self.n_total_frames_list = {}

        subtask_metrics_path = os.path.join(
            output_dir, f"subtask_metrics_{start_ratio}_{end_ratio}_{split}.json"
        )
        if os.path.exists(subtask_metrics_path):
            with open(subtask_metrics_path, "r") as f:
                self.subtask_metrics = json.load(f)
        else:
            self.subtask_metrics = {}

        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.split = split

        # some sanity check
        assert (
            len(self.success_by_snapshot)
            == len(self.spl_by_snapshot)
            == len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_snapshot)} != {len(self.spl_by_snapshot)} != {len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
            == len(self.success_by_snapshot)
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])} != {len(self.success_by_snapshot)}"

        # logging for episode
        self.episode_dir = None

        # logging for subtask
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def save_results(self):
        # some sanity check
        assert (
            len(self.success_by_snapshot)
            == len(self.spl_by_snapshot)
            == len(self.success_by_distance)
            == len(self.spl_by_distance)
        ), f"{len(self.success_by_snapshot)} != {len(self.spl_by_snapshot)} != {len(self.success_by_distance)} != {len(self.spl_by_distance)}"
        assert (
            sum([len(task_res) for task_res in self.success_by_task.values()])
            == sum([len(task_res) for task_res in self.spl_by_task.values()])
            == len(self.success_by_snapshot)
        ), f"{sum([len(task_res) for task_res in self.success_by_task.values()])} != {sum([len(task_res) for task_res in self.spl_by_task.values()])} != {len(self.success_by_snapshot)}"

        with open(
            os.path.join(
                self.output_dir,
                f"success_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_snapshot, f)
        with open(
            os.path.join(
                self.output_dir,
                f"success_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_distance, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_snapshot_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_snapshot, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_distance_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_distance, f)
        with open(
            os.path.join(
                self.output_dir,
                f"success_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.success_by_task, f)
        with open(
            os.path.join(
                self.output_dir,
                f"spl_by_task_{self.start_ratio}_{self.end_ratio}_{self.split}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.spl_by_task, f)
        with open(
            os.path.join(
                self.output_dir,
                f"n_filtered_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_filtered_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_snapshots_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_snapshots_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"n_total_frames_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(self.n_total_frames_list, f, indent=4)
        with open(
            os.path.join(
                self.output_dir,
                f"subtask_metrics_{self.start_ratio}_{self.end_ratio}_{self.split}.json",
            ),
            "w",
        ) as f:
            json.dump(_json_safe(self.subtask_metrics), f, indent=4, allow_nan=False)

    def aggregate_results(self):
        # aggregate the results into a single file
        metric_validity = {}
        filenames_to_merge = [
            "success_by_snapshot",
            "spl_by_snapshot",
            "success_by_distance",
            "spl_by_distance",
        ]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    all_results.update(pickle.load(f))
            all_results, invalid_ids = _sanitize_metric_mapping(all_results)
            metric_validity[filename] = {
                "total": len(all_results),
                "finite": len(all_results) - len(invalid_ids),
                "invalid": len(invalid_ids),
                "invalid_ids": invalid_ids,
                "invalid_treatment": "replaced_with_zero",
            }
            logging.info(
                f"Total {filename} results: {100 * np.mean(list(all_results.values())) if all_results else 0.0:.2f}, len: {len(all_results)}, invalid: {len(invalid_ids)}"
            )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)
        filenames_to_merge = ["success_by_task", "spl_by_task"]
        for filename in filenames_to_merge:
            all_results = {}
            all_results_paths = glob.glob(
                os.path.join(self.output_dir, f"{filename}_*.pkl")
            )
            for results_path in all_results_paths:
                with open(results_path, "rb") as f:
                    separate_stat = pickle.load(f)
                    for task_name, task_res in separate_stat.items():
                        if task_name not in all_results:
                            all_results[task_name] = []
                        all_results[task_name] += task_res
            for task_name, task_res in all_results.items():
                sanitized, invalid_indices = _sanitize_metric_mapping(
                    {str(index): value for index, value in enumerate(task_res)}
                )
                all_results[task_name] = list(sanitized.values())
                metric_validity[f"{filename}:{task_name}"] = {
                    "total": len(task_res),
                    "finite": len(task_res) - len(invalid_indices),
                    "invalid": len(invalid_indices),
                    "invalid_indices": invalid_indices,
                    "invalid_treatment": "replaced_with_zero",
                }
                logging.info(
                    f"Total {filename} results for {task_name}: {100 * np.mean(all_results[task_name]) if task_res else 0.0:.2f}, len: {len(task_res)}, invalid: {len(invalid_indices)}"
                )
            with open(os.path.join(self.output_dir, f"{filename}.pkl"), "wb") as f:
                pickle.dump(all_results, f)

        n_filtered_snapshots_list = {}
        all_n_filtered_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_filtered_snapshots_*.json")
        )
        for n_filtered_snapshots_list_path in all_n_filtered_snapshots_list_paths:
            with open(n_filtered_snapshots_list_path, "r") as f:
                n_filtered_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_filtered_snapshots.json"), "w") as f:
            json.dump(n_filtered_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of filtered snapshots: {np.mean(list(n_filtered_snapshots_list.values())) if n_filtered_snapshots_list else 0.0}"
        )

        n_total_snapshots_list = {}
        all_n_total_snapshots_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_snapshots_*.json")
        )
        for n_total_snapshots_list_path in all_n_total_snapshots_list_paths:
            with open(n_total_snapshots_list_path, "r") as f:
                n_total_snapshots_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_snapshots.json"), "w") as f:
            json.dump(n_total_snapshots_list, f, indent=4)
        logging.info(
            f"Average number of total snapshots: {np.mean(list(n_total_snapshots_list.values())) if n_total_snapshots_list else 0.0}"
        )

        n_total_frames_list = {}
        all_n_total_frames_list_paths = glob.glob(
            os.path.join(self.output_dir, "n_total_frames_*.json")
        )
        for n_total_frames_list_path in all_n_total_frames_list_paths:
            with open(n_total_frames_list_path, "r") as f:
                n_total_frames_list.update(json.load(f))

        with open(os.path.join(self.output_dir, "n_total_frames.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)
        logging.info(
            f"Average number of total frames: {np.mean(list(n_total_frames_list.values())) if n_total_frames_list else 0.0}"
        )

        subtask_metrics = {}
        for metrics_path in glob.glob(
            os.path.join(self.output_dir, "subtask_metrics_*.json")
        ):
            with open(metrics_path, "r") as f:
                subtask_metrics.update(json.load(f))
        with open(os.path.join(self.output_dir, "subtask_metrics.json"), "w") as f:
            json.dump(subtask_metrics, f, indent=4, allow_nan=False)
        with open(os.path.join(self.output_dir, "metric_validity.json"), "w") as f:
            json.dump(metric_validity, f, indent=4, allow_nan=False)
        with open(os.path.join(self.output_dir, "execution_quality.json"), "w") as f:
            json.dump(
                summarize_execution_quality(subtask_metrics),
                f,
                indent=4,
                allow_nan=False,
            )

    def log_subtask_result(
        self,
        success_by_snapshot: bool,
        success_by_distance: bool,
        subtask_id: str,
        gt_subtask_explore_dist: float,
        goal_type: str,
        n_filtered_snapshots,
        n_total_snapshots,
        n_total_frames,
        agent_subtask_distance=None,
        subtask_steps=None,
        subtask_frames=None,
        vlm_telemetry=None,
        final_choice=None,
        selected_topo_id=None,
        timing=None,
        navigation_diagnostics=None,
        execution_outcome=None,
        snapshot_mapping_diagnostics=None,
    ):
        if success_by_snapshot:
            self.success_by_snapshot[subtask_id] = 1.0
        else:
            self.success_by_snapshot[subtask_id] = 0.0
        if success_by_distance:
            self.success_by_distance[subtask_id] = 1.0
        else:
            self.success_by_distance[subtask_id] = 0.0

        # calculate the SPL
        snapshot_spl, snapshot_spl_status = compute_spl_diagnostic(
            success_by_snapshot,
            gt_subtask_explore_dist,
            self.subtask_explore_dist,
        )
        distance_spl, distance_spl_status = compute_spl_diagnostic(
            success_by_distance,
            gt_subtask_explore_dist,
            self.subtask_explore_dist,
        )
        self.spl_by_snapshot[subtask_id] = snapshot_spl
        self.spl_by_distance[subtask_id] = distance_spl

        self.success_by_task[goal_type].append(self.success_by_distance[subtask_id])
        self.spl_by_task[goal_type].append(self.spl_by_distance[subtask_id])

        logging.info(
            f"Subtask {subtask_id} finished, {self.subtask_explore_dist} length"
        )
        logging.info(
            f"Subtask spl by snapshot: {self.spl_by_snapshot[subtask_id]}, spl by distance: {self.spl_by_distance[subtask_id]}"
        )

        logging.info(
            f"Success rate by snapshot: {100 * np.mean(np.asarray(list(self.success_by_snapshot.values()))):.2f}"
        )
        logging.info(
            f"Success rate by distance: {100 * np.mean(np.asarray(list(self.success_by_distance.values()))):.2f}"
        )
        logging.info(
            f"SPL by snapshot: {100 * np.mean(np.asarray(list(self.spl_by_snapshot.values()))):.2f}"
        )
        logging.info(
            f"SPL by distance: {100 * np.mean(np.asarray(list(self.spl_by_distance.values()))):.2f}"
        )

        for task_name, success_list in self.success_by_task.items():
            logging.info(
                f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}"
            )
        for task_name, spl_list in self.spl_by_task.items():
            logging.info(
                f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}"
            )

        logging.info(
            f"Filtered snapshots/Total snapshots/Total frames: {n_filtered_snapshots}/{n_total_snapshots}/{n_total_frames}"
        )
        # save the number of snapshots
        self.n_filtered_snapshots_list[subtask_id] = n_filtered_snapshots
        self.n_total_snapshots_list[subtask_id] = n_total_snapshots
        self.n_total_frames_list[subtask_id] = n_total_frames
        execution_outcome = dict(execution_outcome or {})
        self.subtask_metrics[subtask_id] = _json_safe(
            {
                "subtask_id": subtask_id,
                "goal_type": goal_type,
                "success_by_snapshot": bool(success_by_snapshot),
                "success_by_distance": bool(success_by_distance),
                "spl_by_snapshot": snapshot_spl,
                "spl_by_distance": distance_spl,
                "spl_by_snapshot_status": snapshot_spl_status,
                "spl_by_distance_status": distance_spl_status,
                "gt_shortest_distance_m": gt_subtask_explore_dist,
                "traversed_distance_m": self.subtask_explore_dist,
                "agent_final_distance_m": agent_subtask_distance,
                "subtask_steps": subtask_steps,
                "subtask_frames": subtask_frames,
                "n_filtered_snapshots": n_filtered_snapshots,
                "n_total_snapshots": n_total_snapshots,
                "n_total_frames": n_total_frames,
                "vlm_telemetry": vlm_telemetry or {},
                "final_choice": final_choice,
                "snapshot_mapping_diagnostics": snapshot_mapping_diagnostics or {},
                "selected_topo_id": selected_topo_id,
                "timing": timing or {},
                "navigation_diagnostics": navigation_diagnostics or {},
                "execution_status": execution_outcome.get(
                    "execution_status", "unknown"
                ),
                "termination_reason": execution_outcome.get(
                    "termination_reason", "unknown"
                ),
                "termination_detail": execution_outcome.get(
                    "termination_detail", {}
                ),
                "error_counts": execution_outcome.get("error_counts", {}),
            }
        )

        # clear the subtask logging
        self.subtask_object_observe_dir = None
        self.pts_voxels = np.empty((0, 2))
        self.subtask_explore_dist = 0.0

    def init_episode(
        self,
        episode_id,
    ):
        self.episode_dir = os.path.join(self.output_dir, episode_id)
        eps_frontier_dir = os.path.join(self.episode_dir, "frontier")
        eps_snapshot_dir = os.path.join(self.episode_dir, "snapshot")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(eps_frontier_dir, exist_ok=True)
        os.makedirs(eps_snapshot_dir, exist_ok=True)

        return self.episode_dir, eps_frontier_dir, eps_snapshot_dir

    def init_subtask(
        self,
        subtask_id,
        goal_type,
        subtask_goal,
        pts,
        scene: Scene,  # used to get image goal observation and use its pathfinder
        tsdf_planner: TSDFPlanner,
    ):
        # determine the navigation goals
        goal_category = subtask_goal[0]["object_category"]
        goal_obj_ids = [x["object_id"] for x in subtask_goal]
        goal_obj_ids = [int(x.split("_")[-1]) for x in goal_obj_ids]
        if goal_type != "object":
            assert len(goal_obj_ids) == 1, f"{len(goal_obj_ids)} != 1"

        goal_positions = [x["position"] for x in subtask_goal]
        goal_positions_voxel = [tsdf_planner.habitat2voxel(p) for p in goal_positions]

        viewpoints = [
            view_point["agent_state"]["position"]
            for goal in subtask_goal
            for view_point in goal["view_points"]
        ]
        # get the shortest distance from current position to the viewpoints
        all_distances = []
        for viewpoint in viewpoints:
            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = viewpoint
            found_path = scene.pathfinder.find_path(path)
            if not found_path:
                all_distances.append(np.inf)
            else:
                all_distances.append(path.geodesic_distance)
        gt_subtask_explore_dist = min(all_distances) + 1e-6

        self.subtask_object_observe_dir = os.path.join(
            self.output_dir, subtask_id, "object_observations"
        )
        if os.path.exists(self.subtask_object_observe_dir):
            os.system(f"rm -r {self.subtask_object_observe_dir}")
        os.makedirs(self.subtask_object_observe_dir, exist_ok=False)

        # Prepare metadata for the subtask
        subtask_metadata = {
            "question_id": subtask_id,
            "question": None,
            "image": None,
            "answer": goal_category,
            "goal_obj_ids": goal_obj_ids,  # this is a list of obj id, since for object class type, there will be multiple target objects
            "class": goal_category,
            "goal_positions_voxel": goal_positions_voxel,  # also a list of positions for possible multiple objects
            "task_type": goal_type,
            "viewpoints": viewpoints,
            "gt_subtask_explore_dist": gt_subtask_explore_dist,
        }
        # format question according to the goal type
        if goal_type == "object":
            subtask_metadata["question"] = f"Can you find the {goal_category}?"
        elif goal_type == "description":
            subtask_metadata["question"] = (
                f"Could you find the object exactly described as the '{subtask_goal[0]['lang_desc']}'?"
            )
        else:
            subtask_metadata["question"] = (
                f"Could you find the exact object captured at the center of the following image? You need to pay attention to the environment and find the exact object."
            )
            view_pos_dict = subtask_goal[0]["view_points"][0]["agent_state"]
            obs, _ = scene.get_observation(
                pts=view_pos_dict["position"], rotation=view_pos_dict["rotation"]
            )
            plt.imsave(
                os.path.join(self.output_dir, subtask_id, "image_goal.png"),
                obs["color_sensor"],
            )
            subtask_metadata["image"] = f"{self.output_dir}/{subtask_id}/image_goal.png"

        self.pts_voxels = np.empty((0, 2))
        self.pts_voxels = np.vstack(
            [self.pts_voxels, tsdf_planner.habitat2voxel(pts)[:2]]
        )
        self.subtask_explore_dist = 0.0

        return subtask_metadata

    def log_step(self, pts_voxel):
        self.pts_voxels = np.vstack([self.pts_voxels, pts_voxel])
        self.subtask_explore_dist += (
            np.linalg.norm(self.pts_voxels[-1] - self.pts_voxels[-2]) * self.voxel_size
        )

    def save_topdown_visualization(
        self, global_step, subtask_id, subtask_metadata, goal_obj_ids_mapping, fig
    ):
        assert self.episode_dir is not None
        visualization_path = os.path.join(self.episode_dir, "visualization")
        os.makedirs(visualization_path, exist_ok=True)

        ax1 = fig.axes[0]
        ax1.plot(
            self.pts_voxels[:-1, 1], self.pts_voxels[:-1, 0], linewidth=1, color="white"
        )
        ax1.scatter(self.pts_voxels[0, 1], self.pts_voxels[0, 0], c="white", s=50)

        # add target object bbox
        for goal_id, goal_pos_voxel in zip(
            subtask_metadata["goal_obj_ids"], subtask_metadata["goal_positions_voxel"]
        ):
            color = "green" if len(goal_obj_ids_mapping[goal_id]) > 0 else "red"
            ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)

        fig.tight_layout()
        plt.savefig(os.path.join(visualization_path, f"{global_step}_{subtask_id}.png"))
        plt.close()

    def save_frontier_visualization(
        self,
        global_step,
        subtask_id,
        tsdf_planner: TSDFPlanner,
        max_point_choice: Union[SnapShot, Frontier],
        global_caption,
    ):
        assert self.episode_dir is not None
        frontier_video_path = os.path.join(self.episode_dir, "frontier_video")
        episode_frontier_dir = os.path.join(self.episode_dir, "frontier")
        episode_snapshot_dir = os.path.join(self.episode_dir, "snapshot")
        os.makedirs(frontier_video_path, exist_ok=True)
        num_images = len(tsdf_planner.frontiers)
        if type(max_point_choice) == SnapShot:
            num_images += 1
        side_length = int(np.sqrt(num_images)) + 1
        side_length = max(2, side_length)
        fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
        for h_idx in range(side_length):
            for w_idx in range(side_length):
                axs[h_idx, w_idx].axis("off")
                i = h_idx * side_length + w_idx
                if (i < num_images - 1) or (
                    i < num_images and type(max_point_choice) == Frontier
                ):
                    img_path = os.path.join(
                        episode_frontier_dir, tsdf_planner.frontiers[i].image
                    )
                    img = _load_visualization_image(
                        img_path, tsdf_planner.frontiers[i].feature)
                    if img is not None:
                        axs[h_idx, w_idx].imshow(img)
                    else:
                        axs[h_idx, w_idx].set_title("Image unavailable")
                    if (
                        type(max_point_choice) == Frontier
                        and max_point_choice.image == tsdf_planner.frontiers[i].image
                    ):
                        axs[h_idx, w_idx].set_title("Chosen")
                elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                    img_path = os.path.join(
                        episode_snapshot_dir, max_point_choice.image
                    )
                    img = _load_visualization_image(img_path)
                    if img is not None:
                        axs[h_idx, w_idx].imshow(img)
                    axs[h_idx, w_idx].set_title("Snapshot Chosen")
        fig.suptitle(global_caption, fontsize=16)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(
            os.path.join(frontier_video_path, f"{global_step}_{subtask_id}.png")
        )
        plt.close()
