import os
import json
import unittest

from run_goatbench_evaluation import (
    _direct_first_selection_mode,
    _load_config,
)
from src.persistent_belief_topology import GoalTopoActionType
from src.persistent_belief_topology import (
    active_topology_goal_type_is_listed,
    restrict_active_topology_policy,
    resolve_active_topology_policy,
)


class ActiveTopologyConfigTest(unittest.TestCase):
    def test_place_topology_phase_a_is_shadow_only(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_place_shadow_qwen3vl_dashscope_train.yaml",
        ))
        place = cfg.active_topology.place_topology
        self.assertTrue(place.enabled)
        self.assertEqual(place.stage, "shadow")
        self.assertEqual(place.place_spacing_m, 1.5)
        # Phase A inherits the frozen V7 selection/execution policy; its only
        # added behavioral block is the non-intervening Place recorder.
        self.assertTrue(cfg.active_topology.goal_conditioned_overlay.enabled)
        self.assertTrue(cfg.active_topology.metric_execution.shadow_only)

    def test_place_route_only_keeps_original_hgr_target_selection(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_place_route_only_qwen3vl_dashscope_train.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "stable_only")
        self.assertEqual(
            cfg.active_topology.place_topology.stage, "route_only"
        )
        self.assertNotIn("direct_first_execution", cfg.active_topology)
        self.assertNotIn("goal_conditioned_overlay", cfg.active_topology)
        policy = resolve_active_topology_policy("stable_only", "category")
        self.assertFalse(policy.use_active_view)
        self.assertFalse(policy.use_belief_planner)

    def test_direct_first_anchor_reobserve_uses_original_hgr_once(self):
        image = "history.png"
        self.assertEqual(
            _direct_first_selection_mode(
                image, [], GoalTopoActionType.REVISIT, image
            ),
            "post_anchor_original_hgr",
        )
        self.assertEqual(
            _direct_first_selection_mode(
                image, [], GoalTopoActionType.REVISIT, None
            ),
            "revisit",
        )
        self.assertEqual(
            _direct_first_selection_mode(image, [image], None, None),
            "direct_or_current",
        )

    def test_simple_memory_config_only_enables_minimal_mode(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg", "eval_goatbench_simple_memory_qwen3vl_dashscope.yaml"
        ))
        self.assertEqual(cfg.active_topology.mode, "simple_memory")
        self.assertTrue(cfg.active_topology.enabled)
        self.assertEqual(
            cfg.exp_name,
            "exp_eval_goatbench_simple_memory_qwen3vl30b_dashscope",
        )
        self.assertNotIn("belief", cfg.active_topology)
        self.assertNotIn("verification", cfg.active_topology)

    def test_goal_topomap_config_has_no_belief_state_machine(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg", "eval_goatbench_goal_topomap_qwen3vl_dashscope.yaml"
        ))
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertTrue(cfg.active_topology.enabled)
        self.assertEqual(
            cfg.exp_name,
            "exp_eval_goatbench_goal_topomap_route_v3_qwen3vl30b_dashscope",
        )
        self.assertNotIn("belief", cfg.active_topology)
        self.assertNotIn("verification", cfg.active_topology)
        self.assertEqual(cfg.vlm_model, "qwen3-vl-30b-a3b-instruct")

    def test_goal_topomap_object_priority_is_an_isolated_v4(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_object_priority_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertTrue(cfg.active_topology.object_live_priority)
        self.assertEqual(
            cfg.exp_name,
            "exp_eval_goatbench_goal_topomap_route_v4_object_priority_qwen3vl30b_dashscope",
        )

    def test_goal_topomap_unified_view_is_an_isolated_v5(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_unified_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertTrue(cfg.active_topology.object_live_priority)
        self.assertTrue(cfg.active_topology.unified_task_view)
        self.assertEqual(
            cfg.exp_name,
            "exp_eval_goatbench_goal_topomap_route_v5_unified_qwen3vl30b_dashscope",
        )

    def test_goal_topomap_hybrid_uses_projection_only_for_image(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_hybrid_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertTrue(cfg.active_topology.object_live_priority)
        self.assertTrue(cfg.active_topology.unified_task_view)
        self.assertEqual(list(cfg.active_topology.projection_only_goal_types), ["image"])
        self.assertEqual(
            cfg.exp_name,
            "exp_eval_goatbench_goal_topomap_route_v6_hybrid_qwen3vl30b_dashscope",
        )

        image_policy = resolve_active_topology_policy("goal_topo_map", "image")
        self.assertTrue(image_policy.use_active_view)
        self.assertTrue(image_policy.use_goal_relevance)
        projection_only = cfg.active_topology.projection_only_goal_types
        self.assertTrue(
            active_topology_goal_type_is_listed("image", projection_only)
        )
        self.assertFalse(
            active_topology_goal_type_is_listed("object", projection_only)
        )
        self.assertFalse(
            active_topology_goal_type_is_listed("description", projection_only)
        )

    def test_v7_0_is_diagnostics_only_and_uses_fixed_train_manifest(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_0_diagnostics_qwen3vl_dashscope_train.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertTrue(cfg.active_topology.object_live_priority)
        self.assertTrue(cfg.active_topology.unified_task_view)
        self.assertEqual(list(cfg.active_topology.projection_only_goal_types), ["image"])
        self.assertEqual(
            cfg.episode_manifest,
            "cfg/manifests/goat_train_dev_12x2_seed77.json",
        )
        self.assertIn("v7_0_diagnostics", cfg.exp_name)

    def test_v7_1_a_is_all_goal_shadow_only(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_1_a_shadow_qwen3vl_dashscope_train.yaml",
        ))
        self.assertTrue(cfg.active_topology.metric_execution.enabled)
        self.assertTrue(cfg.active_topology.metric_execution.shadow_only)
        self.assertEqual(
            list(cfg.active_topology.metric_execution.goal_types),
            ["object", "description", "image"],
        )
        self.assertEqual(cfg.active_topology.mode, "goal_topo_map")
        self.assertIn("v7_1_a_shadow", cfg.exp_name)

    def test_v7_1_b_enables_direct_first_for_all_goal_types(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_1_b_direct_first_qwen3vl_dashscope_train.yaml",
        ))
        self.assertTrue(cfg.active_topology.metric_execution.shadow_only)
        self.assertTrue(cfg.active_topology.direct_first_execution.enabled)
        self.assertEqual(
            list(cfg.active_topology.direct_first_execution.goal_types),
            ["object", "description", "image"],
        )
        self.assertEqual(
            list(cfg.active_topology.projection_only_goal_types), ["image"]
        )
        self.assertIn("v7_1_b_direct_first", cfg.exp_name)

    def test_v7_1_c_has_one_shared_predeclared_gate_grid(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_1_c_gate_qwen3vl_dashscope_train_fast.yaml",
        ))
        gate = cfg.active_topology.revisit_gate
        self.assertTrue(gate.enabled)
        self.assertEqual(list(gate.goal_types), ["object", "description", "image"])
        self.assertEqual(gate.selected_profile, "medium")
        self.assertEqual(gate.min_relevance_margin, 0.05)
        self.assertEqual(gate.max_budget_fraction, 0.20)
        self.assertEqual(gate.max_revisit_to_explore_ratio, 1.25)
        self.assertEqual(
            sorted(gate.parameter_grid.keys()), ["lenient", "medium", "strict"]
        )

    def test_v7_1_d_inherits_strict_gate_and_adds_one_commitment_policy(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_1_d_commitment_strict_qwen3vl_dashscope_train_fast.yaml",
        ))
        self.assertEqual(cfg.active_topology.revisit_gate.selected_profile, "strict")
        commitment = cfg.active_topology.target_commitment
        self.assertTrue(commitment.enabled)
        self.assertEqual(commitment.switch_relevance_margin, 0.10)
        self.assertEqual(commitment.max_no_progress_segments, 2)

    def test_final_v7_is_simplified_behavior_preserving_overlay(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train.yaml",
        ))
        self.assertEqual(cfg.active_topology.revisit_gate.selected_profile, "strict")
        self.assertTrue(cfg.active_topology.goal_conditioned_overlay.enabled)
        self.assertEqual(
            list(cfg.active_topology.goal_conditioned_overlay.goal_types),
            ["object", "description", "image"],
        )
        self.assertFalse(cfg.active_topology.get(
            "lightweight_confidence", {}
        ).get("enabled", False))
        self.assertFalse(cfg.active_topology.get(
            "dynamic_goal_topology", {}
        ).get("enabled", False))
        self.assertFalse(cfg.active_topology.get("target_commitment", {}).get(
            "enabled", False
        ))
        self.assertIn("v7_overlay_final", cfg.exp_name)

    def test_final_v7_fast_changes_only_manifest_and_output_identity(self):
        root = os.path.dirname(os.path.dirname(__file__))
        full = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train.yaml",
        ))
        fast = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train_fast.yaml",
        ))
        self.assertEqual(
            full.active_topology.goal_conditioned_overlay,
            fast.active_topology.goal_conditioned_overlay,
        )
        self.assertEqual(
            full.active_topology.revisit_gate,
            fast.active_topology.revisit_gate,
        )
        self.assertNotEqual(full.episode_manifest, fast.episode_manifest)

    def test_revised_v7_binds_gate_to_selected_revisit(self):
        root = os.path.dirname(os.path.dirname(__file__))
        frozen = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train.yaml",
        ))
        revised = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_revised_qwen3vl_dashscope_train.yaml",
        ))
        self.assertFalse(frozen.active_topology.revisit_gate.get(
            "bind_selected_action", False
        ))
        self.assertTrue(
            revised.active_topology.revisit_gate.bind_selected_action
        )
        self.assertEqual(
            frozen.active_topology.goal_conditioned_overlay,
            revised.active_topology.goal_conditioned_overlay,
        )
        self.assertEqual(frozen.episode_manifest, revised.episode_manifest)
        self.assertIn("v7_revised_selected_gate", revised.exp_name)

        fast = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_revised_qwen3vl_dashscope_train_fast.yaml",
        ))
        self.assertTrue(fast.active_topology.revisit_gate.bind_selected_action)
        self.assertEqual(
            revised.active_topology.goal_conditioned_overlay,
            fast.active_topology.goal_conditioned_overlay,
        )
        self.assertNotEqual(revised.episode_manifest, fast.episode_manifest)

    def test_internal_validation_is_frozen_and_scene_disjoint(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_final_qwen3vl_dashscope_train_internal_val.yaml",
        ))
        self.assertEqual(
            cfg.episode_manifest,
            "cfg/manifests/goat_train_internal_val_12x2_seed77.json",
        )
        self.assertTrue(cfg.active_topology.goal_conditioned_overlay.enabled)
        self.assertIn("internal_val", cfg.exp_name)
        with open(os.path.join(
            root, "cfg", "manifests", "goat_train_dev_12x2_seed77.json"
        ), "r", encoding="utf-8") as handle:
            development = json.load(handle)
        with open(os.path.join(
            root, "cfg", "manifests",
            "goat_train_internal_val_12x2_seed77.json",
        ), "r", encoding="utf-8") as handle:
            validation = json.load(handle)
        dev_scenes = {
            item["scene_name"]
            for entries in development["splits"].values()
            for item in entries
        }
        validation_scenes = {
            item["scene_name"]
            for entries in validation["splits"].values()
            for item in entries
        }
        self.assertEqual(len(validation_scenes), 12)
        self.assertFalse(dev_scenes & validation_scenes)
        self.assertEqual(
            [len(validation["splits"][key]) for key in ("1", "2")],
            [12, 12],
        )

    def test_validation2_is_frozen_and_disjoint_from_revealed_scenes(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_revised_qwen3vl_dashscope_train_validation2.yaml",
        ))
        self.assertTrue(cfg.active_topology.revisit_gate.bind_selected_action)
        self.assertEqual(
            cfg.episode_manifest,
            "cfg/manifests/goat_train_validation2_12x2_seed77.json",
        )
        manifests = []
        for filename in (
            "goat_train_dev_12x2_seed77.json",
            "goat_train_internal_val_12x2_seed77.json",
            "goat_train_validation2_12x2_seed77.json",
        ):
            with open(os.path.join(
                root, "cfg", "manifests", filename
            ), "r", encoding="utf-8") as handle:
                manifests.append(json.load(handle))
        scene_sets = [
            {
                item["scene_name"]
                for entries in manifest["splits"].values()
                for item in entries
            }
            for manifest in manifests
        ]
        self.assertEqual([len(items) for items in scene_sets], [12, 12, 12])
        self.assertFalse(scene_sets[0] & scene_sets[2])
        self.assertFalse(scene_sets[1] & scene_sets[2])
        self.assertEqual(
            [
                len(manifests[2]["splits"][key])
                for key in ("1", "2")
            ],
            [12, 12],
        )

    def test_dynamic_v7_is_a_separate_two_stage_c_strict_extension(self):
        root = os.path.dirname(os.path.dirname(__file__))
        shadow = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dynamic_shadow_qwen3vl_dashscope_train_fast.yaml",
        ))
        active = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dynamic_active_qwen3vl_dashscope_train_fast.yaml",
        ))
        self.assertEqual(shadow.active_topology.revisit_gate.selected_profile, "strict")
        self.assertTrue(shadow.active_topology.dynamic_goal_topology.enabled)
        self.assertTrue(shadow.active_topology.dynamic_goal_topology.shadow_only)
        self.assertFalse(active.active_topology.dynamic_goal_topology.shadow_only)
        self.assertFalse(active.active_topology.get(
            "lightweight_confidence", {}
        ).get("enabled", False))

    def test_dynamic_v7_full_keeps_fast_parameters_and_changes_manifest_only(self):
        root = os.path.dirname(os.path.dirname(__file__))
        fast = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dynamic_active_qwen3vl_dashscope_train_fast.yaml",
        ))
        full = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dynamic_active_qwen3vl_dashscope_train.yaml",
        ))
        self.assertEqual(
            full.active_topology.dynamic_goal_topology,
            fast.active_topology.dynamic_goal_topology,
        )
        self.assertEqual(
            full.episode_manifest,
            "cfg/manifests/goat_train_dev_12x2_seed77.json",
        )

    def test_dual_layer_v7_is_isolated_and_shared_by_all_goal_types(self):
        root = os.path.dirname(os.path.dirname(__file__))
        fast = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dual_layer_qwen3vl_dashscope_train_fast.yaml",
        ))
        full = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_dual_layer_qwen3vl_dashscope_train.yaml",
        ))
        self.assertTrue(fast.active_topology.dual_layer_planner.enabled)
        self.assertEqual(
            list(fast.active_topology.dual_layer_planner.goal_types),
            ["object", "description", "image"],
        )
        self.assertTrue(fast.active_topology.dynamic_goal_topology.enabled)
        self.assertFalse(fast.active_topology.dynamic_goal_topology.shadow_only)
        self.assertEqual(
            fast.active_topology.revisit_gate.selected_profile, "strict"
        )
        self.assertTrue(
            fast.active_topology.revisit_gate.bind_selected_action
        )
        self.assertFalse(fast.active_topology.get(
            "target_commitment", {}
        ).get("enabled", False))
        self.assertEqual(
            full.episode_manifest,
            "cfg/manifests/goat_train_dev_12x2_seed77.json",
        )

    def test_v7_2_keeps_dual_layer_policy_and_adds_explore_commitment(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_2_explore_commitment_qwen3vl_dashscope_train_fast.yaml",
        ))
        planner = cfg.active_topology.dual_layer_planner
        self.assertTrue(planner.enabled)
        self.assertTrue(planner.explore_commitment.enabled)
        self.assertEqual(planner.explore_commitment.min_progress_voxels, 1.0)
        self.assertEqual(planner.explore_commitment.max_no_progress_steps, 3)

    def test_v7_2b_adds_only_geometric_frontier_rebinding(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_2b_geometric_rebind_qwen3vl_dashscope_train_fast.yaml",
        ))
        rebind = cfg.active_topology.dual_layer_planner.explore_commitment.geometric_rebind
        self.assertTrue(rebind.enabled)
        self.assertEqual(rebind.max_anchor_distance_voxels, 8.0)

    def test_v7_3_adds_vlm_local_choice_to_the_topology_shortlist(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_goal_topomap_v7_3_topo_shortlist_qwen3vl_dashscope_train_fast.yaml",
        ))
        planner = cfg.active_topology.dual_layer_planner
        self.assertTrue(planner.explore_shortlist.enabled)
        self.assertTrue(planner.explore_commitment.enabled)
        self.assertTrue(planner.explore_commitment.geometric_rebind.enabled)

    def test_projection_only_goal_type_normalizes_object_alias(self):
        self.assertTrue(
            active_topology_goal_type_is_listed("category", ["object"])
        )
        self.assertTrue(
            active_topology_goal_type_is_listed("object", ["category"])
        )

    def test_goal_topomap_legacy_config_remains_unrestricted(self):
        base = resolve_active_topology_policy("goal_topo_map", "image")
        self.assertIs(restrict_active_topology_policy(base, "image", None), base)

    def test_ablation_configs_only_override_mode_and_output_identity(self):
        root = os.path.dirname(os.path.dirname(__file__))
        expected = {
            "eval_goatbench_activetopo_stable_only_qwen3vl_dashscope.yaml": "stable_only",
            "eval_goatbench_activetopo_ca_qwen3vl_dashscope.yaml": "confidence_accessibility",
            "eval_goatbench_activetopo_category_only_qwen3vl_dashscope.yaml": "category_only",
        }
        exp_names = set()
        for filename, mode in expected.items():
            cfg = _load_config(os.path.join(root, "cfg", filename))
            self.assertTrue(cfg.active_topology.enabled)
            self.assertEqual(cfg.active_topology.mode, mode)
            self.assertEqual(cfg.vlm_model, "qwen3-vl-30b-a3b-instruct")
            self.assertEqual(cfg.vlm_max_images, 40)
            self.assertEqual(cfg.seed, 77)
            self.assertTrue(cfg.choose_every_step)
            exp_names.add(str(cfg.exp_name))
            resolve_active_topology_policy(mode, "category")
        self.assertEqual(len(exp_names), len(expected))

    def test_belief_category_config_is_isolated_and_frozen(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_belieftopo_category_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "belief_category")
        self.assertEqual(cfg.active_topology.belief.top_k, 3)
        self.assertEqual(cfg.active_topology.belief.prior, 0.20)
        self.assertEqual(cfg.active_topology.verification.radii_m, [1.0, 1.5, 2.0])
        self.assertEqual(cfg.vlm_max_images, 40)
        self.assertEqual(cfg.seed, 77)
        self.assertIn("belieftopo_category", cfg.exp_name)

    def test_belief_v2_config_enables_only_v2_mechanisms(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_belieftopo_v2_category_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "belief_category_v2")
        self.assertTrue(cfg.active_topology.belief.fast_evidence_path)
        self.assertEqual(
            cfg.active_topology.belief.unknown_method, "weighted_geometric"
        )
        self.assertEqual(
            cfg.active_topology.belief.clip_calibration.warmup_samples, 16
        )
        self.assertEqual(cfg.active_topology.semantic_cache.min_visual_cosine, 0.85)
        self.assertEqual(cfg.active_topology.confirmation.max_steps, 5)
        self.assertEqual(cfg.vlm_model, "qwen3-vl-30b-a3b-instruct")
        self.assertEqual(cfg.seed, 77)
        self.assertIn("belieftopo_v2", cfg.exp_name)

    def test_belief_v3_config_is_separate_and_inherits_v2(self):
        root = os.path.dirname(os.path.dirname(__file__))
        cfg = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_belieftopo_v3_category_qwen3vl_dashscope.yaml",
        ))
        self.assertEqual(cfg.active_topology.mode, "belief_category_v3")
        self.assertTrue(cfg.active_topology.belief.fast_evidence_path)
        self.assertEqual(cfg.active_topology.belief.unknown_method, "weighted_geometric")
        self.assertEqual(cfg.active_topology.snapshot_gate.min_detections, 2)
        self.assertEqual(cfg.active_topology.snapshot_gate.alias_min_posterior, 0.85)
        self.assertFalse(cfg.active_topology.belief.use_vlm_tiebreak)
        self.assertEqual(cfg.active_topology.belief.unknown_action_threshold, 0.75)
        self.assertEqual(
            cfg.active_topology.belief.max_consecutive_unknown_actions, 2
        )
        self.assertEqual(cfg.active_topology.confirmation.max_steps, 8)
        self.assertEqual(cfg.active_topology.snapshot_gate.rejection_ttl_steps, 5)
        self.assertIn("belieftopo_v3", cfg.exp_name)

        smoke = _load_config(os.path.join(
            root, "cfg",
            "eval_goatbench_belieftopo_v3_category_qwen3vl_dashscope_smoke.yaml",
        ))
        self.assertEqual(smoke.active_topology.mode, "belief_category_v3")
        self.assertFalse(smoke.active_topology.belief.use_vlm_tiebreak)
        self.assertTrue(str(smoke.exp_name).endswith("_smoke"))


if __name__ == "__main__":
    unittest.main()
