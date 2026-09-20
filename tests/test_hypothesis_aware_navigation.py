import ast
import copy
import pickle
import random
import unittest
import tempfile
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import supervision as sv
from omegaconf import OmegaConf

from src.dual_dynamic_navigation import BaselineDualDynamicNavigator
from src.dual_dynamic_navigation.unified_navigator import HypothesisAwareNavigator
from src.dual_dynamic_navigation.task_planner import ActionKind, AuthorizedTask, revisit_allowed
from src.dual_dynamic_navigation.feedback import verify_entity
from src.hypothesis_graph import HypothesisGraph, HypothesisNode, NodeType, NodeStatus, CognitiveDependency
from src.place_topology import PlaceTopology
from src.query_vlm_goatbench import query_vlm_for_response
from src.tsdf_planner import SnapShot
from tests import test_known_space_execution as known_tests


class HypothesisAwareNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.goal_image = str(Path(self.temp.name) / "target.png")
        Image.fromarray(np.zeros((8,8,3), np.uint8)).save(self.goal_image)

    def fixture(self, stage="verified"):
        mask = np.ones((10, 7), bool)
        planner = known_tests.KnownSpaceExecutionTests().planner(mask, current=(1, 3))
        planner.hypothesis_graph = HypothesisGraph({})
        graph = PlaceTopology(2., 1., stage="route_only")
        place = graph.observe_pose([1, 3], 0).place_id
        graph.bind_observation("snapshot:a.png", place, [1, 3], 0)
        snap = SnapShot("a.png", (1., 0., 0.), np.array([1, 3]),
                        full_obj_list={7: .9}, cluster=[7], position=np.array([4, 3]))
        snap.visual_prompt = sv.Detections(xyxy=np.array([[1., 1., 6., 6.]]),
            confidence=np.array([.9]), class_id=np.array([0]), data={"obj_id": np.array([7])})
        scene = SimpleNamespace(objects={7: {"bbox": SimpleNamespace(center=np.array([4., 0., 3.])),
                                             "num_detections": 2, "class_name": "chair", "conf": .9}},
            object_id_aliases={}, snapshots={"a.png": snap},
            all_observations={"a.png": np.zeros((8, 8, 3), np.uint8)})
        planner.frontiers = []
        cfg = OmegaConf.create({"planner": vars(known_tests.KnownSpaceExecutionTests().cfg()),
            "egocentric_views": False, "use_full_obj_list": False, "prompt_h": 8, "prompt_w": 8,
            "prefiltering": False, "top_k_categories": 10, "vlm_model": "offline-test"})
        navigator = HypothesisAwareNavigator({"stage": stage})
        navigator.begin_goal("g1")
        navigator.sync(graph, scene, [], "step:0")
        route = {"status": "valid", "terminal": [4., 3.], "look_at": [5., 3.],
                 "approach_place_id": place, "route_places": [place],
                 "certified_path": [[1.,3.],[2.,3.],[3.,3.],[4.,3.]], "total_distance_m": 3.}
        return navigator, scene, planner, graph, cfg, route

    def metadata(self, task="object"):
        return {"question": "find chair", "task_type": task, "class": "chair",
                "image": self.goal_image if task == "image" else None}

    def response(self):
        return "snapshot 0, object 0", [0], {"a.png": [0]}, "visual match", 1

    def task(self, scene, route, kind=ActionKind.APPROACH):
        choice = copy.copy(scene.snapshots["a.png"])
        choice.hierarchical_candidate_crop = np.zeros((5,5,3), np.uint8)
        return AuthorizedTask(kind, "a.png", "7", "digest:a", route, choice)

    def test_original_query_shadow_replay_keeps_pixels_mapping_and_rng(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        observer = BaselineDualDynamicNavigator({"stage": "shadow"})
        observer.begin_goal("g1")
        initial = random.getstate(), np.random.get_state()
        with patch("src.query_vlm_goatbench.explore_step", return_value=self.response()) as request:
            original, _ = query_vlm_for_response(self.metadata(), scene, planner, [], cfg)
            baseline_payload = pickle.dumps(request.call_args.args[0])
            expected_rng = random.getstate(), np.random.get_state()
            random.setstate(initial[0])
            np.random.set_state(initial[1])
            observer.sync(graph, scene, [], "step:0")
            shadow, _ = query_vlm_for_response(self.metadata(), scene, planner, [], cfg)
            shadow_payload = pickle.dumps(request.call_args.args[0])
        self.assertEqual(original.cluster, shadow.cluster)
        self.assertEqual(original.image, shadow.image)
        self.assertEqual(baseline_payload, shadow_payload)
        self.assertEqual(random.getstate(), expected_rng[0])
        np.testing.assert_equal(np.random.get_state(), expected_rng[1])
        self.assertEqual(request.call_count, 2)

    def test_actual_hgr_request_source_authorization_and_approach_install(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        with patch.object(nav.resolver, "object_view", return_value=route), \
             patch("src.query_vlm_goatbench.explore_step", return_value=self.response()) as request:
            selected = nav.select(query_vlm_for_response, self.metadata(), scene, planner,
                                  graph, [], cfg, np.zeros(3), {7}, remaining_steps=10)
        self.assertEqual(selected[0].cluster, [7])
        self.assertEqual(nav.active_intent.task.kind, ActionKind.APPROACH)
        self.assertIn("navigation_task_context", request.call_args.args[0])
        planner.max_point = planner.target_point = None
        self.assertTrue(nav.setup(planner, np.zeros(3), cfg))
        result = nav.executor.step(planner, np.zeros(3), 0., scene.objects, scene.snapshots,
            cfg.planner, certified_route=nav.active_intent.task.route)
        self.assertIsNotNone(result[0])
        planner.get_distance.assert_not_called()

    def test_historical_snapshot_creates_revisit_without_stop_authority(self):
        nav, scene, planner, graph, cfg, route = self.fixture("memory")
        with patch.object(nav.resolver, "object_view", return_value=route), \
             patch("src.query_vlm_goatbench.explore_step", return_value=self.response()):
            nav.select(query_vlm_for_response, self.metadata(), scene, planner, graph, [], cfg,
                       np.zeros(3), set(), remaining_steps=10)
        self.assertEqual(nav.active_intent.task.kind, ActionKind.REVISIT)

    def test_arrived_revisit_promotes_same_intent_without_suppressing_entity(self):
        nav, scene, planner, _, _, route = self.fixture("memory")
        intent = nav.install(
            self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph
        )
        result = nav.promote_revisit(intent.intent_id)
        self.assertEqual(result["source_id"], "a.png")
        self.assertIs(nav.active_intent, intent)
        self.assertEqual(nav.active_intent.task.kind, ActionKind.APPROACH)
        self.assertEqual(nav.active_intent.state, "arrived")
        self.assertEqual(result["transition"], "REVISIT->APPROACH")
        self.assertTrue(result["stop_authorized"])
        self.assertNotIn("7", nav.suppressed)
        self.assertEqual(nav.checked_views["7"], [[4.0, 3.0]])
        self.assertEqual(nav.stats["completed_revisits"], 1)
        self.assertEqual(nav.stats["promoted_revisits"], 1)

    def test_revisit_promotion_requires_matching_authoritative_intent(self):
        nav, scene, planner, _, _, route = self.fixture("memory")
        intent = nav.install(
            self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph
        )
        self.assertIsNone(nav.promote_revisit("stale-intent"))
        self.assertIs(nav.active_intent, intent)
        self.assertEqual(intent.task.kind, ActionKind.REVISIT)

    def test_adaptive_verifies_only_cross_goal_instance_sensitive_revisit(self):
        _, scene, planner, graph, _, route = self.fixture("memory")
        nav = HypothesisAwareNavigator({
            "stage": "adaptive",
            "enable_persistent_intent": True,
            "enable_terminal_verification": True,
            "enable_selective_verification": True,
            "selective_verification_goal_types": ["image", "description"],
            "max_frontier_intent_motion_steps": 3,
        })
        nav.begin_goal("g1")
        nav.sync(graph, scene, [], "g1:step:0")
        nav.begin_goal("g2")

        image_revisit = self.task(scene, route, ActionKind.REVISIT)
        nav.configure_task_policy(image_revisit, self.metadata("image"))
        self.assertTrue(image_revisit.requires_verification)

        object_revisit = self.task(scene, route, ActionKind.REVISIT)
        nav.configure_task_policy(object_revisit, self.metadata("object"))
        self.assertFalse(object_revisit.requires_verification)

        fresh_image = self.task(scene, route, ActionKind.APPROACH)
        nav.configure_task_policy(fresh_image, self.metadata("image"))
        self.assertFalse(fresh_image.requires_verification)
        self.assertEqual(nav.stats["selective_verification_tasks"], 1)

        intent = nav.install(image_revisit, planner.hypothesis_graph)
        self.assertEqual(
            nav.feedback("confirmed", intent.intent_id, "adaptive:verify:1", planner),
            "stop",
        )
        self.assertEqual(intent.verification_views, 1)

    def test_adaptive_frontier_intent_expires_after_bounded_motion_window(self):
        _, scene, planner, _, _, route = self.fixture("memory")
        nav = HypothesisAwareNavigator({
            "stage": "adaptive",
            "enable_persistent_intent": True,
            "enable_terminal_verification": True,
            "enable_selective_verification": True,
            "max_frontier_intent_motion_steps": 3,
        })
        nav.begin_goal("g1")
        intent = nav.install(
            self.task(scene, route, ActionKind.EXPLORE), planner.hypothesis_graph
        )
        intent.motion_steps = 3
        self.assertIsNone(nav.continue_choice(scene, planner))
        self.assertIsNone(nav.active_intent)
        self.assertEqual(nav.stats["bounded_intent_expirations"], 1)

    def test_actual_runner_revisit_arrival_promotes_without_early_continue(self):
        source = Path(__file__).resolve().parents[1] / "run_goatbench_evaluation.py"
        tree = ast.parse(source.read_text())

        def calls_promotion(node):
            return any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "promote_revisit"
                for child in ast.walk(node)
            )

        branches = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and calls_promotion(node)
        ]
        self.assertTrue(branches)
        branch = min(branches, key=lambda node: node.end_lineno - node.lineno)
        self.assertFalse(any(
            isinstance(child, ast.Continue) for child in ast.walk(branch)
        ))

    def test_trace_marks_exact_source_reused_from_prior_goal(self):
        nav, scene, planner, _, _, route = self.fixture("memory")
        nav.begin_goal("g2")
        nav.install(self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph)
        task = nav.to_trace_dict()["active_intent"]["task"]
        self.assertTrue(task["historical_source"])
        self.assertEqual(task["source_first_seen_goal"], "g1")

    def test_wrong_source_and_remaining_budget_do_not_install_intent(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        with patch.object(nav.resolver, "object_view", return_value=route):
            query = Mock(return_value=(SnapShot("wrong.png", (1,0,0), np.array([0,0]), cluster=[7]), 1))
            self.assertIsNone(nav.select(query, self.metadata(), scene, planner, graph, [], cfg,
                np.zeros(3), {7}, remaining_steps=10))
            self.assertIsNone(nav.active_intent)
            query.reset_mock()
            self.assertIsNone(nav.select(query, self.metadata(), scene, planner, graph, [], cfg,
                np.zeros(3), {7}, remaining_steps=0))
            query.assert_not_called()

    def test_feedback_scope_uncertainty_error_and_expired_events(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        intent = nav.install(self.task(scene, route), planner.hypothesis_graph)
        self.assertEqual(nav.feedback("uncertain", "old", "e0", planner), "ignored")
        self.assertEqual(nav.feedback("uncertain", intent.intent_id, "e1", planner), "new_view")
        self.assertEqual(nav.feedback("uncertain", intent.intent_id, "e1", planner), "ignored")
        self.assertEqual(nav.feedback("error", intent.intent_id, "e2", planner), "reselect")
        self.assertNotIn("7", nav.goal_graph.entities.get("7", SimpleNamespace(checked_evidence={})).checked_evidence)
        self.assertIn("7", nav.scene_map.entities)
        self.assertNotIn("7", nav.suppressed)

    def test_original_hgr_mode_preserves_candidate_view_then_authorizes_source(self):
        nav, scene, planner, graph, cfg, route = self.fixture("memory")
        nav = HypothesisAwareNavigator({
            "stage": "memory", "semantic_selection_mode": "original_hgr",
            "enable_candidate_annotations": False,
            "enable_revisit_cost_gate": False,
            "enable_unscored_frontier_fallback": False,
        })
        nav.begin_goal("g1")
        nav.sync(graph, scene, [], "step:0")
        query = Mock(return_value=(scene.snapshots["a.png"], 1))
        with patch.object(nav.resolver, "object_view", return_value=route):
            selected = nav.select(query, self.metadata(), scene, planner, graph, [], cfg,
                                  np.zeros(3), {7}, remaining_steps=0)
        self.assertIsNotNone(selected)
        kwargs = query.call_args.kwargs
        self.assertIsNone(kwargs["eligible_objects_by_snapshot"])
        self.assertIsNone(kwargs["allowed_frontier_indices"])
        self.assertIsNone(kwargs["navigation_task_context"])
        self.assertEqual(nav.active_intent.task.source_id, "a.png")

    def test_original_hgr_mode_request_payload_matches_original_selector(self):
        _, scene, planner, graph, cfg, route = self.fixture("memory")
        nav = HypothesisAwareNavigator({
            "stage": "memory", "semantic_selection_mode": "original_hgr",
            "enable_candidate_annotations": False,
        })
        nav.begin_goal("g1")
        nav.sync(graph, scene, [], "step:0")
        rng = random.getstate(), np.random.get_state()
        with patch("src.query_vlm_goatbench.explore_step", return_value=self.response()) as request:
            query_vlm_for_response(self.metadata(), scene, planner, [], cfg)
            original_payload = pickle.dumps(request.call_args.args[0])
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            with patch.object(nav.resolver, "object_view", return_value=route):
                nav.select(query_vlm_for_response, self.metadata(), scene, planner, graph, [], cfg,
                           np.zeros(3), {7}, remaining_steps=10)
            unified_payload = pickle.dumps(request.call_args.args[0])
        self.assertEqual(unified_payload, original_payload)

    def test_original_hgr_mode_rejects_selected_source_without_known_route(self):
        nav, scene, planner, graph, cfg, _ = self.fixture("memory")
        nav = HypothesisAwareNavigator({
            "stage": "memory", "semantic_selection_mode": "original_hgr",
        })
        nav.begin_goal("g1")
        nav.sync(graph, scene, [], "step:0")
        query = Mock(return_value=(scene.snapshots["a.png"], 1))
        with patch.object(nav.resolver, "object_view",
                          return_value={"status": "unavailable", "reason": "blocked"}):
            self.assertIsNone(nav.select(query, self.metadata(), scene, planner, graph, [], cfg,
                                         np.zeros(3), {7}, remaining_steps=10))
        self.assertEqual(nav.last_selection.reason, "selected_source_not_executable")
        self.assertIsNone(nav.active_intent)

    def test_cost_gated_revisit_reselects_an_authorized_frontier(self):
        nav, scene, planner, graph, cfg, object_route = self.fixture("intent")
        nav = HypothesisAwareNavigator({
            "stage": "intent", "semantic_selection_mode": "original_hgr",
            "enable_revisit_cost_gate": True,
            "enable_persistent_intent": True,
        })
        nav.begin_goal("g1")
        frontier = SimpleNamespace(cluster=[], topo_id="frontier-test",
                                   hypothesis_node_id=None)
        planner.frontiers = [frontier]
        nav.sync(graph, scene, planner.frontiers, "step:0")
        frontier_route = {**object_route, "total_distance_m": 1.0,
                          "terminal": [2., 3.]}
        object_route = {**object_route, "total_distance_m": 8.0}
        query = Mock(side_effect=[(scene.snapshots["a.png"], 1), (frontier, 2)])
        with patch.object(nav.resolver, "object_view", return_value=object_route), \
             patch.object(nav.resolver, "frontier", return_value=frontier_route):
            selected = nav.select(query, self.metadata(), scene, planner, graph, [], cfg,
                                  np.zeros(3), set(), remaining_steps=10)
        self.assertIs(selected[0], frontier)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(query.call_args.kwargs["request_purpose"],
                         "dual_topo_authorization_reselection")
        self.assertEqual(nav.active_intent.task.kind, ActionKind.EXPLORE)
        self.assertEqual(nav.stats["authorization_reselections"], 1)

    def test_revisit_confirmation_transitions_same_entity_without_extra_motion(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        intent = nav.install(self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph)
        self.assertEqual(nav.feedback("confirmed", intent.intent_id, "e1", planner), "stop")
        self.assertEqual(intent.task.kind, ActionKind.APPROACH)
        self.assertEqual(intent.state, "confirmed")
        self.assertEqual(nav.feedback("confirmed", intent.intent_id, "e2", planner), "ignored")

    def test_historical_source_requires_two_confirmed_views(self):
        nav, scene, planner, _, _, route = self.fixture()
        nav.begin_goal("g2")
        intent = nav.install(
            self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph
        )
        self.assertTrue(nav.task_uses_historical_source(intent.task))
        self.assertEqual(nav.feedback("confirmed", intent.intent_id, "e1", planner), "new_view")
        self.assertNotEqual(intent.state, "confirmed")
        self.assertEqual(nav.feedback("confirmed", intent.intent_id, "e2", planner), "stop")
        self.assertEqual(intent.state, "confirmed")
        self.assertEqual(nav.stats["historical_confirmation_deferrals"], 1)

    def test_entity_alias_migrates_source_coverage_and_intent(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        nav.install(self.task(scene, route), planner.hypothesis_graph)
        scene.objects[9] = scene.objects.pop(7)
        scene.object_id_aliases = {7: 9}
        nav.checked_views["7"] = [[4, 3]]
        nav.sync(graph, scene, [], "merge:1")
        choice = nav.continue_choice(scene, planner)
        self.assertEqual(choice.cluster, [9])
        self.assertEqual(nav.active_intent.task.entity_id, "9")
        self.assertEqual(nav.checked_views["9"], [[4, 3]])

    def test_pin_preserves_invalidated_node_until_release(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        hypothesis = HypothesisNode("h1", NodeType.HYPOTHESIS)
        planner.hypothesis_graph.add_node(hypothesis)
        task = self.task(scene, route, ActionKind.EXPLORE)
        task.hypothesis_refs = ("h1",)
        nav.install(task, planner.hypothesis_graph)
        hypothesis.status = NodeStatus.INVALIDATED
        planner.hypothesis_graph._remove_node("h1")
        self.assertIn("h1", planner.hypothesis_graph.nodes)
        nav.release("invalidated", planner.hypothesis_graph)
        self.assertNotIn("h1", planner.hypothesis_graph.nodes)

    def test_actual_verification_payload_binds_entity_and_separates_errors(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        choice = self.task(scene, route).choice
        for task_type in ("object", "description", "image"):
            with self.subTest(task_type=task_type), patch(
                    "src.dual_dynamic_navigation.feedback.validated_request",
                    return_value=({"verdict": "uncertain", "reason": "occluded"}, {})) as request:
                verdict, _ = verify_entity(self.metadata(task_type), np.zeros((8,8,3),np.uint8), choice, cfg)
                self.assertEqual(verdict, "uncertain")
                self.assertIn("SAME physical entity", request.call_args.args[0])
                self.assertEqual(len(request.call_args.args[1]), 3)
        with patch("src.dual_dynamic_navigation.feedback.validated_request", return_value=(None, {})):
            self.assertEqual(verify_entity(self.metadata(), np.zeros((8,8,3),np.uint8), choice, cfg)[0], "error")

    def test_incremental_error_keeps_intent_and_deduplicates_same_physical_view(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        intent = nav.install(self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph)
        query = Mock(return_value=(self.task(scene, route).choice, 1))
        verifier = Mock(return_value=("error", {}))
        args = (query, self.metadata(), scene, planner, graph, [], cfg, np.zeros(3), {7},
                np.zeros((8,8,3),np.uint8), verifier)
        self.assertFalse(nav.assess_incremental(*args))
        scene.all_observations["a.png"] += 1
        self.assertFalse(nav.assess_incremental(*args))
        self.assertIs(nav.active_intent, intent)
        self.assertEqual(query.call_count, 1)

    def test_checkpoint_restores_active_intent_and_rejects_stage_change(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        nav.install(self.task(scene, route), planner.hypothesis_graph)
        restored = pickle.loads(pickle.dumps(nav))
        restored.validate_restore({"stage": "verified"})
        self.assertEqual(restored.active_intent.intent_id, nav.active_intent.intent_id)
        with self.assertRaises(ValueError):
            restored.validate_restore({"stage": "memory"})

    def test_revisit_cost_gate_is_separate_from_object_identity(self):
        self.assertFalse(revisit_allowed({"status": "valid", "total_distance_m": 8.}, [2.], 1.))
        self.assertTrue(revisit_allowed({"status": "valid", "total_distance_m": 8.}, [], 1.))

    def runner_branch(self, predicate):
        source = Path(__file__).resolve().parents[1] / "run_goatbench_evaluation.py"
        node = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.If) and predicate(n))
        return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "runner-branch", "exec")

    def test_actual_runner_motion_branch_uses_the_authoritative_certified_route(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        nav.install(self.task(scene, route), planner.hypothesis_graph)
        cfg.save_visualization = False
        block = self.runner_branch(lambda n: isinstance(n.test, ast.Name)
            and n.test.id == "known_space_execution" and any(isinstance(child, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "return_values" for t in child.targets)
            for child in n.body))
        env = {"known_space_execution": True, "unified_enabled": True, "known_executor": nav.executor,
               "baseline_dual_dynamic": nav, "baseline_dual_dynamic_stage": "verified",
               "tsdf_planner": planner, "pts": np.zeros(3), "angle": 0., "scene": scene,
               "cfg": cfg, "topology_trace": Mock(), "subtask_id": "g1", "global_step": 0}
        exec(block, env)
        self.assertIsNotNone(env["return_values"][0])
        planner.get_distance.assert_not_called()
        self.assertIs(nav.executor.installed_route, nav.active_intent.task.route)
        self.assertTrue(env["topology_trace"].write.call_args.args[0]["audit"]["valid"])

    def test_actual_runner_persistent_waypoint_does_not_clear_target(self):
        nav, scene, planner, graph, cfg, route = self.fixture("intent")
        nav.install(self.task(scene, route, ActionKind.EXPLORE), planner.hypothesis_graph)
        class Frontier:
            pass
        frontier = Frontier()
        planner.max_point = frontier
        target = planner.target_point.copy()
        block = self.runner_branch(lambda n: isinstance(n.test, ast.Attribute) and n.test.attr == "choose_every_step")
        env = {"cfg": SimpleNamespace(choose_every_step=True), "unified_enabled": True,
               "baseline_dual_dynamic": nav, "dual_v2_enabled": False, "object_approach_enabled": False,
               "tsdf_planner": planner, "Frontier": Frontier}
        exec(block, env)
        np.testing.assert_equal(planner.target_point, target)
        self.assertIs(planner.max_point, frontier)

    def test_checkpoint_schema4_actual_geometry_roundtrip_and_contract(self):
        from src.hgr_experiment_state import SCENE_MEMORY, save_checkpoint, load_checkpoint
        import open3d as o3d
        nav, scene, planner, graph, cfg, route = self.fixture()
        nav.install(self.task(scene, route), planner.hypothesis_graph)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array([[1.,2.,3.]]))
        memory_scene = SimpleNamespace(**{key: {} for key in SCENE_MEMORY})
        memory_scene.objects = {7: {"pcd": cloud}}
        config = {"tsdf_grid_size": 1., "navigation_backend": "hgr_dual_topo_verified",
                  "dual_dynamic_navigation": {"stage": "verified"}}
        path = Path(self.temp.name) / "mid-intent.pkl"
        save_checkpoint(path, memory_scene, {}, graph, None, [0,0,0], 0., 1, 0,
                        config, "scene", "0", {}, navigation=nav)
        restored = load_checkpoint(path, config, "scene", "0")
        self.assertEqual(restored["version"], 4)
        np.testing.assert_equal(np.asarray(restored["scene_memory"]["objects"][7]["pcd"].points), [[1,2,3]])
        self.assertEqual(restored["navigation"].active_intent.intent_id, nav.active_intent.intent_id)
        wrong = {**config, "navigation_backend": "hgr_dual_topo_memory"}
        with self.assertRaises(ValueError):
            load_checkpoint(path, wrong, "scene", "0")

    def test_incremental_confirmation_preempts_only_after_same_entity_confirmation(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        old = nav.install(self.task(scene, route, ActionKind.REVISIT), planner.hypothesis_graph)
        query = Mock(return_value=(self.task(scene, route).choice, 1))
        with patch.object(nav.resolver, "object_view", return_value=route):
            matched = nav.assess_incremental(query, self.metadata(), scene, planner, graph, [], cfg,
                np.zeros(3), {7}, np.zeros((8,8,3),np.uint8), Mock(return_value=("confirmed", {})), 10)
        self.assertTrue(matched)
        self.assertNotEqual(nav.active_intent.intent_id, old.intent_id)
        self.assertEqual(nav.active_intent.task.kind, ActionKind.APPROACH)

    def test_static_stage_configs_use_one_backend_and_reject_legacy_mixture(self):
        from run_goatbench_evaluation import _load_config
        from src.dual_dynamic_navigation.config import validate_baseline_backend
        root = Path(__file__).resolve().parents[1]
        for stage in ("memory", "intent", "verified"):
            cfg = _load_config(str(root / "cfg" / f"eval_goatbench_hgr_dual_topo_{stage}.yaml"))
            validate_baseline_backend(cfg)
            self.assertEqual(cfg.dual_dynamic_navigation.stage, stage)
            self.assertFalse(cfg.hgr_dual_topo.enabled)
            cfg.hierarchical_navigation.enabled = True
            with self.assertRaises(ValueError):
                validate_baseline_backend(cfg)

    def test_ablation_configs_expose_the_intended_mechanism_states(self):
        from run_goatbench_evaluation import _load_config
        from src.dual_dynamic_navigation.config import validate_baseline_backend
        root = Path(__file__).resolve().parents[1]
        expected = {
            "memory_semantic_first": ("memory", "original_hgr", False, False, False, False, False),
            "memory_dedup": ("memory", "original_hgr", True, False, False, False, False),
            "memory_cost_gate": ("memory", "original_hgr", True, True, False, False, False),
            "intent_persistent": ("intent", "original_hgr", True, True, True, False, False),
            "intent_preempt": ("intent", "original_hgr", True, True, True, True, False),
            "verified_ablation": ("verified", "original_hgr", True, True, True, False, True),
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                cfg = _load_config(str(
                    root / "cfg" / f"eval_goatbench_hgr_dual_topo_{name}.yaml"
                ))
                validate_baseline_backend(cfg)
                navigation = cfg.dual_dynamic_navigation
                self.assertEqual((
                    navigation.stage, navigation.semantic_selection_mode,
                    navigation.enable_revisit_deduplication,
                    navigation.enable_revisit_cost_gate,
                    navigation.enable_persistent_intent,
                    navigation.enable_incremental_preemption,
                    navigation.enable_terminal_verification,
                ), values)

        adaptive = _load_config(str(
            root / "cfg" / "eval_goatbench_hgr_dual_topo_adaptive_hybrid.yaml"
        ))
        validate_baseline_backend(adaptive)
        navigation = adaptive.dual_dynamic_navigation
        self.assertEqual(navigation.stage, "adaptive")
        self.assertTrue(navigation.enable_revisit_deduplication)
        self.assertFalse(navigation.enable_revisit_cost_gate)
        self.assertTrue(navigation.enable_persistent_intent)
        self.assertEqual(navigation.max_frontier_intent_motion_steps, 3)
        self.assertTrue(navigation.enable_selective_verification)
        self.assertEqual(
            list(navigation.selective_verification_goal_types),
            ["image", "description"],
        )

    def test_multiple_independent_supports_keep_child_until_last_support_fails(self):
        graph = HypothesisGraph({})
        graph.preserve_independent_evidence = True
        for name in ("a", "b", "child"):
            graph.add_node(HypothesisNode(name, NodeType.HYPOTHESIS))
        graph.add_dependency(CognitiveDependency("a", "child", "supports", .9, "test"))
        graph.add_dependency(CognitiveDependency("b", "child", "supports", .9, "test"))
        graph.nodes["a"].mark_falsified("contradiction", 1.)
        graph.cascade_delete("a")
        self.assertEqual(graph.nodes["child"].status, NodeStatus.ACTIVE)
        graph.nodes["b"].mark_falsified("contradiction", 1.)
        graph.cascade_delete("b")
        self.assertNotIn("child", graph.nodes)

    def test_recovery_is_local_then_topological_and_does_not_reset_every_step(self):
        nav, scene, planner, graph, cfg, route = self.fixture()
        intent = nav.install(self.task(scene, route), planner.hypothesis_graph)
        guide = Mock()
        guide.repair.return_value = False
        nav.executor.guidance = guide
        with patch("src.object_approach.resolve_terminal_route", return_value={"status": "unavailable"}) as reroute:
            self.assertFalse(nav.recover(planner, graph, np.zeros(3), 1.5))
            self.assertFalse(nav.recover(planner, graph, np.zeros(3), 1.5))
        self.assertEqual(guide.repair.call_count, 1)
        self.assertEqual(reroute.call_count, 1)
        self.assertEqual(intent.recovery_attempts, 2)

    def test_stall_requires_three_moving_steps_without_voxel_progress(self):
        nav, scene, planner, graph, cfg, route = self.fixture("intent")
        intent = nav.install(self.task(scene, route), planner.hypothesis_graph)
        with patch("src.dual_dynamic_navigation.unified_navigator.grid_path",
                   return_value=np.array([[1., 3.], [2., 3.], [3., 3.]])):
            self.assertFalse(nav.record_motion_progress(planner, graph, np.zeros(3)))
            self.assertFalse(nav.record_motion_progress(planner, graph, np.zeros(3)))
            self.assertFalse(nav.record_motion_progress(planner, graph, np.zeros(3), moved=False))
            self.assertFalse(nav.record_motion_progress(planner, graph, np.zeros(3)))
            self.assertTrue(nav.record_motion_progress(planner, graph, np.zeros(3)))
        self.assertEqual(intent.stalled_steps, 0)
        self.assertEqual(nav.stats["stalls_detected"], 1)


if __name__ == "__main__":
    unittest.main()
