import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.dual_dynamic_navigation import BaselineDualDynamicNavigator, DualDynamicNavigator, GoalGraph, SceneMap
from src.dual_dynamic_navigation.config import validate_baseline_backend
from src.hierarchical_navigator import (
    IntentKind, SemanticAssessment, SemanticCandidate, SemanticConfidence,
    TerminalVerification, VerificationVerdict,
)


def candidate(candidate_id, entity, kind="snapshot", place="place_0", distance=1.0,
              evidence=(), hypotheses=()):
    return SemanticCandidate(
        candidate_id=candidate_id, source_kind=kind, entity_id=str(entity),
        place_id=place, route_id=f"route:{candidate_id}", route_distance_m=distance,
        route_reachable=True, evidence_refs=tuple(evidence), hypothesis_refs=tuple(hypotheses),
        information_gain=0.5,
    )


def assessment(candidate_id, confidence, group=0):
    return SemanticAssessment(candidate_id, SemanticConfidence(confidence), group, "test")


class SceneMapTests(unittest.TestCase):
    def test_l1_persists_entities_and_rebind_is_idempotent(self):
        scene = SceneMap()
        item = candidate("snap:a", "7", evidence=("obs:a",))
        first = scene.update_candidates([item], "event:1")
        duplicate = scene.update_candidates([item], "event:1")
        self.assertEqual(first.revision, duplicate.revision)
        self.assertIn("7", scene.places["place_0"].entity_ids)
        scene.rebind_entity("7", "9", "merge:1")
        scene.rebind_entity("7", "9", "merge:1")
        self.assertEqual(scene.resolve_entity("7"), "9")
        self.assertEqual(scene.entities["9"].evidence_ids, {"obs:a"})
        self.assertIn("9", scene.places["place_0"].entity_ids)
        revision_before = scene.revision
        unchanged = scene.update_candidates([item], "event:2")
        self.assertEqual(unchanged.revision, revision_before)
        self.assertEqual(unchanged.changed_places, ())

    def test_l1_tracks_first_and_latest_evidence_across_goals(self):
        scene = SceneMap()
        item = candidate("snap:a", "7", evidence=("obs:a",))
        scene.update_candidates([item], "event:1", goal_id="goal:a", step=2,
                                evidence_positions={"obs:a": (1.0, 3.0)})
        scene.update_candidates([item], "event:2", goal_id="goal:b", step=7,
                                evidence_positions={"obs:a": (2.0, 4.0)})
        entity = scene.entities["7"]
        self.assertEqual(entity.evidence_first_seen["obs:a"], ("goal:a", 2))
        self.assertEqual(entity.evidence_last_seen["obs:a"], ("goal:b", 7))
        self.assertEqual(entity.evidence_observation_positions["obs:a"], (2.0, 4.0))
        trace = scene.to_trace_dict("goal:b")
        self.assertEqual(trace["evidence_count"], 1)
        self.assertEqual(trace["prior_goal_evidence_count"], 1)
        self.assertEqual(trace["entities_with_prior_goal_evidence"], 1)

    def test_l1_copies_certified_place_edges_and_status_changes(self):
        scene = SceneMap()
        node = SimpleNamespace(observation_ids={"pose:1"})
        edge = SimpleNamespace(
            path_length_m=1.25, status=SimpleNamespace(value="valid"),
            local_path_reference="edge:a:b", last_verified_step=4,
        )
        graph = SimpleNamespace(nodes={"a": node, "b": node}, edges={("a", "b"): edge})
        delta = scene.sync_place_topology(graph, "map:1")
        self.assertEqual(delta.changed_edges, (("a", "b"),))
        self.assertEqual(scene.edges[("a", "b")].status, "valid")
        edge.status = SimpleNamespace(value="invalid")
        scene.sync_place_topology(graph, "map:2")
        self.assertEqual(scene.edges[("a", "b")].status, "invalid")


class GoalGraphTests(unittest.TestCase):
    def test_same_entity_snapshots_form_one_action_with_all_evidence(self):
        graph = GoalGraph("goal")
        far = candidate("snap:far", "7", distance=4.0, evidence=("obs:far",))
        near = candidate("snap:near", "7", distance=1.0, evidence=("obs:near",))
        values = {
            far.candidate_id: assessment(far.candidate_id, "high", 0),
            near.candidate_id: assessment(near.candidate_id, "high", 0),
        }
        projected, projected_values = graph.project([far, near], values, "decision:1")
        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0].candidate_id, "snap:near")
        self.assertEqual(set(projected[0].evidence_refs), {"obs:far", "obs:near"})
        self.assertIn("snap:near", projected_values)

    def test_checked_medium_evidence_is_not_available_until_new_view(self):
        graph = GoalGraph("goal")
        graph.record_verification("verify:1", "7", ["obs:a"], VerificationVerdict.UNCERTAIN)
        self.assertFalse(graph.evidence_available("7", ["obs:a"]))
        self.assertTrue(graph.evidence_available("7", ["obs:a", "obs:b"]))


class DualDynamicNavigatorTests(unittest.TestCase):
    def test_l1_survives_goal_switch_while_l2_is_recreated(self):
        navigator = DualDynamicNavigator({})
        navigator.begin_goal("goal:a")
        item = candidate("snap:a", "7", evidence=("obs:a",))
        navigator.decide([item], {item.candidate_id: assessment(item.candidate_id, "high")}, "place_0")
        old_graph = navigator.goal_graph
        navigator.begin_goal("goal:b")
        self.assertIn("7", navigator.scene_map.entities)
        self.assertIsNot(old_graph, navigator.goal_graph)
        self.assertEqual(navigator.goal_graph.entities, {})

    def test_medium_action_is_removed_after_same_evidence_checked(self):
        navigator = DualDynamicNavigator({})
        navigator.begin_goal("goal")
        item = candidate("snap:a", "7", evidence=("obs:a",))
        values = {item.candidate_id: assessment(item.candidate_id, "medium")}
        decision = navigator.decide([item], values, "place_1")
        self.assertEqual(decision.intent_kind, IntentKind.EVIDENCE_REVISIT)
        navigator.install(decision, terminal=(1, 2))
        navigator.record_verification(
            TerminalVerification(VerificationVerdict.UNCERTAIN, 0.5, "occluded", 1), 2)
        decision = navigator.decide([item], values, "place_1")
        self.assertIsNone(decision.candidate)

    def test_rejected_high_action_requires_new_independent_evidence(self):
        navigator = DualDynamicNavigator({})
        navigator.begin_goal("goal")
        item = candidate("snap:a", "7", evidence=("obs:a",))
        values = {item.candidate_id: assessment(item.candidate_id, "high")}
        navigator.goal_graph.record_verification(
            "verify:reject", "7", ["obs:a"], VerificationVerdict.REJECTED)
        self.assertIsNone(navigator.decide([item], values, "place_0").candidate)
        new_view = candidate("snap:b", "7", evidence=("obs:b",))
        new_values = {new_view.candidate_id: assessment(new_view.candidate_id, "high")}
        self.assertEqual(
            navigator.decide([new_view], new_values, "place_0").intent_kind,
            IntentKind.TARGET_APPROACH,
        )

    def test_backend_checkpoint_roundtrip_preserves_maps_and_intent(self):
        navigator = DualDynamicNavigator({})
        navigator.begin_goal("goal")
        item = candidate("snap:a", "7", evidence=("obs:a",), hypotheses=("h1",))
        decision = navigator.decide(
            [item], {item.candidate_id: assessment(item.candidate_id, "high")}, "place_1")
        navigator.install(decision, terminal=(2, 3))
        restored = pickle.loads(pickle.dumps(navigator))
        self.assertEqual(restored.active_intent.entity_id, "7")
        self.assertEqual(restored.active_intent.evidence_refs, ("obs:a",))
        self.assertIn("7", restored.scene_map.entities)

    def test_research_checkpoint_carries_dual_dynamic_state(self):
        from src.hgr_experiment_state import SCENE_MEMORY, load_checkpoint, save_checkpoint
        navigator = DualDynamicNavigator({})
        navigator.begin_goal("goal")
        item = candidate("snap:a", "7", evidence=("obs:a",))
        decision = navigator.decide(
            [item], {item.candidate_id: assessment(item.candidate_id, "high")}, "place_0")
        navigator.install(decision, terminal=(2, 3))
        scene = SimpleNamespace(**{key: {} for key in SCENE_MEMORY})
        config = {"tsdf_grid_size": 0.1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dual.pkl"
            save_checkpoint(path, scene, {}, None, None, [0, 0, 0], 0.0, 3, 1,
                            config, "scene", "0", {}, navigation=navigator)
            restored = load_checkpoint(path, config, "scene", "0")["navigation"]
        self.assertEqual(restored.backend_name, "dual_dynamic")
        self.assertEqual(restored.active_intent.intent_id, navigator.active_intent.intent_id)
        self.assertIn("7", restored.scene_map.entities)

    def test_new_backend_configs_are_isolated(self):
        from run_goatbench_evaluation import _load_config
        root = os.path.dirname(os.path.dirname(__file__))
        active = _load_config(os.path.join(root, "cfg", "eval_goatbench_dual_dynamic_train.yaml"))
        shadow = _load_config(os.path.join(root, "cfg", "eval_goatbench_dual_dynamic_shadow.yaml"))
        self.assertEqual(active.navigation_backend, "dual_dynamic_route_only")
        self.assertEqual(shadow.navigation_backend, "dual_dynamic_shadow")
        self.assertFalse(active.hgr_dual_topo.enabled)
        self.assertEqual(active.dual_dynamic_navigation.stage, "route_only")
        self.assertFalse(shadow.hgr_dual_topo.enabled)
        self.assertEqual(shadow.dual_dynamic_navigation.stage, "shadow")
        baseline = _load_config(os.path.join(root, "cfg", "eval_goatbench_hgr_baseline_qwen3vl_dashscope_train.yaml"))
        for cfg in (active, shadow):
            validate_baseline_backend(cfg)
            for key in ("vlm_model", "choose_every_step", "hypothesis", "planner", "prefiltering"):
                self.assertEqual(cfg[key], baseline[key])


class BaselineAdapterTests(unittest.TestCase):
    def test_shadow_copies_actual_place_graph_without_mutating_hgr(self):
        import random
        from src.place_topology import PlaceTopology
        graph = PlaceTopology(1.5, 0.1, stage="shadow")
        place = graph.observe_pose([0, 0], 0).place_id
        graph.bind_observation("snapshot:a.png", place, [0, 0], 0)
        graph.observe_frontier("f1", [1, 0], place, 0, 0, 1.0)
        frontiers = [SimpleNamespace(topo_id="f1", hypothesis_node_id="h1")]
        scene = SimpleNamespace(
            snapshots={"a.png": SimpleNamespace(image="a.png", cluster=[7])},
            objects={7: {"num_detections": 2}},
        )
        navigator = BaselineDualDynamicNavigator({"stage": "shadow"})
        navigator.begin_goal("goal:a")
        before = pickle.dumps((graph, scene, frontiers))
        rng_before = random.getstate()
        delta = navigator.sync(graph, scene, frontiers, "step:1")
        record = navigator.observe_hgr_selection(scene.snapshots["a.png"], None, 1)
        self.assertEqual(pickle.dumps((graph, scene, frontiers)), before)
        self.assertEqual(random.getstate(), rng_before)
        self.assertEqual(delta["entity_count_observed"], 1)
        self.assertEqual(delta["candidate_count_observed"], 2)
        self.assertEqual(record.entity_id, "7")
        self.assertEqual(navigator.stats["semantic_requests_added"], 0)
        self.assertEqual(navigator.stats["terminal_verification_requests_added"], 0)
        restored = pickle.loads(pickle.dumps(navigator))
        restored.validate_restore({"stage": "shadow"})
        with self.assertRaises(ValueError):
            restored.validate_restore({"stage": "route_only"})
        old_graph = navigator.goal_graph
        navigator.begin_goal("goal:b")
        self.assertIsNot(navigator.goal_graph, old_graph)
        self.assertIn("7", navigator.scene_map.entities)
        diagnostic = navigator.diagnostics()
        self.assertEqual(diagnostic["scene_map"]["entities"], 1)
        self.assertEqual(diagnostic["goal_graph"]["entities"], 0)
        self.assertNotIn("entities", diagnostic["goal_graph"].get("details", {}))

    def test_incompatible_policy_is_rejected_before_execution(self):
        from copy import deepcopy
        base = {
            "navigation_backend": "dual_dynamic",
            "dual_dynamic_navigation": {"stage": "shadow"},
            "active_topology": {"enabled": True, "mode": "stable_only",
                                "place_topology": {"enabled": True, "stage": "shadow"}},
        }
        validate_baseline_backend(base)
        overrides = [
            ("dual_dynamic_navigation", "preserve_hgr_selector", False),
            ("dual_dynamic_navigation", "preserve_hgr_stop", False),
            ("dual_dynamic_navigation", "object_approach", True),
            ("dual_dynamic_navigation", "stage", "memory_intent"),
            ("hgr_dual_topo", "enabled", True),
            ("hierarchical_navigation", "enabled", True),
            ("place_goal_navigation", "route_guidance", True),
        ]
        for section, key, value in overrides:
            cfg = deepcopy(base)
            cfg.setdefault(section, {})[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_baseline_backend(cfg)
        cfg = deepcopy(base)
        cfg["navigation_backend"] = "dual_dynamic_route_only"
        with self.assertRaises(ValueError):
            validate_baseline_backend(cfg)


if __name__ == "__main__":
    unittest.main()
