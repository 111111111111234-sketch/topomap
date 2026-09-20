import unittest
import json
import pickle
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.hgr_dual_topo import (Candidate, CertifiedExecution, DualTopoRuntime,
    Feedback, GoalState, GridPathCache, RoutePlan)
from src.hgr_timing import StageTimer
from src.hgr_experiment_state import record_decision, save_checkpoint, load_checkpoint, SCENE_MEMORY
from src.place_topology import PlaceTopology
from src.hypothesis_graph import HypothesisGraph, HypothesisNode, NodeType, CognitiveDependency


class UnifiedRouteTests(unittest.TestCase):
    def setUp(self):
        self.graph = PlaceTopology(2., 1., stage="route_only")
        self.a = self.graph.observe_pose([1, 2], 0).place_id
        self.b = self.graph.observe_pose([4, 2], 1).place_id
        self.c = self.graph.observe_pose([7, 2], 2).place_id
        self.graph.current_place_id = self.a
        self.mask = np.ones((12, 9), bool)
        self.runtime = DualTopoRuntime({})
        self.runtime.begin_goal("g")

    def plan(self):
        return self.runtime.resolve(self.graph, self.mask, [1, 2], [8, 2], 1.)

    def test_one_path_for_cost_and_execution_and_no_early_stop(self):
        result = self.plan()
        self.assertEqual(result["status"], "valid")
        plan = RoutePlan(**result["route_plan"])
        self.assertAlmostEqual(plan.total_distance_m, sum(plan.segment_distances_m))
        self.assertAlmostEqual(plan.total_distance_m, np.linalg.norm(np.diff(plan.path, axis=0), axis=1).sum())
        execution = CertifiedExecution.from_plan(plan, self.mask.shape)
        np.testing.assert_array_equal(execution.path, result["certified_path"])
        point, arrived = execution.step([1, 2], 5., 1., self.mask)
        self.assertFalse(arrived)
        self.assertEqual(execution.cursor, 0)  # proposal is not motion feedback
        self.assertTrue(execution.acknowledge(point))
        self.assertEqual(execution.remaining_route()[0], self.b)
        final, arrived = execution.step(point, 5., 1., self.mask)
        self.assertTrue(arrived)
        self.assertTrue(execution.acknowledge(final))
        self.assertEqual(self.graph.current_place_id, self.a)

    def test_failed_motion_never_advances_progress(self):
        execution = CertifiedExecution.from_plan(RoutePlan(**self.plan()["route_plan"]), self.mask.shape)
        execution.step([1, 2], 5., 1., self.mask)
        self.assertFalse(execution.acknowledge([1, 2]))
        self.assertEqual(execution.cursor, 0)

    def test_route_contract_rejects_mismatched_cost_or_terminal(self):
        payload = dict(self.plan()["route_plan"])
        payload["total_distance_m"] += 1.
        with self.assertRaises(ValueError):
            RoutePlan(**payload)
        payload = dict(self.plan()["route_plan"])
        payload["terminal"] = [9, 2]
        with self.assertRaises(ValueError):
            RoutePlan(**payload)

    def test_trace_audit_detects_early_arrival(self):
        from scripts.hgr_v2_experiments import audit_events
        initial = {"event": "v2_execution", "subtask_id": "g", "route_id": "r", "intent_id": "i",
                   "guidance": {"certified_path": [[1, 2], [2, 2], [3, 2]], "progress_index": 0}}
        motion = {"event": "v2_motion_acknowledged", "subtask_id": "g", "route_id": "r", "intent_id": "i",
                  "acknowledged": True, "terminal_arrived": False,
                  "guidance": {"progress_index": 1, "last_segment": [[1, 2], [2, 2]]}}
        self.assertTrue(audit_events([initial, motion])["trace_contract_passed"])
        motion["terminal_arrived"] = True
        report = audit_events([initial, motion])
        self.assertFalse(report["trace_contract_passed"])
        self.assertEqual(report["violations"][0]["reason"], "premature_terminal_arrival")

    def test_wall_and_edge_invalidation_are_not_ignored(self):
        self.mask[5, :] = False
        self.assertEqual(self.plan()["status"], "unavailable")
        self.mask[:] = True
        self.graph.invalidate_edge(self.a, self.b, 9)
        self.assertEqual(self.plan()["status"], "unavailable")

    def test_local_terminal_cost_and_pinned_approach(self):
        result = self.runtime.resolve(self.graph, self.mask, [1, 2], [2, 2], 1.)
        self.assertEqual(result["route_places"], [self.a])
        self.assertAlmostEqual(result["total_distance_m"], 1.)
        retained = self.runtime.resolve(self.graph, self.mask, [1, 2], [2, 2], 1., previous=result)
        self.assertEqual(retained["route_event"], "approach_retained")

    def test_frontier_uses_known_side_terminal_without_mutating_planner(self):
        from src.object_approach import resolve_frontier_approach
        self.graph.dual_topo_runtime = self.runtime
        mask = self.mask.copy()
        mask[8:, :] = False
        planner = SimpleNamespace(island=mask, unoccupied=mask, occupied=~mask, target_point=None)
        frontier = SimpleNamespace(position=np.array([8, 2]), orientation=np.array([1., 0.]))
        result = resolve_frontier_approach(self.graph, planner, frontier, [1, 2], 1.)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["terminal"], [7, 2])
        self.assertEqual(result["frontier_position"], [8, 2])
        self.assertIsNone(planner.target_point)

    def test_v2_replaces_blocked_object_terminal_without_changing_object(self):
        from src.object_approach import resolve_object_approach
        self.graph.dual_topo_runtime = self.runtime
        planner = SimpleNamespace(island=self.mask, unoccupied=self.mask, occupied=np.zeros_like(self.mask),
                                  habitat2voxel=lambda x: x)
        objects = {1: {"bbox": SimpleNamespace(center=np.array([3, 2]))}}
        with patch("src.object_approach.get_proper_observe_point", return_value=np.array([2, 2])):
            before = resolve_object_approach(self.graph, planner, objects, 1, [1, 2], 1.)
        planner.occupied[2, 2] = True
        with patch("src.object_approach.get_proper_observe_point", return_value=np.array([2, 3])):
            after = resolve_object_approach(self.graph, planner, objects, 1, [1, 2], 1., previous=before)
        self.assertEqual(after["status"], "valid")
        self.assertEqual(after["object_id"], before["object_id"])
        self.assertEqual(after["terminal"], [2, 3])

    def test_cache_matches_disabled_and_rechecks_new_shortcuts(self):
        cache = GridPathCache(True)
        mask = self.mask.copy()
        mask[3, 2] = False
        first = cache.query(mask, [1, 2], [6, 2])
        np.testing.assert_array_equal(first, cache.query(mask, [1, 2], [6, 2]))
        self.assertEqual(cache.hits, 1)
        mask[3, 2] = True
        shorter = cache.query(mask, [1, 2], [6, 2])
        self.assertLess(np.linalg.norm(np.diff(shorter, axis=0), axis=1).sum(),
                        np.linalg.norm(np.diff(first, axis=0), axis=1).sum())
        uncached = self.plan()
        self.runtime.cache.enabled = True
        self.assertEqual(uncached["route_plan"], self.plan()["route_plan"])

    def test_replay_record_contains_only_current_geometry_and_cost(self):
        from scripts.hgr_v2_experiments import replay
        with tempfile.TemporaryDirectory() as tmp:
            record_decision(tmp, 1, self.graph, self.mask, [1, 2], self.plan(), None)
            replay(SimpleNamespace(directory=tmp))
            with np.load(Path(tmp)/"decision_1.npz", allow_pickle=False) as payload:
                self.assertEqual(json.loads(str(payload["record"]))["voxel_size"], 1.)
            with self.assertRaises(FileExistsError):
                record_decision(tmp, 1, self.graph, self.mask, [1, 2], self.plan(), None)

    def test_replay_accepts_legacy_continuous_terminal_only_after_exact_rounding(self):
        from scripts.hgr_v2_experiments import replay
        result = self.plan()
        result.pop("route_plan")
        result["cost_basis"] = "current_pose_to_edge_targets_to_terminal"
        result["terminal"] = [8.2, 2.1]
        with tempfile.TemporaryDirectory() as tmp:
            record_decision(tmp, 1, self.graph, self.mask, [1, 2], result, None)
            replay(SimpleNamespace(directory=tmp))
            result["terminal"] = [8.6, 2.1]
            record_decision(tmp, 2, self.graph, self.mask, [1, 2], result, None)
            with self.assertRaisesRegex(ValueError, "Invalid endpoints"):
                replay(SimpleNamespace(directory=tmp))

    def test_source_disappearance_and_goal_switch_do_not_reuse_intent(self):
        frontier = SimpleNamespace(topo_id="f", position=np.array([8, 2]), hypothesis_node_id="h")
        self.runtime.install(frontier, {**self.plan(), "look_at": [9, 2]}, self.mask.shape)
        scene = SimpleNamespace(objects={}, object_id_aliases={})
        planner = SimpleNamespace(frontiers=[])
        self.assertIsNone(self.runtime.refresh_source(scene, planner))
        events = self.runtime.synchronize(scene, planner)
        self.assertEqual(events[0]["kind"], "source_removed")
        self.runtime.begin_goal("new")
        self.assertIsNone(self.runtime.execution)
        self.assertFalse(self.runtime.goal.candidates)

    def test_persistent_frontier_survives_source_mutation_and_removal(self):
        self.runtime.config["persistent_intent"] = True
        frontier = SimpleNamespace(topo_id="f", position=np.array([8, 2]),
                                   orientation=np.array([1., 0.]), hypothesis_node_id="h")
        self.runtime.install(frontier, self.plan(), self.mask.shape)
        frontier.position[:] = 0
        frontier.hypothesis_node_id = None
        scene = SimpleNamespace(objects={}, object_id_aliases={})
        planner = SimpleNamespace(frontiers=[])
        self.assertEqual(self.runtime.synchronize(scene, planner)[0]["kind"], "source_detached")
        self.assertEqual(self.runtime.synchronize(scene, planner), [])
        saved = self.runtime.refresh_source(scene, planner)
        np.testing.assert_array_equal(saved.position, [8, 2])
        self.assertEqual(saved.hypothesis_node_id, "h")
        self.assertEqual(self.runtime.ensure_route(self.graph, self.mask, [1, 2], 1.), "retained")
        feedback = self.runtime.revoke({"hypothesis_node_id": "h"}, "revocation")
        self.assertTrue(feedback["intent_cancelled"])
        self.assertIsNone(self.runtime.execution)

    def test_active_intent_exposes_hypothesis_pin_until_release(self):
        frontier = SimpleNamespace(topo_id="f", position=np.array([8, 2]),
                                   orientation=np.array([1., 0.]), hypothesis_node_id="h")
        self.runtime.install(frontier, self.plan(), self.mask.shape)
        self.assertEqual(self.runtime.active_hypothesis_refs(), {"h"})
        self.runtime.release("done")
        self.assertEqual(self.runtime.active_hypothesis_refs(), set())

    def test_selected_plan_reuse_requires_matching_geometry_and_start(self):
        result = self.plan()
        self.assertTrue(self.runtime.reusable(result, self.graph, self.mask, [1, 2]))
        self.assertFalse(self.runtime.reusable(result, self.graph, self.mask, [2, 2]))
        changed = self.mask.copy()
        changed[0, 0] = False
        self.assertFalse(self.runtime.reusable(result, self.graph, changed, [1, 2]))

    def test_cache_origin_categories_are_exclusive(self):
        cache = GridPathCache(True)
        for context in (("a", 1, 1), ("a", 1, 1), ("a", 1, 2), ("a", 2, 1), ("b", 1, 1)):
            cache.context = context
            cache.query(self.mask, [1, 2], [8, 2])
        self.assertEqual(cache.hits, 4)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hit_kinds, dict.fromkeys(cache.hit_kinds, 1))

    def test_blocked_terminal_cannot_be_retained(self):
        frontier = SimpleNamespace(topo_id="f", position=np.array([8, 2]), hypothesis_node_id="h")
        self.runtime.install(frontier, self.plan(), self.mask.shape)
        blocked = self.mask.copy()
        blocked[8, 2] = False
        self.assertEqual(self.runtime.ensure_route(self.graph, blocked, [1, 2], 1.), "route_unavailable")

    def test_local_obstacle_requires_recovery_before_next_motion(self):
        frontier = SimpleNamespace(topo_id="f", position=np.array([8, 2]), hypothesis_node_id="h")
        self.runtime.install(frontier, self.plan(), self.mask.shape)
        blocked = self.mask.copy()
        blocked[2, 2] = False
        outcome = self.runtime.ensure_route(self.graph, blocked, [1, 2], 1.)
        self.assertIn(outcome, ("local_repair", "same_target_reroute"))
        self.assertTrue(self.runtime.execution.valid(self.graph, blocked))
        self.assertEqual(self.runtime.goal.active_intent.entity_id, "f")

    def test_frozen_cache_replay_is_distinct_from_navigation_acceptance(self):
        from scripts.hgr_v2_experiments import cache_replay
        with tempfile.TemporaryDirectory() as tmp:
            record_decision(tmp, 1, self.graph, self.mask, [1, 2], self.plan(), None)
            output = str(Path(tmp)/"cache.json")
            cache_replay(SimpleNamespace(directory=tmp, output=output, observe_distance=None))
            report = json.loads(Path(output).read_text())
            self.assertTrue(report["route_equivalence_passed"])
            self.assertEqual(len(report["records"]), 1)

    def test_real_critic_cascade_preserves_independent_candidate_and_space(self):
        from src.hypothesis_graph import SemanticDistribution
        from src.semantic_critic import SemanticCritic
        graph = HypothesisGraph({})
        graph.preserve_independent_evidence = True
        node = HypothesisNode("h", NodeType.HYPOTHESIS,
                              semantic_dist=SemanticDistribution(["kitchen"], np.array([1.])))
        graph.add_node(node)
        graph.add_node(HypothesisNode("child", NodeType.HYPOTHESIS))
        graph.add_dependency(CognitiveDependency("h", "child", "semantic", 1., "test"))
        critic = SemanticCritic({"enable_vlm_causality_diagnosis": False}, graph)
        candidate = self.runtime.goal.register(Candidate("f", "f", "EXPLORE", hypothesis_refs={"h"}))
        self.runtime.goal.select(candidate, "route")
        self.runtime.goal.register(Candidate("o", "o", "TARGET_APPROACH",
            observed_evidence_refs={"independent:image"}, hypothesis_refs={"child"}))
        spatial = self.graph.to_trace_dict(include_graph=True)
        report = critic.verify_hypothesis_node_arrival(node,
            {"semantic_class": "office", "detected_objects": ["desk", "monitor"]}, {})
        self.assertTrue(report.is_falsified)
        result = self.runtime.revoke({"hypothesis_node_id": report.node_id,
            "cascade_deleted_nodes": report.cascade_deleted_nodes}, "e")
        self.assertTrue(result["intent_cancelled"])
        self.assertNotIn("child", graph.nodes)
        self.assertEqual(self.runtime.goal.candidates["o"].observed_evidence_refs, {"independent:image"})
        self.assertEqual(self.runtime.goal.candidates["o"].hypothesis_refs, set())
        self.assertEqual(spatial, self.graph.to_trace_dict(include_graph=True))

    def test_audit_does_not_confuse_normal_motion_with_cascade_coverage(self):
        from scripts.hgr_v2_experiments import audit_events
        report = audit_events([])
        self.assertEqual(report["cascade_acceptance"], "not_covered")
        self.assertFalse(report["source_audit"]["passed"])
        self.assertEqual(report["recovery_acceptance"], "not_covered")

    @staticmethod
    def retraction_events():
        before = {
            "f": {"observed_evidence_refs": [], "hypothesis_refs": ["h"]},
            "o": {"observed_evidence_refs": ["image:1"],
                  "hypothesis_refs": ["child", "independent"]},
        }
        after = {
            "f": {"observed_evidence_refs": [], "hypothesis_refs": []},
            "o": {"observed_evidence_refs": ["image:1"],
                  "hypothesis_refs": ["independent"]},
        }
        return [
            {"event": "v2_hypothesis_verification", "subtask_id": "g", "step": 3,
             "result": {"skipped": False, "is_falsified": True,
                        "hypothesis_node_id": "h"}},
            {"event": "v2_hypothesis_retraction", "subtask_id": "g", "step": 3,
             "feedback": {"affected_hypotheses": ["h", "child"],
                          "intent_cancelled": True,
                          "spatial_facts_preserved": True,
                          "candidates_before": before,
                          "candidates_after": after}},
        ]

    def test_source_audit_accepts_exact_semantic_retraction(self):
        from scripts.hgr_v2_experiments import audit_events
        report = audit_events(self.retraction_events())
        self.assertEqual(report["cascade_acceptance"], "passed")
        self.assertTrue(report["source_audit"]["passed"])
        self.assertEqual(report["source_audit"]["retractions_checked"], 1)
        self.assertEqual(report["violations"], [])

    def test_source_audit_rejects_invalid_retraction_invariants(self):
        from copy import deepcopy
        from scripts.hgr_v2_experiments import audit_events
        cases = {
            "retraction_without_matching_falsification": lambda rows: rows[0]["result"].update(is_falsified=False),
            "retraction_hypothesis_not_affected": lambda rows: rows[1]["feedback"].update(affected_hypotheses=["child"]),
            "retraction_did_not_cancel_intent": lambda rows: rows[1]["feedback"].update(intent_cancelled=False),
            "retraction_modified_spatial_facts": lambda rows: rows[1]["feedback"].update(spatial_facts_preserved=False),
            "retraction_changed_candidate_set": lambda rows: rows[1]["feedback"]["candidates_after"].pop("o"),
            "retraction_changed_observed_evidence": lambda rows: rows[1]["feedback"]["candidates_after"]["o"].update(observed_evidence_refs=[]),
            "retraction_reference_mismatch": lambda rows: rows[1]["feedback"]["candidates_after"]["o"].update(hypothesis_refs=[]),
            "malformed_retraction_snapshot": lambda rows: rows[1]["feedback"].update(candidates_after=None),
        }
        for expected, mutate in cases.items():
            with self.subTest(expected):
                rows = deepcopy(self.retraction_events())
                mutate(rows)
                report = audit_events(rows)
                self.assertEqual(report["cascade_acceptance"], "failed")
                self.assertIn(expected, {item["reason"] for item in report["source_audit"]["violations"]})

    def test_source_failure_does_not_relabel_route_contract(self):
        from scripts.hgr_v2_experiments import audit_events
        rows = [
            {"event": "v2_execution", "subtask_id": "g", "step": 2,
             "route_id": "r", "intent_id": "i",
             "guidance": {"certified_path": [[1, 2], [2, 2]],
                          "progress_index": 0}},
            {"event": "v2_motion_acknowledged", "subtask_id": "g", "step": 2,
             "route_id": "r", "intent_id": "i", "acknowledged": True,
             "terminal_arrived": True,
             "guidance": {"progress_index": 1,
                          "last_segment": [[1, 2], [2, 2]]}},
            *self.retraction_events(),
        ]
        rows[-1]["feedback"]["intent_cancelled"] = False
        report = audit_events(rows)
        self.assertTrue(report["trace_contract_passed"])
        self.assertEqual(report["route_violations"], [])
        self.assertEqual(report["cascade_acceptance"], "failed")

    def test_real_local_executor_uses_proposal_without_advancing_cursor(self):
        from src.tsdf_planner import TSDFPlanner
        from unittest.mock import Mock
        guide = CertifiedExecution.from_plan(RoutePlan(**self.plan()["route_plan"]), self.mask.shape)
        planner = TSDFPlanner.__new__(TSDFPlanner)
        planner.max_point = SimpleNamespace()
        planner.target_point, planner.look_at_point = np.array([8, 2]), np.array([9, 2])
        planner._voxel_size, planner._vol_origin = 1., np.zeros(3)
        planner.island = planner.unoccupied = self.mask
        planner.occupied = np.zeros_like(self.mask)
        planner._explore_vol_cpu = np.zeros((*self.mask.shape, 1))
        planner.normal2voxel = lambda _: np.array([1, 2, 0])
        planner.normal2habitat = lambda p: p
        planner.get_distance = Mock(side_effect=AssertionError("unapproved Pathfinder route"))
        config = SimpleNamespace(max_dist_from_cur_phase_1=5., max_dist_from_cur_phase_2=5.,
                                 surrounding_explored_radius=1.)
        result = planner.agent_step(np.zeros(3), 0., {}, {}, None, config,
                                    save_visualization=False, route_guidance=guide)
        self.assertFalse(result[-1])
        self.assertEqual(guide.cursor, 0)
        self.assertTrue(guide.acknowledge(result[2]))
        planner.get_distance.assert_not_called()


class GoalRetractionTests(unittest.TestCase):
    def test_fresh_arrival_uses_terminal_pixels_and_current_detections(self):
        from src.query_vlm_hypothesis import verify_fresh_intent_arrival
        from unittest.mock import Mock
        rgb = np.full((4, 4, 4), 127, dtype=np.uint8)
        scene = Mock()
        scene.get_observation.return_value = ({"color_sensor": rgb, "depth_sensor": np.ones((4, 4))}, None)
        scene.obj_classes.get_classes_arr.return_value = ["desk"]
        boxes = Mock()
        boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        scene.detection_model.predict.return_value = [SimpleNamespace(boxes=boxes)]
        frontier = SimpleNamespace(hypothesis_node_id="h")
        with patch("src.query_vlm_hypothesis.verify_hypothesis_node_arrival",
                   return_value={"skipped": False, "is_falsified": True}) as verify:
            result = verify_fresh_intent_arrival(frontier, scene, Mock(), Mock(), [1, 2, 3], .5)
        scene.get_observation.assert_called_once_with([1, 2, 3], angle=.5)
        np.testing.assert_array_equal(verify.call_args.args[4], rgb[..., :3])
        self.assertEqual(verify.call_args.args[6], ["desk"])
        self.assertEqual(result["observation_source"], "post_motion_terminal")

    def test_failed_arrival_observation_does_not_call_critic(self):
        from src.query_vlm_hypothesis import verify_fresh_intent_arrival
        from unittest.mock import Mock
        scene, critic = Mock(), Mock()
        scene.get_observation.side_effect = RuntimeError("camera unavailable")
        result = verify_fresh_intent_arrival(SimpleNamespace(hypothesis_node_id="h"), scene,
                                            Mock(), critic, [0, 0, 0], 0.)
        self.assertTrue(result["skipped"])
        self.assertNotIn("is_falsified", result)
        critic.verify_hypothesis_node_arrival.assert_not_called()

    def test_positive_fresh_verification_records_independent_observation(self):
        from src.query_vlm_hypothesis import verify_hypothesis_node_arrival
        from src.semantic_critic import SemanticCritic
        from src.hypothesis_graph import SemanticDistribution
        graph = HypothesisGraph({})
        node = HypothesisNode("h", NodeType.HYPOTHESIS,
            semantic_dist=SemanticDistribution(["office"], np.array([1.])))
        graph.add_node(node)
        planner = SimpleNamespace(hypothesis_graph=graph)
        critic = SemanticCritic({"enable_vlm_causality_diagnosis": False}, graph)
        verify_hypothesis_node_arrival(SimpleNamespace(hypothesis_node_id="h"),
            SimpleNamespace(objects={}), planner, critic, np.zeros((2, 2, 3), np.uint8),
            np.ones((2, 2)), ["desk", "monitor"])
        self.assertEqual(len(node.independent_observation_refs), 1)
        self.assertTrue(node.independent_observation_refs[0].startswith("arrival:"))

    def test_new_independent_evidence_updates_already_active_intent(self):
        state = GoalState("g")
        first = state.register(Candidate("f", "f", "EXPLORE", hypothesis_refs={"h"}))
        state.select(first, "route")
        state.register(Candidate("f", "f", "EXPLORE", observed_evidence_refs={"fresh:frame"}, hypothesis_refs={"h"}))
        cancelled = state.apply(Feedback("e", "g", "hypotheses_revoked", "critic", affected_hypotheses=("h",)))
        self.assertFalse(cancelled)
        self.assertEqual(state.active_intent.observed_evidence_refs, {"fresh:frame"})

    def test_retraction_cancels_intent_but_keeps_real_frontier(self):
        state = GoalState("g")
        candidate = state.register(Candidate("f", "f", "EXPLORE", hypothesis_refs={"h"}))
        state.select(candidate, "route")
        event = Feedback("e", "g", "hypotheses_revoked", "critic", affected_hypotheses=("h",))
        self.assertTrue(state.apply(event))
        self.assertIsNone(state.active_intent)
        self.assertIn("f", state.candidates)
        self.assertFalse(state.apply(event))
        self.assertFalse(state.register(Candidate("f", "f", "EXPLORE", hypothesis_refs={"h"})).hypothesis_refs)

    def test_independent_evidence_keeps_intent_and_goal_switch_isolated(self):
        state = GoalState("g")
        candidate = state.register(Candidate("o", "1", "TARGET_APPROACH",
            observed_evidence_refs={"image:1"}, hypothesis_refs={"h"}))
        state.select(candidate, "route")
        state.apply(Feedback("e", "g", "hypotheses_revoked", "critic", affected_hypotheses=("h",)))
        self.assertIsNotNone(state.active_intent)
        state.rebind("1", "2")
        self.assertEqual(state.active_intent.entity_id, "2")
        self.assertEqual(state.active_intent.observed_evidence_refs, {"image:1"})
        other = GoalState("other")
        self.assertFalse(other.apply(Feedback("e", "g", "hypotheses_revoked", "critic", affected_hypotheses=("h",))))
        self.assertFalse(other.revoked_hypotheses)

    def test_graph_cascade_preserves_proven_observation_only_in_v2(self):
        for enabled in (False, True):
            graph = HypothesisGraph({})
            graph.preserve_independent_evidence = enabled
            for name in ("root", "inferred", "observed", "descendant"):
                graph.add_node(HypothesisNode(name, NodeType.HYPOTHESIS))
            graph.nodes["observed"].independent_observation_refs = ["rgb:proof"]
            for parent, child in (("root", "inferred"), ("root", "observed"), ("observed", "descendant")):
                graph.add_dependency(CognitiveDependency(parent, child, "semantic", 1., "test"))
            predicted = graph.cascade_affected_ids("root")
            graph.cascade_delete("root")
            self.assertNotIn("inferred", graph.nodes)
            self.assertEqual("observed" in graph.nodes, enabled)
            self.assertEqual("descendant" in graph.nodes, enabled)
            self.assertEqual(set(predicted), {"inferred"} if enabled else {"inferred", "observed", "descendant"})

    def test_error_feedback_does_not_create_negative_evidence(self):
        state = GoalState("g")
        state.apply(Feedback("error", "g", "request_failed", "timeout"))
        self.assertFalse(state.revoked_hypotheses)
        self.assertFalse(state.observation_coverage)

    def test_exclusive_timing_subtracts_nested_request(self):
        with patch("src.hgr_timing.perf_counter", side_effect=[0., 1., 3., 6., 8.]):
            timer = StageTimer()
            timer.switch("candidates")
            timer.push("model_request")
            timer.pop()
            result = timer.snapshot()
        self.assertEqual(result["stages_seconds"], {"other": 1., "candidates": 4., "model_request": 3.})
        self.assertEqual(result["total_seconds"], 8.)

    def test_checkpoint_roundtrip_and_coordinate_mismatch(self):
        scene = SimpleNamespace(**{key: {} for key in SCENE_MEMORY})
        config = {"tsdf_grid_size": .1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"scene.pkl"
            save_checkpoint(path, scene, {"known_cells": [1, 2]}, None, None,
                            [1, 2, 3], 0., 8, 1, config, "scene", "0", {})
            payload = load_checkpoint(path, config, "scene", "0")
            self.assertEqual(payload["planner"]["known_cells"], [1, 2])
            self.assertEqual(payload["subtask_index"], 1)
            with self.assertRaises(ValueError):
                load_checkpoint(path, {"tsdf_grid_size": .2}, "scene", "0")

    def test_checkpoint_roundtrip_open3d_geometry_and_atomic_failure(self):
        import open3d as o3d
        points = np.asarray([[0., 0., 0.], [1., 2., 3.]])
        colors = np.asarray([[1., 0., 0.], [0., .5, 1.]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        bbox = o3d.geometry.OrientedBoundingBox(
            np.asarray([1., 2., 3.]), np.eye(3), np.asarray([4., 5., 6.]))
        bbox.color = (0.2, 0.4, 0.6)
        memory = {key: {} for key in SCENE_MEMORY}
        memory["objects"] = {7: {"id": 7, "pcd": pcd, "bbox": bbox}}
        scene = SimpleNamespace(**memory)
        config = {"tsdf_grid_size": .1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"scene.pkl"
            save_checkpoint(path, scene, {}, None, None, [1, 2, 3], 0., 8, 1,
                            config, "scene", "0", {})
            payload = load_checkpoint(path, config, "scene", "0")
            restored = payload["scene_memory"]["objects"][7]
            np.testing.assert_allclose(np.asarray(restored["pcd"].points), points)
            np.testing.assert_allclose(np.asarray(restored["pcd"].colors), colors)
            np.testing.assert_allclose(np.asarray(restored["bbox"].center), [1., 2., 3.])
            np.testing.assert_allclose(np.asarray(restored["bbox"].extent), [4., 5., 6.])
            np.testing.assert_allclose(np.asarray(restored["bbox"].color), [.2, .4, .6])

            failed = Path(tmp)/"failed.pkl"
            with self.assertRaises((AttributeError, pickle.PicklingError)):
                save_checkpoint(failed, scene, lambda: None, None, None,
                                [1, 2, 3], 0., 8, 1, config, "scene", "0", {})
            self.assertFalse(failed.exists())
            self.assertFalse(list(Path(tmp).glob(".failed.pkl.*.tmp")))

    def test_stage_configs_keep_correction_commitment_and_cache_separate(self):
        from run_goatbench_evaluation import _load_config
        root = Path(__file__).resolve().parents[1]/"cfg"
        def load(stage):
            return _load_config(str(root/f"eval_goatbench_hgr_dual_topo_v2_{stage}_train.yaml"))
        b, shadow, c, persistent, d = (load(s) for s in ("b", "c_shadow", "c", "c_persistent", "d"))
        self.assertFalse(b.hgr_dual_topo.goal_state)
        self.assertTrue(shadow.hgr_dual_topo.goal_state)
        self.assertFalse(shadow.hgr_dual_topo.cascade_correction)
        self.assertFalse(shadow.hgr_dual_topo.goal_context)
        self.assertTrue(c.hgr_dual_topo.cascade_correction)
        self.assertFalse(c.hgr_dual_topo.persistent_intent)
        self.assertTrue(persistent.hgr_dual_topo.persistent_intent)
        self.assertFalse(persistent.hgr_dual_topo.cache_paths)
        self.assertTrue(d.hgr_dual_topo.cache_paths)
        for config in (b, shadow, c, persistent, d):
            self.assertFalse(config.place_goal_navigation.enabled)
            self.assertEqual(config.planner, b.planner)
            self.assertEqual(config.hypothesis, b.hypothesis)

    def test_memory_export_smoke_uses_shadow_map_and_fixed_single_scene(self):
        from run_goatbench_evaluation import _load_config
        root = Path(__file__).resolve().parents[1]/"cfg"
        config = _load_config(str(
            root/"eval_goatbench_hgr_dual_topo_v2_memory_export_smoke.yaml"
        ))
        self.assertEqual(config.controlled_memory.mode, "export")
        self.assertEqual(config.controlled_memory.prefix_subtasks, 1)
        self.assertEqual(config.active_topology.place_topology.stage, "shadow")
        self.assertEqual(
            config.episode_manifest,
            "cfg/manifests/goat_train_v7_1_d_smoke_aczz_ep0_seed77.json",
        )

    def test_comparison_rejects_equal_but_incomplete_task_sets(self):
        from scripts.hgr_v2_experiments import compare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"episodes": [{"episode_id": "0", "tasks": [["chair", "object", "1"], ["chair", "object", "1"]]}],
                    "goals": {"s.basis.glb_chair": [{"object_category": "chair", "object_id": "1", "children_object_categories": []}]}}
            (root/"s.json").write_text(json.dumps(data))
            manifest = {"dataset_dir": tmp, "splits": {"1": [{"scene_file": "s.json", "scene_name": "s", "episode_id": "0"}]}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            row = {"goal_type": "object", "success_by_distance": True, "spl_by_distance": .5,
                   "traversed_distance_m": 2., "timing": {"total_seconds": 3.}}
            (root/"baseline.json").write_text(json.dumps({"00000-s_0_0": row}))
            (root/"candidate.json").write_text(json.dumps({"00000-s_0_0": row}))
            args = SimpleNamespace(baseline=[str(root/"baseline.json")], candidate=[str(root/"candidate.json")],
                                   manifest=str(root/"manifest.json"), start_subtask=0,
                                   baseline_cold=None, candidate_cold=None, output=str(root/"report.json"))
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                compare(args)

    def test_comparison_rejects_distance_gain_with_identity_regression(self):
        from scripts.hgr_v2_experiments import compare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {"episodes": [{"episode_id": "0", "tasks": [["chair", "object", "1"]]}],
                    "goals": {"s.basis.glb_chair": [{"object_category": "chair", "object_id": "1",
                                                       "children_object_categories": []}]}}
            (root/"s.json").write_text(json.dumps(data))
            manifest = {"dataset_dir": tmp, "splits": {"1": [
                {"scene_file": "s.json", "scene_name": "s", "episode_id": "0"}]}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            baseline = {"goal_type": "object", "success_by_distance": True,
                        "success_by_snapshot": True, "spl_by_distance": .5,
                        "spl_by_snapshot": .5, "traversed_distance_m": 2.,
                        "timing": {"total_seconds": 3.}}
            candidate = {**baseline, "success_by_snapshot": False,
                         "spl_by_distance": .8, "spl_by_snapshot": 0.,
                         "traversed_distance_m": 1., "timing": {"total_seconds": 2.}}
            (root/"baseline.json").write_text(json.dumps({"00000-s_0_0": baseline}))
            (root/"candidate.json").write_text(json.dumps({"00000-s_0_0": candidate}))
            args = SimpleNamespace(baseline=[str(root/"baseline.json")],
                                   candidate=[str(root/"candidate.json")],
                                   manifest=str(root/"manifest.json"), start_subtask=0,
                                   single_subtask=True,
                                   baseline_cold=None, candidate_cold=None,
                                   output=str(root/"report.json"))
            compare(args)
            report = json.loads((root/"report.json").read_text())
            self.assertEqual(report["metrics"]["snapshot_sr"]["difference"], -1.)
            self.assertFalse(report["performance_gate"])


if __name__ == "__main__":
    unittest.main()
