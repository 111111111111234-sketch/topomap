import pickle
import os
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.hierarchical_navigator import (
    ExecutionMode,
    HierarchicalNavigator,
    IntentKind,
    NavigationEvent,
    NavigationEventType,
    NavigatorCommandType,
    SemanticAssessment,
    SemanticCandidate,
    SemanticConfidence,
    VerificationVerdict,
    parse_semantic_assessments,
    parse_terminal_verification,
)


def candidate(
    candidate_id,
    source_kind="snapshot",
    place="place_1",
    distance=1.0,
    reachable=True,
    information=0.0,
    risk=0.0,
    hypotheses=(),
):
    return SemanticCandidate(
        candidate_id=candidate_id,
        source_kind=source_kind,
        entity_id=candidate_id,
        place_id=place,
        route_id=f"route:{candidate_id}",
        route_distance_m=distance,
        route_reachable=reachable,
        information_gain=information,
        failure_risk=risk,
        hypothesis_refs=tuple(hypotheses),
    )


def assessment(candidate_id, confidence, group=0):
    return SemanticAssessment(candidate_id, SemanticConfidence(confidence), group, "test")


class SemanticProtocolTests(unittest.TestCase):
    def test_strict_json_parses_only_known_unique_candidates(self):
        parsed = parse_semantic_assessments(
            '{"assessments":[{"candidate_id":"Frontier 0",'
            '"confidence":"medium","rank_group":1,"evidence":"door"}]}',
            ["Frontier 0"],
        )
        self.assertEqual(parsed["Frontier 0"].confidence, SemanticConfidence.MEDIUM)
        self.assertIsNone(parse_semantic_assessments(
            '{"assessments":[{"candidate_id":"Frontier 9",'
            '"confidence":"high","rank_group":0}]}',
            ["Frontier 0"],
        ))

    def test_semantic_protocol_rejects_empty_or_partial_assessments(self):
        candidate_ids = ["Snapshot 0, Object 0", "Frontier 0"]
        self.assertIsNone(parse_semantic_assessments(
            '{"assessments":[]}', candidate_ids,
        ))
        self.assertIsNone(parse_semantic_assessments(
            '{"assessments":[{"candidate_id":"Frontier 0",'
            '"confidence":"low","rank_group":1}]}',
            candidate_ids,
        ))

    def test_semantic_protocol_accepts_complete_fenced_top_level_array(self):
        parsed = parse_semantic_assessments(
            '```json\n['
            '{"candidate_id":"Snapshot 0, Object 0","confidence":"high",'
            '"rank_group":0,"evidence":"visible"},'
            '{"candidate_id":"Frontier 0","confidence":"medium",'
            '"rank_group":1,"evidence":"useful"}]\n```',
            ["Snapshot 0, Object 0", "Frontier 0"],
        )
        self.assertEqual(parsed["Snapshot 0, Object 0"].confidence, SemanticConfidence.HIGH)
        self.assertEqual(parsed["Frontier 0"].confidence, SemanticConfidence.MEDIUM)

    def test_terminal_parser_is_safe_on_bad_output(self):
        result = parse_terminal_verification("not-json", 1)
        self.assertEqual(result.verdict, VerificationVerdict.ERROR)
        result = parse_terminal_verification(
            '{"verdict":"uncertain","confidence":0.4,"reason":"occluded"}', 2
        )
        self.assertEqual(result.verdict, VerificationVerdict.UNCERTAIN)
        self.assertEqual(result.attempt, 2)

    def test_low_confidence_terminal_verdict_becomes_uncertain(self):
        from src.hierarchical_navigator import query_terminal_verification

        response = '{"verdict":"confirmed","confidence":0.4,"reason":"partial"}'
        with patch("src.eval_utils_gpt_goatbench.call_openai_api", return_value=response):
            result = query_terminal_verification(
                np.zeros((16, 16, 3), dtype=np.uint8),
                "object", "find towel", "towel", "test-model",
            )
        self.assertEqual(result.verdict, VerificationVerdict.UNCERTAIN)

    def test_terminal_verification_binds_selected_candidate_identity(self):
        from src.hierarchical_navigator import query_terminal_verification

        captured = []

        def fake_call(system, content, **kwargs):
            captured.append((system, content, kwargs))
            return '{"verdict":"confirmed","confidence":0.9,"reason":"same entity"}'

        with patch("src.eval_utils_gpt_goatbench.call_openai_api", side_effect=fake_call):
            result = query_terminal_verification(
                np.zeros((16, 16, 3), dtype=np.uint8),
                "object", "find sink", "sink", "test-model",
                candidate_image=np.ones((8, 8, 3), dtype=np.uint8),
                candidate_label="refrigerator",
                require_candidate_identity=True,
            )
        self.assertEqual(result.verdict, VerificationVerdict.CONFIRMED)
        self.assertIn("same physical candidate", captured[0][0])
        self.assertIn("refrigerator", str(captured[0][1]))
        self.assertEqual(len(captured[0][1]), 2)

    def test_semantic_request_is_route_blind_and_cached_by_evidence(self):
        from src.eval_utils_gpt_goatbench import assess_navigation_step

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        step = {
            "question": "find the towel",
            "task_type": "object",
            "class": "towel",
            "image": None,
            "obj_map": {1: "towel"},
            "snapshot_objects": {"frame.png": [1]},
            "snapshot_source_indices": [0],
            "snapshot_imgs": {"frame.png": {
                "full_img": image,
                "object_crop": [{"obj_class": "towel", "obj_id": 1, "crop": image}],
            }},
            "frontier_imgs": [image],
            "frontier_stable_ids": ["frontier-stable"],
            "frontier_semantic_predictions": [None],
            "hgr_topology_context": {"total_distance_m": 999.0},
        }
        cfg = OmegaConf.create({
            "prefiltering": False,
            "top_k_categories": 10,
            "use_full_obj_list": True,
            "egocentric_views": False,
            "vlm_model": "test-model",
        })
        navigator = HierarchicalNavigator({"semantic_parse_attempts": 2})
        response = (
            '{"assessments":['
            '{"candidate_id":"Snapshot 0, Object 0","confidence":"high","rank_group":0},'
            '{"candidate_id":"Frontier 0","confidence":"low","rank_group":1}]}'
        )
        captured = []

        def fake_call(system, content, **kwargs):
            captured.append((system, content, kwargs))
            return response

        with patch("src.eval_utils_gpt_goatbench.call_openai_api", side_effect=fake_call):
            first = assess_navigation_step(step, cfg, navigator)
            second = assess_navigation_step(step, cfg, navigator)
        self.assertEqual(len(captured), 1)
        rendered = str(captured[0][1])
        self.assertNotIn("total_distance_m", rendered)
        self.assertEqual(first[4], second[4])
        self.assertEqual(first[0]["Snapshot 0, Object 0"].confidence, SemanticConfidence.HIGH)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.navigator = HierarchicalNavigator({"max_verification_attempts": 2})
        self.navigator.begin_goal("goal")

    def test_short_low_confidence_object_cannot_beat_high_identity(self):
        high = candidate("correct", distance=8.0)
        low = candidate("wrong", distance=0.1)
        decision = self.navigator.decide(
            [low, high],
            {
                "correct": assessment("correct", "high"),
                "wrong": assessment("wrong", "low"),
            },
            "place_0",
        )
        self.assertEqual(decision.candidate.candidate_id, "correct")
        self.assertEqual(decision.intent_kind, IntentKind.TARGET_APPROACH)

    def test_geometry_breaks_only_semantic_group_tie(self):
        far = candidate("far", distance=5.0)
        near = candidate("near", distance=1.0)
        decision = self.navigator.decide(
            [far, near],
            {
                "far": assessment("far", "high", 0),
                "near": assessment("near", "high", 0),
            },
            "place_0",
        )
        self.assertEqual(decision.candidate.candidate_id, "near")

        decision = self.navigator.decide(
            [far, near],
            {
                "far": assessment("far", "high", 0),
                "near": assessment("near", "high", 1),
            },
            "place_0",
        )
        self.assertEqual(decision.candidate.candidate_id, "far")

    def test_medium_object_is_revisit_not_target(self):
        item = candidate("possible", place="place_2")
        decision = self.navigator.decide(
            [item], {"possible": assessment("possible", "medium")}, "place_0"
        )
        self.assertEqual(decision.intent_kind, IntentKind.EVIDENCE_REVISIT)
        self.assertEqual(decision.execution_mode, ExecutionMode.TOPO_TSDF)

    def test_same_place_is_native_local_execution(self):
        item = candidate("target", place="place_0")
        decision = self.navigator.decide(
            [item], {"target": assessment("target", "high")}, "place_0"
        )
        self.assertEqual(decision.execution_mode, ExecutionMode.LOCAL_TSDF)

    def test_frontier_information_gain_breaks_semantic_tie(self):
        low_info = candidate("f0", "frontier", information=0.1)
        high_info = candidate("f1", "frontier", information=0.9, distance=3.0)
        decision = self.navigator.decide(
            [low_info, high_info],
            {
                "f0": assessment("f0", "medium"),
                "f1": assessment("f1", "medium"),
            },
            "place_0",
        )
        self.assertEqual(decision.candidate.candidate_id, "f1")
        self.assertEqual(decision.intent_kind, IntentKind.FRONTIER_EXPLORE)

    def test_no_match_frontier_is_last_resort_exploration(self):
        item = candidate("f0", "frontier", information=0.4)
        decision = self.navigator.decide(
            [item], {"f0": assessment("f0", "no_match")}, "place_0"
        )
        self.assertEqual(decision.candidate.candidate_id, "f0")
        self.assertEqual(decision.intent_kind, IntentKind.FRONTIER_EXPLORE)

    def test_unreachable_candidates_are_not_selected(self):
        blocked = candidate("blocked", reachable=False)
        open_frontier = candidate("frontier", "frontier")
        decision = self.navigator.decide(
            [blocked, open_frontier],
            {
                "blocked": assessment("blocked", "high"),
                "frontier": assessment("frontier", "low"),
            },
            "place_0",
        )
        self.assertEqual(decision.candidate.candidate_id, "frontier")


class EventLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.navigator = HierarchicalNavigator({"max_verification_attempts": 2})
        self.navigator.begin_goal("goal")
        item = candidate("target", place="place_0", hypotheses=("hypothesis-1",))
        decision = self.navigator.decide(
            [item], {"target": assessment("target", "high")}, "place_0"
        )
        self.intent = self.navigator.install(decision, terminal=(2, 3))

    def test_waypoint_only_advances_execution_without_reselection(self):
        before = self.navigator.stats["high_level_replans"]
        command = self.navigator.on_event(
            NavigationEvent(NavigationEventType.WAYPOINT_ARRIVED, 3)
        )
        self.assertEqual(command.command, NavigatorCommandType.CONTINUE)
        self.assertEqual(self.navigator.stats["high_level_replans"], before)
        self.assertIs(self.navigator.active_intent, self.intent)

    def test_terminal_requires_verification_and_confirmation_stops(self):
        command = self.navigator.on_event(
            NavigationEvent(NavigationEventType.TERMINAL_ARRIVED, 4)
        )
        self.assertEqual(command.command, NavigatorCommandType.VERIFY)
        result = parse_terminal_verification(
            '{"verdict":"confirmed","confidence":0.95,"reason":"visible"}', 1
        )
        command = self.navigator.record_verification(result, 4)
        self.assertEqual(command.command, NavigatorCommandType.STOP)

    def test_uncertain_gets_one_retry_then_reselects_without_negative(self):
        result = parse_terminal_verification(
            '{"verdict":"uncertain","confidence":0.5,"reason":"occluded"}', 1
        )
        first = self.navigator.record_verification(result, 5)
        self.assertEqual(first.command, NavigatorCommandType.VERIFY)
        second = self.navigator.record_verification(result, 6)
        self.assertEqual(second.command, NavigatorCommandType.RESELECT)
        self.assertFalse(self.navigator.suppressed_evidence)

    def test_error_suppresses_request_digest_but_not_hypothesis(self):
        result = parse_terminal_verification(None, 1)
        command = self.navigator.record_verification(result, 5, "digest")
        self.assertEqual(command.command, NavigatorCommandType.RESELECT)
        self.assertIn("digest", self.navigator.suppressed_evidence)
        self.assertEqual(self.intent.hypothesis_refs, ("hypothesis-1",))

    def test_source_invalidation_releases_intent(self):
        command = self.navigator.on_event(NavigationEvent(
            NavigationEventType.SOURCE_INVALIDATED, 5, "removed"
        ))
        self.assertEqual(command.command, NavigatorCommandType.RESELECT)
        self.assertIsNone(self.navigator.active_intent)

    def test_contracts_remain_checkpoint_serializable(self):
        restored = pickle.loads(pickle.dumps(self.navigator))
        self.assertEqual(restored.active_intent.candidate_id, "target")
        self.assertEqual(restored.active_intent.terminal, (2.0, 3.0))


class LocalExecutionAdapterTests(unittest.TestCase):
    def test_same_place_route_uses_native_controller_and_keeps_route_validity(self):
        from src.hgr_dual_topo import DualTopoRuntime
        from src.place_topology import PlaceTopology

        graph = PlaceTopology(2.0, 1.0, stage="route_only")
        graph.observe_pose([2, 2], 0)
        mask = np.ones((8, 8), dtype=bool)
        runtime = DualTopoRuntime({"local_same_place": True})
        runtime.begin_goal("goal")
        approach = runtime.resolve(graph, mask, [2, 2], [4, 2], 1.0)

        class FrontierChoice:
            topo_id = "frontier"
            hypothesis_node_id = None

        runtime.install(FrontierChoice(), approach, mask.shape)
        self.assertTrue(runtime.uses_local_controller)
        self.assertEqual(
            runtime.ensure_route(graph, mask, [2, 2], 1.0),
            "local_direct_retained",
        )
        self.assertTrue(runtime.acknowledge_motion([3, 2], target_arrived=False))
        self.assertTrue(runtime.acknowledge_motion([4, 2], target_arrived=True))
        self.assertEqual(runtime.execution.cursor, len(runtime.execution.path) - 1)

    def test_route_audit_accepts_prevalidated_native_local_motion(self):
        from scripts.hgr_v2_experiments import audit_events

        events = [
            {
                "event": "v2_execution", "subtask_id": "g", "step": 1,
                "intent_id": "i", "route_id": "r", "execution_mode": "local_tsdf",
                "reason": "local_direct_retained",
                "guidance": {"certified_path": [[1, 1], [2, 1]], "progress_index": 0},
            },
            {
                "event": "v2_motion_acknowledged", "subtask_id": "g", "step": 1,
                "intent_id": "i", "route_id": "r", "execution_mode": "local_tsdf",
                "acknowledged": True, "terminal_arrived": True,
                "guidance": {"progress_index": 1},
            },
        ]
        result = audit_events(events)
        self.assertTrue(result["trace_contract_passed"])
        self.assertEqual(result["route_violations"], [])


class HierarchicalConfigTests(unittest.TestCase):
    def test_active_and_shadow_configs_are_isolated(self):
        from run_goatbench_evaluation import _load_config

        root = os.path.dirname(os.path.dirname(__file__))
        active = _load_config(os.path.join(
            root, "cfg", "eval_goatbench_hierarchical_navigation_train.yaml"
        ))
        shadow = _load_config(os.path.join(
            root, "cfg", "eval_goatbench_hierarchical_navigation_shadow.yaml"
        ))
        self.assertTrue(active.hierarchical_navigation.enabled)
        self.assertFalse(active.hierarchical_navigation.shadow_only)
        self.assertFalse(active.hgr_dual_topo.enabled)
        self.assertEqual(active.active_topology.mode, "stable_only")
        self.assertTrue(shadow.hierarchical_navigation.shadow_only)
        self.assertTrue(shadow.hgr_dual_topo.enabled)


if __name__ == "__main__":
    unittest.main()
