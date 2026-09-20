import unittest

import numpy as np

from src.goal_belief import (
    BeliefActionType,
    EvidenceSource,
    ExecutableNode,
    GoalBeliefMemory,
    VerificationOption,
    category_compatibility_score,
    evaluate_snapshot_acceptance,
    normalize_goal_key,
)


class GoalBeliefMemoryTest(unittest.TestCase):
    def setUp(self):
        self.memory = GoalBeliefMemory({
            "prior": 0.20,
            "top_k": 3,
            "elimination_threshold": 0.05,
            "min_negative_verify_views": 2,
        })
        self.goal = normalize_goal_key("chair")

    def evidence(self, evidence_id, source, score, point, heading=0.0,
                 node="frontier_0", observation=None):
        return self.memory.make_evidence(
            evidence_id, node, self.goal, source, score, 0,
            viewpoint=np.asarray(point, dtype=float), heading=heading,
            observation_id=observation or evidence_id,
        )

    def test_id_and_observation_deduplication(self):
        item = self.evidence("e0", EvidenceSource.CLIP_GOAL, 0.9, [0, 0],
                             observation="frame0")
        self.assertTrue(self.memory.add_evidence(item))
        self.assertFalse(self.memory.add_evidence(item))
        same_source_observation = self.evidence(
            "e1", EvidenceSource.CLIP_GOAL, 0.8, [0, 0], observation="frame0"
        )
        self.assertFalse(self.memory.add_evidence(same_source_observation))
        other_source = self.evidence(
            "e2", EvidenceSource.DETECTOR, 0.8, [0, 0], observation="frame0"
        )
        self.assertTrue(self.memory.add_evidence(other_source))

    def test_same_view_is_correlated_independent_view_strengthens(self):
        self.memory.add_evidence(self.evidence(
            "a", EvidenceSource.DETECTOR, 0.9, [0, 0]
        ))
        first = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.memory.add_evidence(self.evidence(
            "b", EvidenceSource.DETECTOR, 0.9, [0.2, 0.1]
        ))
        correlated = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.assertAlmostEqual(first.posterior, correlated.posterior, places=6)
        self.memory.add_evidence(self.evidence(
            "c", EvidenceSource.DETECTOR, 0.9, [1.0, 0.0]
        ))
        independent = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.assertGreater(independent.posterior, correlated.posterior)
        self.assertEqual(independent.independent_view_count, 2)

    def test_verify_replaces_weak_same_cluster_and_conflict_is_visible(self):
        self.memory.add_evidence(self.evidence(
            "clip", EvidenceSource.CLIP_GOAL, 0.9, [0, 0]
        ))
        positive = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.memory.add_evidence(self.evidence(
            "verify", EvidenceSource.ACTIVE_VERIFY_VLM, 0.1, [0, 0]
        ))
        corrected = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.assertLess(corrected.posterior, positive.posterior)
        self.assertGreater(corrected.conflict, 0.0)
        trace = self.memory.evidence_trace("frontier_0", self.goal)
        self.assertEqual(len(trace["clusters"][0]["evidence"]), 2)

    def test_two_independent_negative_verify_views_eliminate_without_delete(self):
        for index, point in enumerate(([0, 0], [1, 0])):
            self.memory.add_evidence(self.evidence(
                f"verify-{index}", EvidenceSource.ACTIVE_VERIFY_VLM,
                0.05, point
            ))
        belief = self.memory.belief("frontier_0", self.goal, 1.0, 0.5)
        self.assertTrue(belief.eliminated)
        self.assertLess(belief.posterior, 0.05)
        self.assertEqual(len(self.memory.clusters("frontier_0", self.goal)), 2)

    def test_independent_nodes_topk_unknown_and_goal_isolation(self):
        nodes = []
        for index, score in enumerate((0.95, 0.85, 0.75, 0.65)):
            node_id = f"frontier_{index}"
            self.memory.add_evidence(self.evidence(
                f"e-{index}", EvidenceSource.DETECTOR, score,
                [index, 0], node=node_id
            ))
            nodes.append(ExecutableNode(
                node_id, "frontier", index, 1.0, 0.8, index / 4.0
            ))
        result = self.memory.build_hypotheses(self.goal, nodes, 1)
        self.assertEqual(len(result.hypotheses), 3)
        self.assertEqual(result.hypotheses[0].node_id, "frontier_0")
        self.assertGreaterEqual(result.unknown_probability, 0.0)
        self.assertLessEqual(result.unknown_probability, 1.0)
        # A temporarily absent executable source cannot appear in the action set.
        absent = self.memory.build_hypotheses(self.goal, nodes[1:], 2)
        self.assertNotIn("frontier_0", absent.executable_node_ids)
        other_goal = self.memory.belief(
            "frontier_0", normalize_goal_key("table"), 1.0, 0.8
        )
        self.assertAlmostEqual(other_goal.posterior, 0.20, places=6)

    def test_action_thresholds_margin_and_verify_budget(self):
        self.memory.add_evidence(self.evidence(
            "strong", EvidenceSource.DETECTOR, 0.95, [0, 0], node="snapshot_0"
        ))
        self.memory.add_evidence(self.evidence(
            "strong2", EvidenceSource.DETECTOR, 0.95, [1, 0], node="snapshot_0"
        ))
        facts = [
            ExecutableNode("snapshot_0", "snapshot", "s", 1.0, 1.0, 0.0),
            ExecutableNode("frontier_0", "frontier", 0, 1.0, 1.0, 0.2,
                           map_information_gain=1.0),
        ]
        hypotheses = self.memory.build_hypotheses(self.goal, facts, 0)
        actions = self.memory.build_actions(hypotheses, facts, step=0)
        self.assertEqual(actions[0].action_type, BeliefActionType.NAVIGATE)
        direct, tied = self.memory.choose_actions_for_tiebreak(actions, 0.0)
        self.assertIsNotNone(direct)
        self.assertEqual(tied, [])

        medium = GoalBeliefMemory({"prior": 0.5})
        medium_goal = normalize_goal_key("lamp")
        medium.add_evidence(medium.make_evidence(
            "m", "frontier_0", medium_goal, EvidenceSource.CLIP_GOAL,
            0.6, 0, viewpoint=[0, 0], observation_id="m"
        ))
        medium_facts = [ExecutableNode(
            "frontier_0", "frontier", 0, 1.0, 1.0, 0.1
        )]
        medium_hypotheses = medium.build_hypotheses(medium_goal, medium_facts, 0)
        option = VerificationOption(np.array([1, 0]), np.array([0, 0]), 1, 1, 0)
        verify = medium.build_actions(
            medium_hypotheses, medium_facts,
            verification_options={"frontier_0": option}, step=0,
        )[0]
        self.assertEqual(verify.action_type, BeliefActionType.VERIFY)
        medium.record_verify_attempt("frontier_0", medium_goal, 0)
        self.assertFalse(medium.can_verify("frontier_0", medium_goal, 3))
        self.assertTrue(medium.can_verify("frontier_0", medium_goal, 5))

        context_facts = [ExecutableNode(
            "frontier_0", "snapshot_context", "s0", 1.0, 1.0, 0.1
        )]
        context_hypotheses = medium.build_hypotheses(
            medium_goal, context_facts, 6
        )
        context_actions = medium.build_actions(
            context_hypotheses,
            context_facts,
            verification_options={"frontier_0": option},
            step=6,
        )
        self.assertEqual(context_actions[0].action_type, BeliefActionType.VERIFY)
        no_view = medium.build_actions(
            context_hypotheses, context_facts, step=6
        )
        self.assertEqual(no_view, [])

    def test_mock_two_subtasks_reuse_category_then_episode_reset(self):
        # No API participates when the planner has a clear utility winner.
        self.memory.add_evidence(self.evidence(
            "chair-view", EvidenceSource.DETECTOR, 0.95, [0, 0]
        ))
        fact = ExecutableNode(
            "frontier_0", "frontier", 0, 1.0, 1.0, 0.0,
            map_information_gain=1.0,
        )
        first = self.memory.build_hypotheses(self.goal, [fact], 1)
        second = self.memory.build_hypotheses(
            normalize_goal_key("chair"), [fact], 2
        )
        self.assertAlmostEqual(
            first.hypotheses[0].posterior,
            second.hypotheses[0].posterior,
        )
        action = self.memory.build_actions(second, [fact], step=2)
        chosen, ties = self.memory.choose_actions_for_tiebreak(action)
        self.assertIsNotNone(chosen)
        self.assertEqual(ties, [])

        new_episode = GoalBeliefMemory({"prior": 0.20})
        reset = new_episode.belief("frontier_0", self.goal, 1.0, 1.0)
        self.assertAlmostEqual(reset.posterior, 0.20, places=6)

    def test_v2_fast_duplicate_and_posterior_cache_invalidation(self):
        memory = GoalBeliefMemory({"fast_evidence_path": True})
        item = memory.make_evidence(
            "fast", "frontier_0", self.goal, EvidenceSource.DETECTOR,
            0.9, 0, viewpoint=[0, 0], observation_id="frame",
        )
        self.assertFalse(memory.contains_evidence(item, count_fast_skip=True))
        self.assertTrue(memory.add_evidence(item))
        first = memory.belief("frontier_0", self.goal, 0.5, 0.5)
        updates = memory.stats["posterior_updates"]
        self.assertIs(first, memory.belief("frontier_0", self.goal, 0.5, 0.5))
        self.assertEqual(memory.stats["posterior_updates"], updates)
        self.assertTrue(memory.contains_evidence(item, count_fast_skip=True))
        self.assertEqual(memory.stats["evidence_fast_skipped"], 1)
        # Structural confidence and reachability are part of the cache key.
        self.assertIsNot(first, memory.belief("frontier_0", self.goal, 0.6, 0.5))
        new_item = memory.make_evidence(
            "new", "frontier_0", self.goal, EvidenceSource.DETECTOR,
            0.8, 1, viewpoint=[1, 0], observation_id="frame-2",
        )
        self.assertTrue(memory.add_evidence(new_item))
        self.assertIsNot(first, memory.belief("frontier_0", self.goal, 0.5, 0.5))

    def test_v3_fusion_cache_reused_when_only_structure_changes(self):
        memory = GoalBeliefMemory({"fast_evidence_path": True})
        memory.add_evidence(memory.make_evidence(
            "fusion", "frontier_0", self.goal, EvidenceSource.DETECTOR,
            0.9, 0, viewpoint=[0, 0], observation_id="fusion",
        ))
        memory.belief("frontier_0", self.goal, 0.5, 0.5)
        misses = memory.stats["evidence_fusion_cache_misses"]
        memory.belief("frontier_0", self.goal, 0.6, 0.7)
        self.assertEqual(memory.stats["evidence_fusion_cache_misses"], misses)
        self.assertGreater(memory.stats["evidence_fusion_cache_hits"], 0)
        memory.add_evidence(memory.make_evidence(
            "fusion-2", "frontier_0", self.goal, EvidenceSource.DETECTOR,
            0.8, 1, viewpoint=[1, 0], observation_id="fusion-2",
        ))
        memory.belief("frontier_0", self.goal, 0.6, 0.7)
        self.assertEqual(memory.stats["evidence_fusion_cache_misses"], misses + 1)

    def test_v2_unknown_ignores_duplicate_candidate_count(self):
        memory = GoalBeliefMemory({"unknown_method": "weighted_geometric"})
        memory.add_evidence(memory.make_evidence(
            "u", "frontier_0", self.goal, EvidenceSource.DETECTOR,
            0.9, 0, viewpoint=[0, 0], observation_id="u",
        ))
        one = [ExecutableNode("frontier_0", "frontier", 0, 1.0, 0.8, 0.1)]
        # A copied candidate with the same belief must not make the geometric
        # missing rate artificially smaller.
        memory.add_evidence(memory.make_evidence(
            "u-copy", "frontier_1", self.goal, EvidenceSource.DETECTOR,
            0.9, 0, viewpoint=[0, 0], observation_id="u-copy",
        ))
        duplicated = one + [ExecutableNode(
            "frontier_1", "frontier", 1, 1.0, 0.8, 0.1
        )]
        unknown_one = memory.build_hypotheses(self.goal, one, 0).unknown_probability
        unknown_two = memory.build_hypotheses(self.goal, duplicated, 1).unknown_probability
        self.assertAlmostEqual(unknown_one, unknown_two, places=7)
        exact_duplicate = memory.build_hypotheses(
            self.goal, one + [one[0]], 2
        ).unknown_probability
        self.assertAlmostEqual(unknown_one, exact_duplicate, places=7)
        empty = GoalBeliefMemory({"unknown_method": "weighted_geometric"})
        self.assertEqual(
            empty.build_hypotheses(self.goal, one, 0).unknown_probability, 1.0
        )

    def test_v3_controlled_unknown_budget_and_resets(self):
        memory = GoalBeliefMemory({"unknown_method": "weighted_geometric"})
        fact = ExecutableNode(
            "frontier_0", "frontier", 0, 1.0, 1.0, 0.1,
            map_information_gain=1.0,
        )
        hypotheses = memory.build_hypotheses(self.goal, [fact, fact], 0)
        ordinary = memory.build_actions(
            hypotheses, [fact], step=0, include_unknown=False
        )
        action, decision = memory.build_controlled_unknown_action(
            hypotheses, [fact, fact], ordinary, step=0,
        )
        self.assertTrue(decision.eligible)
        memory.record_action_selection(action)
        action, _ = memory.build_controlled_unknown_action(
            hypotheses, [fact], ordinary, step=1,
        )
        memory.record_action_selection(action)
        action, decision = memory.build_controlled_unknown_action(
            hypotheses, [fact], ordinary, step=2,
        )
        self.assertIsNone(action)
        self.assertEqual(decision.reason, "consecutive_budget")
        new_fact = ExecutableNode(
            "frontier_1", "frontier", 1, 1.0, 1.0, 0.1,
            map_information_gain=1.0,
        )
        new_hypotheses = memory.build_hypotheses(
            self.goal, [fact, new_fact], 3
        )
        action, decision = memory.build_controlled_unknown_action(
            new_hypotheses, [fact, new_fact], ordinary, step=3,
        )
        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reset_reason, "new_executable_node")

    def test_v3_deterministic_priority_never_requests_tiebreak(self):
        memory = GoalBeliefMemory({"prior": 0.2})
        facts = [ExecutableNode(
            "frontier_0", "frontier", 0, 1.0, 1.0, 0.1,
            map_information_gain=0.5,
        )]
        hypotheses = memory.build_hypotheses(self.goal, facts, 0)
        actions = memory.build_actions(
            hypotheses, facts, step=0, include_unknown=False
        )
        selected = memory.choose_deterministic_action(actions)
        self.assertEqual(selected.action_type, BeliefActionType.EXPLORE)
        self.assertEqual(memory.stats["vlm_tiebreak_choices"], 0)

    def test_v2_clip_warmup_freeze_replay_and_goal_isolation(self):
        memory = GoalBeliefMemory({
            "clip_calibration": {"warmup_samples": 16, "temperature": 1.5}
        })
        replay = []
        event = None
        for index in range(16):
            replay, event = memory.observe_clip(
                f"clip-{index}", f"frontier_{index}", self.goal,
                0.10 + index * 0.01, index,
                viewpoint=[index, 0], observation_id=f"frame-{index}",
            )
            if index < 15:
                self.assertEqual(replay, [])
        self.assertEqual(event["event"], "freeze")
        self.assertEqual(len(replay), 16)
        self.assertTrue(all(0.20 <= item.score <= 0.80 for item in replay))
        self.assertAlmostEqual(float(np.median([item.score for item in replay])), 0.5, places=2)
        other_goal = normalize_goal_key("table")
        other, other_event = memory.observe_clip(
            "other", "frontier_0", other_goal, 0.9, 20,
            viewpoint=[0, 0], observation_id="other",
        )
        self.assertEqual(other, [])
        self.assertEqual(other_event["event"], "warmup")

    def test_confirmation_requires_current_frontier_and_has_end_reasons(self):
        memory = GoalBeliefMemory({"prior": 0.8})
        state = memory.create_confirmation(
            "frontier_0", self.goal, 10, 0.8, duration_steps=5,
        )
        self.assertIsNone(memory.build_confirmation_action(state, []))
        fact = ExecutableNode(
            "frontier_0", "frontier", 3, 1.0, 1.0, 0.2
        )
        action = memory.build_confirmation_action(state, [fact])
        self.assertEqual(action.action_type, BeliefActionType.CONFIRM_APPROACH)
        self.assertEqual(action.source_index, 3)
        memory.record_confirmation_approach(self.goal)
        memory.record_confirmation_approach(self.goal)
        self.assertIsNone(memory.build_confirmation_action(state, [fact]))
        memory.resolve_confirmation(self.goal, "no_snapshot_after_approach_budget")
        events = memory.pop_confirmation_events()
        self.assertEqual(events[-1]["reason"], "no_snapshot_after_approach_budget")

        expired = memory.create_confirmation("frontier_0", self.goal, 20, 0.8)
        self.assertIsNone(memory.active_confirmation(self.goal, 26))
        self.assertEqual(expired.end_reason, "expired")

    def test_v3_confirmation_waits_for_observation_and_does_not_overwrite(self):
        memory = GoalBeliefMemory({"prior": 0.8})
        state = memory.create_confirmation(
            "frontier_0", self.goal, 10, 0.8, duration_steps=8,
            positive_source="detector", reuse_existing=True,
        )
        updated = memory.create_confirmation(
            "frontier_0", self.goal, 11, 0.9, duration_steps=8,
            positive_source="active_verify_vlm", reuse_existing=True,
        )
        self.assertIs(state, updated)
        self.assertEqual(updated.created_step, 10)
        self.assertEqual(updated.expires_step, 18)
        self.assertEqual(updated.max_posterior, 0.9)
        fact = ExecutableNode("frontier_0", "frontier", 0, 1, 1, 0.1)
        memory.record_confirmation_approach(
            self.goal, 12, wait_for_observation=True
        )
        self.assertIsNone(memory.build_confirmation_action(state, [fact]))
        memory.record_confirmation_observation(self.goal, 12, False)
        self.assertTrue(state.awaiting_observation)
        memory.record_confirmation_observation(self.goal, 13, False)
        self.assertFalse(state.awaiting_observation)
        memory.record_confirmation_approach(
            self.goal, 14, wait_for_observation=True
        )
        memory.record_confirmation_observation(self.goal, 15, False)
        self.assertFalse(state.active)
        self.assertEqual(state.end_reason, "no_snapshot_after_observation")

        first = memory.create_confirmation(
            "frontier_0", self.goal, 20, 0.8, reuse_existing=True
        )
        second = memory.create_confirmation(
            "frontier_1", self.goal, 21, 0.9, reuse_existing=True
        )
        self.assertEqual(first.end_reason, "superseded")
        self.assertTrue(second.active)

    def test_v3_rejected_snapshot_ttl_and_compatible_release(self):
        memory = GoalBeliefMemory()
        rejected = evaluate_snapshot_acceptance(
            target_label="rug", detected_label="towel",
            detector_confidence=0.8, detection_count=2, posterior=0.8,
            candidate_score=0.0, best_candidate_score=0.0,
        )
        memory.reject_snapshot(self.goal, 7, 10, rejected, duration_steps=5)
        self.assertTrue(memory.is_snapshot_rejected(
            self.goal, 7, 11, compatibility=0.0,
            detector_confidence=0.8, detection_count=2,
        )[0])
        suppressed, reason = memory.is_snapshot_rejected(
            self.goal, 7, 12, compatibility=1.0,
            detector_confidence=0.9, detection_count=3,
        )
        self.assertFalse(suppressed)
        self.assertEqual(reason, "stronger_compatible_detection")
        memory.reject_snapshot(self.goal, 8, 20, rejected, duration_steps=5)
        self.assertFalse(memory.is_snapshot_rejected(
            self.goal, 8, 26
        )[0])

    def test_confirmed_real_snapshot_can_navigate_below_snapshot_threshold(self):
        memory = GoalBeliefMemory({"prior": 0.20})
        memory.add_evidence(memory.make_evidence(
            "positive-verify", "frontier_0", self.goal,
            EvidenceSource.ACTIVE_VERIFY_VLM, 0.95, 0,
            viewpoint=[0, 0], observation_id="verify",
        ))
        confirmed = memory.belief("frontier_0", self.goal, 1.0, 1.0)
        self.assertGreaterEqual(confirmed.posterior, 0.75)
        state = memory.create_confirmation(
            "frontier_0", self.goal, 0, confirmed.posterior
        )
        snapshot_fact = ExecutableNode(
            "observed_0", "snapshot", "real-snapshot", 1.0, 1.0, 0.1
        )
        action = memory.build_confirmation_snapshot_action(
            state, snapshot_fact, confirmed
        )
        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, BeliefActionType.NAVIGATE)
        self.assertLess(action.belief.posterior, 0.75)
        self.assertEqual(action.source_index, "real-snapshot")
        self.assertIsNone(memory.build_confirmation_snapshot_action(
            state,
            ExecutableNode("ghost", "snapshot", None, 1, 1, 0, reachable=False),
            confirmed,
        ))

    def test_v3_failure_case_category_compatibility_is_conservative(self):
        self.assertEqual(category_compatibility_score("plant", "potted plant"), 0.90)
        self.assertEqual(category_compatibility_score("rug", "carpet"), 0.90)
        self.assertEqual(category_compatibility_score("hanging clothes", "clothes"), 0.90)
        # Observed baseline mistakes must never pass as aliases.
        self.assertEqual(category_compatibility_score("rug", "towel"), 0.0)
        self.assertEqual(category_compatibility_score("mirror", "picture"), 0.0)
        self.assertEqual(category_compatibility_score("microwave", "tv"), 0.0)

    def test_v3_snapshot_gate_rejects_false_and_unstable_terminal_choices(self):
        false_choice = evaluate_snapshot_acceptance(
            target_label="rug", detected_label="towel",
            detector_confidence=0.99, detection_count=10, posterior=0.99,
            candidate_score=0.99, best_candidate_score=0.99,
        )
        self.assertFalse(false_choice.accepted)
        self.assertEqual(false_choice.reason, "incompatible_category")
        unstable = evaluate_snapshot_acceptance(
            target_label="pillow", detected_label="pillow",
            detector_confidence=0.8, detection_count=1, posterior=0.8,
            candidate_score=0.4, best_candidate_score=0.4,
        )
        self.assertFalse(unstable.accepted)
        self.assertEqual(unstable.reason, "insufficient_multiview_detections")
        wrong_instance = evaluate_snapshot_acceptance(
            target_label="pillow", detected_label="pillow",
            detector_confidence=0.8, detection_count=3, posterior=0.8,
            candidate_score=0.5, best_candidate_score=0.8,
        )
        self.assertFalse(wrong_instance.accepted)
        self.assertEqual(wrong_instance.reason, "less_stable_instance_available")
        preserved_success = evaluate_snapshot_acceptance(
            target_label="refrigerator", detected_label="refrigerator",
            detector_confidence=0.8, detection_count=3, posterior=0.45,
            candidate_score=0.8, best_candidate_score=0.8,
        )
        self.assertTrue(preserved_success.accepted)

    def test_v3_alias_requires_stronger_evidence_or_confirmation(self):
        weak_alias = evaluate_snapshot_acceptance(
            target_label="plant", detected_label="potted plant",
            detector_confidence=0.8, detection_count=3, posterior=0.7,
            candidate_score=0.7, best_candidate_score=0.7,
        )
        self.assertFalse(weak_alias.accepted)
        self.assertEqual(weak_alias.reason, "alias_requires_stronger_evidence")
        confirmed = evaluate_snapshot_acceptance(
            target_label="plant", detected_label="potted plant",
            detector_confidence=0.8, detection_count=1, posterior=0.7,
            candidate_score=0.7, best_candidate_score=0.7,
            confirmed=True,
        )
        self.assertTrue(confirmed.accepted)
        self.assertEqual(confirmed.reason, "active_confirmation")
        confirmed_wrong_instance = evaluate_snapshot_acceptance(
            target_label="plant", detected_label="potted plant",
            detector_confidence=0.8, detection_count=1, posterior=0.7,
            candidate_score=0.4, best_candidate_score=0.8,
            confirmed=True,
        )
        self.assertFalse(confirmed_wrong_instance.accepted)
        self.assertEqual(
            confirmed_wrong_instance.reason, "less_stable_instance_available"
        )


if __name__ == "__main__":
    unittest.main()
