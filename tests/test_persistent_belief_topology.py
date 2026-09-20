import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from src.persistent_belief_topology import (
    ActiveTopologyMode,
    ActiveTopologyTraceWriter,
    ActiveTopoState,
    GoalContext,
    GoalTopoActionType,
    PersistentBeliefTopology,
    VerificationEvidence,
    resolve_active_topology_policy,
)


CFG = {
    "visited_merge_distance": 0.75,
    "observed_merge": {
        "max_distance": 0.75,
        "min_category_jaccard": 0.30,
        "empty_category_max_distance": 0.35,
    },
    "frontier_matching": {
        "min_score": 0.60,
        "distance_scale": 1.0,
        "weights": {"distance": 0.45, "region_iou": 0.25, "direction": 0.15, "visual": 0.15},
    },
    "confidence": {
        "decay": 0.995,
        "positive_delta": 1.0,
        "negative_delta": 1.2,
        "prune_threshold": 0.10,
        "min_negative_evidence": 2,
        "positive_grace_steps": 5,
        "dependency_propagation": 0.70,
    },
    "accessibility": {"block_steps": 5, "suppress_after_failures": 3},
    "active_view": {"verify_relevance": 0.55, "verify_confidence": 0.55, "fold_relevance": 0.20, "no_progress_steps": 12},
    "utility": {"relevance": 1.0, "confidence": 0.4, "accessibility": 0.4, "path_cost": 0.1, "revisit": 0.2, "information_gain": 0.05},
}


def frontier(x, region=None, embedding=None):
    item = SimpleNamespace(
        position=np.array([x, 0.0]),
        orientation=np.array([1.0, 0.0]),
        region=np.asarray(region if region is not None else [[1, 0], [0, 0]], dtype=bool),
        semantic_dist=None,
        hypothesis_node_id=None,
        feature=None,
    )
    item.embedding = embedding
    return item


class PersistentBeliefTopologyTest(unittest.TestCase):
    def setUp(self):
        self.topology = PersistentBeliefTopology(CFG, voxel_size=0.1)

    def test_visited_and_observed_merge(self):
        visited_a = self.topology.upsert_visited([0, 0], 0)
        visited_b = self.topology.upsert_visited([5, 0], 1)
        self.assertEqual(visited_a, visited_b)  # 0.5 m
        snapshot = SimpleNamespace(image="s0", obs_point=np.array([0, 0]))
        observed_a = self.topology.upsert_observed(
            snapshot, [0, 0], ["chair", "table"], 0, visited_a, "snapshot:s0"
        )
        # Same evidence is idempotent even though the caller scans all snapshots each step.
        observed_same = self.topology.upsert_observed(
            snapshot, [2, 0], ["chair"], 1, visited_a, "snapshot:s0"
        )
        self.assertEqual(observed_a, observed_same)
        self.assertEqual(self.topology.nodes[observed_a].view_count, 1)
        observed_b = self.topology.upsert_observed(
            SimpleNamespace(image="s1"), [4, 0], ["chair"], 2, visited_a, "snapshot:s1"
        )
        self.assertEqual(observed_a, observed_b)
        observed_c = self.topology.upsert_observed(
            SimpleNamespace(image="s2"), [20, 0], ["bed"], 3, visited_a, "snapshot:s2"
        )
        self.assertNotEqual(observed_a, observed_c)

    def test_ablation_mode_policy_is_explicit_and_goal_conditioned(self):
        stable = resolve_active_topology_policy("stable_only", "object")
        self.assertEqual(stable.mode, ActiveTopologyMode.STABLE_ONLY)
        self.assertFalse(stable.use_active_view)
        self.assertFalse(stable.use_goal_relevance)

        simple = resolve_active_topology_policy("simple_memory", "description")
        self.assertEqual(simple.mode, ActiveTopologyMode.SIMPLE_MEMORY)
        self.assertTrue(simple.use_active_view)
        self.assertFalse(simple.use_goal_relevance)
        self.assertFalse(simple.use_belief_planner)

        goal_topo = resolve_active_topology_policy("goal_topo_map", "image")
        self.assertEqual(goal_topo.mode, ActiveTopologyMode.GOAL_TOPO_MAP)
        self.assertTrue(goal_topo.use_active_view)
        self.assertTrue(goal_topo.use_goal_relevance)
        self.assertFalse(goal_topo.use_belief_planner)

        ca = resolve_active_topology_policy("confidence_accessibility", "description")
        self.assertTrue(ca.use_active_view)
        self.assertFalse(ca.use_goal_relevance)

        category = resolve_active_topology_policy("category_only", "object")
        self.assertTrue(category.use_active_view)
        self.assertTrue(category.use_goal_relevance)
        description = resolve_active_topology_policy("category_only", "description")
        self.assertFalse(description.use_active_view)
        self.assertFalse(description.use_goal_relevance)

        full = resolve_active_topology_policy("full", "image")
        self.assertTrue(full.use_active_view)
        self.assertTrue(full.use_goal_relevance)

        belief_category = resolve_active_topology_policy("belief_category", "object")
        self.assertFalse(belief_category.use_active_view)
        self.assertTrue(belief_category.use_goal_relevance)
        self.assertTrue(belief_category.use_belief_planner)
        belief_description = resolve_active_topology_policy(
            "belief_category", "description"
        )
        self.assertFalse(belief_description.use_active_view)
        self.assertFalse(belief_description.use_goal_relevance)
        self.assertFalse(belief_description.use_belief_planner)
        belief_v2 = resolve_active_topology_policy("belief_category_v2", "object")
        self.assertEqual(belief_v2.mode, ActiveTopologyMode.BELIEF_CATEGORY_V2)
        self.assertTrue(belief_v2.use_belief_planner)
        belief_v3 = resolve_active_topology_policy("belief_category_v3", "object")
        self.assertEqual(belief_v3.mode, ActiveTopologyMode.BELIEF_CATEGORY_V3)
        self.assertTrue(belief_v3.use_belief_planner)
        with self.assertRaises(ValueError):
            resolve_active_topology_policy("typo", "category")

    def test_confidence_accessibility_view_has_no_relevance_states(self):
        items = [frontier(10), frontier(30)]
        mapping = self.topology.match_frontiers(
            items, 0, {0: np.array([1.0, 0.0]), 1: np.array([-1.0, 0.0])}
        )
        self.topology.nodes[mapping[0]].map_log_odds = 1.0
        self.topology.nodes[mapping[1]].map_log_odds = -1.0
        goal = GoalContext("g", "category", category="chair", text_embedding=np.array([1.0, 0.0]))
        self.topology.set_goal(goal)
        view = self.topology.build_active_view(
            goal, items, {0: 1.0, 1: 2.0}, 0, use_goal_relevance=False
        )
        self.assertFalse(view.uses_goal_relevance)
        self.assertEqual(
            {candidate.relevance for candidate in view.all_candidates}, {0.5}
        )
        self.assertEqual(
            {candidate.state for candidate in view.all_candidates},
            {ActiveTopoState.ACTIVE},
        )
        self.assertGreater(view.candidates[0].confidence, view.candidates[1].confidence)

    def test_simple_memory_preserves_order_and_hides_recent_failure(self):
        items = [frontier(10), frontier(30), frontier(50)]
        mapping = self.topology.match_frontiers(items, 0)
        goal = GoalContext("g", "description", description="red chair")
        self.topology.set_goal(goal)

        initial = self.topology.build_simple_memory_view(goal, items, 0)
        self.assertEqual(
            [item.source_frontier_index for item in initial.candidates],
            [0, 1, 2],
        )
        self.assertFalse(initial.include_prompt_metadata)

        self.topology.record_navigation_result(mapping[1], False, 1)
        blocked = self.topology.build_simple_memory_view(goal, items, 2)
        self.assertEqual(
            [item.source_frontier_index for item in blocked.candidates],
            [0, 2],
        )
        unblocked = self.topology.build_simple_memory_view(goal, items, 7)
        self.assertEqual(
            [item.source_frontier_index for item in unblocked.candidates],
            [0, 1, 2],
        )

    def test_simple_memory_recovers_one_candidate_when_all_are_blocked(self):
        items = [frontier(10), frontier(30)]
        mapping = self.topology.match_frontiers(items, 0)
        goal = GoalContext("g", "category", category="chair")
        self.topology.record_navigation_result(mapping[0], False, 1)
        self.topology.record_navigation_result(mapping[1], False, 1)
        view = self.topology.build_simple_memory_view(goal, items, 2)
        self.assertEqual(len(view.candidates), 1)
        self.assertEqual(view.recovery_level, 1)
        self.assertEqual(view.candidates[0].source_frontier_index, 0)

    def test_goal_topology_projects_connected_view_without_mutating_global_map(self):
        visited_a = self.topology.upsert_visited([0, 0], 0)
        chair_id = self.topology.upsert_observed(
            SimpleNamespace(image="chair.png"), [0, 0], ["chair", "table"],
            0, visited_a, "snapshot:chair.png", np.array([1.0, 0.0]),
        )
        visited_b = self.topology.upsert_visited([10, 0], 1)
        bed_id = self.topology.upsert_observed(
            SimpleNamespace(image="bed.png"), [10, 0], ["bed"],
            1, visited_b, "snapshot:bed.png", np.array([-1.0, 0.0]),
        )
        items = [frontier(20), frontier(30)]
        self.topology.match_frontiers(
            items, 2,
            {0: np.array([-1.0, 0.0]), 1: np.array([1.0, 0.0])},
        )
        node_ids_before = set(self.topology.nodes)
        edge_ids_before = set(self.topology.edges)

        chair_goal = GoalContext(
            "chair", "category", category="chair",
            text_embedding=np.array([1.0, 0.0]),
        )
        chair_view = self.topology.build_goal_topology_view(chair_goal, items, 2)
        self.assertEqual(chair_view.evidence_node_ids, [chair_id])
        self.assertIn(visited_a, chair_view.connector_node_ids)
        self.assertIn(visited_b, chair_view.included_node_ids)
        self.assertEqual(
            [candidate.source_frontier_index for candidate in chair_view.candidates],
            [1, 0],
        )
        self.assertEqual(len(chair_view.candidates), len(items))
        self.assertFalse(chair_view.include_prompt_metadata)

        bed_goal = GoalContext(
            "bed", "category", category="bed",
            text_embedding=np.array([-1.0, 0.0]),
        )
        bed_view = self.topology.build_goal_topology_view(bed_goal, items, 3)
        self.assertEqual(bed_view.evidence_node_ids, [bed_id])
        self.assertEqual(
            [candidate.source_frontier_index for candidate in bed_view.candidates],
            [0, 1],
        )
        self.assertEqual(set(self.topology.nodes), node_ids_before)
        self.assertEqual(set(self.topology.edges), edge_ids_before)

    def test_goal_topology_does_not_leak_category_into_image_goal(self):
        visited = self.topology.upsert_visited([0, 0], 0)
        chair_id = self.topology.upsert_observed(
            SimpleNamespace(image="chair.png"), [0, 0], ["chair"],
            0, visited, "snapshot:chair.png", np.array([-1.0, 0.0]),
        )
        image_goal = GoalContext(
            "image", "image", category="chair",
            image_embedding=np.array([1.0, 0.0]),
        )
        relevance = self.topology.node_relevance(chair_id, image_goal)
        self.assertEqual(relevance, 0.0)
        view = self.topology.build_goal_topology_view(image_goal, [], 1)
        self.assertEqual(view.topology_context["goal"], "image target")

    def test_goal_task_view_unifies_direct_revisit_and_explore(self):
        visited = self.topology.upsert_visited([0, 0], 0)
        historical_id = self.topology.upsert_observed(
            SimpleNamespace(image="old.png"), [0, 0], ["chair"],
            0, visited, "snapshot:old.png",
        )
        direct_id = self.topology.upsert_observed(
            SimpleNamespace(image="now.png"), [10, 0], ["chair"],
            1, visited, "snapshot:now.png",
        )
        fresh_id = self.topology.upsert_observed(
            SimpleNamespace(image="fresh.png"), [20, 0], ["lamp"],
            1, visited, "snapshot:fresh.png",
        )
        items = [frontier(20)]
        frontier_id = self.topology.match_frontiers(items, 1)[0]
        goal = GoalContext("chair", "category", category="chair")

        view = self.topology.build_goal_task_view(
            goal, items, ["old.png", "fresh.png"], 1,
            direct_snapshot=(direct_id, "now.png", 7),
            current_snapshot_images=["fresh.png"],
        )

        self.assertEqual(
            [action.action_type for action in view.actions],
            [
                GoalTopoActionType.DIRECT,
                GoalTopoActionType.DIRECT,
                GoalTopoActionType.REVISIT,
                GoalTopoActionType.EXPLORE,
            ],
        )
        self.assertEqual(view.direct_actions[0].topo_node_id, direct_id)
        self.assertEqual(
            view.resolve_snapshot("fresh.png").topo_node_id, fresh_id
        )
        self.assertEqual(
            view.resolve_snapshot("fresh.png").action_type,
            GoalTopoActionType.DIRECT,
        )
        self.assertEqual(
            view.resolve_snapshot("old.png", 99).topo_node_id,
            historical_id,
        )
        self.assertEqual(view.resolve_frontier(0).topo_node_id, frontier_id)
        self.assertIs(self.topology.current_task_view, view)

    def test_frontier_stable_identity_and_one_to_one(self):
        image = np.array([1.0, 0.0, 0.0])
        first = [frontier(10)]
        first_map = self.topology.match_frontiers(first, 0, {0: image})
        stable_id = first_map[0]
        second = [frontier(11), frontier(12)]
        # Visual features may be absent; remaining weights are re-normalized.
        second_map = self.topology.match_frontiers(second, 1, {})
        self.assertIn(stable_id, second_map.values())
        self.assertEqual(len(set(second_map.values())), 2)
        # Temporary disappearance keeps the historical node alive.
        self.topology.match_frontiers([], 2)
        self.assertIn(stable_id, self.topology.nodes)

    def test_confidence_accessibility_and_goal_switch(self):
        item = frontier(10)
        node_id = self.topology.match_frontiers([item], 0)[0]
        initial_c = self.topology.nodes[node_id].map_confidence
        self.assertEqual(initial_c, 0.5)
        self.topology.record_navigation_result(node_id, False, 1)
        self.assertEqual(self.topology.nodes[node_id].blocked_until_step, 6)
        self.topology.record_navigation_result(node_id, False, 7)
        self.topology.record_navigation_result(node_id, False, 13)
        self.assertEqual(self.topology.nodes[node_id].consecutive_failures, 3)
        self.topology.record_navigation_result(node_id, True, 14)
        self.assertEqual(self.topology.nodes[node_id].consecutive_failures, 0)
        self.assertEqual(self.topology.nodes[node_id].blocked_until_step, -1)
        c_before, a_before = self.topology.nodes[node_id].map_confidence, self.topology.nodes[node_id].accessibility
        self.topology.set_goal(GoalContext("a", "category", category="chair"))
        self.topology.set_goal(GoalContext("b", "description", description="red chair"))
        self.assertEqual(c_before, self.topology.nodes[node_id].map_confidence)
        self.assertEqual(a_before, self.topology.nodes[node_id].accessibility)

    def test_active_view_utility_recovery_and_mapping(self):
        items = [frontier(10), frontier(30)]
        self.topology.match_frontiers(items, 0, {0: np.array([1, 0]), 1: np.array([-1, 0])})
        goal = GoalContext("g", "category", category="chair", text_embedding=np.array([1, 0]))
        self.topology.set_goal(goal)
        view = self.topology.build_active_view(goal, items, {0: 1.0, 1: 4.0}, 0)
        self.assertTrue(view.candidates)
        self.assertIs(view.resolve_frontier(0, items), items[view.candidates[0].source_frontier_index])
        utilities = [candidate.utility for candidate in view.candidates]
        self.assertEqual(utilities, sorted(utilities, reverse=True))
        self.topology.recovery.level = 2
        recovered = self.topology.build_active_view(goal, items, {0: 1.0, 1: 4.0}, 1, recovery_level=2)
        self.assertGreaterEqual(len(recovered.candidates), len(view.candidates))

    def test_all_states_and_recovery_candidate_scopes(self):
        items = [frontier(10), frontier(30), frontier(50), frontier(70)]
        embeddings = {
            0: np.array([0.225, np.sqrt(1.0 - 0.225 ** 2)]),  # R = 0.50 -> ACTIVE
            1: np.array([1.0, 0.0]),   # R = 1.00 and low C -> VERIFY
            2: np.array([-1.0, 0.0]),  # R = 0.00 -> FOLDED
            3: np.array([0.225, np.sqrt(1.0 - 0.225 ** 2)]),  # failures -> SUPPRESSED
        }
        mapping = self.topology.match_frontiers(items, 0, embeddings)
        self.topology.nodes[mapping[1]].map_log_odds = 0.0
        self.topology.nodes[mapping[2]].map_log_odds = 1.0
        self.topology.nodes[mapping[3]].consecutive_failures = 3
        goal = GoalContext("g", "category", category="chair", text_embedding=np.array([1.0, 0.0]))
        self.topology.set_goal(goal)

        level0 = self.topology.build_active_view(
            goal, items, {index: float(index + 1) for index in range(4)}, 1,
            recovery_level=0,
        )
        states0 = {candidate.source_frontier_index: candidate.state
                   for candidate in level0.all_candidates}
        self.assertEqual(states0, {
            0: ActiveTopoState.ACTIVE,
            1: ActiveTopoState.VERIFY,
            2: ActiveTopoState.FOLDED,
            3: ActiveTopoState.SUPPRESSED,
        })
        self.assertEqual(
            {candidate.source_frontier_index for candidate in level0.candidates},
            {0, 1},
        )

        level1 = self.topology.build_active_view(
            goal, items, {index: float(index + 1) for index in range(4)}, 1,
            recovery_level=1,
        )
        self.assertEqual(
            {candidate.source_frontier_index for candidate in level1.candidates},
            {0, 1, 2},
        )
        self.assertEqual(
            next(candidate.state for candidate in level1.candidates
                 if candidate.source_frontier_index == 2),
            ActiveTopoState.FOLDED,
        )

        self.topology.nodes[mapping[2]].nav_failure_count = 1
        level1_low_accessibility = self.topology.build_active_view(
            goal, items, {index: float(index + 1) for index in range(4)}, 1,
            recovery_level=1,
        )
        self.assertNotIn(
            2,
            {candidate.source_frontier_index for candidate in level1_low_accessibility.candidates},
        )
        level2 = self.topology.build_active_view(
            goal, items, {index: float(index + 1) for index in range(4)}, 1,
            recovery_level=2,
        )
        self.assertEqual(
            {candidate.source_frontier_index for candidate in level2.candidates},
            {0, 1, 2},
        )

    def test_empty_normal_view_automatically_recovers_folded_node(self):
        items = [frontier(10)]
        self.topology.match_frontiers(items, 0, {0: np.array([-1.0, 0.0])})
        self.topology.match_frontiers(items, 1, {0: np.array([-1.0, 0.0])})
        goal = GoalContext("g", "category", category="chair", text_embedding=np.array([1.0, 0.0]))
        self.topology.set_goal(goal)
        view = self.topology.build_active_view(goal, items, {0: 1.0}, 2)
        self.assertEqual(view.recovery_level, 1)
        self.assertEqual(self.topology.recovery.level, 1)
        self.assertEqual(len(view.candidates), 1)
        self.assertEqual(view.candidates[0].state, ActiveTopoState.FOLDED)

    def test_new_relevant_frontier_is_verify_then_stable_match_raises_confidence(self):
        items = [frontier(10)]
        embedding = {0: np.array([1.0, 0.0])}
        node_id = self.topology.match_frontiers(items, 0, embedding)[0]
        goal = GoalContext("g", "category", category="chair", text_embedding=np.array([1.0, 0.0]))
        self.topology.set_goal(goal)
        first = self.topology.build_active_view(goal, items, {0: 1.0}, 0)
        self.assertEqual(first.all_candidates[0].state, ActiveTopoState.VERIFY)
        self.assertEqual(first.all_candidates[0].confidence, 0.5)

        self.assertEqual(self.topology.match_frontiers(items, 1, embedding)[0], node_id)
        second = self.topology.build_active_view(goal, items, {0: 1.0}, 1)
        self.assertEqual(second.all_candidates[0].state, ActiveTopoState.ACTIVE)
        self.assertGreater(second.all_candidates[0].confidence, 0.55)

    def test_geodesic_progress_updates_accessibility_without_consuming_node(self):
        item = frontier(10)
        node_id = self.topology.match_frontiers([item], 0)[0]
        self.assertFalse(self.topology.record_selection(node_id, 5.0, 0))
        self.assertTrue(self.topology.record_selection(node_id, 4.8, 1))
        node = self.topology.nodes[node_id]
        self.assertGreater(node.accessibility, 0.5)
        self.assertFalse(node.consumed)
        self.assertEqual(self.topology.stats["navigation_progresses"], 1)

    def test_missing_relevance_features_remain_unknown_not_folded(self):
        items = [frontier(10)]
        self.topology.match_frontiers(items, 0)
        goal = GoalContext("g", "description", description="red chair")
        self.topology.set_goal(goal)
        view = self.topology.build_active_view(goal, items, {0: 1.0}, 0)
        self.assertEqual(view.all_candidates[0].relevance, 0.5)
        self.assertEqual(view.all_candidates[0].state, ActiveTopoState.ACTIVE)

    def test_soft_verification_propagates_without_early_prune(self):
        node_id = self.topology.match_frontiers([frontier(10)], 0)[0]
        self.topology.bind_hypothesis(node_id, "hg0", 0.8)
        child_id = "hypothesis:hg0"
        before = self.topology.nodes[child_id].map_confidence
        evidence = VerificationEvidence(node_id, False, 1.0, 0.9, "mismatch", "neg:1")
        pruned = self.topology.apply_verification(node_id, evidence, 1, "soft")
        self.assertLess(self.topology.nodes[child_id].map_confidence, before)
        self.assertEqual(pruned, [])

    def test_negative_evidence_is_deduplicated_and_prunes_only_after_thresholds(self):
        node_id = self.topology.match_frontiers([frontier(10)], 0)[0]
        first = VerificationEvidence(node_id, False, 1.0, 0.9, "bad", "negative:a")
        self.topology.apply_verification(node_id, first, 6, "soft")
        after_first = self.topology.nodes[node_id].map_log_odds
        self.topology.apply_verification(node_id, first, 6, "soft")
        self.assertEqual(after_first, self.topology.nodes[node_id].map_log_odds)
        second = VerificationEvidence(node_id, False, 1.0, 0.9, "bad", "negative:b")
        self.assertIn(node_id, self.topology.apply_verification(node_id, second, 7, "soft"))
        self.assertNotIn(node_id, self.topology.nodes)

    def test_recovery_budget_escalates_and_progress_resets(self):
        for _ in range(11):
            self.assertFalse(self.topology.recovery.record_step(False, 12))
        self.assertTrue(self.topology.recovery.record_step(False, 12))
        self.assertEqual(self.topology.recovery.level, 1)
        self.topology.recovery.record_progress()
        self.assertEqual(self.topology.recovery.level, 0)
        self.assertEqual(self.topology.recovery.no_progress_steps, 0)

    def test_trace_is_jsonl_and_summary_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = ActiveTopologyTraceWriter(directory, "scene/episode")
            writer.write({"event": "step", "position": np.array([1, 2])})
            writer.save_summary({"ok": True})
            with open(writer.trace_path, encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.readline())["event"], "step")
            self.assertTrue(os.path.exists(writer.summary_path))

    def test_two_subtasks_share_episode_facts_and_new_episode_clears(self):
        item = frontier(10)
        stable_id = self.topology.match_frontiers([item], 0)[0]
        self.topology.record_navigation_result(stable_id, False, 1)
        accessibility = self.topology.nodes[stable_id].accessibility
        self.topology.set_goal(GoalContext("subtask-0", "category", category="chair"))
        self.topology.set_goal(GoalContext("subtask-1", "category", category="bed"))
        self.assertIn(stable_id, self.topology.nodes)
        self.assertEqual(accessibility, self.topology.nodes[stable_id].accessibility)
        next_episode = PersistentBeliefTopology(CFG, voxel_size=0.1)
        self.assertNotIn(stable_id, next_episode.nodes)

    def test_safe_snapshot_route_uses_only_visited_waypoints(self):
        visited_a = self.topology.upsert_visited([0, 0], 0)
        observed = self.topology.upsert_observed(
            SimpleNamespace(image="history.png"),
            [0, 0],
            ["chair"],
            0,
            visited_a,
            "snapshot:history.png",
        )
        visited_b = self.topology.upsert_visited([10, 0], 1)
        visited_c = self.topology.upsert_visited([20, 0], 2)
        # A historical frontier exists in the fact graph, but it must never
        # become an executable return waypoint.
        frontier_id = self.topology.match_frontiers([frontier(15)], 2)[0]

        self.assertEqual(
            self.topology.observed_node_for_snapshot("history.png"), observed
        )
        route, waypoint = self.topology.plan_safe_route_to_observed(observed)
        self.assertEqual(route, [visited_c, visited_b, visited_a, observed])
        self.assertEqual(waypoint, visited_b)
        self.assertAlmostEqual(self.topology.route_length_m(route), 2.0)
        self.assertNotIn(frontier_id, route)
        self.assertTrue(all(
            self.topology.nodes[node_id].node_type.value == "visited"
            for node_id in route[:-1]
        ))

    def test_snapshot_route_at_current_anchor_needs_no_intermediate_waypoint(self):
        visited = self.topology.upsert_visited([0, 0], 0)
        observed = self.topology.upsert_observed(
            SimpleNamespace(image="current.png"),
            [0, 0],
            ["table"],
            0,
            visited,
            "snapshot:current.png",
        )
        route, waypoint = self.topology.plan_safe_route_to_observed(observed)
        self.assertEqual(route, [visited, observed])
        self.assertIsNone(waypoint)
        self.assertIsNone(
            self.topology.observed_node_for_snapshot("missing.png")
        )
        self.assertEqual(
            self.topology.plan_safe_route_to_observed("observed:missing"),
            ([], None),
        )

    def test_goal_topology_keeps_snapshot_origin_anchor_on_rescan(self):
        origin = self.topology.upsert_visited([0, 0], 0)
        snapshot = SimpleNamespace(image="fixed-origin.png")
        observed = self.topology.upsert_observed(
            snapshot,
            [0, 0],
            ["chair"],
            0,
            origin,
            "snapshot:fixed-origin.png",
            reanchor_existing_evidence=False,
        )
        current = self.topology.upsert_visited([10, 0], 1)
        self.assertEqual(
            self.topology.upsert_observed(
                snapshot,
                [0, 0],
                ["chair"],
                1,
                current,
                "snapshot:fixed-origin.png",
                reanchor_existing_evidence=False,
            ),
            observed,
        )

        route, waypoint = self.topology.plan_safe_route_to_observed(observed)
        self.assertEqual(route, [current, origin, observed])
        self.assertEqual(waypoint, origin)
        self.assertEqual(self.topology.stats["observed_reanchor_skips"], 1)
        self.assertEqual(
            self.topology.capture_anchor_for_snapshot("fixed-origin.png"), origin
        )

    def test_legacy_modes_can_still_reanchor_existing_snapshot(self):
        origin = self.topology.upsert_visited([0, 0], 0)
        snapshot = SimpleNamespace(image="legacy.png")
        observed = self.topology.upsert_observed(
            snapshot, [0, 0], ["chair"], 0, origin, "snapshot:legacy.png"
        )
        current = self.topology.upsert_visited([10, 0], 1)
        self.topology.upsert_observed(
            snapshot, [0, 0], ["chair"], 1, current, "snapshot:legacy.png"
        )
        route, waypoint = self.topology.plan_safe_route_to_observed(observed)
        self.assertEqual(route, [current, observed])
        self.assertIsNone(waypoint)

    def test_frontier_semantic_cache_hit_and_invalidation(self):
        embedding = np.array([1.0, 0.0])
        item = frontier(10, embedding=embedding)
        stable_id = self.topology.match_frontiers(
            [item], 0, visual_embeddings={0: embedding}
        )[0]
        semantic_dist = SimpleNamespace(
            categories=["kitchen"], probabilities=np.array([1.0])
        )
        self.assertFalse(self.topology.store_frontier_semantic_cache(
            stable_id, semantic_dist, embedding, item.position, 0, "heuristic"
        ))
        self.assertTrue(self.topology.store_frontier_semantic_cache(
            stable_id, semantic_dist, embedding, item.position, 0, "vlm"
        ))
        cached, reason = self.topology.lookup_frontier_semantic_cache(
            stable_id, np.array([0.99, 0.01]), np.array([10.5, 0.0])
        )
        self.assertEqual(reason, "hit")
        self.assertEqual(cached.categories, ["kitchen"])
        cached.categories[0] = "mutated"
        cached_again, _ = self.topology.lookup_frontier_semantic_cache(
            stable_id, embedding, item.position
        )
        self.assertEqual(cached_again.categories, ["kitchen"])
        self.assertEqual(self.topology.lookup_frontier_semantic_cache(
            stable_id, np.array([-1.0, 0.0]), item.position
        )[1], "visual")
        self.assertEqual(self.topology.lookup_frontier_semantic_cache(
            stable_id, embedding, np.array([30.0, 0.0])
        )[1], "position")


if __name__ == "__main__":
    unittest.main()
