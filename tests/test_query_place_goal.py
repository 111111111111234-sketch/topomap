import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from src.place_goal_navigation import PlaceGoalNavigation
from src.query_place_goal import (
    parse_goal_response, select_place_goal,
    verify_place_goal,
)
from tests.test_place_goal_navigation import candidate


class QueryPlaceGoalTest(unittest.TestCase):
    def test_old_view_id_is_not_executable_and_retry_corrects_with_current_alias(self):
        state = PlaceGoalNavigation("g", .1, evidence_coverage=True)
        item = candidate()
        item.candidate_id = "verify:40@place_0/view:2"
        state.candidates = [item]
        old = "verify:40@place_0/view:3"
        state.feedback = [{"candidate_id": old, "entity_id": "object:40",
                           "outcome": "verification_rejected"}]
        cfg = {"place_goal_navigation": {"evidence_coverage": True}}
        with patch("src.query_place_goal.candidate_visual", return_value=(None, [], "crop")), patch(
            "src.query_place_goal._goal_content", return_value=[]
        ), patch("src.query_place_goal._request", side_effect=[
            json.dumps({"candidate_id": old, "support": .99}),
            '{"candidate_id":"choice_0","support":0.8}'
        ]) as request:
            selected, diag = select_place_goal(state, {}, None, None, [], cfg)
            self.assertEqual(selected[0].candidate_id, item.candidate_id)
            self.assertEqual(diag["request_attempts"], 2)
            initial = str(request.call_args_list[0].args[1])
            self.assertNotIn(old, initial)
            self.assertNotIn(item.candidate_id, initial)
            self.assertIn("verification_rejected", initial)
            self.assertIn("choice_0", initial)
            correction = request.call_args_list[1].args[1][-1][0]
            self.assertIn("failed validation", correction)
            self.assertIn("choice_0", correction)
            self.assertEqual(state.feedback[0]["candidate_id"], old)

    def test_unhashable_candidate_id_is_invalid_output(self):
        self.assertIsNone(parse_goal_response('{"candidate_id":[],"support":0.9}', {"v"}))

    def test_all_candidates_are_present_but_unchanged_topology_is_not_requeried(self):
        verifies = [candidate(f"object:{i}") for i in range(20)]
        for i, item in enumerate(verifies):
            item.candidate_id = f"verify:{i}"
            item.graph_cost_m = float(i)
        explores = [candidate(f"frontier_{i}") for i in range(5)]
        for i, item in enumerate(explores):
            item.candidate_id = f"explore:frontier_{i}"
            item.intent = "Explore"
            item.graph_cost_m = float(i)
        state = PlaceGoalNavigation("g", .1, topological_actions=True,
                                    event_driven_selection=True)
        state.candidates = verifies + explores
        with patch("src.query_place_goal.candidate_visual", return_value=(None, [], None)), patch(
            "src.query_place_goal._goal_content", return_value=[]
        ), patch("src.query_place_goal._request", return_value='{"candidate_id":"verify:19","support":0.8}') as request:
            selected, diagnostic = select_place_goal(state, {}, None, None, [], {})
            self.assertEqual(selected[0].candidate_id, "verify:19")
            self.assertEqual(diagnostic["presented_ids"], [c.candidate_id for c in state.candidates])
            selected, diagnostic = select_place_goal(state, {}, None, None, [], {})
            self.assertIsNone(selected)
            self.assertEqual(diagnostic["reason"], "no_goal_topology_change")
            self.assertEqual(request.call_count, 1)

    def test_parser_never_accepts_missing_id_or_nonfinite_support(self):
        for text in ['{"candidate_id":"ghost","support":0.9}',
                     '{"candidate_id":"verify:1","support":NaN}',
                     '{"candidate_id":"verify:1","support":true}',
                     '{"candidate_id":"verify:1","support":1.1}',
                     'matched', '{"verdict":"yes"}']:
            self.assertIsNone(parse_goal_response(text, {"verify:1"}))
        self.assertIsNotNone(parse_goal_response('```json\n{"candidate_id":"verify:1","support":0.8}\n```', {"verify:1"}))

    def test_all_goal_types_share_selection_and_current_view_verification(self):
        for goal_type in ("object", "description", "image"):
            cfg = OmegaConf.create({"vlm_model": "same-model"})
            state = PlaceGoalNavigation(goal_type, .1)
            state.candidates = [candidate()]
            metadata = {"task_type": goal_type, "question": "find the target"}
            with patch("src.query_place_goal._goal_content", side_effect=lambda _: [("same goal adapter",)]), patch(
                "src.query_place_goal.candidate_visual", return_value=(SimpleNamespace(cluster=[1]), [], "historical-crop")
            ), patch("src.query_place_goal._request", side_effect=[
                '{"candidate_id":"verify:1@place_1","support":0.8}', '{"verdict":"matched","reason":"visible now"}'
            ]) as request:
                selected, diagnostic = select_place_goal(state, metadata, None, None, [], cfg)
                self.assertEqual(selected[0].intent, "Verify")
                self.assertEqual(diagnostic["presented_ids"], ["verify:1@place_1"])
                verdict, _ = verify_place_goal(metadata, np.zeros((16, 16, 3), np.uint8), selected[2], cfg)
                self.assertEqual(verdict, "matched")
                self.assertEqual(request.call_args_list[0].args[-1], "place_goal_select")
                self.assertEqual(request.call_args_list[1].args[-1], "place_goal_verify")
                content = request.call_args_list[1].args[1]
                self.assertEqual(content[1][1], "historical-crop")
                self.assertNotEqual(content[2][1], "historical-crop")

    def test_model_failure_cannot_confirm_a_historical_target(self):
        cfg = OmegaConf.create({"vlm_model": "same-model"})
        with patch("src.query_place_goal._goal_content", return_value=[]), patch(
            "src.query_place_goal._request", return_value=None
        ):
            verdict, diagnostic = verify_place_goal({}, np.zeros((16, 16, 3), np.uint8), "crop", cfg)
            self.assertEqual(verdict, "uncertain")
            self.assertIn("failed", diagnostic["reason"])
        self.assertEqual(verify_place_goal({}, None, None, cfg)[0], "uncertain")

    def test_missing_source_is_reported_and_never_presented(self):
        state = PlaceGoalNavigation("g", .1)
        state.candidates = [candidate()]
        with patch("src.query_place_goal.candidate_visual", return_value=None), patch(
            "src.query_place_goal._request"
        ) as request:
            result, diagnostic = select_place_goal(state, {}, None, None, [], {})
            self.assertIsNone(result)
            self.assertEqual(diagnostic["reason"], "no_executable_visual_candidate")
            self.assertEqual(len(state.exclusions), 1)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
