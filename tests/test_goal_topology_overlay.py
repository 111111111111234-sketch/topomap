import unittest
from types import SimpleNamespace

from src.goal_topology_overlay import GoalTopologyOverlay
from src.place_goal_navigation import GoalCandidate
from src.place_topology import PlaceTopology


def action(place_id="place_0", entity="object:1"):
    return GoalCandidate(
        f"verify:1@{place_id}", "Verify", entity, "old.png", 1, None,
        place_id, "snapshot:old.png", [1., 2.], [2., 2.], [3., 2.],
        [place_id], 0., 0., 1., evidence_place_id=place_id,
    )


class GoalTopologyOverlayTest(unittest.TestCase):
    def test_state_is_on_places_and_decisions_require_revisions(self):
        graph = PlaceTopology(2., 1., stage="route_only")
        place_id = graph.observe_pose([1, 2], 0, "obs").place_id
        overlay = GoalTopologyOverlay("goal")
        candidate = action(place_id)
        overlay.project(graph, [candidate], {})
        self.assertEqual(overlay.places[place_id].search_state, "evidence_available")
        self.assertTrue(overlay.decision_required())
        overlay.mark_decision()
        revision = overlay.revision
        overlay.project(graph, [candidate], {})
        self.assertEqual(overlay.revision, revision)
        self.assertFalse(overlay.decision_required())
        overlay.mark_checked(candidate, "verification_uncertain", 3)
        overlay.project(graph, [], {})
        self.assertTrue(overlay.decision_required())
        self.assertEqual(
            overlay.places[place_id].search_state,
            "exhausted_for_current_topology",
        )

    def test_entity_rebind_moves_checked_topological_action(self):
        overlay = GoalTopologyOverlay("goal")
        candidate = action(entity="object:1")
        overlay.mark_checked(candidate, "verification_rejected", 1)
        overlay.rebind_entity("object:1", "object:2")
        self.assertFalse(overlay.is_checked("Verify", "object:1", "place_0"))
        self.assertTrue(overlay.is_checked("Verify", "object:2", "place_0"))


if __name__ == "__main__":
    unittest.main()
