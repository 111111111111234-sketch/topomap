import inspect
import unittest
from types import SimpleNamespace

from src.hypothesis_graph import NodeType
from src.tsdf_planner import TSDFPlanner


class DeferredHypothesisPredictionTest(unittest.TestCase):
    def test_update_frontier_default_preserves_original_eager_behavior(self):
        parameter = inspect.signature(
            TSDFPlanner.update_frontier_map
        ).parameters["defer_hypothesis_prediction"]
        self.assertIs(parameter.default, False)

    def test_eager_cleanup_preserves_pinned_hypothesis_node(self):
        planner = TSDFPlanner.__new__(TSDFPlanner)
        old_frontier = SimpleNamespace(frontier_id=1)
        current_frontier = SimpleNamespace(
            frontier_id=2, hypothesis_node_id="current", semantic_dist=None
        )
        planner.frontiers = [current_frontier]
        nodes = {
            "pinned": SimpleNamespace(
                node_type=NodeType.HYPOTHESIS, associated_object=old_frontier
            ),
            "current": SimpleNamespace(
                node_type=NodeType.HYPOTHESIS,
                associated_object=current_frontier,
            ),
        }
        planner.hypothesis_graph = SimpleNamespace(
            nodes=nodes, _remove_node=lambda node_id: nodes.pop(node_id)
        )

        planner._predict_hypothesis_node_semantics(
            scene=SimpleNamespace(),
            pts_habitat=None,
            preserve_node_ids={"pinned"},
        )

        self.assertIn("pinned", nodes)
        self.assertIn("current", nodes)

    def test_cached_frontier_rebinds_and_only_miss_is_predicted(self):
        planner = TSDFPlanner.__new__(TSDFPlanner)
        cached = SimpleNamespace(frontier_id=1, hypothesis_node_id=None)
        missing = SimpleNamespace(frontier_id=2, hypothesis_node_id=None)
        planner.frontiers = [cached, missing]
        planner.hypothesis_graph = SimpleNamespace(nodes={}, _remove_node=lambda node_id: None)
        planner.observation_history = []
        planner._build_spatial_context = lambda scene, pts: {}
        predicted_dist = SimpleNamespace(name="predicted")

        class Predictor:
            last_prediction_sources = {2: "vlm"}

            def batch_predict_frontiers(self, frontiers, spatial_context, observed_history):
                self.frontier_ids = [item.frontier_id for item in frontiers]
                return {2: predicted_dist}

        predictor = Predictor()
        planner.hypothesis_node_predictor = predictor
        attached = []

        def attach(frontier, semantic_dist):
            frontier.hypothesis_node_id = f"hypothesis-{frontier.frontier_id}"
            planner.hypothesis_graph.nodes[frontier.hypothesis_node_id] = object()
            attached.append((frontier.frontier_id, semantic_dist))

        planner._attach_hypothesis_node = attach
        cached_dist = SimpleNamespace(name="cached")
        sources = planner._predict_hypothesis_node_semantics(
            scene=SimpleNamespace(), pts_habitat=None,
            cached_semantics={1: cached_dist},
        )
        self.assertEqual(predictor.frontier_ids, [2])
        self.assertEqual([item[0] for item in attached], [1, 2])
        self.assertEqual(sources, {1: "cache", 2: "vlm"})


if __name__ == "__main__":
    unittest.main()
