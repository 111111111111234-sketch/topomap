import unittest

import numpy as np
import torch

from src.hypothesis_graph import (
    CognitiveDependency,
    HypothesisGraph,
    HypothesisNode,
    NodeType,
    SemanticDistribution,
)
from src.semantic_critic import SemanticCritic


def build_graph():
    cfg = {
        "semantic_residual_threshold": 0.5,
        "enable_vlm_causality_diagnosis": False,
        "enable_cascade_deletion": True,
    }
    graph = HypothesisGraph(cfg)
    distribution = SemanticDistribution(["bedroom"], np.array([1.0]))
    root = HypothesisNode("root", NodeType.HYPOTHESIS, semantic_dist=distribution)
    child = HypothesisNode("child", NodeType.HYPOTHESIS, semantic_dist=distribution)
    graph.add_node(root)
    graph.add_node(child)
    graph.add_dependency(CognitiveDependency("root", "child", "test", 0.8, "test"))
    return cfg, graph, root


class SemanticCriticModesTest(unittest.TestCase):
    def test_feature_residual_moves_inputs_to_clip_device_and_dtype(self):
        class StrictClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))

            def encode_image(self, tensor):
                self.last_device = tensor.device
                self.last_dtype = tensor.dtype
                return torch.tensor([[1.0, 0.0]], device=tensor.device, dtype=tensor.dtype)

        cfg, graph, _ = build_graph()
        critic = SemanticCritic(cfg, graph)
        model = StrictClip()
        critic.set_clip_model(
            model, lambda _image: torch.ones((3, 2, 2), dtype=torch.float32)
        )
        residual = critic._compute_feature_residual(
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
        self.assertEqual(model.last_device, model.anchor.device)
        self.assertEqual(model.last_dtype, model.anchor.dtype)
        self.assertAlmostEqual(residual, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_feature_residual_accepts_cpu_evidence_with_cuda_clip(self):
        class TinyClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))

            def encode_image(self, tensor):
                return torch.stack((tensor.mean(), tensor.sum())).unsqueeze(0)

        cfg, graph, _ = build_graph()
        critic = SemanticCritic(cfg, graph)
        model = TinyClip().cuda()
        critic.set_clip_model(
            model, lambda _image: torch.ones((3, 2, 2), dtype=torch.float32)
        )
        residual = critic._compute_feature_residual(
            torch.tensor([1.0, 2.0]),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
        self.assertTrue(np.isfinite(residual))
        self.assertEqual(critic.stats["feature_residual_errors"], 0)

    def test_feature_residual_fallback_is_counted(self):
        class BrokenClip(torch.nn.Module):
            def encode_image(self, _tensor):
                raise RuntimeError("broken test encoder")

        cfg, graph, _ = build_graph()
        critic = SemanticCritic(cfg, graph)
        critic.set_clip_model(
            BrokenClip(), lambda _image: torch.ones((3, 2, 2), dtype=torch.float32)
        )
        self.assertEqual(critic._compute_feature_residual(
            torch.tensor([1.0, 0.0]), np.zeros((2, 2, 3), dtype=np.uint8)
        ), 0.5)
        self.assertEqual(critic.stats["feature_residual_errors"], 1)

    def test_feature_residual_accepts_numpy_evidence(self):
        class TinyClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))

            def encode_image(self, _tensor):
                return torch.tensor([[1.0, 0.0]])

        critic = SemanticCritic({}, HypothesisGraph({}))
        model = TinyClip()
        critic.set_clip_model(model, lambda _: torch.ones(3, 2, 2))
        residual = critic._compute_feature_residual(
            np.array([1.0, 0.0], dtype=np.float32),
            np.zeros((4, 4, 3), dtype=np.uint8),
        )
        self.assertAlmostEqual(residual, 0.0)
        self.assertEqual(critic.stats["feature_residual_errors"], 0)

    def test_feature_residual_encodes_historical_rgb_evidence(self):
        class TinyClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1))

            def encode_image(self, tensor):
                return torch.tensor([[1.0, 0.0]], device=tensor.device)

        critic = SemanticCritic({}, HypothesisGraph({}))
        critic.set_clip_model(TinyClip(), lambda _: torch.ones(3, 2, 2))
        residual = critic._compute_feature_residual(
            np.zeros((360, 360, 3), dtype=np.uint8),
            np.zeros((360, 360, 3), dtype=np.uint8),
        )
        self.assertAlmostEqual(residual, 0.0)
        self.assertEqual(critic.stats["feature_residual_errors"], 0)

    def test_evidence_only_has_no_graph_side_effect(self):
        cfg, graph, root = build_graph()
        critic = SemanticCritic(cfg, graph)
        report = critic.verify_hypothesis_node_arrival(
            root,
            {"semantic_class": "bathroom", "detected_objects": ["toilet"]},
            {},
            apply_graph_update=False,
        )
        self.assertTrue(report.is_falsified)
        self.assertIn("root", graph.nodes)
        self.assertIn("child", graph.nodes)
        self.assertEqual(graph.nodes["root"].node_type, NodeType.HYPOTHESIS)

    def test_default_hard_mode_preserves_cascade(self):
        cfg, graph, root = build_graph()
        critic = SemanticCritic(cfg, graph)
        report = critic.verify_hypothesis_node_arrival(
            root,
            {"semantic_class": "bathroom", "detected_objects": ["toilet"]},
            {},
        )
        self.assertTrue(report.is_falsified)
        self.assertIn("root", graph.nodes)
        self.assertNotIn("child", graph.nodes)
        self.assertEqual(graph.nodes["root"].node_type, NodeType.FALSIFIED)


if __name__ == "__main__":
    unittest.main()
