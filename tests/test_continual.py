"""Comprehensive unit tests for continual learning models."""
import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.continual import ContinualLearner, MLP
from data.mnist_tasks import make_tasks


class TestContinualModels(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 16
        self.x = torch.randn(self.batch_size, 1, 28, 28)
        self.y = torch.randint(0, 10, (self.batch_size,))
        self.dataset = TensorDataset(self.x, self.y)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size)

    def test_mlp_forward(self):
        mlp = MLP(hidden=64)
        logits, features = mlp(self.x)
        self.assertEqual(logits.shape, (self.batch_size, 10))
        self.assertEqual(features.shape, (self.batch_size, 128))

    def test_methods_initialization(self):
        methods = ["baseline", "ewc", "si", "replay", "nsm"]
        for m in methods:
            learner = ContinualLearner(method=m, device=self.device, hidden=64)
            self.assertEqual(learner.method, m)
            logits, _ = learner.model(self.x)
            self.assertEqual(logits.shape, (self.batch_size, 10))

    def test_ewc_accumulation_and_penalty(self):
        learner = ContinualLearner(method="ewc", device=self.device, hidden=64)
        learner.ewc_accum = {n: torch.zeros_like(p) for n, p in learner.model.named_parameters()}
        learner.ewc_n = 1

        # Simulate gradients
        logits, _ = learner.model(self.x)
        loss = F.cross_entropy(logits, self.y)
        loss.backward()
        for n, p in learner.model.named_parameters():
            if p.grad is not None:
                learner.ewc_accum[n] += p.grad.detach() ** 2

        learner.consolidate_ewc_from_accum()
        self.assertEqual(len(learner.ewc_params), 1)
        self.assertEqual(len(learner.ewc_fisher), 1)

        # Penalty before moving params should be 0
        pen = learner.ewc_penalty()
        self.assertAlmostEqual(pen.item(), 0.0, places=4)

    def test_si_lifecycle(self):
        learner = ContinualLearner(method="si", device=self.device, hidden=64)
        learner.si_init()
        self.assertEqual(len(learner.si_prev_params), len(list(learner.model.named_parameters())))

        # Forward + backward + update
        logits, _ = learner.model(self.x)
        loss = F.cross_entropy(logits, self.y)
        loss.backward()
        learner.si_update(lr=0.01)

        learner.si_consolidate()
        self.assertEqual(len(learner.si_omega), len(list(learner.model.named_parameters())))
        pen = learner.si_penalty()
        self.assertGreaterEqual(pen.item(), 0.0)

    def test_replay_buffer(self):
        learner = ContinualLearner(method="replay", device=self.device, hidden=64, replay_size=100)
        learner.add_replay(self.x, self.y)
        self.assertEqual(len(learner.replay_buffer), self.batch_size)

        rx, ry = learner.get_replay_batch(8)
        self.assertEqual(rx.shape, (8, 1, 28, 28))
        self.assertEqual(ry.shape, (8,))

    def test_nsm_penalty(self):
        learner = ContinualLearner(method="nsm", device=self.device, hidden=64)
        learner.ewc_accum = {n: torch.zeros_like(p) for n, p in learner.model.named_parameters()}
        learner.ewc_n = 1
        learner.consolidate_ewc_from_accum()
        learner.si_init()
        learner.si_consolidate()
        learner.update_synaptic_gate()

        self.assertTrue(len(learner.synaptic_gate) > 0)
        pen = learner.nsm_penalty()
        self.assertGreaterEqual(pen.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
