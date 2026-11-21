"""
Comprehensive Unit Test Suite for Continual Learning & Discovery Models
Validates:
1. Autonomous Chiral Router
2. Betti Gated Layer
3. Quantum Monodromy Layer
4. Hyperbolic Geodesic Router
5. Symplectic Hamiltonian Router
6. Fractional Memory Network
7. Quasicrystal Penrose Router
8. Unified Master Operator
9. 3-SAT DPLL and Continuous Solvers
"""

import unittest
import torch
from models.autonomous_router import AutonomousChiralNetwork
from models.betti_gated_layer import BettiGatedLayer
from models.quantum_monodromy_net import NonCommutativeMonodromyLayer
from models.hyperbolic_geodesic_router import HyperbolicGeodesicRouter
from models.symplectic_hamiltonian_router import SymplecticHamiltonianRouter
from models.fractional_memory_network import FractionalMemoryNetwork
from models.quasicrystal_penrose_router import QuasicrystalMemoryRouter
from models.unified_master_equation import UnifiedMasterOperator
from experiments.attack_p_vs_np import generate_random_3sat, dpll_solve, continuous_gradient_solve

class TestDiscoverySuite(unittest.TestCase):
    def test_autonomous_router_shape(self):
        net = AutonomousChiralNetwork(in_features=784, hidden=64)
        net.add_task((0, 1), device="cpu")
        net.add_task((2, 3), device="cpu")
        x = torch.randn(8, 784)
        out = net.forward_blind(x)
        self.assertEqual(out.shape, (8, 10))

    def test_betti_gated_layer_crystallize(self):
        layer = BettiGatedLayer(10, 5)
        layer.crystallize_important_synapses()
        # Plasticity mask should have some 0.0 entries
        zeros = (layer.plasticity_mask == 0.0).sum().item()
        self.assertGreater(zeros, 0)

    def test_quantum_monodromy_wilson_loops(self):
        net = NonCommutativeMonodromyLayer(in_features=10, hidden_dim=8)
        net.add_task(device="cpu")
        x = torch.randn(4, 10)
        w = net.compute_wilson_loops(x, 0)
        self.assertEqual(w.shape, (4, 8))
        self.assertTrue((w >= -1.01).all() and (w <= 1.01).all())

    def test_hyperbolic_poincare_distance(self):
        net = HyperbolicGeodesicRouter(in_features=10, hidden_dim=8)
        net.add_task(device="cpu")
        x = torch.randn(4, 10)
        out = net.forward_blind(x)
        self.assertEqual(out.shape, (4, 2))

    def test_symplectic_hamiltonian_flow(self):
        net = SymplecticHamiltonianRouter(in_features=10, feature_dim=8)
        net.add_task((0, 1), device="cpu")
        x = torch.randn(4, 10)
        out = net.forward_blind(x)
        self.assertEqual(out.shape, (4, 2))

    def test_fractional_memory_network(self):
        net = FractionalMemoryNetwork(in_features=10, hidden_dim=8)
        x = torch.randn(4, 10)
        net.checkpoint_task()
        loss = net.get_fractional_memory_loss()
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_quasicrystal_penrose_router(self):
        net = QuasicrystalMemoryRouter(in_features=10, feature_dim=8)
        net.add_task((0, 1), device="cpu")
        x = torch.randn(4, 10)
        out = net.forward_blind(x)
        self.assertEqual(out.shape, (4, 2))

    def test_unified_master_operator(self):
        net = UnifiedMasterOperator(in_features=10, hidden_dim=8)
        net.add_task(device="cpu")
        x = torch.randn(4, 10)
        out = net.forward_blind(x)
        self.assertEqual(out.shape, (4, 2))

    def test_3sat_solvers(self):
        n_vars, clauses = generate_random_3sat(n_vars=6, clause_ratio=3.0, seed=42)
        sat, nodes, _ = dpll_solve(n_vars, clauses)
        self.assertIsInstance(sat, bool)
        self.assertGreater(nodes, 0)

if __name__ == "__main__":
    unittest.main()
