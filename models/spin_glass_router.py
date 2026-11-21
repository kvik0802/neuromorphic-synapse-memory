"""
Spin Glass Frustration Router (SGFR)
====================================
Solving Zero-Replay Class-IL via Edwards-Anderson Spin Glass Ground States.

Mathematical Foundations:
Let each task k define a disordered Ising spin glass system of N binary spin variables:
    s_i in {-1, +1},  i = 1, ..., N

The Hamiltonian of task k is governed by the Edwards-Anderson disordered spin glass:
    H_k(s; x) = - sum_{i < j} J_{ij}^{(k)} s_i s_j - sum_i h_i(x) s_i
where:
  J_{ij}^{(k)} is a symmetric quenched coupling matrix defining the frustrated energy landscape
  h(x) in R^N is the magnetic field induced by the continuous input representation x = f(w, x)

Phase Transitions & Ground State Energy:
In a spin glass, the interaction couplings J_{ij} contain geometric frustration:
    J_{ij} J_{jk} J_{ki} < 0 (frustrated plaquette)
Frustration creates an exponentially large number of metastable valleys (Parisi Ultrametricity).

When an input x belongs to task k:
The external field h(x) is aligned with the ferromagnetic/spin-glass valley of J^{(k)}, 
rapidly driving the system into a deep, low-temperature Edwards-Anderson ground state:
    E_0^{(k)}(x) = min_{s} H_k(s; x) << 0

When an out-of-distribution input x' arrives from another task:
The field h(x') strongly conflicts with the quenched couplings J^{(k)}, creating massive 
frustration and pinning the spins in a high-energy excited state:
    E_0^{(k)}(x') >> E_0^{(k)}(x)

Autonomous Blind Class-IL Routing (Zero Task IDs, Zero Replay):
Given a blind sample x without task ID:
1. Initialize continuous relaxed spins m in [-1, +1]^N via mean-field Thouless-Anderson-Palmer (TAP) dynamics:
       m_i = tanh( beta * ( sum_j J_{ij}^{(k)} m_j + h_i(x) - Onsager_reaction_term ) )
2. Measure the ground state free energy:
       F_k(x) = H_k(m; x)
3. Route strictly to the task that achieves the deepest spin glass ground state:
       k* = argmin_k F_k(x)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpinGlassModule(nn.Module):
    def __init__(self, spin_dim: int, active_classes: tuple[int, int]):
        super().__init__()
        self.spin_dim = spin_dim
        self.active_classes = active_classes
        
        # Quenched disordered coupling matrix J (symmetric, zero diagonal)
        J_raw = torch.randn(spin_dim, spin_dim) / math.sqrt(spin_dim)
        J_sym = 0.5 * (J_raw + J_raw.t())
        J_sym.fill_diagonal_(0.0)
        self.J = nn.Parameter(J_sym)
        
        # Task classification head from mean-field spin configurations
        self.head = nn.Linear(spin_dim, 2)

    def relax_tap_spins(self, h: torch.Tensor, steps: int = 6, beta: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean-field Thouless-Anderson-Palmer (TAP) relaxation to find the ground state spins:
        m_i = tanh( beta * ( J m + h ) )
        """
        batch_size = h.size(0)
        m = torch.tanh(h) # Initialize spins aligned with local magnetic field
        
        # Symmetric zero-diagonal coupling
        J_eff = 0.5 * (self.J + self.J.t())
        J_eff = J_eff * (1.0 - torch.eye(self.spin_dim, device=h.device))
        
        for _ in range(steps):
            # Local cavity field: b = J * m + h
            cavity_field = torch.matmul(m, J_eff) + h
            # Mean-field spin update
            m = torch.tanh(beta * cavity_field)
            
        # Ground-state spin glass Hamiltonian energy:
        # H = - 1/2 m^T J m - h^T m
        interaction_energy = -0.5 * torch.sum(m * torch.matmul(m, J_eff), dim=-1)
        field_energy = -torch.sum(h * m, dim=-1)
        total_energy = interaction_energy + field_energy # [batch]
        
        return m, total_energy


class SpinGlassFrustrationRouter(nn.Module):
    def __init__(self, in_features: int = 784, spin_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.spin_dim = spin_dim
        
        # Magnetic field encoder: maps input x to continuous field vector h in R^N
        self.field_encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, spin_dim),
            nn.LayerNorm(spin_dim)
        )
        
        self.spin_modules = nn.ModuleList()

    def add_task(self, active_classes: tuple[int, int], device=None):
        # Freeze past spin modules
        for mod in self.spin_modules:
            for p in mod.parameters():
                p.requires_grad = False
                
        mod = SpinGlassModule(self.spin_dim, active_classes).to(device)
        self.spin_modules.append(mod)

    def forward_train(self, x: torch.Tensor, task_id: int):
        h = self.field_encoder(x)
        mod = self.spin_modules[task_id]
        m, _ = mod.relax_tap_spins(h, steps=5)
        return mod.head(m)

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Blind Class-IL inference via Spin Glass Ground State Energy.
        Routes to the task whose quenched disordered couplings produce the deepest ground state.
        """
        h = self.field_encoder(x)
        batch_size = x.size(0)
        n_tasks = len(self.spin_modules)
        
        all_energies = []
        all_logits = []
        
        for k in range(n_tasks):
            mod = self.spin_modules[k]
            m, energy = mod.relax_tap_spins(h, steps=6)
            logits = mod.head(m)
            
            all_energies.append(energy)
            all_logits.append(logits)
            
        # [n_tasks, batch]
        stacked_energies = torch.stack(all_energies, dim=0)
        # Deepest ground state (lowest energy / maximum resonance)
        best_tasks = torch.argmin(stacked_energies, dim=0) # [batch]
        
        global_logits = torch.full((batch_size, n_tasks * 2), -1e4, device=x.device)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                start_c = k * 2
                end_c = start_c + 2
                global_logits[mask, start_c:end_c] = all_logits[k][mask]
                
        return global_logits
