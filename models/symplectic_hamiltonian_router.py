"""
Symplectic Hamiltonian Flow Router (SHFR)
=========================================
Continual Learning formulated as Energy-Conserving Symplectic Mechanics.

Mathematical Formulation:
Let the hidden state of the network be a phase space coordinate:
    z = (q, p) in R^{2D}
where:
  q in R^D is the generalized feature coordinate (spatial configuration)
  p in R^D is the conjugate momentum coordinate (velocity of representation)

Each continual task k defines a conservative task Hamiltonian H_k(q, p):
    H_k(q, p) = 1/2 p^T M_k^{-1} p + V_k(q)
where:
  V_k(q) is a task potential well with minimum at task center mu_k
  M_k is the generalized mass matrix

Hamilton's Equations of Motion:
    dq/dt =  partial H_k / partial p = M_k^{-1} p
    dp/dt = -partial H_k / partial q = -grad V_k(q)

Fundamental Symplectic Conservation Theorem:
Under time evolution Phi_t^{H_k}, the symplectic 2-form omega = sum dq_i wedge dp_i is INVARIANT:
    (Phi_t^{H_k})* omega = omega (Liouville's theorem / Phase space volume conservation)
Because energy H_k(q(t), p(t)) is strictly conserved along task k's flow trajectories:
    dH_k / dt == 0
For inputs from task k: the Hamiltonian remains bounded inside a low-energy invariant torus (KAM Theorem).
For out-of-distribution inputs from other tasks: the Hamiltonian diverges or scatters!

Autonomous Blind Class-IL Routing:
Given a test sample x without task ID:
Initialize (q_0, p_0) = (f(x), 0).
Integrate Hamilton's equations for tau steps using a symplectic Störmer-Verlet leapfrog integrator.
Compute the phase space Action Integral:
    S_k(x) = int_0^tau ( p^T dq/dt - H_k(q, p) ) dt
The input belongs to the task that minimizes action dissipation (Principle of Least Action):
    k* = argmin_k | S_k(x) |
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SymplecticHamiltonianModule(nn.Module):
    def __init__(self, feature_dim: int, active_classes: tuple[int, int]):
        super().__init__()
        self.feature_dim = feature_dim
        self.active_classes = active_classes
        
        # Learnable potential well parameters: V_k(q) = 1/2 (q - mu)^T K (q - mu)
        self.register_buffer("mu", torch.zeros(feature_dim))
        self.log_stiffness = nn.Parameter(torch.zeros(feature_dim))
        
        # Classification head for in-task classes
        self.head = nn.Linear(feature_dim, 2)

    def potential_V(self, q: torch.Tensor) -> torch.Tensor:
        """Potential energy V(q) = sum_i 1/2 k_i (q_i - mu_i)^2"""
        k = torch.exp(self.log_stiffness).unsqueeze(0)
        diff = q - self.mu.unsqueeze(0)
        return 0.5 * torch.sum(k * (diff ** 2), dim=-1)

    def grad_potential(self, q: torch.Tensor) -> torch.Tensor:
        """dV/dq = k * (q - mu)"""
        k = torch.exp(self.log_stiffness).unsqueeze(0)
        return k * (q - self.mu.unsqueeze(0))

    def symplectic_leapfrog_integrate(self, q0: torch.Tensor, steps: int = 5, dt: float = 0.1):
        """
        Symplectic Störmer-Verlet integrator preserving phase space volume exactly.
        q_{n+1/2} = q_n + dt/2 * p_n
        p_{n+1}   = p_n - dt * grad_V(q_{n+1/2})
        q_{n+1}   = q_{n+1/2} + dt/2 * p_{n+1}
        """
        batch_size = q0.size(0)
        q = q0.clone()
        p = torch.zeros_like(q) # Initial zero momentum
        
        total_action = torch.zeros(batch_size, device=q0.device)
        
        for _ in range(steps):
            # Half-step position update
            q_half = q + 0.5 * dt * p
            # Full-step momentum update from conservative force -dV/dq
            force = -self.grad_potential(q_half)
            p = p + dt * force
            # Second half-step position update
            q = q_half + 0.5 * dt * p
            
            # Lagrangian action increment: L = Kinetic - Potential = 1/2 p^2 - V(q)
            kinetic = 0.5 * torch.sum(p ** 2, dim=-1)
            potential = self.potential_V(q)
            total_action = total_action + torch.abs(kinetic - potential) * dt
            
        return q, total_action

    def fit_equilibrium(self, loader, encoder, device):
        """Find the mechanical equilibrium centroid mu on task training data."""
        self.eval()
        feats = []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.view(bx.size(0), -1).to(device).float()
                q = encoder(bx)
                feats.append(q)
        all_q = torch.cat(feats, dim=0)
        self.mu.copy_(torch.mean(all_q, dim=0))


class SymplecticHamiltonianRouter(nn.Module):
    def __init__(self, in_features: int = 784, feature_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.feature_dim = feature_dim
        
        # Generalized coordinate encoder: maps raw inputs to phase space position q
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.hamiltonians = nn.ModuleList()

    def add_task(self, active_classes: tuple[int, int], device=None):
        # Freeze past task potential parameters
        for h in self.hamiltonians:
            for p in h.parameters():
                p.requires_grad = False
                
        mod = SymplecticHamiltonianModule(self.feature_dim, active_classes).to(device)
        self.hamiltonians.append(mod)

    def forward_train(self, x: torch.Tensor, task_id: int):
        q = self.encoder(x)
        return self.hamiltonians[task_id].head(q)

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autonomous Principle of Least Action Routing:
        Simulates Hamiltonian flow under each task's dynamics.
        Selects the task whose phase space trajectory has minimal action divergence.
        """
        q0 = self.encoder(x)
        batch_size = x.size(0)
        n_tasks = len(self.hamiltonians)
        
        all_actions = []
        all_logits = []
        
        for k in range(n_tasks):
            h_mod = self.hamiltonians[k]
            q_final, action = h_mod.symplectic_leapfrog_integrate(q0, steps=5, dt=0.05)
            logits = h_mod.head(q_final)
            
            all_actions.append(action)
            all_logits.append(logits)
            
        # [n_tasks, batch]
        stacked_actions = torch.stack(all_actions, dim=0)
        # Principle of Least Action: argmin action integral
        best_tasks = torch.argmin(stacked_actions, dim=0) # [batch]
        
        # Build global 10-class blind prediction
        global_logits = torch.full((batch_size, n_tasks * 2), -1e4, device=x.device)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                start_c = k * 2
                end_c = start_c + 2
                global_logits[mask, start_c:end_c] = all_logits[k][mask]
                
        return global_logits
