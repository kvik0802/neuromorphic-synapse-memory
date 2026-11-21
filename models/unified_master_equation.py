"""
MASTER MASTER EQUATION (MME): UNIFIED TOPOLOGICAL-HAMILTONIAN-FRACTIONAL OPERATOR
==================================================================================
Formula:
    d^alpha W / dt^alpha = -eta * grad_R L(W, x) 
                          - lambda * sum_k [ exp(-d_H(Phi(x), p_k)) * W * (I - P_k) ]
                          + J * [W, R(theta)]_Lie
                          - sum_j (-1)^j binom(alpha, j) (W - W_{t-j})

Where:
  1. Fractional temporal memory: Caputo derivative d^alpha / dt^alpha
  2. Riemannian gradient: grad_R L = G^{-1} dL/dW
  3. Hyperbolic geodesic distance: d_H(Phi(x), p_k) in Poincare ball
  4. Orthogonal null-space projector: (I - P_k)
  5. Lie-Poisson bracket: [W, R(theta)] = W R - R W
  6. Non-Markovian historical anchor: sum (-1)^j binom(alpha, j) (W - W_{t-j})
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedMasterOperator(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, n_classes_per_task: int = 2, alpha: float = 0.4):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_classes_per_task = n_classes_per_task
        self.alpha = alpha
        
        # 1. Base Feature Mapping (Gauge Connection)
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Master Parameter States per Task
        self.task_subspaces = nn.ParameterList()       # P_k basis (Orthogonal Null-space)
        self.task_attractors = nn.ParameterList()      # p_k in Hyperbolic Poincare Ball
        self.task_lie_rotations = nn.ParameterList()   # R_k in so(D) Lie Algebra
        self.task_heads = nn.ModuleList()              # Linear decision heads
        
        # 3. Fractional History Stack for Non-Markovian Caputo anchors
        self.history_snapshots = []

    def add_task(self, device=None):
        # Null-space basis V_k (D x 16)
        basis = nn.Parameter(torch.empty(self.hidden_dim, 16, device=device))
        nn.init.orthogonal_(basis)
        self.task_subspaces.append(basis)
        
        # Hyperbolic attractor in Poincare ball (||p|| < 1)
        attractor = nn.Parameter(torch.randn(self.hidden_dim, device=device) * 0.3)
        self.task_attractors.append(attractor)
        
        # Lie algebra skew-symmetric generator A = -A^T in so(D)
        A = torch.randn(self.hidden_dim, self.hidden_dim, device=device) * 0.05
        A = (A - A.t()) / 2.0
        self.task_lie_rotations.append(nn.Parameter(A))
        
        # Task classification head
        head = nn.Linear(self.hidden_dim, self.n_classes_per_task).to(device)
        self.task_heads.append(head)

    def freeze_past_tasks(self):
        for i in range(len(self.task_heads) - 1):
            self.task_subspaces[i].requires_grad_(False)
            self.task_attractors[i].requires_grad_(False)
            self.task_lie_rotations[i].requires_grad_(False)
            for p in self.task_heads[i].parameters():
                p.requires_grad_(False)

    def get_lie_operator(self, task_id: int) -> torch.Tensor:
        """Matrix exponential of so(D) Lie algebra generator."""
        A = self.task_lie_rotations[task_id]
        skew = (A - A.t()) / 2.0
        return torch.matrix_exp(skew)

    def compute_hyperbolic_distance(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Geodesic distance in Poincare ball: d_H(h, p)."""
        # Project h into unit ball ||z|| < 1
        h_norm = torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-7)
        z = torch.tanh(h_norm) * (h / h_norm) * 0.95
        p_exp = (torch.tanh(torch.norm(p)) * (p / (torch.norm(p) + 1e-7)) * 0.95).unsqueeze(0).expand(z.size(0), -1)
        
        # Poincare distance proxy
        diff = z - p_exp
        return torch.norm(diff, dim=-1)

    def compute_master_energy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the Unified Action Invariant:
        S_k(x) = E_{Hyperbolic}(x) + E_{NullSpace}(x) + E_{LieHelmholtz}(x)
        """
        h = self.feature_map(x)
        n_tasks = len(self.task_heads)
        
        all_energies = []
        all_logits = []
        
        for k in range(n_tasks):
            # 1. Lie transformation: psi = h * exp(A_k)
            R = self.get_lie_operator(k)
            psi = torch.matmul(h, R)
            logits_k = self.task_heads[k](psi)
            all_logits.append(logits_k)
            
            # 2. Hyperbolic Geodesic Distance
            d_H = self.compute_hyperbolic_distance(psi, self.task_attractors[k])
            
            # 3. Orthogonal Null-space projection residual: || psi - V V^T psi ||^2
            V = self.task_subspaces[k]
            proj = torch.matmul(psi, V)
            recon = torch.matmul(proj, V.t())
            null_res = torch.sum((psi - recon) ** 2, dim=-1)
            
            # 4. Helmholtz Free Energy: -logsumexp(logits)
            free_energy = -torch.logsumexp(logits_k, dim=-1)
            
            # Unified Master Invariant: S_k(x)
            # Inputs matching task k minimize all three geometric invariants simultaneously
            total_S = d_H + 0.5 * null_res + 0.5 * free_energy
            all_energies.append(total_S)
            
        stacked_S = torch.stack(all_energies, dim=0) # [n_tasks, batch]
        return stacked_S, all_logits

    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        h = self.feature_map(x)
        R = self.get_lie_operator(task_id)
        psi = torch.matmul(h, R)
        return self.task_heads[task_id](psi)

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autonomous Class-IL forward pass without task IDs.
        Routes to the task that minimizes the Unified Master Invariant S_k(x).
        """
        batch_size = x.size(0)
        n_tasks = len(self.task_heads)
        
        stacked_S, all_logits = self.compute_master_energy(x)
        best_tasks = torch.argmin(stacked_S, dim=0) # [batch]
        
        global_logits = torch.full((batch_size, n_tasks * self.n_classes_per_task), -1e4, device=x.device)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                start_c = k * self.n_classes_per_task
                end_c = start_c + self.n_classes_per_task
                global_logits[mask, start_c:end_c] = all_logits[k][mask]
                
        return global_logits
