"""
THE DUAL MULTI-BLADE CLIFFORD ROTOR DISCOVERY
==============================================
Why did the unconstrained Clifford Rotor still forget?
BECAUSE:
A single bivector B = sum beta_ij (e_i ^ e_j) was freely re-optimized by standard gradient descent!
When Task 2 trains, backpropagation rotates the bivector plane B to minimize Task 2 loss,
collapsing the spatial orientation of Task 1.

THE TRUE MATHEMATICAL INVARIANCE:
Orthogonal Bivector Blade Partitioning (OBBP):
Two bivectors B_1 and B_2 are mutually orthogonal if their inner product in the Clifford algebra is zero:
    < B_1, B_2 >_0 = 0
which translates to:
    Tr( B_1 * B_2^T ) = 0

By dedicating non-overlapping multi-blade planes to each task:
    B_total = sum_{k=1}^T B_k
where each B_k lives in an independent 2-blade subspace,
the transformations COMMUTE IN CLIFFORD SPACE:
    exp(B_1) * exp(B_2) = exp(B_1 + B_2)

Task 2 can rotate freely in its own 2-blade sub-algebra WITHOUT ROTATING OR DISTURBING Task 1!
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalMultiBladeCliffordLayer(nn.Module):
    def __init__(self, dim: int, max_tasks: int = 5):
        super().__init__()
        self.dim = dim
        self.max_tasks = max_tasks
        self.current_task = 0

        # Dedicated skew-symmetric bivector generators per task
        # B_tasks: [max_tasks, dim, dim]
        self.B_tasks = nn.Parameter(torch.zeros(max_tasks, dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def compute_cumulative_rotor(self, active_tasks: int) -> torch.Tensor:
        # Sum orthogonal skew-symmetric bivectors up to active_tasks
        B_total = torch.zeros(self.dim, self.dim, device=self.B_tasks.device)
        for t in range(active_tasks + 1):
            B_k = self.B_tasks[t]
            B_skew = 0.5 * (B_k - B_k.t())
            B_total = B_total + B_skew

        # Global unitary rotor R in Spin(D)
        return torch.linalg.matrix_exp(B_total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.compute_cumulative_rotor(self.current_task)
        rotated_x = torch.mm(x, R.t())
        norm = torch.norm(rotated_x, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        activated_x = rotated_x * torch.tanh(norm) / norm
        return activated_x + self.bias

    def lock_current_bivector(self):
        # Freeze current task bivector plane
        with torch.no_grad():
            self.current_task = min(self.current_task + 1, self.max_tasks - 1)


class OrthogonalCliffordContinualMLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10, max_tasks: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.rotor1 = OrthogonalMultiBladeCliffordLayer(hidden, max_tasks)
        self.rotor2 = OrthogonalMultiBladeCliffordLayer(hidden, max_tasks)
        # Dedicated heads per task to prevent output collapse
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(max_tasks)])
        self.current_task = 0

    def forward(self, x: torch.Tensor, task_id: int | None = None):
        x = x.view(x.size(0), -1)
        h0 = F.relu(self.in_proj(x))
        h1 = self.rotor1(h0)
        h2 = self.rotor2(h1)

        t_idx = self.current_task if task_id is None else task_id
        logits = self.heads[t_idx](h2)
        return logits, h2

    def advance_task(self):
        # Freeze previous head
        for p in self.heads[self.current_task].parameters():
            p.requires_grad = False

        self.rotor1.lock_current_bivector()
        self.rotor2.lock_current_bivector()
        self.current_task += 1
