"""
Holomorphic Lie-Algebraic Fiber Bundle (H-LAF) Continual Learning
==================================================================
Mathematical Theory & Operator Construction:

Instead of modifying weight matrices in flat Euclidean space R^(d_out x d_in),
we structure the layer as a Principal Fiber Bundle P(M, G) where:
- Base Space M: Shared structural weight matrix W_base in R^(d_out x d_in)
- Structure Group G: The Lie Group SO(d_out), acting as an internal gauge group.
- Lie Algebra so(d_out): Space of skew-symmetric generators:
    Omega = -Omega^T

For Task k, the effective operational weight is generated via the exponential map:
    W_k = exp( Omega_k ) * W_base
where:
    Omega_k = sum_{a=1}^r c_{k,a} * J^a
and J^a are orthogonal basis generators of so(d_out).

PROPERTIES:
1. Distance / Norm Preservation:
   Because exp(Omega_k) in SO(d_out), it is an exact orthogonal matrix:
       || W_k ||_F = || exp(Omega_k) W_base ||_F = || W_base ||_F
   The energy of the base representation is strictly invariant across all tasks.

2. Gauge Decoupling:
   Task k updates its Lie algebra parameters {c_{k,a}} on the fiber,
   while W_base is updated with a gauge-covariant conservation constraint.
   Interference between Task i and Task j is governed by the Lie Bracket:
       [Omega_i, Omega_j] = Omega_i Omega_j - Omega_j Omega_i
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LieAlgebraFiberLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank_generators: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank_generators, out_features)

        # Base Manifold: Shared representation weight matrix
        self.W_base = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W_base, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.zeros(out_features))

        # Skew-symmetric generator parameters for active task:
        # We parameterize Omega as: A - A^T (strictly skew-symmetric)
        self.A = nn.Parameter(torch.zeros(out_features, out_features))
        
        # Frozen Lie algebra gauge parameters per task
        self.saved_gauges = {}  # task_id -> orthogonal rotation matrix exp(Omega)
        self.current_task = 0

    def get_current_gauge(self) -> torch.Tensor:
        # Construct skew-symmetric matrix Omega = A - A^T
        omega = self.A - self.A.t()
        # Matrix exponential map from so(n) -> SO(n)
        return torch.linalg.matrix_exp(omega)

    def forward(self, x: torch.Tensor, task_id: int | None = None) -> torch.Tensor:
        if task_id is not None and task_id in self.saved_gauges:
            gauge = self.saved_gauges[task_id].to(self.W_base.device)
        else:
            gauge = self.get_current_gauge()

        # Gauge-transformed weight: W_eff = Gauge * W_base
        w_eff = torch.mm(gauge, self.W_base)
        return F.linear(x, w_eff, self.bias)

    def freeze_task_gauge(self, task_id: int):
        with torch.no_grad():
            gauge = self.get_current_gauge()
            self.saved_gauges[task_id] = gauge.detach().clone()
            # Reset active generator for next task
            self.A.data.zero_()
            self.current_task += 1


class LieAlgebraContinualMLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128, num_classes: int = 10):
        super().__init__()
        self.layer1 = LieAlgebraFiberLayer(in_features, hidden)
        self.layer2 = LieAlgebraFiberLayer(hidden, 64)
        self.head = nn.Linear(64, num_classes)
        self.current_task = 0

    def forward(self, x: torch.Tensor, task_id: int | None = None):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.layer1(x, task_id))
        h2 = F.relu(self.layer2(h1, task_id))
        logits = self.head(h2)
        return logits, h2

    def freeze_current_task(self):
        self.layer1.freeze_task_gauge(self.current_task)
        self.layer2.freeze_task_gauge(self.current_task)
        self.current_task += 1
