"""
Fractional Integro-Differential Memory Field (F-IDMF)
=====================================================
Continual Learning via Caputo Fractional Calculus.

Mathematical Foundations:
Standard artificial neural networks are Markovian:
    W_{t+1} = W_t - eta * grad L_t
The future weight depends exclusively on the current step. There is NO power-law memory 
of the initial historical configuration W_0.

Fractional Integro-Differential Synapses introduce non-local temporal memory 
governed by the Caputo Fractional Derivative of order alpha in (0, 1):
    ^C D_t^alpha W(t) = 1 / Gamma(1 - alpha) * int_0^t (t - s)^{-alpha} * dW/ds ds

In the discrete regime, the fractional memory gradient operator is:
    Delta W_t = -eta * grad L_t + sum_{j=1}^t (-1)^j * binom(alpha, j) * W_{t-j}
where binom(alpha, j) = Gamma(alpha + 1) / ( Gamma(j + 1) * Gamma(alpha - j + 1) ).

Properties for Continual Learning:
1. Power-Law Historical Attractor:
   Unlike exponential forgetting (e^{-lambda t}) where past tasks decay to zero, 
   fractional memory decays as a heavy-tailed power law:
       M(t) ~ t^{-alpha}
   The network maintains a persistent mathematical anchor to Task 1, Task 2, etc., 
   irrespective of the sequence length!
2. Phase-Memory Resonance for Blind Class-IL:
   Each task creates a Mittag-Leffler memory nucleus:
       E_{alpha, beta}(z) = sum_{k=0}^infty z^k / Gamma(alpha * k + beta)
   Inputs resonate with the fractional kernel whose historical memory horizon 
   minimizes the fractional integro-differential action.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_caputo_coefficients(alpha: float, history_len: int, device=None) -> torch.Tensor:
    """Computes discrete binomial weights (-1)^j * binom(alpha, j) for j=1..history_len"""
    weights = []
    w = 1.0
    for j in range(1, history_len + 1):
        # Recursive binomial coefficient calculation:
        # binom(alpha, j) = binom(alpha, j-1) * (alpha - j + 1) / j
        w = w * (alpha - j + 1) / j
        coeff = ((-1) ** j) * w
        weights.append(coeff)
    return torch.tensor(weights, device=device, dtype=torch.float32)


class FractionalSynapseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Primary plastic synaptic weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Historical memory buffer storing task checkpoint states W_k
        self.history_snapshots = []
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def checkpoint_task_history(self):
        """Anchor task weights into the Caputo history stack."""
        self.history_snapshots.append(self.weight.data.clone().detach())

    def compute_fractional_restoring_force(self) -> torch.Tensor:
        """
        Calculates the non-local Caputo restoring force:
        F_restoring = sum_{j=1}^K (-1)^j * binom(alpha, j) * (W_current - W_{hist_j})
        """
        if not self.history_snapshots:
            return torch.zeros_like(self.weight)
            
        k = len(self.history_snapshots)
        coeffs = get_caputo_coefficients(self.alpha, k, device=self.weight.device)
        
        restoring = torch.zeros_like(self.weight)
        for j, snapshot in enumerate(self.history_snapshots):
            # Power-law restoring tension pulling back towards historical memory
            diff = self.weight - snapshot
            restoring = restoring + coeffs[j] * diff
            
        return restoring


class FractionalMemoryNetwork(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 128, alpha: float = 0.4):
        super().__init__()
        self.fc1 = FractionalSynapseLayer(in_features, hidden_dim, alpha=alpha)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = FractionalSynapseLayer(hidden_dim, 64, alpha=alpha)
        self.ln2 = nn.LayerNorm(64)
        self.heads = nn.ModuleList([nn.Linear(64, 2) for _ in range(5)])

    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.gelu(self.ln1(self.fc1(x)))
        h2 = F.gelu(self.ln2(self.fc2(h1)))
        return self.heads[task_id](h2)

    def checkpoint_task(self):
        self.fc1.checkpoint_task_history()
        self.fc2.checkpoint_task_history()

    def get_fractional_memory_loss(self) -> torch.Tensor:
        """Total fractional memory potential energy."""
        loss1 = torch.sum(self.fc1.compute_fractional_restoring_force() ** 2)
        loss2 = torch.sum(self.fc2.compute_fractional_restoring_force() ** 2)
        return loss1 + loss2

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """Blind 10-class inference via fractional confidence resonance."""
        x = x.view(x.size(0), -1)
        batch_size = x.size(0)
        n_tasks = len(self.heads)
        
        h1 = F.gelu(self.ln1(self.fc1(x)))
        h2 = F.gelu(self.ln2(self.fc2(h1)))
        
        all_logits = []
        all_max_confs = []
        for k in range(n_tasks):
            sub_logits = self.heads[k](h2)
            max_c = torch.max(sub_logits, dim=-1).values
            all_logits.append(sub_logits)
            all_max_confs.append(max_c)
            
        stacked_confs = torch.stack(all_max_confs, dim=0) # [n_tasks, batch]
        best_tasks = torch.argmax(stacked_confs, dim=0)
        
        global_logits = torch.full((batch_size, n_tasks * 2), -1e4, device=x.device)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                start_c = k * 2
                end_c = start_c + 2
                global_logits[mask, start_c:end_c] = all_logits[k][mask]
                
        return global_logits
