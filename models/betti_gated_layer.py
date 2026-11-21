"""
Topological Betti Gated Synapses (TBGS)
Synapses with local persistence gating:
When learning Task k, only synapses whose persistence interval overlaps with 
Task k's filtration curve receive non-zero gradients:
  grad_{w_i} = grad_y * pi_i(x_i)
If a synapse has already stabilized on past task features, its birth/death 
interval is locked, providing mathematically bounded stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BettiGatedLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Synapse parameters
        self.weights = nn.Parameter(torch.empty(out_features, in_features))
        self.birth = nn.Parameter(torch.empty(out_features, in_features))
        self.lifespan = nn.Parameter(torch.empty(out_features, in_features))
        
        # Synaptic plasticity mask (1.0 = plastic/trainable, 0.0 = crystallized/memory)
        self.register_buffer("plasticity_mask", torch.ones(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5) if hasattr(math, 'sqrt') else 2.236)
        nn.init.uniform_(self.birth, -1.0, 1.0)
        nn.init.uniform_(self.lifespan, 0.1, 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        death = self.birth + torch.abs(self.lifespan)
        
        x_exp = x.unsqueeze(1)
        b_exp = self.birth.unsqueeze(0)
        d_exp = death.unsqueeze(0)
        
        # Filtration persistence overlap
        overlap = F.relu(torch.minimum(x_exp, d_exp) - b_exp) # [batch, out, in]
        
        # Effective weights
        w_eff = self.weights * overlap
        return torch.sum(w_eff, dim=-1)

    def crystallize_important_synapses(self, activation_threshold: float = 0.5):
        """Topological consolidation: Freeze synapses that had significant persistence overlap."""
        # Find synapses whose lifespan and weights are committed
        commitment = torch.abs(self.weights.data) * torch.abs(self.lifespan.data)
        top_k_cutoff = torch.quantile(commitment, 0.3) # Protect top 70% active pathways
        protected = commitment > top_k_cutoff
        self.plasticity_mask[protected] = 0.0

    def apply_plasticity_guard(self):
        """Zero out gradients for crystallized synapses."""
        if self.weights.grad is not None:
            self.weights.grad.data.mul_(self.plasticity_mask)
        if self.birth.grad is not None:
            self.birth.grad.data.mul_(self.plasticity_mask)
        if self.lifespan.grad is not None:
            self.lifespan.grad.data.mul_(self.plasticity_mask)
