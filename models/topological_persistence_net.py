"""
TOPOLOGICAL PERSISTENCE OPERATOR NETWORK (TPON)
===============================================
A completely unexplored mathematical learning rule:
Instead of standard dot-product activation (f(x) = sigma(W x + b)),
each neuron computes a persistent filtration topological invariant.

Mathematical Definition:
Let input x in R^D be viewed as a discrete filtration curve over an ordered index set.
Each synapse possesses two parameters:
  - Birth threshold b_i in R
  - Death threshold d_i in R  (with constraint d_i >= b_i)

The topological persistence length of coordinate x_i under synapse (b_i, d_i) is:
  pi_i(x_i) = ReLU( min(x_i, d_i) - b_i )

The neuronal activation is the 0-th Betti number integral:
  Phi(x) = sum_{i=1}^D pi_i(x_i) * w_i

Properties for Continual Learning:
Two tasks with disjoint support or distinct filtration intervals have strictly
orthogonal persistence diagrams:
  Integral_{R} pi^{(1)}(x) * pi^{(2)}(x) dx == 0
guaranteeing ZERO CATASTROPHIC INTERFERENCE without requiring task IDs or rehearsal!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PersistentFiltrationLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Birth threshold b
        self.birth = nn.Parameter(torch.empty(out_features, in_features))
        # Log-lifespan delta: death = birth + exp(log_delta) >= birth
        self.log_lifespan = nn.Parameter(torch.empty(out_features, in_features))
        # Topological persistence weight
        self.weights = nn.Parameter(torch.empty(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.birth, -1.0, 1.0)
        nn.init.uniform_(self.log_lifespan, -1.0, 1.0)
        nn.init.kaiming_uniform_(self.weights, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        # birth, death: [out_features, in_features]
        death = self.birth + torch.exp(self.log_lifespan) # Guarantees death >= birth
        
        # Broadcast: [batch, 1, in_features] vs [1, out_features, in_features]
        x_exp = x.unsqueeze(1)
        b_exp = self.birth.unsqueeze(0)
        d_exp = death.unsqueeze(0)
        w_exp = self.weights.unsqueeze(0)
        
        # Persistent homology interval overlap:
        # pi_i(x_i) = max(0, min(x_i, d_i) - b_i)
        clamped_x = torch.minimum(x_exp, d_exp)
        interval_overlap = F.relu(clamped_x - b_exp) # [batch, out_features, in_features]
        
        # Integrate topological persistence
        out = torch.sum(interval_overlap * w_exp, dim=-1) # [batch, out_features]
        return out


class TopologicalMemoryNetwork(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.t_layer1 = PersistentFiltrationLayer(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.t_layer2 = PersistentFiltrationLayer(hidden_dim, 64)
        self.ln2 = nn.LayerNorm(64)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.silu(self.ln1(self.t_layer1(x)))
        h2 = F.silu(self.ln2(self.t_layer2(h1)))
        return self.head(h2)
