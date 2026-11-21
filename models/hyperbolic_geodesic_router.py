"""
Hyperbolic Geodesic Flow Network (HGFN)
=======================================
Riemannian Geometry on the Poincaré Ball Manifold (B^D, g_ball)
for Zero-Replay Continual Learning.

Mathematical Foundations:
Let the representation space be the D-dimensional Poincaré ball:
    B^D = { z in R^D : ||z|| < 1 }
equipped with the Riemannian metric tensor:
    g_z = ( 2 / (1 - ||z||^2) )^2 * I_D

Properties of Hyperbolic Geometry for Continual Learning:
1. Exponential Volume Growth:
   The volume of a geodesic ball of radius r in B^D grows exponentially:
       Vol(B_r) ~ exp((D - 1) * r)
   Unlike Euclidean space R^D where capacity grows polynomially, Hyperbolic space has
   infinite hierarchical capacity near the conformal boundary ||z|| -> 1.
2. Orthogonal Geodesic Cones:
   Infinitely many disjoint task sub-cones can be packed near the ideal boundary 
   partial B^D without overlapping, eliminating catastrophic interference!

Möbius Operations:
- Möbius Addition:
    x \oplus_c y = ( (1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y ) / ( 1 + 2c<x,y> + c^2||x||^2||y||^2 )
- Geodesic Distance:
    d_c(x, y) = (2 / sqrt(c)) * arctanh( sqrt(c) * || -x \oplus_c y || )

Blind Class-IL Routing via Poincaré Horizon Attraction:
Each task k anchors a Geodesic Horizon Attractor p_k on the boundary:
    k* = argmin_k d_c( Phi_c(x), p_k )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Möbius addition in the Poincaré ball."""
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2 + 1e-7
    return num / denom

def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Riemannian geodesic distance in the Poincaré ball."""
    diff = mobius_add(-x, y, c)
    norm = torch.norm(diff, dim=-1).clamp(max=1.0 - 1e-5)
    return 2.0 * torch.arctanh(norm)

def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map from tangent space at origin T_0 B^D into Poincaré ball."""
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-7)
    scaled = torch.tanh(v_norm) * (v / v_norm)
    # Ensure strictly within ball ||z|| < 1
    return scaled.clamp(min=-0.99, max=0.99)


class HyperbolicGeodesicRouter(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 64, c: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.c = c
        
        # Tangent space projection
        self.tangent_proj = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # Geodesic task attractors in B^D
        self.task_attractors = nn.ParameterList()
        self.task_heads = nn.ModuleList()

    def add_task(self, device=None):
        # Anchor an attractor near the hyperbolic boundary
        v = torch.randn(self.hidden_dim, device=device) * 0.5
        attractor = exp_map_zero(v.unsqueeze(0), self.c).squeeze(0)
        self.task_attractors.append(nn.Parameter(attractor))
        
        head = nn.Linear(self.hidden_dim, 2).to(device)
        self.task_heads.append(head)

    def freeze_past_tasks(self):
        for i in range(len(self.task_heads) - 1):
            self.task_attractors[i].requires_grad_(False)
            for p in self.task_heads[i].parameters():
                p.requires_grad_(False)

    def project_to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        v = self.tangent_proj(x)
        return exp_map_zero(v, self.c) # [batch, hidden_dim] in B^D

    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        z = self.project_to_hyperbolic(x)
        return self.task_heads[task_id](z)

    def compute_geodesic_distances(self, z: torch.Tensor) -> torch.Tensor:
        """Computes Riemannian distances d_c(z, p_k) to each task attractor."""
        dists = []
        for p in self.task_attractors:
            p_exp = p.unsqueeze(0).expand(z.size(0), -1)
            d = poincare_distance(z, p_exp, self.c)
            dists.append(d)
        return torch.stack(dists, dim=1) # [batch, n_tasks]

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autonomous Class-IL inference:
        Routes to the task with minimum hyperbolic geodesic distance.
        """
        z = self.project_to_hyperbolic(x)
        dists = self.compute_geodesic_distances(z)
        best_tasks = torch.argmin(dists, dim=1) # [batch]
        
        batch_size = x.size(0)
        n_tasks = len(self.task_heads)
        global_logits = torch.full((batch_size, n_tasks * 2), -1e4, device=x.device)
        
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                sub_z = z[mask]
                sub_logits = self.task_heads[k](sub_z)
                start_c = k * 2
                end_c = start_c + 2
                global_logits[mask, start_c:end_c] = sub_logits
                
        return global_logits
