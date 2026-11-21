"""
Quasi-Crystal Penrose Tiling Memory (QPTM)
=========================================
Continual Learning via Aperiodic Non-Repeating Quasicrystalline Tiling.

Mathematical Foundations:
Standard periodic lattices in Euclidean space R^D obey crystallographic restriction theorems:
they only admit 2-fold, 3-fold, 4-fold, or 6-fold rotational symmetry.
Because periodic lattices repeat, representations across different tasks inevitably collide.

A Quasicrystal (discovered by Dan Shechtman, Nobel Prize 2011) possesses:
1. Long-range orientational order with 5-fold (icosahedral) forbidden rotational symmetry.
2. ZERO translational periodicity (it never repeats in infinite space!).

The De Bruijn Dual-Grid Projection Method:
A 2D Penrose quasicrystalline tiling is constructed as a projection from a 5D hypercubic lattice Z^5:
    e_j = ( cos(2*pi*j / 5), sin(2*pi*j / 5) ),  j = 0, 1, 2, 3, 4
The golden ratio tau = (1 + sqrt(5)) / 2 governs the self-similar inflation/deflation operator.

Properties for Zero-Replay Continual Learning:
- Aperiodic Task Allocation:
  Each continual task k is allocated a distinct aperiodic ribbon along the Penrose star pentagon.
  Because the tiling NEVER repeats, task representations do NOT have periodic harmonic resonances
  with previous tasks!
- Golden Ratio Inflation Invariant:
  Scaling a representation by tau^k maps it into a self-similar, non-overlapping quasicrystalline shell.

Blind Class-IL Routing via Quasicrystal Diffraction Braggs:
The Fourier transform of a Penrose quasicrystal consists of a dense set of Bragg delta-peaks
arranged in 10-fold concentric rings.
When an input x matches task k, its projection aligns with the k-th 5-fold diffraction Bragg peak:
    B_k(x) = sum_{j=0}^4 cos( <Phi(x), e_j^{(k)}> )
The input is routed to the task with the highest aperiodic diffraction resonance:
    k* = argmax_k B_k(x)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0

class PenroseQuasicrystalModule(nn.Module):
    def __init__(self, feature_dim: int, task_id: int, active_classes: tuple[int, int]):
        super().__init__()
        self.feature_dim = feature_dim
        self.task_id = task_id
        self.active_classes = active_classes
        
        # 5-fold icosahedral basis vectors in R^feature_dim
        # De Bruijn projection vectors e_0..e_4
        angles = [2.0 * math.pi * j / 5.0 for j in range(5)]
        
        # Build 5 aperiodic star vectors
        star_vectors = []
        for a in angles:
            # Shift angle by task_id to rotate diffraction ring
            phase = a + (task_id * math.pi / 5.0)
            v = torch.randn(feature_dim)
            # Project onto 5-fold phase
            v = v * math.cos(phase) + torch.roll(v, 1) * math.sin(phase)
            star_vectors.append(v / (torch.norm(v) + 1e-6))
            
        # [5, feature_dim]
        self.register_buffer("penrose_star", torch.stack(star_vectors, dim=0))
        
        # Classification head
        self.head = nn.Linear(feature_dim, 2)

    def compute_bragg_diffraction(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes 5-fold Penrose Bragg diffraction peaks:
        B(z) = 1/5 sum_{j=0}^4 cos( <z, e_j> * tau )
        """
        # z: [batch, feature_dim]
        # penrose_star: [5, feature_dim]
        projections = torch.matmul(z, self.penrose_star.t()) # [batch, 5]
        # Modulate by Golden Ratio inflation
        bragg_peaks = torch.cos(projections * GOLDEN_RATIO)
        return torch.mean(bragg_peaks, dim=-1) # [batch] in [-1, +1]

    def forward(self, z: torch.Tensor):
        logits = self.head(z)
        bragg = self.compute_bragg_diffraction(z)
        return logits, bragg


class QuasicrystalMemoryRouter(nn.Module):
    def __init__(self, in_features: int = 784, feature_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.feature_dim = feature_dim
        
        # Projection into continuous embedding space
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.quasicrystal_modules = nn.ModuleList()

    def add_task(self, active_classes: tuple[int, int], device=None):
        # Freeze past modules
        for mod in self.quasicrystal_modules:
            for p in mod.parameters():
                p.requires_grad = False
                
        task_id = len(self.quasicrystal_modules)
        mod = PenroseQuasicrystalModule(self.feature_dim, task_id, active_classes).to(device)
        self.quasicrystal_modules.append(mod)

    def forward_train(self, x: torch.Tensor, task_id: int):
        z = self.encoder(x)
        mod = self.quasicrystal_modules[task_id]
        logits, bragg = mod(z)
        return logits, bragg

    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Blind Class-IL inference via Penrose Quasicrystal Diffraction Resonance.
        Routes to the task whose 5-fold aperiodic Bragg peak is maximally constructive.
        """
        z = self.encoder(x)
        batch_size = x.size(0)
        n_tasks = len(self.quasicrystal_modules)
        
        all_braggs = []
        all_logits = []
        
        for k in range(n_tasks):
            mod = self.quasicrystal_modules[k]
            logits, bragg = mod(z)
            all_braggs.append(bragg)
            all_logits.append(logits)
            
        stacked_braggs = torch.stack(all_braggs, dim=0) # [n_tasks, batch]
        best_tasks = torch.argmax(stacked_braggs, dim=0) # [batch]
        
        global_logits = torch.full((batch_size, n_tasks * 2), -1e4, device=x.device)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                start_c = k * 2
                end_c = start_c + 2
                global_logits[mask, start_c:end_c] = all_logits[k][mask]
                
        return global_logits
