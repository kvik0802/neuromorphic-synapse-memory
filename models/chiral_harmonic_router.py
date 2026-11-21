"""
Chiral Spinor Harmonic Resonator (CSHR)
Unifying Chiral Gauge Invariance with Free-Energy Harmonic Routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChiralHarmonicRouter(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dim: int = 128, n_classes_per_task: int = 2):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.n_classes_per_task = n_classes_per_task
        
        # Shared input projection
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Chiral Spinor Generators for each task k
        self.chiral_bivectors = nn.ParameterList()
        self.task_heads = nn.ModuleList()
        
    def add_task(self, device=None):
        # Skew-symmetric generator A = -A^T in so(N) Lie algebra
        # dim = hidden_dim // 2
        d = self.hidden_dim
        A = torch.randn(d, d, device=device) * 0.05
        # Antisymmetrize
        A = (A - A.t()) / 2.0
        self.chiral_bivectors.append(nn.Parameter(A))
        
        # Task classification head
        head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.n_classes_per_task)
        ).to(device)
        self.task_heads.append(head)
        
    def freeze_past_tasks(self):
        for i in range(len(self.task_heads) - 1):
            self.chiral_bivectors[i].requires_grad_(False)
            for p in self.task_heads[i].parameters():
                p.requires_grad_(False)
                
    def get_spinor_rotation(self, task_id: int) -> torch.Tensor:
        """Matrix exponential of skew-symmetric bivector Lie algebra element: R = exp(A) in SO(D)"""
        A = self.chiral_bivectors[task_id]
        skew = (A - A.t()) / 2.0
        return torch.matrix_exp(skew)
        
    def forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        h = self.encoder(x)
        R = self.get_spinor_rotation(task_id)
        # Apply chiral gauge rotation
        psi = torch.matmul(h, R)
        logits = self.task_heads[task_id](psi)
        return logits
        
    def compute_task_energies(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates Helmholtz free energy: E_k(x) = -LogSumExp(logits_k(x))
        Because R_k is strictly in SO(D), the norm ||psi|| is preserved: ||psi|| = ||h||.
        """
        h = self.encoder(x)
        energies = []
        for k in range(len(self.task_heads)):
            R = self.get_spinor_rotation(k)
            psi = torch.matmul(h, R)
            logits = self.task_heads[k](psi)
            free_energy = -torch.logsumexp(logits, dim=-1) # [batch]
            energies.append(free_energy)
        return torch.stack(energies, dim=1) # [batch, n_tasks]
        
    def forward_blind(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure blind inference across all classes without task ID.
        Routes to the task with lowest Helmholtz free energy (maximum confidence resonance).
        """
        energies = self.compute_task_energies(x) # [batch, n_tasks]
        best_tasks = torch.argmin(energies, dim=1)
        
        batch_size = x.size(0)
        n_tasks = len(self.task_heads)
        total_classes = n_tasks * self.n_classes_per_task
        
        global_logits = torch.full((batch_size, total_classes), -1e4, device=x.device)
        
        h = self.encoder(x)
        for k in range(n_tasks):
            mask = (best_tasks == k)
            if mask.any():
                sub_h = h[mask]
                R = self.get_spinor_rotation(k)
                psi = torch.matmul(sub_h, R)
                sub_logits = self.task_heads[k](psi)
                start_c = k * self.n_classes_per_task
                end_c = start_c + self.n_classes_per_task
                global_logits[mask, start_c:end_c] = sub_logits
                
        return global_logits
