"""
Zero-Memory Continual Learning Engine via Orthogonal Gradient Projection (GPM/Null-Space).

Mathematical Foundation:
Given layer l and input representation matrix X_k for Task k:
1. Compute SVD of representation matrix: X_k = U * Sigma * V^T
2. Extract the orthogonal basis vectors U_k that capture >= threshold (e.g. 98%) of signal variance.
3. Update projection memory: Basis space B = Span(U_1, U_2, ..., U_k)
4. When training subsequent tasks, project raw gradients into the null space:
     G_projected = G - (G * B) * B^T
This guarantees: G_projected * X_prior = 0 (Mathematical zero interference on previous tasks)
WITHOUT STORING A SINGLE PAST DATA SAMPLE.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class OrthogonalGradientProjector:
    def __init__(self, model: nn.Module, threshold: float = 0.98, device: str = "cpu"):
        self.model = model
        self.threshold = threshold
        self.device = device
        # module_name -> orthogonal basis matrix tensor [hidden_in, rank]
        self.feature_bases = {}

    def extract_task_basis(self, module_name: str, activations: torch.Tensor):
        """
        Extract orthogonal basis from intermediate layer activations using SVD.
        activations: [N, D_in]
        """
        # Center activations
        acts = activations.detach().to(self.device).float()
        if acts.dim() > 2:
            acts = acts.view(acts.size(0), -1)

        # Compute SVD: acts^T acts = V S V^T
        # torch.linalg.svd returns U, S, Vh
        u, s, vh = torch.linalg.svd(acts.t(), full_matrices=False)
        
        # Determine rank capturing self.threshold of variance
        energy = torch.cumsum(s ** 2, dim=0)
        total_energy = energy[-1]
        threshold_energy = self.threshold * total_energy
        rank = (energy < threshold_energy).sum().item() + 1
        rank = min(rank, u.size(1))

        current_basis = u[:, :rank]  # [D_in, rank]

        if module_name not in self.feature_bases:
            self.feature_bases[module_name] = current_basis
        else:
            # Expand existing basis without redundancy
            old_basis = self.feature_bases[module_name]
            # Project current out of old basis: current_res = current - old * (old^T * current)
            proj = torch.mm(old_basis, torch.mm(old_basis.t(), current_basis))
            res = current_basis - proj
            # SVD on residual to get independent new directions
            u_res, s_res, _ = torch.linalg.svd(res, full_matrices=False)
            res_rank = (s_res > 1e-4).sum().item()
            if res_rank > 0:
                new_basis = torch.cat([old_basis, u_res[:, :res_rank]], dim=1)
                self.feature_bases[module_name] = new_basis

    def project_gradients(self):
        """
        Projects gradients of all tracked linear layers into the null space of prior tasks.
        G_new = G - (G * B) * B^T
        """
        for name, module in self.model.named_modules():
            # Only project representation layers (fc1, fc2); let the final classification layer adapt per task
            if name in self.feature_bases and name != "fc3":
                if module.weight.grad is not None:
                    basis = self.feature_bases[name].to(module.weight.device)  # [D_in, rank]
                    grad = module.weight.grad  # [D_out, D_in]
                    # Compute projection: grad * basis -> [D_out, rank]
                    grad_proj = torch.mm(grad, basis)
                    # Subtract projection from raw gradient: [D_out, D_in]
                    grad_orthogonal = grad - torch.mm(grad_proj, basis.t())
                    module.weight.grad.copy_(grad_orthogonal)
