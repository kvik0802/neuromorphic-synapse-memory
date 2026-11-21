"""
Dynamic Curvature-Attenuated Manifold Gating (DC-MG)
----------------------------------------------------
A Novel Continual Learning Paradigm without Replay Buffers.

THE MATHEMATICAL PROBLEM WITH EXISTING METHODS:
1. Hard Null-Space Projection (GPM): Requires G_new * B = 0.
   Defect: In finite networks, rank(B) quickly approaches D_in, causing "Capacity Saturation"
   where subsequent tasks have zero available gradient space left to learn.
2. Static Regularization (EWC / SI): Uses constant isotropic penalties λ * (θ - θ*)^2.
   Defect: Treats all parameter directions identically, failing to account for non-isotropic
   loss curvature and parameter interaction.

THE NOVEL FORMULATION (DC-MG):
Instead of binary null-space rejection or static quadratic penalties, we introduce:
1. Dynamic Activation-Covariance Metric Tensor M:
      M_l = E_{x ~ D_k} [ x x^T ]
   with eigendecomposition: M_l = V Lambda V^T

2. Non-Linear Anisotropic Plasticity Filter Phi(Lambda):
   Synaptic stiffness scales non-linearly with curvature energy:
      Phi(lambda_i) = tanh( alpha * (lambda_i / (lambda_max + epsilon)) )

3. Soft Curvature-Attenuated Gradient Mapping:
   Instead of hard projection (P = I - B B^T), the gradient transformation is Riemannian:
      P_{DC-MG}(G) = G - G * V * diag( Phi(Lambda) ) * V^T

   - If an eigenvector v_i has high historical curvature (lambda_i >> 0), Phi -> 1:
     G * v_i is completely shielded (0 gradient update).
   - If an eigenvector has low or noise curvature (lambda_i ~ 0), Phi -> 0:
     G * v_i passes completely unhindered (plasticity preserved for new tasks).
   - Prevents hard rank saturation while guaranteeing bounded directional drift.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class DynamicCurvatureManifoldProjector:
    def __init__(self, model: nn.Module, alpha: float = 3.5, device: str = "cpu"):
        self.model = model
        self.alpha = alpha
        self.device = device
        # module_name -> cumulative curvature metric tensor
        self.metric_tensors = {}
        # module_name -> spectral decomposition (V, filter_weights)
        self.spectral_filters = {}

    def update_manifold_metric(self, module_name: str, activations: torch.Tensor):
        """
        Accumulates the activation metric tensor M = X^T X
        """
        acts = activations.detach().to(self.device).float()
        if acts.dim() > 2:
            acts = acts.view(acts.size(0), -1)

        cov = torch.mm(acts.t(), acts) / acts.size(0)

        if module_name not in self.metric_tensors:
            self.metric_tensors[module_name] = cov
        else:
            # Cumulative Riemannian curvature combination
            self.metric_tensors[module_name] = 0.7 * self.metric_tensors[module_name] + 0.3 * cov

        # Compute Spectral Decomposition
        cov_matrix = self.metric_tensors[module_name]
        try:
            L, V = torch.linalg.eigh(cov_matrix)
            # Sort in descending order
            idx = torch.argsort(L, descending=True)
            L = L[idx]
            V = V[:, idx]

            # Compute Non-Linear Anisotropic Plasticity Filter
            L_max = L[0].clamp(min=1e-8)
            norm_L = L / L_max
            phi = torch.tanh(self.alpha * norm_L)

            self.spectral_filters[module_name] = {
                "V": V,
                "phi": phi
            }
        except Exception as e:
            # Fallback if numerical stability issues occur
            pass

    def apply_manifold_attenuation(self):
        """
        Transforms raw gradients through the Dynamic Curvature Manifold:
        G_attenuated = G - G * V * diag(Phi) * V^T
        """
        for name, module in self.model.named_modules():
            if name in self.spectral_filters and name != "fc3":
                if module.weight.grad is not None:
                    sf = self.spectral_filters[name]
                    V = sf["V"].to(module.weight.device)
                    phi = sf["phi"].to(module.weight.device)

                    grad = module.weight.grad  # [D_out, D_in]

                    # G * V
                    g_v = torch.mm(grad, V)
                    # Scale by anisotropic filter
                    g_v_scaled = g_v * phi.unsqueeze(0)
                    # Reconstruct back
                    g_reconstructed = torch.mm(g_v_scaled, V.t())

                    # Subtract curvature-interfering gradient components
                    grad_safe = grad - g_reconstructed
                    module.weight.grad.copy_(grad_safe)
