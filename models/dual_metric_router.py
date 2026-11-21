"""
Dual-Metric Manifold Router (DMMR)
Combines:
1. Mahalanobis Manifold Distance in hidden feature space: D_M(h, mu_k, Sigma_k)
2. Energy Logit Margin: E_margin(logits) = max(logits_k) - second_max(logits_k)
3. Temperature-Scaled Softmax Normalization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualMetricModule(nn.Module):
    def __init__(self, in_features: int, hidden: int, active_classes: tuple[int, int]):
        super().__init__()
        self.active_classes = active_classes
        self.fc1 = nn.Linear(in_features, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, 64)
        self.ln2 = nn.LayerNorm(64)
        self.head = nn.Linear(64, 10)
        
        # Manifold prototype and variance (computed after task training)
        self.register_buffer("centroid", torch.zeros(64))
        self.register_buffer("inv_std", torch.ones(64))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.gelu(self.ln1(self.fc1(x)))
        h2 = F.gelu(self.ln2(self.fc2(h1)))
        return h2

    def forward(self, x: torch.Tensor):
        h = self.extract_features(x)
        logits = self.head(h)
        c1, c2 = self.active_classes
        
        # Dual Metric Scoring:
        # 1. Logit Margin between the two active task classes
        active_logits = torch.stack([logits[:, c1], logits[:, c2]], dim=1) # [batch, 2]
        margin = torch.abs(active_logits[:, 0] - active_logits[:, 1]) # [batch]
        max_active_logit = torch.max(active_logits, dim=1).values     # [batch]
        
        # 2. Normalized Mahalanobis Distance to the task centroid
        diff = (h - self.centroid.unsqueeze(0)) * self.inv_std.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1) # [batch]
        # Radial basis function affinity (0 to 1)
        rbf_affinity = torch.exp(-0.05 * dist_sq)
        
        # Composite autonomous resonance score
        resonance = max_active_logit * rbf_affinity + 0.5 * margin
        return logits, resonance

    def update_manifold_statistics(self, loader, device):
        """Fit empirical Gaussian centroid and variance on task training data."""
        self.eval()
        feats = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.view(bx.size(0), -1).to(device).float()
                h = self.extract_features(bx)
                feats.append(h)
        all_h = torch.cat(feats, dim=0)
        mu = torch.mean(all_h, dim=0)
        std = torch.std(all_h, dim=0) + 1e-4
        self.centroid.copy_(mu)
        self.inv_std.copy_(1.0 / std)


class DualMetricNetwork(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.modules_list = nn.ModuleList()
        self.current_task = 0

    def add_task(self, active_classes: tuple[int, int], device: str = "cpu"):
        for m in self.modules_list:
            for p in m.parameters():
                p.requires_grad = False
        new_mod = DualMetricModule(self.in_features, self.hidden, active_classes).to(device)
        self.modules_list.append(new_mod)
        self.current_task = len(self.modules_list) - 1

    def forward_train(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        logits, _ = self.modules_list[self.current_task](x)
        return logits

    def forward_blind(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        all_logits = []
        all_scores = []
        
        for mod in self.modules_list:
            logits, score = mod(x)
            all_logits.append(logits)
            all_scores.append(score)
            
        # [num_tasks, batch]
        stacked_scores = torch.stack(all_scores, dim=0)
        best_tasks = torch.argmax(stacked_scores, dim=0) # [batch]
        
        stacked_logits = torch.stack(all_logits, dim=0) # [num_tasks, batch, 10]
        batch_size = x.size(0)
        
        final_logits = torch.zeros(batch_size, 10, device=x.device)
        for i in range(batch_size):
            chosen_task = best_tasks[i]
            final_logits[i] = stacked_logits[chosen_task, i]
            
        return final_logits
