"""
Autonomous Energy-Tuned Chiral Router (A-ETCR) for Class-Incremental Learning
=============================================================================
THE REAL PROBLEM TO SOLVE:
In Class-Incremental Learning, the model is NOT told which task an image comes from.
If you use task-specific modules, how does the network know which module to activate 
when shown a blind image?

THE MATHEMATICAL ARCHITECTURE:
1. Each task module learns a Chiral Bivector representation:
      h_k(x) in R^64
2. We compute the Task-Reconstruction Energy:
      E_k(x) = -LogSumExp( logits_k(x) )
   Statistical Physics Principle:
   In-distribution samples for Task k produce low free energy (high confidence logits).
   Out-of-distribution samples produce high free energy (flat, noisy logits).
3. The Blind Router:
      k* = argmin_k E_k(x)  (or argmax confidence)
   The model routes the blind test sample to the module that has the highest 
   internal resonance, completely autonomously without task labels!
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutonomousChiralModule(nn.Module):
    def __init__(self, in_features: int, hidden: int, active_classes: tuple[int, int]):
        super().__init__()
        self.active_classes = active_classes
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 64)
        # Classifies the 10 global classes
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.head(h2)
        # Compute in-task logit energy
        c1, c2 = self.active_classes
        task_confidence = torch.max(logits[:, c1], logits[:, c2])
        return logits, task_confidence


class AutonomousChiralNetwork(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.modules_list = nn.ModuleList()
        self.current_task = 0

    def add_task(self, active_classes: tuple[int, int], device: str = "cpu"):
        # Freeze existing modules
        for m in self.modules_list:
            for p in m.parameters():
                p.requires_grad = False

        new_mod = AutonomousChiralModule(self.in_features, self.hidden, active_classes).to(device)
        self.modules_list.append(new_mod)
        self.current_task = len(self.modules_list) - 1

    def forward_train(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        active_mod = self.modules_list[self.current_task]
        logits, _ = active_mod(x)
        return logits

    def forward_blind(self, x: torch.Tensor):
        """
        CLASS-INCREMENTAL BLIND INFERENCE:
        No task ID given. Evaluates over all modules and dynamically 
        routes based on autonomous energy resonance.
        """
        x = x.view(x.size(0), -1)
        all_logits = []
        all_confs = []

        for mod in self.modules_list:
            logits, conf = mod(x)
            all_logits.append(logits)
            all_confs.append(conf)

        # [num_tasks, N]
        stacked_confs = torch.stack(all_confs, dim=0)
        # Best task per sample: [N]
        best_tasks = torch.argmax(stacked_confs, dim=0)

        # Stack logits: [num_tasks, N, 10]
        stacked_logits = torch.stack(all_logits, dim=0)

        # Select per sample
        final_logits = torch.zeros(x.size(0), 10, device=x.device)
        for i in range(x.size(0)):
            chosen_task = best_tasks[i]
            final_logits[i] = stacked_logits[chosen_task, i]

        return final_logits
