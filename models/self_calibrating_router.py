"""
Energy Contrastive Self-Calibrating Router (EC-SCR)
Solving Class-IL Overconfidence via Unsupervised Contrastive Calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCalibratingModule(nn.Module):
    def __init__(self, in_features: int, hidden: int, active_classes: tuple[int, int]):
        super().__init__()
        self.active_classes = active_classes
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.head = nn.Linear(64, 10)
        
        # Calibration scalar
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("mean_logit", torch.tensor(0.0))
        self.register_buffer("std_logit", torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        logits = self.head(h)
        c1, c2 = self.active_classes
        
        # Focus strictly on the 2 task classes
        sub_logits = torch.stack([logits[:, c1], logits[:, c2]], dim=1) # [batch, 2]
        
        # Softmax margin within this task
        probs = F.softmax(sub_logits, dim=-1)
        conf = torch.max(probs, dim=-1).values # [0.5 to 1.0]
        
        # Raw maximum logit
        raw_max = torch.max(sub_logits, dim=-1).values
        # Z-score normalized confidence
        z_score = (raw_max - self.mean_logit) / (self.std_logit + 1e-4)
        
        # Self-calibrated resonance: z_score * confidence
        resonance = z_score * conf
        return logits, resonance

    def calibrate(self, loader, device):
        self.eval()
        c1, c2 = self.active_classes
        all_max_logits = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.view(bx.size(0), -1).to(device).float()
                h = self.encoder(bx)
                logits = self.head(h)
                sub_logits = torch.stack([logits[:, c1], logits[:, c2]], dim=1)
                max_l = torch.max(sub_logits, dim=-1).values
                all_max_logits.append(max_l)
                
        all_l = torch.cat(all_max_logits, dim=0)
        self.mean_logit.copy_(torch.mean(all_l))
        self.std_logit.copy_(torch.std(all_l) + 1e-4)


class SelfCalibratingNetwork(nn.Module):
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
        new_mod = SelfCalibratingModule(self.in_features, self.hidden, active_classes).to(device)
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
            
        stacked_scores = torch.stack(all_scores, dim=0) # [n_tasks, batch]
        best_tasks = torch.argmax(stacked_scores, dim=0)
        
        stacked_logits = torch.stack(all_logits, dim=0)
        batch_size = x.size(0)
        final_logits = torch.zeros(batch_size, 10, device=x.device)
        
        for i in range(batch_size):
            chosen_task = best_tasks[i]
            final_logits[i] = stacked_logits[chosen_task, i]
            
        return final_logits
