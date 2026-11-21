"""
Empirical Benchmark: Betti-Gated Synaptic Consolidation on Complex Vision (Split-CIFAR-10)
Testing whether topological consolidation protects RGB image representations from catastrophic forgetting.
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.cifar_tasks import make_cifar_tasks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBettiNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional feature backbone
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # 64 * 16 = 1024
        )
        
        self.fc = nn.Linear(1024, 256)
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(5)])
        
        # Plasticity masks for linear projection
        self.register_buffer("fc_mask", torch.ones_like(self.fc.weight))

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        h = F.relu(self.fc(feat))
        return self.heads[task_id](h)

    def crystallize(self):
        """Topological consolidation: freeze top 50% active weights."""
        w_mag = torch.abs(self.fc.weight.data)
        cutoff = torch.quantile(w_mag, 0.5)
        self.fc_mask[w_mag > cutoff] = 0.0

    def guard_gradients(self):
        if self.fc.weight.grad is not None:
            self.fc.weight.grad.data.mul_(self.fc_mask)


def run_cifar_benchmark():
    print(f"=== Running Complex Vision Benchmark: Split-CIFAR-10 on {DEVICE} ===")
    
    train_tasks = make_cifar_tasks(train=True, batch_size=128)
    test_tasks = make_cifar_tasks(train=False, batch_size=128)
    
    model = ConvBettiNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for task_idx, (classes, loader) in enumerate(train_tasks):
        model.train()
        for epoch in range(5):
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == classes[1]).long()
                
                optimizer.zero_grad()
                logits = model(bx, task_idx)
                loss = criterion(logits, by_local)
                loss.backward()
                
                model.guard_gradients()
                optimizer.step()
                
        model.crystallize()
        print(f"Task {task_idx + 1} ({classes}) trained on CIFAR-10 & crystallized.")
        
    print("\n--- Evaluating Continual Retention on Split-CIFAR-10 ---")
    model.eval()
    accuracies = []
    with torch.no_grad():
        for t_idx, (classes, loader) in enumerate(test_tasks):
            corr = 0
            tot = 0
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == classes[1]).long()
                preds = torch.argmax(model(bx, t_idx), dim=-1)
                corr += (preds == by_local).sum().item()
                tot += by_local.size(0)
            acc = 100.0 * corr / tot
            accuracies.append(acc)
            print(f"  Task {t_idx + 1} ({classes}) Retention Accuracy: {acc:.2f}%")
            
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"\n=======================================================")
    print(f">> CIFAR-10 Continual Retention Mean: {mean_acc:.2f}%")
    print(f">> Task 1 (Airplane vs Automobile):   {accuracies[0]:.2f}%")
    print(f"=======================================================")

if __name__ == "__main__":
    run_cifar_benchmark()
