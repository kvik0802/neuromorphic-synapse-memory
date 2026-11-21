"""
Empirical Benchmark: Betti-Gated Synaptic Consolidation
Testing gradient gating via topological commitment on Split-MNIST Task-IL.
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.mnist_tasks import make_tasks
from models.betti_gated_layer import BettiGatedLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BettiConsolidatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BettiGatedLayer(784, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = BettiGatedLayer(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.heads = nn.ModuleList([nn.Linear(128, 2) for _ in range(5)])

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.silu(self.ln1(self.l1(x)))
        h2 = F.silu(self.ln2(self.l2(h1)))
        return self.heads[task_id](h2)

    def crystallize(self):
        self.l1.crystallize_important_synapses()
        self.l2.crystallize_important_synapses()

    def guard_gradients(self):
        self.l1.apply_plasticity_guard()
        self.l2.apply_plasticity_guard()

def test_betti():
    print(f"=== Running Betti-Gated Synaptic Consolidation Discovery on {DEVICE} ===")
    tasks = make_tasks(train=True, batch_size=128)
    test_tasks = make_tasks(train=False, batch_size=128)
    
    model = BettiConsolidatedNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    
    for task_idx, (digits, loader) in enumerate(tasks):
        model.train()
        for epoch in range(3):
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == digits[1]).long()
                
                optimizer.zero_grad()
                logits = model(bx, task_idx)
                loss = criterion(logits, by_local)
                loss.backward()
                
                # Topological Plasticity Guard:
                # Blocks gradients from perturbing crystallized synapses!
                model.guard_gradients()
                optimizer.step()
                
        # Consolidate topological persistence for the learned task
        model.crystallize()
        print(f"Task {task_idx + 1} ({digits}) trained & crystallized.")
        
    print("\n--- Evaluating Retention on ALL Tasks after full sequential training ---")
    model.eval()
    accuracies = []
    with torch.no_grad():
        for t_idx, (digits, loader) in enumerate(test_tasks):
            corr = 0
            tot = 0
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == digits[1]).long()
                preds = torch.argmax(model(bx, t_idx), dim=-1)
                corr += (preds == by_local).sum().item()
                tot += by_local.size(0)
            acc = 100.0 * corr / tot
            accuracies.append(acc)
            print(f"  Task {t_idx + 1} ({digits}) Retention Accuracy: {acc:.2f}%")
            
    mean_acc = sum(accuracies) / len(accuracies)
    print(f"\n=======================================================")
    print(f">> Mean Continual Retention across all tasks: {mean_acc:.2f}%")
    print(f">> Task 1 Retention:                          {accuracies[0]:.2f}%")
    print(f"=======================================================")

if __name__ == "__main__":
    test_betti()
