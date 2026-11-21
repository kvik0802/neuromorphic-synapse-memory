"""
Empirical Discovery Experiment: Topological Persistence Operator Network (TPON)
Testing whether a persistent-filtration interval overlap neuron learns Split-MNIST
and preserves backward transfer compared to standard dot-product neurons.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from data.mnist_tasks import make_tasks
from models.topological_persistence_net import TopologicalMemoryNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_tpon():
    print(f"=== Testing Topological Persistence Operator Network (TPON) on {DEVICE} ===")
    tasks = make_tasks(train=True, batch_size=128)
    test_tasks = make_tasks(train=False, batch_size=128)
    
    model = TopologicalMemoryNetwork(in_features=784, hidden_dim=128, num_classes=10).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Model initialized. Training sequentially across all 5 tasks...")
    
    for task_idx, (digits, loader) in enumerate(tasks):
        model.train()
        for epoch in range(3):
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE).long()
                
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
        # Evaluate current task accuracy
        model.eval()
        corr = 0
        tot = 0
        with torch.no_grad():
            for bx, by in test_tasks[task_idx][1]:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE).long()
                preds = torch.argmax(model(bx), dim=-1)
                corr += (preds == by).sum().item()
                tot += by.size(0)
        acc = 100.0 * corr / tot
        print(f"Task {task_idx + 1} ({digits}) Immediate Test Accuracy: {acc:.2f}%")
        
    # Evaluate retention on Task 1 (digits 0, 1) after training all 5 tasks
    print("\n--- Measuring Catastrophic Forgetting on Task 1 (Digits 0, 1) ---")
    corr1 = 0
    tot1 = 0
    with torch.no_grad():
        for bx, by in test_tasks[0][1]:
            bx = bx.to(DEVICE)
            by = by.to(DEVICE).long()
            preds = torch.argmax(model(bx), dim=-1)
            corr1 += (preds == by).sum().item()
            tot1 += by.size(0)
    final_acc1 = 100.0 * corr1 / tot1
    print(f">> Task 1 Retention Accuracy after learning all subsequent tasks: {final_acc1:.2f}%")

if __name__ == "__main__":
    test_tpon()
