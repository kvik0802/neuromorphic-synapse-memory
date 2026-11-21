"""
Discovery Benchmark: Fractional Integro-Differential Memory Field (F-IDMF)
Testing Caputo Fractional Calculus restoring forces for zero-replay continual learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from data.mnist_tasks import make_tasks, DATA_DIR, get_transform
from models.fractional_memory_network import FractionalMemoryNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

def run_fractional_benchmark(epochs_per_task: int = 5):
    print(f"=== Starting Fractional Integro-Differential Memory (F-IDMF) Discovery on {DEVICE} ===")
    
    train_tasks = make_tasks(train=True, batch_size=128)
    test_tasks = make_tasks(train=False, batch_size=128)
    
    # Order alpha=0.4 controls power-law decay memory rate
    model = FractionalMemoryNetwork(in_features=784, hidden_dim=128, alpha=0.4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for task_id, (digits, loader) in enumerate(train_tasks):
        model.train()
        for epoch in range(epochs_per_task):
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == digits[1]).long()
                
                optimizer.zero_grad()
                logits = model.forward_task(bx, task_id)
                loss_cls = criterion(logits, by_local)
                
                # Caputo Fractional Restoring Force Loss:
                # Penalizes divergence from the power-law fractional memory trajectory
                frac_loss = model.get_fractional_memory_loss()
                
                total_loss = loss_cls + 0.05 * frac_loss
                total_loss.backward()
                optimizer.step()
                
        # Consolidate this task into the Caputo memory history stack
        model.checkpoint_task()
        print(f"Task {task_id + 1} ({digits}) consolidated via Caputo fractional history integral.")
        
    print("\n--- 1. Evaluating Task-IL Continual Retention across all tasks ---")
    model.eval()
    task_il_accs = []
    with torch.no_grad():
        for t_idx, (digits, loader) in enumerate(test_tasks):
            corr = 0
            tot = 0
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by_local = (by.to(DEVICE) == digits[1]).long()
                preds = torch.argmax(model.forward_task(bx, t_idx), dim=-1)
                corr += (preds == by_local).sum().item()
                tot += by_local.size(0)
            acc = 100.0 * corr / tot
            task_il_accs.append(acc)
            print(f"  Task {t_idx + 1} ({digits}) Task-IL Accuracy: {acc:.2f}%")
            
    mean_task_il = sum(task_il_accs) / len(task_il_accs)
    print(f">> Task-IL Mean Retention: {mean_task_il:.2f}%\n")
    
    print("--- 2. Evaluating Autonomous Class-IL Blind Benchmark (Zero Replay, No Task IDs) ---")
    test_loader = get_full_test_loader()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            by = by.to(DEVICE).long()
            
            global_logits = model.forward_blind(bx)
            preds = torch.argmax(global_logits, dim=-1)
            
            total_correct += (preds == by).sum().item()
            total_samples += by.size(0)
            
    blind_acc = 100.0 * total_correct / total_samples
    print(f"\n=======================================================")
    print(f">> F-IDMF Autonomous Class-IL 10-Class Blind Accuracy: {blind_acc:.2f}%")
    print(f">> Standard Neural Network Baseline:                   19.39%")
    print(f">> Multiplier Over Standard AI:                        {blind_acc / 19.39:.2f}x")
    print(f"=======================================================")

if __name__ == "__main__":
    run_fractional_benchmark()
