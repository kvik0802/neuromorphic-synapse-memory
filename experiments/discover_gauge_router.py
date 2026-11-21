"""
Empirical Benchmark: Exact Holomorphic Gauge Field (EHGF) Router
Zero-Replay Autonomous Continual Learning (Class-IL 10-Class Blind Benchmark)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from data.mnist_tasks import make_tasks, DATA_DIR, get_transform
from models.holomorphic_gauge_router import ExactHolomorphicGaugeRouter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

def train_gauge_router(epochs_per_task: int = 5):
    print(f"=== Starting Exact Holomorphic Gauge Field (EHGF) Discovery on {DEVICE} ===")
    
    train_tasks = make_tasks(train=True, batch_size=64)
    model = ExactHolomorphicGaugeRouter(in_features=784, hidden_dim=128, n_classes_per_task=2).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    for task_id, (digits, loader) in enumerate(train_tasks):
        model.add_task(device=DEVICE)
        model.freeze_past_tasks()
        
        # Optimize only current subspace basis, and current head
        active_params = list(model.task_heads[task_id].parameters()) + [model.task_subspaces[task_id]]
        if task_id == 0:
            active_params += list(model.feature_map.parameters())
            
        optimizer = optim.AdamW(active_params, lr=2e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs_per_task):
            for bx, by in loader:
                bx = bx.view(bx.size(0), -1).to(DEVICE).float()
                # Local class label: 0 or 1
                by_local = (by.to(DEVICE) == digits[1]).long()
                
                optimizer.zero_grad()
                logits = model.forward_task(bx, task_id)
                loss_cls = criterion(logits, by_local)
                
                # QR decomposition ensures strictly orthonormal projection basis
                with torch.no_grad():
                    basis_q, _ = torch.linalg.qr(model.task_subspaces[task_id].data)
                    model.task_subspaces[task_id].data.copy_(basis_q)
                
                h = model.feature_map(bx)
                basis = model.task_subspaces[task_id]
                proj = torch.matmul(h, basis)
                recon = torch.matmul(proj, basis.t())
                subspace_loss = torch.mean(torch.sum((h - recon) ** 2, dim=-1))
                
                total_loss = loss_cls + 1.0 * subspace_loss
                total_loss.backward()
                optimizer.step()
                
        print(f"Task {task_id + 1} (Digits {digits}) registered with gauge subspace projection.")
        
    # --- Class-IL Blind Benchmark ---
    print("\n--- Evaluating Class-IL Blind Inference (Zero Replay, No Task IDs) ---")
    model.eval()
    test_loader = get_full_test_loader()
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.view(bx.size(0), -1).to(DEVICE).float()
            by = by.to(DEVICE).long() # Global digits 0..9
            
            global_logits = model.forward_blind(bx)
            preds = torch.argmax(global_logits, dim=-1)
            
            total_correct += (preds == by).sum().item()
            total_samples += by.size(0)
            
    overall_acc = 100.0 * total_correct / total_samples
    print(f"\n=======================================================")
    print(f">> EHGF Autonomous Class-IL 10-Class Blind Accuracy: {overall_acc:.2f}%")
    print(f">> Standard Neural Network Baseline:                  19.39%")
    print(f">> Relative Multiplier Over Standard AI:              {overall_acc / 19.39:.2f}x")
    print(f"=======================================================")

if __name__ == "__main__":
    train_gauge_router()
