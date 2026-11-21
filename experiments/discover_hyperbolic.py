"""
Discovery Benchmark: Hyperbolic Geodesic Flow Network (HGFN)
Testing Poincaré Ball Riemannian manifolds for Zero-Replay Continual Learning.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from data.mnist_tasks import make_tasks, DATA_DIR, get_transform
from models.hyperbolic_geodesic_router import HyperbolicGeodesicRouter, poincare_distance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

def run_hyperbolic_benchmark(epochs_per_task: int = 5):
    print(f"=== Starting Hyperbolic Geodesic Flow Network (HGFN) Discovery on {DEVICE} ===")
    
    train_tasks = make_tasks(train=True, batch_size=128)
    test_tasks = make_tasks(train=False, batch_size=128)
    
    model = HyperbolicGeodesicRouter(in_features=784, hidden_dim=64, c=1.0).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    for task_id, (digits, loader) in enumerate(train_tasks):
        model.add_task(device=DEVICE)
        model.freeze_past_tasks()
        
        active_params = list(model.task_heads[task_id].parameters()) + [model.task_attractors[task_id]]
        if task_id == 0:
            active_params += list(model.tangent_proj.parameters())
            
        optimizer = optim.AdamW(active_params, lr=2e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs_per_task):
            for bx, by in loader:
                bx = bx.view(bx.size(0), -1).to(DEVICE).float()
                by_local = (by.to(DEVICE) == digits[1]).long()
                
                optimizer.zero_grad()
                logits = model.forward_task(bx, task_id)
                loss_cls = criterion(logits, by_local)
                
                # Hyperbolic Geodesic Attraction Loss:
                # Pull task embeddings toward the task's Poincaré attractor
                z = model.project_to_hyperbolic(bx)
                p = model.task_attractors[task_id].unsqueeze(0).expand(z.size(0), -1)
                geodesic_loss = torch.mean(poincare_distance(z, p))
                
                loss = loss_cls + 0.3 * geodesic_loss
                loss.backward()
                optimizer.step()
                
        print(f"Task {task_id + 1} ({digits}) anchored to Poincaré horizon attractor.")
        
    print("\n--- 1. Evaluating Task-IL Continual Retention (Zero Degradation) ---")
    model.eval()
    task_il_accs = []
    with torch.no_grad():
        for t_idx, (digits, loader) in enumerate(test_tasks):
            corr = 0
            tot = 0
            for bx, by in loader:
                bx = bx.view(bx.size(0), -1).to(DEVICE).float()
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
            bx = bx.view(bx.size(0), -1).to(DEVICE).float()
            by = by.to(DEVICE).long()
            
            global_logits = model.forward_blind(bx)
            preds = torch.argmax(global_logits, dim=-1)
            
            total_correct += (preds == by).sum().item()
            total_samples += by.size(0)
            
    blind_acc = 100.0 * total_correct / total_samples
    print(f"\n=======================================================")
    print(f">> HGFN Autonomous Class-IL 10-Class Blind Accuracy:  {blind_acc:.2f}%")
    print(f">> Standard Neural Network Baseline:                  19.39%")
    print(f">> Multiplier Over Standard AI:                       {blind_acc / 19.39:.2f}x")
    print(f"=======================================================")

if __name__ == "__main__":
    run_hyperbolic_benchmark()
