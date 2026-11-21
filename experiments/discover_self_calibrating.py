"""
Discovery Benchmark: Energy Contrastive Self-Calibrating Router (EC-SCR)
Testing Z-Score In-Distribution Normalization for Zero-Replay Class-IL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from data.mnist_tasks import make_tasks, DATA_DIR, get_transform
from models.self_calibrating_router import SelfCalibratingNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

def run_calibrated_benchmark(epochs_per_task: int = 5):
    print(f"=== Starting Energy Contrastive Self-Calibrating Router (EC-SCR) Discovery on {DEVICE} ===")
    
    train_tasks = make_tasks(train=True, batch_size=64)
    model = SelfCalibratingNetwork(in_features=784, hidden=128).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    for task_id, (digits, loader) in enumerate(train_tasks):
        model.add_task(digits, device=DEVICE)
        active_mod = model.modules_list[task_id]
        optimizer = optim.AdamW(active_mod.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs_per_task):
            for bx, by in loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE).long()
                
                optimizer.zero_grad()
                logits = model.forward_train(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
        # Calibrate in-distribution Z-score statistics
        active_mod.calibrate(loader, DEVICE)
        print(f"Trained Task {task_id + 1} (Digits {digits}) and computed in-distribution Z-score calibration.")
        
    print("\n--- Evaluating Class-IL Blind Inference (Zero Replay, No Task IDs) ---")
    model.eval()
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
            
    overall_acc = 100.0 * total_correct / total_samples
    print(f"\n=======================================================")
    print(f">> EC-SCR Autonomous Class-IL 10-Class Blind Accuracy: {overall_acc:.2f}%")
    print(f">> Standard Neural Network Baseline:                  19.39%")
    print(f">> Previous Best Router (Argmax):                     72.68%")
    print(f">> Relative Multiplier Over Standard AI:              {overall_acc / 19.39:.2f}x")
    print(f"=======================================================")

if __name__ == "__main__":
    run_calibrated_benchmark()
