"""
CLASS-INCREMENTAL SCIENTIFIC SHOWDOWN
======================================
Comparing on 10-Class Blind Evaluation (ZERO TASK IDs AT TEST TIME):
1. Naive Sequential SGD
2. Autonomous Energy-Tuned Chiral Router (Our Architecture)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data.mnist_tasks import make_tasks, DATA_DIR, get_transform
from models.autonomous_router import AutonomousChiralNetwork
from models.continual import MLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return DataLoader(test_ds, batch_size=128, shuffle=False)

def run_scientific_comparison():
    train_tasks = make_tasks(train=True)
    full_test_loader = get_full_test_loader()

    print(f"=========================================================")
    print(f"RUNNING SCIENTIFIC TEST: CLASS-INCREMENTAL LEARNING (CLASS-IL)")
    print(f"Hardware: {DEVICE} (NVIDIA RTX 5050)")
    print(f"Rule: Zero Task IDs at test time. Zero Replay Buffer.")
    print(f"=========================================================\n")

    # 1. BASELINE: Standard Neural Network (MLP)
    print("--- 1. Evaluating Standard Neural Network (Baseline) ---")
    baseline = MLP(hidden=128).to(DEVICE)
    b_opt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    
    for ti, (digits, loader) in enumerate(train_tasks):
        baseline.train()
        for epoch in range(2):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                b_opt.zero_grad()
                logits, _ = baseline(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                b_opt.step()
        print(f"Trained Task {ti+1} (Classes {digits})")

    # Blind Evaluation of Baseline on all 10 classes
    baseline.eval()
    b_correct = 0; b_total = 0
    with torch.no_grad():
        for x, y in full_test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = baseline(x)
            b_correct += (logits.argmax(1) == y).sum().item()
            b_total += y.size(0)
    baseline_acc = b_correct / max(b_total, 1)
    print(f">> Standard Neural Network Blind 10-Class Accuracy: {baseline_acc * 100:.2f}%\n")

    # 2. OUR ARCHITECTURE: Autonomous Chiral Router
    print("--- 2. Evaluating Autonomous Energy-Tuned Chiral Router (Ours) ---")
    auto_net = AutonomousChiralNetwork(in_features=784, hidden=128).to(DEVICE)

    for ti, (digits, loader) in enumerate(train_tasks):
        auto_net.add_task(digits, device=DEVICE)
        # Train active module
        opt = torch.optim.Adam([p for p in auto_net.parameters() if p.requires_grad], lr=1e-3)
        auto_net.train()
        for epoch in range(2):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = auto_net.forward_train(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
        print(f"Trained Module {ti+1} (Classes {digits})")

    # Blind Evaluation of Autonomous Router on all 10 classes
    auto_net.eval()
    a_correct = 0; a_total = 0
    with torch.no_grad():
        for x, y in full_test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = auto_net.forward_blind(x)
            a_correct += (logits.argmax(1) == y).sum().item()
            a_total += y.size(0)
    auto_acc = a_correct / max(a_total, 1)
    print(f">> Autonomous Chiral Router Blind 10-Class Accuracy: {auto_acc * 100:.2f}%\n")

    print(f"=========================================================")
    print(f"FINAL CLASS-INCREMENTAL BENCHMARK (10 Classes Blind):")
    print(f"Standard AI Model:    {baseline_acc * 100:.2f}%")
    print(f"Our Autonomous Net:   {auto_acc * 100:.2f}%")
    print(f"Zero Replay Buffer:   100% Stored-Data Free")
    print(f"=========================================================")

if __name__ == "__main__":
    run_scientific_comparison()
