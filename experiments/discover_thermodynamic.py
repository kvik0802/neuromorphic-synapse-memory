"""
THERMODYNAMIC ROUTER DISCOVERY EXPERIMENT (CLASS-IL BLIND EVALUATION)
=====================================================================
Testing whether Gibbs-Boltzmann Thermodynamic Soft Energy Routing 
beats hard argmax routing and reaches the next level of Class-IL performance.
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
from models.thermodynamic_router import ThermodynamicContinualNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_full_test_loader():
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=get_transform())
    return DataLoader(test_ds, batch_size=128, shuffle=False)

def run():
    train_tasks = make_tasks(train=True)
    full_test_loader = get_full_test_loader()

    print(f"=== Running Thermodynamic Free-Energy Discovery (TS-FEMR) on {DEVICE} ===")
    net = ThermodynamicContinualNetwork(in_features=784, hidden=128, routing_temp=1.5).to(DEVICE)

    for ti, (digits, loader) in enumerate(train_tasks):
        net.add_task(digits, device=DEVICE)
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)

        net.train()
        for epoch in range(2):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits = net.forward_train(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
        print(f"Trained Thermodynamic Module {ti+1} (Classes {digits})")

    # Blind Evaluation on all 10 classes
    net.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in full_test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = net.forward_blind_thermodynamic(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    acc = correct / max(total, 1)
    print(f"\n=======================================================")
    print(f">> Thermodynamic Free-Energy Router 10-Class Blind Accuracy: {acc * 100:.2f}%")
    print(f">> Standard Neural Network Baseline:                         19.39%")
    print(f">> Previous Hard Argmax Router:                               72.68%")
    print(f"=======================================================")

if __name__ == "__main__":
    run()
