"""
DISCOVERY EXPERIMENTAL PROTOCOL: CLIFFORD-ROTOR CONTINUAL LEARNING
==================================================================
Testing whether Clifford Bivector Rotors eliminate catastrophic forgetting 
under sequential task training without any data replay.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.mnist_tasks import make_tasks
from models.clifford_rotor import CliffordRotorMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_tasks(model, test_tasks):
    model.eval()
    accs = []
    with torch.no_grad():
        for digits, loader in test_tasks:
            correct = 0; total = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            accs.append(correct / max(total, 1))
    return accs

def run_pcrd_discovery():
    train_tasks = make_tasks(train=True)
    test_tasks = make_tasks(train=False)

    model = CliffordRotorMLP(in_features=784, hidden=128, num_classes=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    acc_matrix = []

    print(f"=== Starting Discovery Run on {DEVICE} ===")
    for ti in range(len(train_tasks)):
        digits, loader = train_tasks[ti]
        print(f"\n--- Training Task {ti+1} (Digits {digits}) ---")
        model.train()
        for epoch in range(2):
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits, h = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()

        accs = evaluate_tasks(model, test_tasks)
        acc_matrix.append(accs)
        print(f"Accuracy after Task {ti+1}: " + " ".join(f"{a:.2f}" for a in accs))

    print("\n=== FINAL DISCOVERY MATRIX ===")
    for idx, row in enumerate(acc_matrix):
        print(f"Post-Task {idx+1}: {[round(x, 4) for x in row]}")

if __name__ == "__main__":
    run_pcrd_discovery()
