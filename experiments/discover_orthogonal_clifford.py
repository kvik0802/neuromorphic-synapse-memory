"""
DISCOVERY EXPERIMENT 2: ORTHOGONAL MULTI-BLADE CLIFFORD ROTOR RUN
=================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.mnist_tasks import make_tasks
from models.orthogonal_clifford import OrthogonalCliffordContinualMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_all(model, test_tasks):
    model.eval()
    accs = []
    with torch.no_grad():
        for t_idx, (digits, loader) in enumerate(test_tasks):
            correct = 0; total = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x, task_id=t_idx)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            accs.append(correct / max(total, 1))
    return accs

def run():
    train_tasks = make_tasks(train=True)
    test_tasks = make_tasks(train=False)

    model = OrthogonalCliffordContinualMLP(in_features=784, hidden=128, num_classes=10, max_tasks=len(train_tasks)).to(DEVICE)
    acc_matrix = []

    print(f"=== Running Orthogonal Clifford Multi-Blade Discovery on {DEVICE} ===")
    for ti in range(len(train_tasks)):
        digits, loader = train_tasks[ti]
        print(f"\n--- Training Task {ti+1} (Digits {digits}) ---")

        # Freeze input projection after task 1 so shared coordinate frame stays rigid
        if ti == 1:
            for p in model.in_proj.parameters():
                p.requires_grad = False

        # Optimize only active parameters
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        model.train()
        for epoch in range(2):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logits, _ = model(x, task_id=ti)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

        accs = evaluate_all(model, test_tasks)
        acc_matrix.append(accs)
        print(f"Accuracy after Task {ti+1}: " + " ".join(f"{a:.2f}" for a in accs))

        if ti < len(train_tasks) - 1:
            model.advance_task()

    print("\n=== FINAL DISCOVERY MATRIX (ORTHOGONAL CLIFFORD MULTI-BLADE) ===")
    for idx, row in enumerate(acc_matrix):
        print(f"Post-Task {idx+1}: {[round(x, 4) for x in row]}")

    task1_final = acc_matrix[-1][0]
    task1_initial = acc_matrix[0][0]
    forgetting = task1_initial - task1_final
    print(f"\nTask 1 Initial: {task1_initial:.4f} -> Final: {task1_final:.4f} (Forgetting: {forgetting:.4f})")

if __name__ == "__main__":
    run()
