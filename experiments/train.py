from __future__ import annotations
import os
from typing import Dict, List
import torch
import torch.nn.functional as F
from data.mnist_tasks import make_tasks
from models.continual import ContinualLearner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, tasks):
    model.model.eval()
    accs = []
    with torch.no_grad():
        for digits, loader in tasks:
            correct = 0; total = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model.model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
            accs.append(correct / max(total, 1))
    return accs

def train_task(model, loader, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.model.train()
    if model.method in ("si","nsm"):
        model.si_init()
    if model.method in ("ewc","nsm"):
        model.ewc_accum = {n: torch.zeros_like(p) for n, p in model.model.named_parameters()}
        model.ewc_n = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, _ = model.model(x)
            loss = F.cross_entropy(logits, y)
            if model.method == "ewc":
                loss = loss + model.ewc_penalty()
            elif model.method == "si":
                loss = loss + model.si_penalty()
                model.si_update(lr)
            elif model.method == "nsm":
                loss = loss + model.nsm_penalty()
                # replay already handled below
            if model.method in ("replay","nsm"):
                rx, ry = model.get_replay_batch(32)
                if rx is not None:
                    rlogits, _ = model.model(rx)
                    loss = loss + F.cross_entropy(rlogits, ry)
            loss.backward()
            if model.method in ("ewc","nsm"):
                for n, p in model.model.named_parameters():
                    if p.grad is not None:
                        model.ewc_accum[n] += p.grad.detach() ** 2
                model.ewc_n += 1
            if model.method == "orthogonal":
                model.projector.project_gradients()
            opt.step()
            if model.method in ("si","nsm"):
                model.si_update(lr)
    if model.method == "ewc":
        model.consolidate_ewc_from_accum()
    elif model.method == "si":
        model.si_consolidate()
    elif model.method == "nsm":
        model.consolidate_ewc_from_accum()
        model.si_consolidate()
        model.update_synaptic_gate()
    elif model.method == "orthogonal":
        model.consolidate_orthogonal_basis()
    if model.method in ("replay","nsm"):
        xs, ys = [], []
        for x, y in loader:
            xs.append(x); ys.append(y)
            if len(xs)*x.size(0) > 500:
                break
        if xs:
            model.add_replay(torch.cat(xs)[:500], torch.cat(ys)[:500])

def run_experiment(method, epochs=2):
    train_tasks = make_tasks(train=True)
    test_tasks = make_tasks(train=False)
    model = ContinualLearner(method, device=DEVICE)
    model.to(DEVICE)
    n_tasks = len(train_tasks)
    acc_matrix = []
    for ti in range(n_tasks):
        train_task(model, train_tasks[ti][1], epochs=epochs)
        accs = evaluate(model, test_tasks)
        acc_matrix.append(accs)
        print(f"[{method}] after task {ti+1}: " + " ".join(f"{a:.2f}" for a in accs))
    forgetting = 0.0
    if n_tasks > 1:
        drops = []
        for t in range(n_tasks-1):
            best = acc_matrix[t][t]
            final = acc_matrix[-1][t]
            drops.append(best - final)
        forgetting = sum(drops)/len(drops)

    # Backward Transfer (BWT): average difference between final accuracy and initial accuracy after learning each task
    bwt = 0.0
    if n_tasks > 1:
        bwt_drops = [acc_matrix[-1][t] - acc_matrix[t][t] for t in range(n_tasks - 1)]
        bwt = sum(bwt_drops) / len(bwt_drops)

    return {
        "method": method,
        "acc_matrix": acc_matrix,
        "forgetting": round(forgetting, 4),
        "bwt": round(bwt, 4),
        "final_avg": round(sum(acc_matrix[-1]) / len(acc_matrix[-1]), 4),
    }
