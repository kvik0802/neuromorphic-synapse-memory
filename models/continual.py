"""Models for continual learning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2), h2  # logits, features


class ContinualLearner:
    """Wrapper holding model + method-specific state."""
    def __init__(self, method: str, device: str = "cpu", hidden: int = 256, replay_size: int = 2000):
        self.method = method
        self.device = device
        self.model = MLP(hidden).to(device)
        self.replay_size = replay_size
        self.replay_buffer = []  # list of (x, y)
        self.ewc_accum = None
        self.ewc_n = 0

        # EWC state
        self.ewc_params = {}  # task -> {name: param tensor}
        self.ewc_fisher = {}  # task -> {name: fisher tensor}
        # SI state
        self.si_omega = {}  # param name -> importance
        self.si_prev_params = {}
        self.si_W = {}  # accumulators
        # NSM: synaptic gate
        self.synaptic_gate = {}
        # Zero-Replay Orthogonal Gradient Projection Engine
        from models.orthogonal_projection import OrthogonalGradientProjector
        self.projector = OrthogonalGradientProjector(self.model, threshold=0.98, device=device)
        self.recorded_activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.recorded_activations:
                    self.recorded_activations[name] = []
                # Keep small sample of input activations to extract basis
                if len(self.recorded_activations[name]) < 20:
                    self.recorded_activations[name].append(input[0].detach())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(get_hook(name))

    def consolidate_orthogonal_basis(self):
        for name, act_list in self.recorded_activations.items():
            if act_list:
                all_acts = torch.cat(act_list, dim=0)
                self.projector.extract_task_basis(name, all_acts)
        self.recorded_activations = {}

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.projector.device = device
        return self

    # --- EWC ---
    def compute_fisher(self, loader, n_samples: int = 200):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += x.size(0)
            if count >= n_samples:
                break
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    def consolidate_ewc_from_accum(self):
        if self.ewc_accum is None or self.ewc_n == 0:
            return
        fisher = {n: (v / self.ewc_n) for n, v in self.ewc_accum.items()}
        tid = len(self.ewc_params)
        self.ewc_params[tid] = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.ewc_fisher[tid] = fisher
        self.ewc_accum = None

    def ewc_penalty(self, lam: float = 100000.0) -> torch.Tensor:
        if not self.ewc_params:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for tid in self.ewc_params:
            for n, p in self.model.named_parameters():
                penalty = penalty + (self.ewc_fisher[tid][n] * (p - self.ewc_params[tid][n]) ** 2).sum()
        return lam * penalty

    # --- SI ---
    def si_init(self):
        self.si_prev_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.si_W = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

    def si_update(self, lr: float):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            delta = (p.detach() - self.si_prev_params[n]).detach()
            self.si_W[n] += (-p.grad.detach() * delta)
            self.si_prev_params[n] = p.detach().clone()

    def si_consolidate(self):
        for n, p in self.model.named_parameters():
            denom = (p.detach() - self.si_prev_params[n]).pow(2) + 1e-3  # approx
            # use accumulated W
            omega = self.si_W[n] / (denom + 1e-8)
            if n in self.si_omega:
                self.si_omega[n] = self.si_omega[n] + omega
            else:
                self.si_omega[n] = omega

    def si_penalty(self, c: float = 1.0) -> torch.Tensor:
        if not self.si_omega:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.si_omega:
                penalty = penalty + (self.si_omega[n] * (p - self.si_prev_params.get(n, p)) ** 2).sum() if n in self.si_prev_params else torch.tensor(0.0, device=self.device)
        return c * penalty

    # --- Replay ---
    def add_replay(self, x, y, max_per_task: int = 200):
        for xi, yi in zip(x, y):
            self.replay_buffer.append((xi.detach().cpu(), yi.detach().cpu()))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer = self.replay_buffer[-self.replay_size:]

    def get_replay_batch(self, n: int):
        if not self.replay_buffer:
            return None, None
        import random
        sample = random.sample(self.replay_buffer, min(n, len(self.replay_buffer)))
        xs = torch.stack([s[0] for s in sample]).to(self.device)
        ys = torch.tensor([s[1] for s in sample], device=self.device)
        return xs, ys

    # --- NSM: synaptic gate update ---
    def update_synaptic_gate(self):
        # gate = importance (from si_omega + fisher) normalized
        gate = {}
        for n, p in self.model.named_parameters():
            imp = torch.zeros_like(p)
            if n in self.si_omega:
                imp = imp + self.si_omega[n].to(p.device)
            for tid in self.ewc_fisher:
                if n in self.ewc_fisher[tid]:
                    imp = imp + self.ewc_fisher[tid][n]
            # normalize
            m = imp.abs().max()
            if m > 0:
                imp = imp / m
            gate[n] = imp.detach()
        self.synaptic_gate = gate

    def nsm_penalty(self, lam: float = 5000.0) -> torch.Tensor:
        if not self.synaptic_gate:
            return torch.tensor(0.0, device=self.device)
        penalty = torch.tensor(0.0, device=self.device)
        for tid in self.ewc_params:
            for n, p in self.model.named_parameters():
                if n in self.synaptic_gate:
                    penalty = penalty + (self.synaptic_gate[n] * (p - self.ewc_params[tid][n]) ** 2).sum()
        return lam * penalty
