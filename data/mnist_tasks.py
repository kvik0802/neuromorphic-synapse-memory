"""Data utilities: MNIST split into continual tasks."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torchvision import datasets, transforms

DATA_DIR = "data"
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]  # 5 tasks, 2 digits each


def get_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def load_mnist(train: bool = True):
    return datasets.MNIST(DATA_DIR, train=train, download=True, transform=get_transform())


def filter_task(dataset, digits: Tuple[int, int]):
    idx = [i for i, (_, y) in enumerate(dataset) if y in digits]
    return torch.utils.data.Subset(dataset, idx)


def make_tasks(train: bool = True, batch_size: int = 128):
    ds = load_mnist(train)
    tasks = []
    for d in TASKS:
        sub = filter_task(ds, d)
        loader = torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=train)
        tasks.append((d, loader))
    return tasks
