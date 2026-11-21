"""Permuted MNIST dataset generator for continual learning."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def make_permuted_mnist_tasks(n_tasks: int = 5, batch_size: int = 64, data_root: str = "data", train: bool = True):
    """
    Creates n_tasks where each task applies a fixed random pixel permutation
    to standard MNIST (all 10 digits per task).
    """
    dataset = datasets.MNIST(data_root, train=train, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, labels = next(iter(loader))
    flat_data = data.view(data.size(0), -1)

    tasks = []
    rng = np.random.RandomState(42)

    for i in range(n_tasks):
        if i == 0:
            perm = np.arange(784)
        else:
            perm = rng.permutation(784)
        permuted_x = flat_data[:, perm].view(-1, 1, 28, 28)
        task_ds = TensorDataset(permuted_x, labels)
        task_loader = DataLoader(task_ds, batch_size=batch_size, shuffle=train)
        tasks.append((f"perm_{i}", task_loader))

    return tasks
