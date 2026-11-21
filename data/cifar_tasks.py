"""
Continuous Vision Data Loader for Continual Learning: Split-CIFAR-10
Provides 5 sequential tasks of 2 classes each (RGB 32x32 images).
"""

from typing import List, Tuple
import torch
from torchvision import datasets, transforms

DATA_DIR = "data"
CIFAR_TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

def get_cifar_transform(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

def load_cifar10(train: bool = True):
    return datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=get_cifar_transform(train))

def filter_cifar_task(dataset, classes: Tuple[int, int]):
    idx = [i for i, (_, y) in enumerate(dataset) if y in classes]
    return torch.utils.data.Subset(dataset, idx)

def make_cifar_tasks(train: bool = True, batch_size: int = 128):
    ds = load_cifar10(train)
    tasks = []
    for c in CIFAR_TASKS:
        sub = filter_cifar_task(ds, c)
        loader = torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=train, num_workers=0)
        tasks.append((c, loader))
    return tasks
