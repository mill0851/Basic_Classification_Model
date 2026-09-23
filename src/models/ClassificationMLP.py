"""
This python file contains a generic classification MLP. Much of the code is taken from
the capstone project of Robert Miller which can be found here: https://github.com/mill0851/capstone_SiC_gratings

Basic Implementation Details:
> input_dim, hidden_dim, and n_layers can be configured at instantiation
> ReLU is used for nonlinear activation
> BCE is used as the loss function (sigmoid activation is applied internally)
> 

Created: 09/22/2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy

class ClassificationMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            n_layers: int
    ):
        super().__init__()

        layers = []

        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.shared = nn.Sequential(*layers)
        self.classification_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        internal = self.shared(x)
        logits = self.classification_head(internal)
        return logits.squeeze(-1)

def create_dataloaders(
        dataset,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        batch_size: int,
        pre_split: dict | None = None):

    # Ensure ratios workout correctly
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must add to 1.0"

    if pre_split is not None:
        # implement if we wanna do some kfold cross val
        pass 

    # How many samples in each set?
    n_total = len(dataset)
    n_train = round(train_ratio * n_total)
    n_val = round(val_ratio * n_total)

    # seed for reproduceability
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)

    # Extract indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Split up dataset
    train_set = Subset(dataset, train_indices.tolist())
    val_set = Subset(dataset, val_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_classification(
        model: ClassificationMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
        patience: int,
        output_path: str):

    # Configuration for Training Loop
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()        # this combines sigmoid and BCE!

    # History for diagnostics and early stopping
    history = {
        "train_loss": [],
        "val_loss": [],
        "stop_epoch": 0,
        "best_cls_loss": float('inf'),
        "patience_counter": 0
    }

    for epoch in range(epochs):

        # Training Loop
        model.train()
        train_loss = 0
        train_samples = 0

