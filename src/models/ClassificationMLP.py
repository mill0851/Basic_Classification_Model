"""
This python file contains a generic classification MLP. Much of the code is taken from
the capstone project of Robert Miller which can be found here: https://github.com/mill0851/capstone_SiC_gratings

Basic Implementation Details:
> input_dim, hidden_dim, and n_layers can be configured at instantiation
> ReLU is used for nonlinear activation
> BCE is used as the loss function (sigmoid activation is applied internally)
> passing `seed` at instantiation makes the weight initialization reproducible

Created: 09/22/2026
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy

from src.data.ClassificationData import ClassificationData

class ClassificationMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            n_layers: int,
            seed: int | None = None
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

        if seed is not None:
            self.initialize_weights(seed)

    def initialize_weights(self, seed: int):
        """Re-initialize every Linear layer from an explicitly seeded generator.

        This reproduces the distribution PyTorch already uses for nn.Linear -
        kaiming_uniform_(a=sqrt(5)) on the weight reduces to U(-b, b) with
        b = 1/sqrt(fan_in), which is also the bias bound - but draws it from a
        local generator instead of the global RNG. Seeding the global RNG would
        work too, but it would leak into anything else drawing randomness later
        (dataloader shuffling, dropout), so the same call would not stay
        reproducible as the script grows.
        """
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in self.modules():
                if not isinstance(module, nn.Linear):
                    continue
                bound = 1.0 / math.sqrt(module.in_features)
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, x):
        internal = self.shared(x)
        logits = self.classification_head(internal)
        return logits.squeeze(-1)

def create_dataloaders(
        dataset: ClassificationData,
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
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

        for feature, classification in train_loader:

            # Zero + Forward pass
            optimizer.zero_grad()
            cls_logits = model(feature)

            # Loss + Backprop
            loss_cls = bce(cls_logits, classification)
            loss_cls.backward()
            optimizer.step()

            # Logging
            batch_size = feature.size(0)
            train_loss += (loss_cls.item() * batch_size)
            train_samples += batch_size

        # Evaluation Loop
        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for feature, classification in val_loader:

                # Forward pass
                cls_logits = model(feature)

                # Loss
                loss_cls = bce(cls_logits, classification)

                # Logging
                batch_size = feature.size(0)
                val_loss += (loss_cls.item() * batch_size)
                val_samples += batch_size

        # Convert to per sample epoch average losses
        val_loss /= val_samples
        train_loss /= train_samples

        # Log Losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Update best epoch if needed
        if val_loss < history['best_cls_loss']:
            history["best_cls_loss"] = val_loss
            history["patience_counter"] = 0
            history["stop_epoch"] = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "history": copy.deepcopy(history)
            }, f"{output_path}/best_model_state.pt")
        else:
            history["patience_counter"] += 1
            if history["patience_counter"] >= patience:
                print(f"Early Stopping triggered at epoch {epoch+1}")
                break

    checkpoint = torch.load(f"{output_path}/best_model_state.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.state_dict(), checkpoint["history"]







