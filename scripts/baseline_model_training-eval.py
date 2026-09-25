from src.data.ClassificationData import ClassificationData
from src.models.ClassificationMLP import (
    train_classification,
    create_dataloaders,
    ClassificationMLP
)
from src.utils.helpers import import_data
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

data_path = "./data/gaussian_2d/gaussian_2d_overlap.csv"
output_path = "./outputs/gaussian_2d_overlap_results"

coordinates, classifications = import_data(data_path)

dataset = ClassificationData(coordinates, classifications)

train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    seed = 1234,
    train_ratio=0.75,
    val_ratio=0.15,
    test_ratio=0.10,
    batch_size=16
)

untrained_model = ClassificationMLP(2, 32, 2)

trained_model_state, history = train_classification(
    model=untrained_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=200,
    lr=0.001,
    patience=50,
    output_path=output_path
)

trained_model = ClassificationMLP(2, 32, 2)
trained_model.load_state_dict(trained_model_state)

# Plot the saved training history
train_losses = history.get("train_loss", [])
val_losses = history.get("val_loss", [])
if train_losses or val_losses:
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    if train_losses:
        axes[0].plot(range(1, len(train_losses) + 1), train_losses)
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")

    if val_losses:
        axes[1].plot(range(1, len(val_losses) + 1), val_losses)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation loss")

    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "training_loss.png"))
    plt.close(figure)

# Evaluate the test set using sigmoid(logit) >= 0.5 as the positive threshold.
trained_model.eval()
confusion_matrix = torch.zeros(2, 2, dtype=torch.int64)
with torch.no_grad():
    for features, labels in test_loader:
        logits = trained_model(features)
        probabilities = torch.sigmoid(logits)
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            probabilities = probabilities[:, 1]
        predictions = (probabilities >= 0.5).long()
        labels = labels.long().view(-1)
        for actual, predicted in zip(labels, predictions):
            confusion_matrix[actual, predicted] += 1

plt.figure()
plt.imshow(confusion_matrix.numpy(), cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix")
plt.xticks([0, 1], [0, 1])
plt.yticks([0, 1], [0, 1])
for row in range(2):
    for column in range(2):
        plt.text(column, row, int(confusion_matrix[row, column]),
                 ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
plt.close()






