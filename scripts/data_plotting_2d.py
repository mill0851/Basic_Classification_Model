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

data_path = "./data/moons_2d/moons_2d_narrow.csv"
output_path = "./outputs/moons_2d_narrow_results"

coordinates, classifications = import_data(data_path)

dataset = ClassificationData(coordinates, classifications)
coordinates = dataset.data
labels = dataset.labels

coordinates = coordinates.detach().cpu().numpy()
labels = labels.detach().cpu().numpy().reshape(-1)

plt.figure(figsize=(8, 6))
plt.scatter(
    coordinates[labels == 0, 0],
    coordinates[labels == 0, 1],
    color="green",
    label="Class 0",
)
plt.scatter(
    coordinates[labels == 1, 0],
    coordinates[labels == 1, 1],
    color="red",
    label="Class 1",
)
plt.xlabel("X1 coordinate", fontsize=16)
plt.ylabel("X2 coordinate", fontsize=16)
plt.title("2D Classification Data - Moons (Narrow)", fontsize=18)
plt.grid(True)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_path, "data_visualization.png"))
plt.show()