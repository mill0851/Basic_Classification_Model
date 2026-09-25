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

data_path = "./data/gaussian_3d/gaussian_3d_narrow.csv"
output_path = "./outputs/gaussian_3d_narrow_results"

coordinates, classifications = import_data(data_path)

dataset = ClassificationData(coordinates, classifications)
coordinates = dataset.data
labels = dataset.labels

coordinates = coordinates.detach().cpu().numpy()
labels = labels.detach().cpu().numpy().reshape(-1)

figure = plt.figure(figsize=(8, 6))
axis = figure.add_subplot(111, projection="3d")
axis.scatter(
    coordinates[labels == 0, 0],
    coordinates[labels == 0, 1],
    coordinates[labels == 0, 2],
    color="green",
    label="Label 0",
)
axis.scatter(
    coordinates[labels == 1, 0],
    coordinates[labels == 1, 1],
    coordinates[labels == 1, 2],
    color="red",
    label="Label 1",
)
axis.set_xlabel("X1 coordinate")
axis.set_ylabel("X2 coordinate")
axis.set_zlabel("X3 coordinate")
axis.set_title("3D Classification Data")
axis.grid(True)
axis.legend()
figure.tight_layout()
figure.savefig(os.path.join(output_path, "data_visualization.png"))
plt.show()