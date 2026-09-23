import numpy as np
import torch
from torch.utils.data import Dataset


class ClassificationData(Dataset):
    """PyTorch dataset for features and binary class labels."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
    ) -> None:
        
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError("Data must have shape (n_samples, n_features).")
        if labels.ndim != 1:
            raise ValueError("Labels must have shape (n_samples,).")
        if len(data) != len(labels):
            raise ValueError("Data and labels must contain the same number of samples.")
        if len(data) == 0:
            raise ValueError("The dataset cannot be empty.")

        self.normalize = normalize

        mean = data.mean(axis=0)
        std = data.std(axis=0)
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.std = torch.from_numpy(std.astype(np.float32))

        if normalize:
            data = (data - mean) / std

        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def denormalize_feat(self, features: torch.Tensor) -> torch.Tensor:
        
        if not self.normalize:
            return features

        mean = self.mean.to(device=features.device, dtype=features.dtype)
        std = self.std.to(device=features.device, dtype=features.dtype)

        return features * std + mean
