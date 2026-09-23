import torch
from torch.utils.data import Dataset
from typing import List
import numpy as np
import pandas as pd

class ClassificationData(Dataset):
    
    def __init__(
            self,
            df: pd.DataFrame,
            normalize: bool=True
    ):
        
        df = df.sort_index()
        arr = df.values.astype(np.float32)

        self.normalize = normalize

        # Normalize
        if self.normalize:
            self.mean = arr.mean(axis=0)
            self.std = arr.std(axis=0)
            arr_norm = (arr - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        # Convert to tensor
        self.data = torch.tensor(arr, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def denormalize_feat(self, y):
        if not self.normalize:
            return y
        return y * torch.tensor(self.std) + torch.tensor(self.mean)
