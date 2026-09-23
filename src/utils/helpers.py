import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def import_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This funciton is used to to import the datasets for this project.
    some important notes on the datasets for this project:

    for N-dimensional dataset the first N columns are class 0 the second N are class 1
    (x1, ... , xN)(x1, ... , xN)
    (--Class 0---)(--Class 1---)

    Args:
        path (str): path the to data file to be imported

    Returns:
        np.ndarray: _description_
    """

    # Import data table
    data_table = pd.read_csv(path, header=None)

    if data_table.shape[1] % 2 != 0:
        raise ValueError("Must have 2N columns")

    n_cord = data_table.shape[1] // 2

    # Seperate class 0 and class 1 coordinates
    class_0 = data_table.iloc[:, :n_cord].dropna().to_numpy(dtype=np.float32)
    class_1 = data_table.iloc[:, n_cord:].dropna().to_numpy(dtype=np.float32)

    # Generate coordinate and classification arrays
    coordinates = np.vstack([class_0, class_1])
    classifications = np.concatenate([
        np.zeros(len(class_0), dtype=np.int64),
        np.ones(len(class_1), dtype=np.int64),
    ])

    # Shuffle indices
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(coordinates))

    coordinates = coordinates[indices]
    classifications = classifications[indices]

    print(f"Coordinates table shape: {coordinates.shape}")
    print(f"Classification table shape: {classifications.shape}")

    return (coordinates, classifications)


