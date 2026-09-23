import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def import_data(path: str) -> np.ndarray:

    data_table = pd.read_csv(path)
    
