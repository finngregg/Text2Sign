"""
print_numpy.py: used to confirm that the JSON-files have been correcly compiled into a numpy file using save_files.py

"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import random

if __name__ == '__main__':
    old = np.load
    np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)
    # set path to numpy file
    kp_files = np.load("../json_train_saved_numpy/raw_data.npy").item()
    print(kp_files)