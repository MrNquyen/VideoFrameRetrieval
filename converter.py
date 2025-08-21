import numpy as np
import os
from tqdm import tqdm
import torch

path = '/data2/npl/ViInfographicCaps/contest/AIC/save'

def turn_into_cpu(subset_path):
    for path in tqdm(os.listdir(subset_path), desc="Converting to CPU "):
        full_path = os.path.join(subset_path, path)
        data = np.load(full_path, allow_pickle=True).item()
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.cpu()   # đưa về CPU
        np.save(full_path, data)

for subset in os.listdir(path):
    subset_path = os.path.join(path, subset)
    if os.path.isdir(subset_path):
        turn_into_cpu(subset_path)

# turn_into_cpu('ViInfographicCaps/contest/AIC/save/L21_V003')