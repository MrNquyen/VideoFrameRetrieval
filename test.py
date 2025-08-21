import numpy as np
import json

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# data_path = '/data2/npl/Speech2Text/data2/features/clip-features-32-aic25-b1/clip-features-32/L21_V001.npy'
# data = np.load(data_path, allow_pickle=True)
# print(len(data[0]))

data_path = '/data2/npl/Speech2Text/data2/features/media-info-aic25-b1/media-info/L30_V095.json'
data = load_json(data_path)
print(data)