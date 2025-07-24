import json
import numpy as np
import yaml
import cv2
import functools

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

#---- Load json
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        json_content = json.load(file)
        return json_content
    
#---- Save json
def save_json(path, content):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=3)


#---- Load numpy
def load_npy(path):
    return np.load(path, allow_pickle=True)


#---- Load vocab
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
        return vocab

#---- Get name of the image
def get_img_name(name):
    return name.split(".")[0]

#---- Load yaml file
def load_yml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

@functools.lru_cache(maxsize=256)  # adjust size as needed
def load_img_cache(path):
    # Read binary first
    with open(path, 'rb') as f:
        img_bytes = f.read()
    # Convert to NumPy array and decode
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def load_list_images_fast(image_paths, num_workers=8, desc="Loading images"):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(tqdm(executor.map(load_img_cache, image_paths), total=len(image_paths), desc=desc))
    return images