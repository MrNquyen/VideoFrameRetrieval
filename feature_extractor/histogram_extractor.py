
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from icecream import ic

class HistogramExtractor:
    def convert_image_type(self, image):
        if type(image) == Image.Image:
            image = np.array(image)
            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=-1)
        elif type(image) == torch.Tensor:
            if len(image.shape) < 3:
                image = image.unsqueeze(-1)
            image = image.numpy()
        return image

    def extract_hist(self, image):
        image = self.convert_image_type(image)
        num_channels = np.array(image).shape[2]
        feature = []
        for i in range(num_channels):
            channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            feature.append(channel_hist)
        feature = np.concat(feature, axis=0)
        feature = torch.tensor(feature).norm(dim=-1, keepdim=True)
        return feature 