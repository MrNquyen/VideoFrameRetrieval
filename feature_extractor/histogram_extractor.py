
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class HistogramExtractor:
    def convert_image_type(self, image):
        if type(image) == Image:
            image = np.array(image)
        elif type(image) == torch.Tensor:
            image = image.numpy()
        return image

    def extract_hist(self, image):
        image = self.convert_image_type(image)
        num_channels = 3
        feature = []
        for i in range(num_channels):
            channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            feature.append(channel_hist)
        feature = np.concat(feature, axis=0)
        return feature 