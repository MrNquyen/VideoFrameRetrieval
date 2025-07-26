import diffusers
import torch
import os
import numpy as np
from typing import List
from tqdm import tqdm
from torchvision import transforms
from utils.transform import Transform
from PIL import Image

class DepthEstimationExtractor:
    def __init__(self, model_name: str = 'prs-eth/marigold-depth-v1-0'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def convert_image_type(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        return image

    def load_model(self):
        try: 
            self.model = diffusers.MarigoldDepthPipeline.from_pretrained(
                self.model_name, 
                variant="fp16", 
                torch_dtype=torch.float16
            ).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model - {e}")
        
    def convert_depth(self, image_list: List[np.ndarray], batch_size: int):
        depth_images = []
        for start_index in tqdm(range(0, len(image_list), batch_size), desc="Encoding images"):
            batch_images = image_list[start_index:start_index + batch_size]
            processed_batch_images = []
            for image in batch_images:
                try:
                    img = self.convert_image_type(image).convert("RGB")
                    img = self.preprocess(img)  # Add batch dimension
                    processed_batch_images.append(img)
                except Exception as e:
                    print(f"Failed to process image: {e}")
            
            if not processed_batch_images:
                print(f"No valid images to extract depth in batch starting at index {start_index}.")
                continue

            # Perform encoding
            depth = self.model(processed_batch_images)
            depth_16bit = self.model.image_processor.export_depth_to_16bit_png(depth.prediction)
            depth_images.extend(depth_16bit)

        if not depth_images:
            raise ValueError("No valid images were successfully extracted depth.")
        return depth_images
