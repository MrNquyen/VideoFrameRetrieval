import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List
from torchvision import transforms

class CLIPEncoderBase:
    def __init__(self, pretrained: str = 'laion400m_e32'):
        self.model_name = 'ViT-L-14'
        self.pretrained = pretrained
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained
            )
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name} with weights {pretrained}: {e}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def convert_image_type(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        return image


class CLIPImageEncoder(CLIPEncoderBase):
    def __init__(self, pretrained = 'laion400m_e32'):
        super().__init__(pretrained)

    def encode_image(self, image_list: List[np.ndarray], batch_size: int):
        encoding_list = []
        for start_index in tqdm(range(0, len(image_list), batch_size), desc="Encoding images"):
            batch_images = image_list[start_index:start_index + batch_size]
            processed_batch_images = []
            for image in batch_images:
                try:
                    img = self.convert_image_type(image)
                    img = self.preprocess(img)  # Add batch dimension
                    processed_batch_images.append(img)
                except Exception as e:
                    print(f"Failed to process image: {e}")
            
            if not processed_batch_images:
                print(f"No valid images to encode in batch starting at index {start_index}.")
                continue

            image_tensor = torch.stack(processed_batch_images, dim=0).to(self.device)

            # Perform encoding
            with torch.inference_mode():
                image_features = self.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                encoding_list.extend(image_features.cpu().numpy().astype(np.float32))

        if not encoding_list:
            raise ValueError("No valid images were successfully encoded.")
        return encoding_list # (768,)
    

class CLIPTextEncoder(CLIPEncoderBase):
    def __init__(self, pretrained = 'laion400m_e32'):
        super().__init__(pretrained)

    def encode_text(self, text_list: List[str], batch_size: int):
        encoding_list = []
        for start_index in tqdm(range(0, len(text_list), batch_size), desc="Encoding texts"):
            batch_texts = text_list[start_index:start_index + batch_size]
            if not batch_texts:
                print(f"No valid texts to encode in batch starting at index {start_index}.")
                continue

            # Perform encoding
            with torch.inference_mode():
                text_features = self.model.encode_documents(batch_texts)
                encoding_list.extend(text_features.cpu().numpy().astype(np.float32))

        if not encoding_list:
            raise ValueError("No valid texts were successfully encoded.")
        return encoding_list # (768,)
