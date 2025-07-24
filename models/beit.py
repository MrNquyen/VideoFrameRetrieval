import torch
from torch import nn
from utils.registry import registry
import torch.nn.functional as F
import numpy as np
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from utils.beit.unilm.beit3.modeling_finetune import beit3_base_patch16_224_retrieval, beit3_large_patch16_224_nlvr2
from torchvision.transforms.functional import InterpolationMode
from utils.utils import load_img_cache
from utils.transform import Transform
from tqdm import tqdm

class BEiTImangeEncoder:
    def __init__(self, feat_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = registry.get_config("beit")
        self.writer = registry.get_writer("common")
        self.transform = Transform()
        self.feat_type = feat_type

    #-- BUILD
    def build_task(self):
        beit_type_config = self.config.get(self.feat_type, None)
        if beit_type_config==None:
            self.writer.LOG_ERROR(f"Feature type {self.feat_type} unavailable")
            assert ValueError

        beit_model_path = beit_type_config["model_path"]
        beit_tokenizer_path = beit_type_config["tokenizer_path"]
        if self.feat_type=="retrieval":
            self.model = beit3_large_patch16_224_nlvr2(pretrained=True)
        elif self.feat_type=="semantic":
            self.model = beit3_large_patch16_224_nlvr2(pretrained=True)
        
        checkpoint = torch.load(beit_model_path, map_location=self.device)
        self.tokenizer = XLMRobertaTokenizer(beit_tokenizer_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    #-- FUNCTION
    def process_image_from_path(self, image_path: str):
        """Transform a single image."""
        # transform = transforms.Compose([
        #     transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        # ])
        transform = self.transform.transform_from_PIL()
        try:
            with load_img_cache(image_path).convert('RGB') as img:
                return transform(img).to(self.device)
        except Exception as e:
            self.writer.LOG_ERROR(f"Failed to process image {image_path}: {e}")
            return None

    def process_image(self, img: np.ndarray):
        """Transform a single image."""
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        # ])
        transform = self.transform.transform_from_ndarray()
        if img.shape[0] > 3:
            img = img.transpose(2, 0, 1)
        try:
            return transform(img).to(self.device)
        except Exception as e:
            self.writer.LOG_ERROR(f"Failed to process image: {e}")
            return None
        
    #-- Encode frame
    def encode_frames(self, frames, batch_size=4):
        """
            Function:
            ---------
                Encode all frames in one single video shot

            Params:
            ------
                frames: List[np.ndarray] - W, H, C
                    - Frame from frame splitting modules
        """
        encoding_list = []
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(frames), batch_size), desc="Processing and encoding images"):
                batch_frames = frames[start_idx:start_idx + batch_size]
                batch_tensors = []

                # Preprocess images in the batch
                for frame_id, frame in enumerate(frames):
                    try:
                        image_tensor = self.process_image(frame)
                        if image_tensor is not None:
                            batch_tensors.append(image_tensor)
                    except Exception as e:
                        print(f"Failed to process frame {start_idx + frame_id}: {e}")

                if not batch_tensors:
                    print(f"No valid images in batch {start_idx}-{start_idx + batch_size}. Skipping.")
                    continue

                # Stack tensors and move to device
                batch_images = torch.stack(batch_tensors).to(self.device)

                # Encode images
                try:
                    image_features, _ = self.model(image=batch_images, only_infer=True)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    encoding_list.extend(image_features.cpu().numpy().astype(np.float32))
                except Exception as e:
                    print(f"Error during encoding batch {start_idx}-{start_idx + batch_size}: {e}")

        return encoding_list
