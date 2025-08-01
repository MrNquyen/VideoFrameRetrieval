import torch
import numpy as np
from utils.registry import registry
from icecream import  ic
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
from utils.transform import Transform

class ImageCaptioner:
    def __init__(self):
        self.config = registry.get_config("captioner")
        self.device = registry.get_args("device")
        self.transform = Transform(image_size=448)
        #~ Load model
        self.load_model()

    def convert_image_type(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        return image

    def load_model(self):
        model_path = self.config["model_path"]
        ic(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(self.device)
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=224, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def tokenize_image(self, image, max_num=12):
        image = self.convert_image_type(image).convert("RGB")
        images = self.dynamic_preprocess(
            image=image, 
            use_thumbnail=True, 
            max_num=max_num
        )
        transform = self.transform.transform_from_PIL()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def caption(self, image):
        pixel_values = self.tokenize_image(image, max_num=12).to(torch.float16).to(self.device)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        question = '<image>\nPlease describe the image shortly.'
        response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
        return response


#===================IMAGE CAPTIONER GEMINI=================
class ImageCaptionerGemini:
    def __init__(self):
        