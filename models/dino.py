import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.registry import registry
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class DinoDetector:
    def __init__(self):
        self.config = registry.get_config("object_extractor")["dino"]
        self.box_threshold = self.config["box_threshold"] 
        self.model_path = self.config["model_path"]
        self.device = registry.get_args("device")
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
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_path).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model - {e}")
        
    def zs_detect(self, images, objects):
        images = [self.convert_image_type(image) for image in images]
        inputs = [
            self.processor(images=image, text=text_label, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)
            for image, text_label in zip(images, objects)
        ]
        results = []
        for id, item_input in enumerate(inputs):
            with torch.no_grad():
                outputs = self.model(**item_input)
            item_results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                target_sizes=[images[id].size[::-1]]
            )

            result = item_results[0]
            results.append(result)
            for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
                box = [round(x, 2) for x in box.tolist()]
                print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        return results