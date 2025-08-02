from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os
from utils.registry import registry  # Sử dụng registry để lấy config

class ObjectFeatureExtractor:
    def __init__(self):
        self.config = registry.get_module("config", name="base")  # Lấy config_base từ registry
        self.device = registry.get_args("device")
        try:
            # Giữ nguyên cách tải YOLOv12m.pt
            self.yolo = YOLO("yolov12m.pt").to(self.device)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def extract_features(self, image_path: str) -> list:
        try:
            image = Image.open(image_path).convert("RGB")
            yolo_config = registry.get_module("config", name="yolo")
            results = self.yolo.predict(source=image, imgsz=yolo_config["imgsz"], conf=yolo_config["conf"])[0]

            embeddings = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cropped = image.crop((x1, y1, x2, y2))
                inputs = self.clip_processor(images=cropped, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    embedding = self.clip_model.get_image_features(**inputs).cpu().numpy()
                
                class_label = results.names[int(box.cls)]
                embeddings.append({
                    "class_label": class_label,
                    "box_coords": (x1, y1, x2, y2),
                    "confidence": box.conf.item(),
                    "v_obj": embedding
                })
            
            if embeddings:
                storage_config = registry.get_module("config", name="storage")
                feature_dir = storage_config.get("features_dir", "AIC2025\\save\\features")
                os.makedirs(feature_dir, exist_ok=True)
                base_name = os.path.basename(image_path).replace(".webp", "")
                for i, emb in enumerate(embeddings):
                    np.save(os.path.join(feature_dir, f"object_{base_name}_{emb['class_label']}_{i}.npy"), emb["v_obj"])

            return embeddings

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []

    def __del__(self):
        del self.yolo
        del self.clip_model
        if self.device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Đảm bảo config đã được đăng ký trước khi chạy (thường trong main.py)
    from utils.configs import Config
    config = Config("D:\\AIC2025\\config\\config.yaml")
    config.build_registry()  # Đăng ký config vào registry
    extractor = ObjectFeatureExtractor()
    keyframe_path = "AIC2025\\save\\keyframes\\frame_0013.webp"
    features = extractor.extract_features(keyframe_path)
    print(f"Extracted features for {keyframe_path}: {features}")
