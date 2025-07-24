from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os

class ObjectFeatureExtractor:
    def __init__(self):
        # Khởi tạo mô hình một lần
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.yolo = YOLO("yolov12m.pt").to(self.device)  # Tải YOLOv12-M
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def extract_features(self, image_path: str) -> list:
        """
        Trích xuất Object Features từ một ảnh.
        
        Args:
            image_path (str): Đường dẫn đến ảnh (ví dụ: D:\\AIC2025\\save\\keyframes\\frame_0013.webp)
        
        Returns:
            list: Danh sách các đối tượng với thông tin (class_label, box_coords, confidence, v_obj)
        """
        try:
            # Mở và chuẩn bị ảnh
            image = Image.open(image_path).convert("RGB")
            results = self.yolo.predict(source=image, imgsz=640, conf=0.5)[0]  # Sử dụng predict thay vì gọi trực tiếp

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
            
            # (Tùy chọn) Lưu vector
            if embeddings:
                base_name = os.path.basename(image_path).replace(".webp", "")
                feature_dir = "D:\\AIC2025\\save\\features"
                os.makedirs(feature_dir, exist_ok=True)
                for i, emb in enumerate(embeddings):
                    np.save(os.path.join(feature_dir, f"object_{base_name}_{emb['class_label']}_{i}.npy"), emb["v_obj"])

            return embeddings

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []

    def __del__(self):
        # Giải phóng bộ nhớ
        del self.yolo
        del self.clip_model
        if self.device == "cuda":
            torch.cuda.empty_cache()

# Ví dụ sử dụng
if __name__ == "__main__":
    extractor = ObjectFeatureExtractor()
    keyframe_path = "D:\\AIC2025\\save\\keyframes\\frame_0013.webp"
    embeddings = extractor.extract_features(keyframe_path)
    print(f"Extracted features for {keyframe_path}: {embeddings}")
