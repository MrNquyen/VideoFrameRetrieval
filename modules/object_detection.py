from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import os

def object_detection(keyframe_path: str) -> list:
    # Tải mô hình YOLOv12-M
    yolo = YOLO("yolov12m.pt")  # Đảm bảo tải pretrained weights
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Phát hiện đối tượng
    image = Image.open(keyframe_path).convert("RGB")  # Chuyển .webp sang RGB nếu cần
    results = yolo(image, imgsz=640, conf=0.5)  # Confidence threshold = 0.5
    
    embeddings = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        cropped = image.crop((x1, y1, x2, y2))
        inputs = clip_processor(images=cropped, return_tensors="pt")
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs).cpu().numpy()
        
        # Lưu embedding
        class_label = results[0].names[int(box.cls)]
        embedding_path = f"D:\\AIC2025\\save\\features\\object_{os.path.basename(keyframe_path)}_{class_label}.npy"
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        np.save(embedding_path, embedding)
        
        embeddings.append({
            "class_label": class_label,
            "box_coords": (x1, y1, x2, y2),
            "confidence": box.conf.item(),
            "v_obj": embedding
        })
    return embeddings

# Ví dụ sử dụng
if __name__ == "__main__":
    keyframe_path = "D:\\AIC2025\\save\\keyframes\\frame_0013.webp"
    embeddings = object_detection(keyframe_path)
    print(embeddings)
