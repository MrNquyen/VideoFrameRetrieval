import cv2
import torch
import numpy as np
from PIL import Image
from utils.registry import registry
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

class ObjectExtractor:
    """
    Dùng RTDetrV2ForObjectDetection để detect và trích hidden_states.
    """
    def __init__(self):
        self.logger = registry.get_writer("common")
        self.device = registry.get_args("device")
        self.config = registry.get_config("object_extractor")["detr"]
        self.threshold = self.config["threshold"]
        
        # Load processor & model
        model_name = self.config["model_name"]
        resolution = self.config["resolution"]
        threshold = self.config["threshold"]
        self.processor = RTDetrImageProcessor.from_pretrained(
            model_name,
            size={"height": resolution, "width": resolution}
        )
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        # Enable hidden_states
        self.model.config.output_hidden_states = True
        self.model.config.return_dict = True
        self.model.to(self.device).eval()

        self.logger.info(f"Loaded RT-DETR v2 {model_name} @res={resolution}, thr={threshold}")

    def forward_batch(self, frames: list[np.ndarray]):
        """
        Args:
          frames: list of BGR np.ndarray
        Returns:
          feats_list: list of Tensor[num_objs, hidden_size]
          boxes_list: list of np.ndarray[num_objs,4]
          labels_list: list of np.ndarray[num_objs]
          scores_list: list of np.ndarray[num_objs]
        """
        # 1) Preprocess
        imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)

        # 2) Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3) Hidden states & raw outputs
        hs = outputs.decoder_hidden_states[-1]   # (B, Q, hidden_size)
        logits = outputs.logits                  # (B, Q, num_classes+1)
        boxes_norm = outputs.pred_boxes          # (B, Q, 4)

        # 4) Compute class probabilities
        probs = logits.softmax(-1)
        scores_per_box, labels_per_box = probs[..., :-1].max(-1)  # (B, Q)

        feats_list, boxes_list, labels_list, scores_list = [], [], [], []
        B, Q, Hdim = hs.shape
        for i in range(B):
            scores_all = scores_per_box[i]
            labels_all = labels_per_box[i]
            pos_inds = (scores_all > self.threshold).nonzero(as_tuple=False).flatten()

            if pos_inds.numel() == 0:
                feats_list.append(torch.empty((0, Hdim), dtype=hs.dtype))
                boxes_list.append(np.zeros((0, 4), dtype=np.int32))
                labels_list.append(np.zeros((0,), dtype=np.int32))
                scores_list.append(np.zeros((0,), dtype=np.float32))
                self.logger.info(f"Frame {i}: objs=0, hidden={Hdim}")
                continue

            # extract features
            feats = hs[i][pos_inds].cpu()
            # extract confidences
            confs = scores_all[pos_inds].cpu().numpy().astype(np.float32)
            # extract labels
            labs = labels_all[pos_inds].cpu().numpy().astype(np.int32)
            # extract and convert boxes
            b_norm = boxes_norm[i][pos_inds].cpu().numpy()
            H, W = frames[i].shape[:2]
            xyxy = []
            for xc, yc, wn, hn in b_norm:
                cx, cy = float(xc) * W, float(yc) * H
                bw, bh = float(wn) * W, float(hn) * H
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                x2, y2 = int(cx + bw/2), int(cy + bh/2)
                xyxy.append([x1, y1, x2, y2])
            boxes_arr = np.array(xyxy, dtype=np.int32)

            feats_list.append(feats)
            boxes_list.append(boxes_arr)
            labels_list.append(labs)
            scores_list.append(confs)
            self.logger.info(f"Frame {i}: objs={len(confs)}, hidden={Hdim}")

        return feats_list, boxes_list, labels_list, scores_list