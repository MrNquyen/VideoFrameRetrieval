import cv2
import argparse
import torch
import os
import numpy as np
import torch.nn.functional as F

from typing import List
from tqdm import tqdm
from utils.registry import registry
from feature_extractor.histogram_extractor import HistogramExtractor
from models.beit import BEiTImangeEncoder
from models.clip import CLIPImageEncoder, CLIPTextEncoder
from models.marigold_depth import DepthEstimationExtractor
from modules.image_captioning import ImageCaptioner
from utils.utils import cosine_similarity

class FrameSplitter:
    def __init__(self, interval: int):
        self.writer = registry.get_writer("common")
        self.interval = interval

    def split_frames(
            self,
            source  : str,
            save_dir: str = None, 
            is_saved: bool = False
        ):
        """
            Spliting video into frame

            Parameters:
            -----------
            - source: mp4 video path
            - save_dir: directory where frames is saved
        """
        cap = cv2.VideoCapture(source)
        if is_saved:
            if save_dir==None:
                self.writer.LOG_ERROR("Please provide valid save dir")
                raise ValueError
                if not os.path.exists(save_dir):
                    self.writer.LOG_INFO("Create save directory")
                    os.mkdir(save_dir)

        #-- Setup config
        interval_sec = 2
        fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_skip_frames = interval_sec * fps

        #-- Split frame
        frames = []
        frame_id = 0
        saved_count = 0
        if cap.isOpened() == False:
            self.writer.LOG_INFO('Cap is not open')

        # print(f"Start splitting - Total frames: {total_frames}")
        while(cap.isOpened()):
            ret, frame = self.cap_frame(cap, frame_id=frame_id)
            # print(f"Frame {frame_id} - Type: {type(frame)}")
            #~ Save frame
            if is_saved and frame is not None:
                save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
                self.save_frame(save_path, frame)

            #~ Yield each frame
            if frame is not None: frames.append(frame)
            if not ret:
                #~~ Save last frame
                if frame_id - num_skip_frames < total_frames:
                    ret, frame = self.cap_frame(cap, frame_id=total_frames - 1)
                    if is_saved and frame is not None:
                        save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
                        self.save_frame(save_path, frame)
                    saved_count += 1
                    if frame is not None: frames.append(frame)
                break
            frame_id += num_skip_frames
            saved_count += 1

        self.writer.LOG_INFO(f"Splitting {saved_count} frames from video")
        cap.release()

        frames = [frame for frame in frames if frame is not None]
        return frames

    def cap_frame(self, cap, frame_id):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        return ret, frame

    def save_frame(self, save_path, frame):
        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])


class FrameSelection:
    def __init__(self):
        self.frame_splitter = FrameSplitter()
        self.frame_captioner = ImageCaptioner()
        self.frame_depth_extractor = DepthEstimationExtractor()
        self.beit_encoder = BEiTImangeEncoder()
        self.image_encoder = CLIPImageEncoder()
        self.text_encoder = CLIPTextEncoder()
        self.histogram_extractor = HistogramExtractor()


    #-- Frame selection
    def frame_selection(self, source, batch_size):
        frames = self.frame_splitter.split_frames(source=source)
        features = {
            "color_histogram": [],
            "depth_histogram": [],
            "clip_text_features": [],
            "clip_image_features": [],
            "beit_features": [],
            "num_frames": None
        }

        for start_index in tqdm(range(0, len(frames), batch_size), desc="Get features embedding of video frames"):
            batch_frames = frames[start_index:start_index + batch_size]
            
            #~ Low-level features
            batch_depth_frames = self.frame_depth_extractor.convert_depth(
                image_list=batch_frames,
                batch_size=batch_size
            )
            color_hist_feat = [self.histogram_extractor.extract_hist(image=frame) for frame in batch_frames]
            depth_hist_feat = [self.histogram_extractor.extract_hist(image=frame) for frame in batch_depth_frames]
            
            #~ High-level features
            batch_captions = [self.frame_captioner.caption(frame) for frame in batch_captions]
            batch_text_clip_features = self.text_encoder.encode_text(
                text_list=batch_captions,
                batch_size=batch_size
            )
            batch_frame_clip_features = self.image_encoder.encode_image(
                image_list=batch_frames,
                batch_size=batch_size
            )
            batch_beit_features = self.beit_encoder.encode_frames(
                frames=batch_frames,
                batch_size=batch_size
            )
            
            # Save to common
            features["color_histogram"].extend(color_hist_feat)
            features["depth_histogram"].extend(depth_hist_feat)
            features["clip_text_features"].extend(batch_text_clip_features)
            features["clip_image_features"].extend(batch_frame_clip_features)
            features["beit_features"].extend(batch_beit_features)
        features["num_frames"] = len(frames)

        #~ Selection
        list_keyframe_id = self.selection_lowlevel_features(features=features)
        list_keyframe_id = self.selection_highlevel_features(
            features=features,
            prev_list_keyframe_id=list_keyframe_id
        )
        return frames[list_keyframe_id]


    def selection_lowlevel_features(
            self, 
            features: dict, 
            threshold: float=0.5
        ):
        last_keyframe_id = 0
        list_keyframe_id = [last_keyframe_id]
        color_histogram = torch.tensor(features["color_histogram"])
        depth_histogram = torch.tensor(features["depth_histogram"])
        lowlevel_features = torch.concat([
            color_histogram,
            depth_histogram
        ], dim=-1)
        for id in range(1, features["num_frames"]):
            similarity = cosine_similarity(
                input1=lowlevel_features[last_keyframe_id],
                input2=lowlevel_features[id]
            )
            
            if similarity <= threshold: # New keyframes
                list_keyframe_id.append(id)
                last_keyframe_id = id
        return list_keyframe_id
        

    def selection_highlevel_features(
            self, 
            features: dict, 
            prev_list_keyframe_id: List[int], 
            threshold: float=0.5
        ):
        last_keyframe_id = prev_list_keyframe_id[0]
        final_list_keyframe_id = [last_keyframe_id]
        clip_image_features = torch.tensor(features["clip_image_features"])
        beit_features = torch.tensor(features["beit_features"])
        highlevel_features = torch.concat([
            clip_image_features,
            beit_features
        ], dim=-1)
        for id in prev_list_keyframe_id[1:]:
            similarity = cosine_similarity(
                input1=highlevel_features[last_keyframe_id],
                input2=highlevel_features[id]
            )
            
            if similarity <= threshold: # New keyframes
                final_list_keyframe_id.append(id)
                last_keyframe_id = id
        return final_list_keyframe_id
            
