import torch
import os
from icecream import ic
from tqdm import tqdm
from utils.registry import registry
from utils.configs import Config
from utils.logger import Logger
from utils.registry import registry
from utils.flags import Flags
from utils.utils import load_yml
from modules.milvus import Milvus, milvus_config, insert_dummy_data
from models.gemini import Gemini
from utils.preprocess import lemmalizer, remove_stopwords, parse_element
from models.dino import DinoDetector

class FiveBrosRetrieverBase:
    def __init__(self, args):
        #~ Configuration
        self.args = args
        self.device = args.device
        #~ Build
        self.build()

    #-- BUILD
    def build(self):
        self.build_logger()
        self.build_config()
        self.build_registry()

    def build_config(self):
        self.config = Config(load_yml(args.config))
        
    def build_logger(self):
        self.writer = Logger(name="all")

    def build_registry(self):
        #~ Build writer
        registry.set_module("writer", name="common", instance=self.writer)
        #~ Build args
        registry.set_module("args", name=None, instance=self.args)
        #~ Build config
        self.config.build_registry()


# ================================SPLIT FRAMES=====================
#~ Example split video frames
from modules.frame_splitter import FrameSelection
from modules.image_captioning import ImageCaptioner

class FiveBrosSplitFrames:
    def __init__(self):
        super().__init__()
        self.config = registry.get_config("frame_splitter")
        self.add_modules()

    def add_modules(self):
        self.frame_selection = FrameSelection(interval=2)
        self.frame_captioner = ImageCaptioner()

    def caption_frames(self, frames):
        captions = []
        for frame in tqdm(frames, desc="Captioning frames"):
            output = self.frame_captioner.caption(frame)
            captions.append(output)
        return captions

    def split_frames(self, id, video_path):
        frames = self.frame_selection.frame_selection(
            id=id,
            source=video_path,
            save_dir=self.config["save_dir"],
            is_saved_all=True,
            is_saved_selected=True
        )
        return frames

# ================================SYSTEMS=====================
class System(FiveBrosRetrieverBase):
    def __init__(self, args):
        super().__init__(args)
        self.video_paths = [
            os.path.join(
                self.args.video_data_save_dir,
                video_name
            )
            for video_name in 
            os.listdir(self.args.video_data_save_dir)
        ]
        self.milvus_instance = Milvus(milvus_config)

    def build_modules(self):
        self.frame_splitter = FiveBrosSplitFrames()
        self.gemini = Gemini()
        self.lemmalizer = lemmalizer
        self.dino_detector = DinoDetector()

    #-- Frame Selection
    def frame_selection(self):
        for video_id, video_path in tqdm(enumerate(self.video_paths)):
            keyframe_ids, keyframes, keyframe_features, timestamps = self.frame_splitter.split_frames(id=video_id, video_path=video_path)
            
            features_dict = {
                keyframe_id: {
                    "visual_feature": keyframe_features["beit_features"][idx],
                    "text_feature": keyframe_features["clip_text_features"][idx],
                    "objects": [],
                    "concepts": [],
                    "timestamp": timestamp,
                }
                for idx, keyframe_id, timestamp in enumerate(zip(keyframe_ids, timestamps))
            }
            insert_dummy_data(
                milvus_instance=self.milvus_instance,
                features_dict=features_dict
            )

if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()
    print(args)


    #~ Our selection
    video_path = "data/video.mp4"
    fbros_selection = FiveBrosSplitFrames()
    frames = fbros_selection.split_frames(id=0, video_path=video_path)