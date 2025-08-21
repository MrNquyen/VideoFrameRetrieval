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
from modules.milvus import Milvus, milvus_config, insert_dummy_data, MilvusLight, MilvusLightV2
from models.gemini import Gemini
from utils.preprocess import lemmalizer, remove_stopwords, parse_element
from models.dino import DinoDetector
from models.beit import BEiTImangeEncoder
from models.clip import CLIPImageEncoder, CLIPTextEncoder
from utils.utils import load_img_cache
from icecream import ic
import numpy as np
from models.bert_text import TextEncoderBERT

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


# ================================SYSTEMS=====================
class System(FiveBrosRetrieverBase):
    def __init__(self, args):
        super().__init__(args)
        self.build_modules()
        # self.milvus_instance = Milvus(milvus_config)
        # self.milvus_instance = MilvusLight(milvus_config)
        # self.milvus_instance = MilvusLightV2(milvus_config)

    def build_modules(self):
        self.gemini = Gemini()
        self.beit_encoder = BEiTImangeEncoder(feat_type="retrieval")
        self.image_encoder = CLIPImageEncoder()
        self.text_encoder = CLIPTextEncoder()
        self.text_encoder_bert = TextEncoderBERT()


    
    def get_features(self, image_dir, batch_size=2):
        image_names = os.listdir(image_dir)
        # image_paths = [os.path.join(image_dir, name) for name in image_names]
        save_dir = f"save/textcaps_features"
        all_exist_npy_file = os.listdir(save_dir)
        for start_index in tqdm(range(0, len(image_names), batch_size), desc="Get features embedding of video frames"):
            batch_image_names = image_names[start_index:start_index + batch_size]
            batch_image_paths = [os.path.join(image_dir, name) for name in batch_image_names]
            batch_image_names = [name.split(".")[0] for name in batch_image_names]
            batch_image_names = [
                name
                for name in batch_image_names 
                if f"{name}.npy" not in all_exist_npy_file
            ]
            if len(batch_image_names) == 0:
                continue
            
            batch_images = [load_img_cache(image_path) for image_path in batch_image_paths]
            
            # Generate Captions
            batch_captions = self.gemini.captioning(batch_images)
            batch_captions = self.gemini.parapharasing(batch_captions)

            # Image BEIT, CLIP Text Features and CLIP Image Features
            batch_text_clip_features = self.text_encoder.encode_text(
                text_list=batch_captions,
                batch_size=batch_size
            )

            batch_text_bert_features = self.text_encoder_bert.encode_texts(
                texts=batch_captions
            )

            batch_frame_clip_features = self.image_encoder.encode_image(
                image_list=batch_images,
                batch_size=batch_size
            )

            batch_beit_features = self.beit_encoder.encode_frames(
                frames=batch_images,
                batch_size=batch_size
            )

            # Features
            for i, name in enumerate(batch_image_names):
                features = {
                    "text_clip_features": batch_text_clip_features[i],
                    "text_bert_features": batch_text_bert_features[i],
                    "frame_clip_features": batch_frame_clip_features[i],
                    "beit_features": batch_beit_features[i],
                    "gemini_captions": batch_captions[i]
                }
                save_path = f"{save_dir}/{name}.npy"
                np.save(save_path, features)

if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()

    #~ Our selection
    video_path = "data/video.mp4"
    # fbros_selection = FiveBrosSplitFrames()
    # frames = fbros_selection.split_frames(id=0, video_path=video_path)
    system = System(args=args)
    system.get_features(image_dir="/data2/npl/ViInfographicCaps/contest/AIC/data/textcaps/images/test_images")