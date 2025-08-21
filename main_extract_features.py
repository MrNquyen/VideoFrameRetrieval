import torch
import os
from icecream import ic
import numpy as np
from tqdm import tqdm
from utils.registry import registry
from utils.configs import Config
from utils.logger import Logger
from utils.registry import registry
from utils.flags import Flags
from utils.utils import load_yml
from models.gemini import Gemini
from utils.template import CAPTION_PROMPT
from models.bert_text import TextEncoderBERT
from modules.image_captioning import ImageCaptioner
from utils.preprocess import lemmalizer, remove_stopwords, parse_element
from models.dino import DinoDetector
from models.beit import BEiTImangeEncoder
from models.clip import CLIPImageEncoder, CLIPTextEncoder
from utils.utils import load_image, get_all_image_path

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

    def build_modules(self):
        self.gemini = Gemini()
        self.image_captioner = ImageCaptioner()
        self.beit_encoder = BEiTImangeEncoder(feat_type="retrieval")
        self.image_encoder = CLIPImageEncoder()
        self.text_encoder = CLIPTextEncoder()
        self.text_encoder_bert = TextEncoderBERT()

    
    def get_features(self, image_paths, save_dir, batch_size=4):
        all_exist_npy_file = os.listdir(save_dir)
        for start_index in tqdm(range(0, len(image_paths), batch_size), desc="Get features embedding of video frames"):
            batch_image_paths = image_paths[start_index:start_index + batch_size]
            batch_image_names = [os.path.basename(image_path).split(".")[0] for image_path in batch_image_paths]
            batch_subset_names =  [os.path.basename(os.path.dirname(image_path)) for image_path in batch_image_paths]
            batch_ids = [
                idx
                for idx, (name, subset_name) in enumerate(zip(batch_image_names, batch_subset_names)) 
                if f"{subset_name}_{name}.npy" not in all_exist_npy_file
            ]
            batch_image_paths = np.array(batch_image_paths)[batch_ids]
            batch_image_names = np.array(batch_image_names)[batch_ids]
            batch_subset_names = np.array(batch_subset_names)[batch_ids]

            if len(batch_image_paths) == 0:
                continue
            
            batch_images = [np.array(load_image(image_path)) for image_path in batch_image_paths]
            
            # BEIT
            batch_beit_features = self.beit_encoder.encode_frames(
                frames=batch_images,
                batch_size=batch_size
            )

            # Generate Captions
            # batch_captions = [self.image_captioner.caption(image) for image in batch_images]
            batch_captions = [self.image_captioner.caption(image=image, prompt=CAPTION_PROMPT) for image in batch_images]
            # batch_captions = self.gemini.parapharasing(batch_captions)

            # Image BEIT, CLIP Text Features and CLIP Image Features
            batch_text_clip_features = self.text_encoder.encode_text(
                text_list=batch_captions,
                batch_size=batch_size
            )

            batch_text_bert_features = self.text_encoder_bert.encode_texts(
                texts=batch_captions
            )

            # batch_frame_clip_features = self.image_encoder.encode_image(
            #     image_list=batch_images,
            #     batch_size=batch_size
            # )

            # Features
            for i, (subset_name, name) in enumerate(zip(batch_subset_names, batch_image_names)):
                features = {
                    "text_clip_features": batch_text_clip_features[i],
                    "text_bert_features": batch_text_bert_features[i],
                    # "frame_clip_features": batch_frame_clip_features[i],
                    "beit_features": batch_beit_features[i],
                    "captions": batch_captions[i]
                }
                save_path = f"{save_dir}/{subset_name}_{name}.npy"
                np.save(save_path, features)

def load_txt(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    return lines

if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()

    #~ Our selection
    aic_keyframes_dir = "/data2/npl/ViInfographicCaps/contest/AIC/data/frames/keyframes"
    system = System(args=args)
    save_root = "/data2/npl/ViInfographicCaps/contest/AIC/save" 
    # aic_subsets = os.listdir(aic_keyframes_dir)
    subset_list_path = args.subset_list
    aic_subsets = load_txt(subset_list_path)

    for subset in tqdm(aic_subsets, desc="Extract by subset"):
        aic_subset_keyframes_dir = os.path.join(aic_keyframes_dir, subset)
        subset_keyframes_paths = get_all_image_path(aic_subset_keyframes_dir)
        save_subset_dir = os.path.join(save_root, subset)
        os.makedirs(save_subset_dir, exist_ok=True)
        # ic(subset_keyframes_paths)
        # raise
        system.get_features(
            image_paths=subset_keyframes_paths,
            save_dir=save_subset_dir
        )