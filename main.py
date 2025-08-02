import torch
from icecream import ic
from tqdm import tqdm
from utils.registry import registry
from utils.configs import Config
from utils.logger import Logger
from utils.registry import registry
from utils.flags import Flags
from utils.utils import load_yml

class FiveBrosRetrieverBase:
    def __init__(self, args):
        #~ Configuration
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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


# ================================TEST MODULE HERE=====================
#~ Example split video frames
from modules.frame_splitter import FrameSelection
from modules.object_detection import object_detection  # Thêm module mới
from modules.image_captioning import ImageCaptioner

class FiveBrosSplitFrames(FiveBrosRetrieverBase):
    def __init__(self, args):
        super().__init__(args)
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

    def split_frames(self, video_path):
        frames = self.frame_selection.frame_selection(
            source=video_path,
            save_dir=self.config["save_dir"],
            is_saved_all=True,
            is_saved_selected=True
        )
        return frames


if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()
    print(args)

    #~ Our selection
    video_path = "data/video.mp4"
    fbros_selection = FiveBrosSplitFrames(args=args)
    frames = fbros_selection.split_frames(video_path=video_path)
