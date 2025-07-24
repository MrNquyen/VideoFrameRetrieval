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


# ================================TEST MODULE HERE=====================
#~ Example split video frames
from modules.frame_splitter import FrameSplitter
class FiveBrosSplitFrames(FiveBrosRetrieverBase):
    def __init__(self, args):
        super().__init__(args)

    def add_modules(self):
        self.frame_splitter = FrameSplitter(interval=2)

    def split_frames(self, video_path):
        frames = self.frame_splitter.split_frames(
            source=video_path
        )
        return frames


if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()

    #~ Our splitter
    video_path = "data/video.mp4"
    fbros_splitter = FiveBrosSplitFrames(args=args)
    frames = fbros_splitter.split_frames(video_path=video_path)
