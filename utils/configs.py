from utils.registry import registry
class Config():
    def __init__(self, config):
        self.config_base = config
        self.build_config()

    def build_config(self):
        self.config_base = self.config_base
        self.config_beit = self.config_base["beit"]
        self.config_clip = self.config_base["clip"]
        self.config_transform = self.config_base["transform"]
        self.config_captioner = self.config_base["captioner"]
        self.config_frame_splitter = self.config_base["frame_splitter"]
        self.config_depth_estimation = self.config_base["depth_estimation"]
        self.config_yolo = self.config_base.get("yolo", {})
        self.config_storage = self.config_base.get("storage", {})
        self.config_object_extractor = self.config_base.get("object_extractor", {})
        
    def build_registry(self):
        registry.set_module("config", name="base", instance=self.config_base)
        registry.set_module("config", name="beit", instance=self.config_beit)
        registry.set_module("config", name="clip", instance=self.config_clip)
        registry.set_module("config", name="captioner", instance=self.config_captioner)
        registry.set_module("config", name="transform", instance=self.config_transform)
        registry.set_module("config", name="frame_splitter", instance=self.config_frame_splitter)
        registry.set_module("config", name="depth_estimation", instance=self.config_depth_estimation)
        registry.set_module("config", name="object_extractor", instance=self.config_object_extractor)
            
        registry.set_module("config", name="yolo", instance=self.config_yolo)
        registry.set_module("config", name="storage", instance=self.config_storage)
            
