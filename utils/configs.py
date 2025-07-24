from utils.registry import registry
class Config():
    def __init__(self, config):
        self.config_base = config
        self.build_config()

    def build_config(self):
        self.config_base = self.config_base
        self.config_beit = self.config_base["beit"]
        self.config_transform = self.config_base["transform"]
        self.config_captioner = self.config_base["captioner"]
        
    def build_registry(self):
        registry.set_module("config", name="base", instance=self.config_base)
        registry.set_module("config", name="beit", instance=self.config_beit)
        registry.set_module("config", name="captioner", instance=self.config_captioner)
        registry.set_module("config", name="transform", instance=self.config_transform)
            