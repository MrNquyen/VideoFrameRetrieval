from utils.registry import registry
from utils.configs import Config
from utils.logger import Logger
from utils.registry import registry
from utils.flags import Flags
from utils.utils import load_yml

class FiveBrosRetriever:
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



if __name__=="__main__":
    flag = Flags()
    args = flag.get_parser()

    #~ Our Retriever
    fbros_sys = FiveBrosRetriever(args=args)
