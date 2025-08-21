import argparse
class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser.parse_args()

    def add_core_args(self):
        # TODO: Update default values
        self.parser.add_argument_group("Core Arguments")

        self.parser.add_argument(
            "--config", type=str, default="config/config.yaml", required=False, help="config yaml file"
        )
        self.parser.add_argument(
            "--save_dir", type=str, default="data", required=False, help="Data include videos"
        )
        self.parser.add_argument(
            "--device", type=str, default="cuda:0", required=False, help="Cuda device"
        )
        self.parser.add_argument(
            "--subset_list", type=str, required=False, help="Subset List .txt"
        )