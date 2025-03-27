import os
import argparse 
import importlib.util
import warnings

from self_play.self_play import SelfPlay

warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path):
    '''
    Load the config file from the given path
    '''
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Select config file using --config")
    parser.add_argument("--config", type=str, default="baselines/purposed_model+hybrid.py", help="Path to the config file")
    args = parser.parse_args()

    # load the config file
    config_file_path = os.path.join('configs', args.config)
    config = load_config(config_file_path)

    # run the self play training
    self_play = SelfPlay(config = config)
    self_play.train()