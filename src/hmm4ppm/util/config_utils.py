import os
import yaml

def read_config(config_path: os.PathLike):

    with open(config_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return cfg