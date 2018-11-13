import sys
from pathlib import Path
from importlib import import_module

def build_model(model_path):
    path = Path(model_path)
    experiment_dir = path.parent
    sys.path.append(str(experiment_dir.resolve()))
    module = import_module(str(path.stem))

    model = module.Model()
    return model

def build_config(config_path):
    path = Path(config_path)
    experiment_dir = path.parent
    sys.path.append(str(experiment_dir.resolve()))
    module = import_module(str(path.stem))

    config = module.Config()
    return config
