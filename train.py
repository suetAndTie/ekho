import os
import argparse
import torch
import torch.optim
from dataset.datasource import TextDataSource, MelSpecDataSource, LinearSpecDataSource
from dataset.dataset import FileDataset, PyTorchDataset
import util.util as ut
from config import config



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/processed', help="Directory containing the dataset")
    parser.add_argument('--experiment_dir', default='experiment/example', help="Directory containing model and configs")
    parser.add_argument('--resume', default=None,
                        help="Optional, name of the file in --experiment_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()
    model_path = os.path.join(args.experiment_dir, 'model.py')
    config_path = os.path.join(args.experiment_dir, 'config.py')
    if not os.path.isfile(model_path):
        raise IOError("No model.py found at {}".format(model_path))
    if not os.path.isfile(config_path):
        raise IOError("No config.py found at {}".format(config_path))

    # Load dataset
    text = FileDataset(TextDataSource(args.data_dir))
    mel = FileDataset(MelSpecDataSource(args.data_dir))
    linear = FileDataset(LinearSpecDataSource(args.data_dir))
    dataset = PyTorchDataset(text, mel, linear)

    # TODO: Sampler
    # TODO: Dataloader
    # TODO: Trainer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ut.build_model(model_path).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    try:
        pass
        # TODO: train
    except KeyboardInterrupt:
        pass
        # TODO: save checkpoint
