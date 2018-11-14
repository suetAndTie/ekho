import os
import argparse
import torch
import torch.optim
import util.util as ut    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/LibraSpeech', help="Directory containing the dataset")
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

    config = ut.build_config(config_path)
    # TODO: Dataset
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
