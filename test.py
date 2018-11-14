import argparse
import lib.util



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
    parser.add_argument('--experiment_dir', default='model/test', help="Directory containing model and configs")
    parser.add_argument('--resume', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    args = parser.parse_args()
    model_path = os.path.join(args.experiment_dir, 'model.py')
    config_path = os.path.join(args.experiment_dir, 'config.py')
    assert os.path.isfile(model_path), "No model.py found at {}".format(model_path)
    assert os.path.isfile(config_path), "No config.py found at {}".format(config_path)
