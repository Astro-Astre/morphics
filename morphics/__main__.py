# __main__.py
import argparse
from morphics.train import main, get_data_config, init_rand_seed

if __name__ == "__main__":
    init_rand_seed(1926)
    data_config = get_data_config()
    parser = argparse.ArgumentParser(description='Galaxy Classification Training')
    parser.add_argument('--train_file', type=str, default=data_config.train_file,
                        help='Path to the training data annotations file')
    parser.add_argument('--valid_file', type=str, default=data_config.valid_file,
                        help='Path to the validation data annotations file')
    parser.add_argument('--save_dir', type=str, default=data_config.save_dir,
                        help='Directory to save logs, checkpoints, and trained models')
    parser.add_argument('--epochs', type=int, default=data_config.epochs,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=data_config.batch_size,
                        help='Batch size for training and validation')
    parser.add_argument('--WORKERS', type=int, default=data_config.WORKERS,
                        help='Number of workers for data loading')
    parser.add_argument('--optimizer', type=str, default=data_config.optimizer,
                        help='Optimizer for training')
    parser.add_argument('--optimizer_parm', type=dict, default=data_config.optimizer_parm,
                        help='Optimizer parameters')
    parser.add_argument('--transfer', type=callable, default=data_config.transfer,
                        help='Transforms to apply to the input data')

    args = parser.parse_args()
    main(args)
