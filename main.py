import argparse
import os
import logging
from train import Train
from eval import Eval


def define_logging(args, logger):
    handler = logging.FileHandler(os.path.join(args.save_path, 'logging.log'))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General Arguments
    parser.add_argument('-id', '--device_id', default=None, type=str,
                        help='Set the device (GPU id).')
    parser.add_argument('-sp', '--saved_path', type=str,
                        help='Path to save results', default='result')
    parser.add_argument('-se', '--seed', default=42, type=int,
                        help='Global random seed')
    parser.add_argument('-ta', '--task_type', default='classification', type=str,
                        choices=['classification', 'regression'],
                        help='The task type')
    parser.add_argument('-nj', '--n_jobs', default=1, type=int,
                        help='Number of workers')
    parser.add_argument('-st', '--split', default='random', type=str,
                        choices=['random', 'scaffold', 'customized'],
                        help='The split type')
    parser.add_argument('-mo', '--mode', default='train', type=str,
                        choices=['train', 'predict'])
    parser.add_argument('-ra', '--ratio', default='0.8 0.1 0.1', type=str,
                        help='The split ratio of train_valid_test split')
    # Training Arguments
    parser.add_argument('-al', '--alpha', default=0.5, type=int,
                        help='The alpha coefficient used in loss function')
    parser.add_argument('-ep', '--epoch', default=200, type=int,
                        help='Number of epochs for training')
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float,
                        help='learning rate to use')
    parser.add_argument('-bs', '--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('-wd', '--weight_decay', default=0.0, type=float,
                        help='weight decay to use')
    parser.add_argument('-pe', '--print_every', default=1, type=int,
                        help='Print results')
    parser.add_argument('-cm', '--check_metric', default='loss', type=str,
                        choices=['loss', 'AUC', 'AUPR', 'MSE', 'R2', 'Comprehensive'],
                        help='The metric used to determine best model')
    # Model Arguments
    parser.add_argument('-hs', '--hidden_size', default=128, type=int,
                        help='The dimension of hidden tensor in the model')
    parser.add_argument('-nl', '--num_layer', default=2, type=int,
                        help='The number of EGNN layer')
    parser.add_argument('-dp', '--dropout', default=0.0, type=float,
                        help='The rate of dropout layer')

    args = parser.parse_args()
    args.data_path = '../data'
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    task_list = os.listdir(path=os.path.join(args.data_path, 'target'))
    for count, task in enumerate(task_list):
        args.task = task.split('.csv')[0]
        args.save_path = os.path.join('../result', args.task,
                                      args.saved_path + '_' + str(args.seed))
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        define_logging(args, logger)
        logger.info(f'current target: {args.task}, {count}/{len(task_list)} targets')

        frac = args.ratio.split(' ')
        args.frac_train, args.frac_val, args.frac_test = float(frac[0]), float(frac[1]), float(frac[2])

        if args.mode == 'train':
            Train(args, logger)
        else:
            Eval(args, logger)

        logger.handlers.clear()


if __name__ == '__main__':
    main()
