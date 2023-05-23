import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from preprocessing import load_dataset, dataset_split
from train import eval_one_epoch
from model import Model
from utils import collate, set_seed, get_loss


def Eval(args, logger):
    if args.device_id != 'cpu':
        logger.info('Training on GPU')
        device = torch.device('cuda:{}'.format(args.device_id))
    else:
        logger.info('Training on CPU')
        device = torch.device('cpu')

    argsDict = args.__dict__
    with open(os.path.join(args.save_path, 'setting.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
    logger.info("Configuration: {}".format(args))

    set_seed(args.seed)
    dataset = load_dataset(args)
    criterion = get_loss(args)

    # if args.k == 0:
    train, val, test = dataset_split(dataset, args)
    logger.info(f"Test dataset: {len(test)}")
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate,
                             shuffle=False, drop_last=False)
    model = Model(in_size=dataset[0][1].ndata['node_feat'].shape[-1],
                  hidden_size=args.hidden_size,
                  edge_feat_size=dataset[0][1].edata['edge_feat'].shape[-1],
                  num_layer=args.num_layer,
                  dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model.pkl')))

    score = {'loss': [], 'epoch': [],
             'metric_1': [], 'metric_2': [],
             'val_metric_1': [], 'val_metric_2': []}

    if args.mode == 'predict':
        test_result = eval_one_epoch(args, device, model, test_loader, score)
        logger.info("Test {}: {:.3f}, {}: {:.3f}".format(
            args.metric[0], score['val_metric_1'][-1],
            args.metric[1], score['val_metric_2'][-1]))
        pd.DataFrame(np.array(test_result).T,
                     columns=['SMILES', 'Predict', 'Label']).to_csv(os.path.join(
            args.save_path, 'predict.csv'), index=False)

    logger.info(f'Training done, file saved in {args.save_path}')
    return
