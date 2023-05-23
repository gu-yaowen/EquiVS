import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dgllife.utils import EarlyStopping
from preprocessing import load_dataset, dataset_split
from model import Model
from utils import collate, get_loss, get_metrics, set_seed, \
    plot_training_curve, checkpoint


def train_one_epoch(args, epoch, device, model,
                    data_loader, score, optimizer, criterion):
    model.train()
    labels, preds, losses = [], [], []
    for batch_id, batch_data in enumerate(data_loader):
        smiles, graphs, label = batch_data
        graphs = graphs.to(device)
        label = torch.tensor(label).float().to(device)
        node_feat = graphs.ndata.pop('node_feat')
        edge_feat = graphs.edata.pop('edge_feat')
        coord_feat = graphs.ndata.pop('coord_feat')
        # logits: predicted results, ins_pred: predicted results for  conformers, attn: attn coef for conformers
        logits, ins_pred, attn = model(graphs, node_feat, edge_feat, coord_feat)
        logits = logits.squeeze()

        loss = criterion(logits.squeeze(), ins_pred.squeeze(dim=-1), label)
        if args.task == 'classification':
            logits = torch.sigmoid(logits)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        labels.extend(label.detach().cpu().numpy().tolist())
        if logits.shape == 0:
            preds.extend([logits.item()])
        else:
            try:
                preds.extend(logits.detach().cpu().numpy().tolist())
            except:
                print(logits)
                preds.extend([logits.item()])

    score['loss'] += [np.mean(losses)]
    score['epoch'] += [epoch]
    metrics = get_metrics(labels, preds, args.metric)
    score['metric_1'] += [metrics[0]]
    score['metric_2'] += [metrics[1]]


def eval_one_epoch(args, device, model, data_loader, score):
    model.eval()
    smiles_list, label_list, pred_list = [], [], []
    for batch_id, batch_data in enumerate(data_loader):
        smiles, graphs, label = batch_data
        graphs = graphs.to(device)
        label = torch.tensor(label).float().to(device)
        node_feat = graphs.ndata.pop('node_feat')
        edge_feat = graphs.edata.pop('edge_feat')
        coord_feat = graphs.ndata.pop('coord_feat')
        logits, ins_pred, attn = model(graphs, node_feat, edge_feat, coord_feat)
        logits = logits.squeeze()
        if args.task == 'classification':
            logits = torch.sigmoid(logits)
        smiles_list.extend(smiles)
        label_list.extend(label.detach().cpu().numpy().tolist())
        if logits.shape == 0:
            pred_list.extend([logits.item()])
        else:
            try:
                pred_list.extend(logits.detach().cpu().numpy().tolist())
            except:
                print(logits)
                pred_list.extend([logits.item()])

    metrics = get_metrics(label_list, pred_list, args.metric)
    score['val_metric_1'] += [metrics[0]]
    score['val_metric_2'] += [metrics[1]]
    return [smiles_list, pred_list, label_list]


def Train(args, logger):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

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

    train, val, test = dataset_split(dataset, args)
    logger.info(f"Train dataset: {len(train)}; Val dataset: {len(val)}; Test dataset: {len(test)}")
    train_loader = DataLoader(train, batch_size=args.batch_size, collate_fn=collate,
                              shuffle=True, drop_last=False)
    if len(val) >= 2:
        val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate,
                                shuffle=False, drop_last=False)
    else:
        val_loader = None
    test_loader = DataLoader(test, batch_size=args.batch_size, collate_fn=collate,
                             shuffle=False, drop_last=False)

    model = Model(in_size=dataset[0][1].ndata['node_feat'].shape[-1],
                  hidden_size=args.hidden_size,
                  edge_feat_size=dataset[0][1].edata['edge_feat'].shape[-1],
                  num_layer=args.num_layer,
                  dropout=args.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    criterion = get_loss(args)
    stopper = EarlyStopping(patience=30,
                            filename=os.path.join(args.save_path, 'model.pkl'),
                            metric='roc_auc_score')

    score = {'loss': [], 'epoch': [],
             'metric_1': [], 'metric_2': [],
             'val_metric_1': [], 'val_metric_2': []}
    for epoch in range(args.epoch):
        train_one_epoch(args, epoch, device, model,
                        train_loader, score, optimizer, criterion)
        if epoch % args.print_every == 0:
            if val_loader:
                _ = eval_one_epoch(args, device, model, val_loader, score)
            else:
                _ = eval_one_epoch(args, device, model, test_loader, score)

            logger.info("Epoch {}/{}: loss: {:.3f}, train {}: {:.3f}, {}: {:.3f} "
                        " val {}: {:.3f}, {}: {:.3f}".format(
                epoch + 1, args.epoch, score['loss'][-1],
                args.metric[0], score['metric_1'][-1],
                args.metric[1], score['metric_2'][-1],
                args.metric[0], score['val_metric_1'][-1],
                args.metric[1], score['val_metric_2'][-1]))
        m = checkpoint(args, model, score, args.metric)
        if m:
            best_model = m
            early_stop = stopper.step(score['val_metric_1'][-1], model)
            # if early_stop:
            #     break
    # stopper.load_checkpoint(model)
    plot_training_curve(args, score)
    test_result = eval_one_epoch(args, device, best_model, test_loader, score)
    logger.info("Test {}: {:.3f}, {}: {:.3f}".format(
        args.metric[0], score['val_metric_1'][-1],
        args.metric[1], score['val_metric_2'][-1]))
    pd.DataFrame(np.array(test_result).T,
                 columns=['SMILES', 'Predict', 'Label']
                 ).to_csv(os.path.join(args.save_path, 'predict.csv'), index=False)
    logger.info(f'Training done, file saved in {args.save_path}')
