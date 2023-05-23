import torch
import numpy as np
import pandas as pd


def conformer_attn(data_loader, model, device):
    model.eval()
    smiles, labels, preds, attns = [], [], [], []
    all_pred = []
    for batch_id, batch_data in enumerate(data_loader):
        smi, graphs, label = batch_data
        graphs = graphs.to(device)
        label = torch.tensor(label).float().to(device)
        node_feat = graphs.ndata.pop('node_feat')
        edge_feat = graphs.edata.pop('edge_feat')
        coord_feat = graphs.ndata.pop('coord_feat')
        logits, _, attn = model(graphs, node_feat,
                                edge_feat, coord_feat)
        smiles.extend(smi)
        labels.extend(label.detach().cpu().numpy().tolist())
        preds.extend(logits.squeeze().detach().cpu().numpy().tolist())
        attns.extend(attn.squeeze().detach().cpu().numpy().tolist())
        all_pred.extend(_.squeeze().detach().cpu().numpy().tolist())
    return pd.DataFrame(np.hstack((np.array([smiles, labels, preds]).T,
                                   np.array(attns))),
                        columns=['SMILES', 'Label',
                                 'Predict'] +
                                [f'Attn_{i}' for i in range(10)]), all_pred


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

