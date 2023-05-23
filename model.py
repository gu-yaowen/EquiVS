from layer import *
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class Model(nn.Module):
    def __init__(self, in_size, hidden_size, edge_feat_size, num_layer, dropout=0.):
        super(Model, self).__init__()
        self.num_layer = num_layer
        self.egnn = nn.ModuleDict()
        self.Pooling = nn.ModuleDict()
        self.gcn = GraphConv(in_size, hidden_size)
        self.Pooling[f'Pooling_GCN'] = WeightedSumAndMax(hidden_size)
        for l in range(num_layer):
            self.egnn[f'EGNNLayer_{l}'] = EGNNConv(hidden_size, hidden_size,
                                                   hidden_size, edge_feat_size, dropout)
            self.Pooling[f'Pooling_{l}'] = WeightedSumAndMax(hidden_size)
        self.BagLayer = BagLayer(hidden_size * (num_layer+1) * 2, hidden_size * 2)
        self.InstanceLayer = InstanceLayer(hidden_size * (num_layer+1) * 2, hidden_size * 2)
        self.bag_predict = MLP(hidden_size * 4, dropout=dropout)

    def forward(self, graphs, node_feats, edge_feats, coord_feats, num_conf=10):
        # Split the batch graphs based on coordinate features 
        # which represent different conformers
        graph_skip_emb = {}
        node_feats = self.gcn(graphs, node_feats)
        gcn_emb = self.Pooling[f'Pooling_GCN'](graphs, node_feats)
        for idx in range(num_conf):
            coord_feat_ = coord_feats[:, idx * 3: (idx + 1) * 3]
            node_feats_ = node_feats

            graph_skip_emb[f'Conf_{idx}'] = [gcn_emb]

            for idx_layer in range(len(self.egnn)):
                node_feats_, coord_feat_ = self.egnn[f'EGNNLayer_{idx_layer}'](graphs, node_feats_,
                                                                               coord_feat_, edge_feats)
                graph_skip_emb[f'Conf_{idx}'].append(self.Pooling[f'Pooling_{idx_layer}'](graphs, node_feats_))
            graph_skip_emb[f'Conf_{idx}'] = torch.concat(graph_skip_emb[f'Conf_{idx}'], dim=-1)

        graph_emb = torch.stack(list(graph_skip_emb.values()), dim=1)
        bag_emb, attn = self.BagLayer(graph_emb, output_attn=True)
        bag_predict = self.bag_predict(torch.concat([bag_emb, gcn_emb], dim=-1))
        all_ins_pred = self.InstanceLayer(graph_emb, attn)
        return bag_predict, all_ins_pred, attn
