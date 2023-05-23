import torch
import torch.nn as nn
import dgl.function as fn
from dgl.readout import sum_nodes, softmax_nodes
from dgl.nn.pytorch import GraphConv


class EGNNConv(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.
    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, dp=0.):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dp)
        self.bn = nn.BatchNorm1d(out_size)

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']

        return {'msg_x': msg_x, 'msg_h': msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        r"""
        Description
        -----------
        Compute EGNN layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        node_feat : torch.Tensor
            The input feature of shape :math:`(N, h_n)`. :math:`N` is the number of
            nodes, and :math:`h_n` must be the same as in_size.
        coord_feat : torch.Tensor
            The coordinate feature of shape :math:`(N, h_x)`. :math:`N` is the
            number of nodes, and :math:`h_x` can be any positive integer.
        edge_feat : torch.Tensor, optional
            The edge feature of shape :math:`(M, h_e)`. :math:`M` is the number of
            edges, and :math:`h_e` must be the same as edge_feat_size.

        Returns
        -------
        node_feat_out : torch.Tensor
            The output node feature of shape :math:`(N, h_n')` where :math:`h_n'`
            is the same as out_size.
        coord_feat_out: torch.Tensor
            The output coordinate feature of shape :math:`(N, h_x)` where :math:`h_x`
            is the same as the input coordinate feature dimension.
        """
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            graph.edata['x_diff'] = graph.edata['x_diff'] / (graph.edata['radial'].sqrt() + 1e-30)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            h = self.dropout(h)

            h = self.bn(h)
            x = coord_feat + x_neigh

            return h, x


class GlobalAttentionPooling(nn.Module):
    r"""Global Attention Pooling from `Gated Graph Sequence Neural Networks

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.
    """

    def __init__(self, in_size, hidden_size):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = nn.Linear(in_size, 1, bias=False)
        self.feat_nn = nn.Linear(in_size, hidden_size)

    def forward(self, graph, feat, get_attention=False):
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input node feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.
        get_attention : bool, optional
            Whether to return the attention values from gate_nn. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        torch.Tensor, optional
            The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
            nodes in the graph. This is returned only when :attr:`get_attention` is ``True``.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat)

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            if get_attention:
                return readout, gate
            else:
                return readout


class InstanceLayer(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(InstanceLayer, self).__init__()

        self.project = nn.Sequential(nn.Linear(in_size, hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, ins_emb, attn, SimSiam=False):
        predicts = self.project(ins_emb)
        attn_pred = attn * predicts
        # topk_out = torch.mean(attn_pred.topk(k=self.topk, dim=1)[0], dim=1)
        return predicts


class BagLayer(nn.Module):
    r"""
    Description
    -----------
    Attention based multi-instance learning layer.

    Parameters
    ----------
    in_size : int
        Input feature size.
    hidden_size : int
        Hidden feature size.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, in_size, hidden_size):
        super(BagLayer, self).__init__()

        self.project = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.Tanh(),
                                     nn.Linear(hidden_size, 1, bias=False))
        self.linear = nn.Linear(in_size, hidden_size)

    def forward(self, ins_emb, output_attn=False):
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        node_feat : torch.Tensor
            The input node feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.
        output_attn : bool, optional
            Whether to return the attention values from gate_nn. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        torch.Tensor, optional
            The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
            nodes in the graph. This is returned only when :attr:`get_attention` is ``True``.
        """
        ins_emb = self.linear(ins_emb)
        attn = torch.softmax(self.project(ins_emb), dim=1)
        bag_emb = (ins_emb * attn).sum(dim=1)
        if output_attn:
            return bag_emb, attn
        else:
            return bag_emb


class MLP(nn.Module):
    """MLP for predicting drug-disease associations.
    """

    def __init__(self, in_size, dropout=0.):
        super(MLP, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.linear1 = nn.Linear(in_size, in_size, bias=True)
        self.linear2 = nn.Linear(in_size, 1, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, bag_emb):
        h = self.linear1(bag_emb)
        if self.dropout:
            h = self.dropout(h)
        outputs = self.linear2(h)
        return outputs


class MLP_2(nn.Module):
    """MLP for predicting drug-disease associations.
    """

    def __init__(self, in_size, dropout=0.):
        super(MLP_2, self).__init__()
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.linear = nn.Linear(in_size, 1, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, bag_emb):
        outputs = self.linear(bag_emb)
        return outputs


class EGNNConv_2(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, dp=0.):
        super(EGNNConv_2, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()
        self.dropout = nn.Dropout(dp)

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size)
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [edges.src['h'], edges.dst['h'], edges.data['radial'], edges.data['a']],
                dim=-1
            )
        else:
            f = torch.cat([edges.src['h'], edges.dst['h'], edges.data['radial']], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data['x_diff']

        return {'msg_x': msg_x, 'msg_h': msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = node_feat
            # coordinate feature
            graph.ndata['x'] = coord_feat
            # edge feature
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata['a'] = edge_feat
            # get coordinate diff & radial features
            graph.apply_edges(fn.u_sub_v('x', 'x', 'x_diff'))
            graph.edata['radial'] = graph.edata['x_diff'].square().sum(dim=1).unsqueeze(-1)
            # normalize coordinate difference
            graph.edata['x_diff'] = graph.edata['x_diff'] / (graph.edata['radial'].sqrt() + 1e-30)
            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e('msg_x', 'm'), fn.mean('m', 'x_neigh'))
            graph.update_all(fn.copy_e('msg_h', 'm'), fn.sum('m', 'h_neigh'))

            h_neigh, x_neigh = graph.ndata['h_neigh'], graph.ndata['x_neigh']

            h = self.node_mlp(
                torch.cat([node_feat, h_neigh], dim=-1)
            )
            h = self.dropout(h)
            x = coord_feat + x_neigh
            return h, x


class GCNLayer(nn.Module):
    r"""Single GCN layer from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Parameters
    ----------
    in_feats : int
        Number of input node features.
    out_feats : int
        Number of output node features.
    gnn_norm : str
        The message passing normalizer, which can be `'right'`, `'both'` or `'none'`. The
        `'right'` normalizer divides the aggregated messages by each node's in-degree.
        The `'both'` normalizer corresponds to the symmetric adjacency normalization in
        the original GCN paper. The `'none'` normalizer simply sums the messages.
        Default to be 'none'.
    activation : activation function
        Default to be None.
    residual : bool
        Whether to use residual connection, default to be True.
    batchnorm : bool
        Whether to use batch normalization on the output,
        default to be True.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """

    def __init__(self, in_feats, out_feats, gnn_norm='none', activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=gnn_norm, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match in_feats in initialization

        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output node feature size, which must match out_feats in initialization
        """
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats
