from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.nn.conv import DirGNNConv
import torch
from config import SHARE_BACKBONE_WEIGHTS
from torch.nn import Dropout


class HeteroGNN(torch.nn.Module):
    def __init__(self, num_input_features, hidden_channels, normalize=False, share_weights=SHARE_BACKBONE_WEIGHTS):
        super(HeteroGNN, self).__init__()
        self.dropout = Dropout(0.15)

        edge_types = [
            ('note', 'forward', 'note'),
            ('note', 'onset', 'note'),
            ('note', 'sustain', 'note'),
            ('note', 'rest', 'note')
        ]
        gcn_conv = GCNConv(num_input_features, hidden_channels, normalize)
        self.conv_1 = HeteroConv({
            edge_type: GCNConv(num_input_features, hidden_channels, normalize)
            if not share_weights
            else gcn_conv
            for edge_type in edge_types
        })
        self.norm_1 = LayerNorm(hidden_channels)

        gcn_conv_2 = GCNConv(hidden_channels, hidden_channels, normalize)
        self.conv_2 = HeteroConv({
            edge_type: GCNConv(hidden_channels, hidden_channels, normalize)
            if not share_weights
            else gcn_conv_2
            for edge_type in edge_types
        }, aggr='sum')
        self.norm_2 = LayerNorm(hidden_channels)

    def forward(self, x, edge_index_dict, attribute_dict):
        x = self.conv_1(x, edge_index_dict, attribute_dict)
        x['note'] = self.norm_1(x['note'])
        x['note'] = self.dropout(x['note'])
        x = self.conv_2(x, edge_index_dict, attribute_dict)
        x['note'] = self.norm_2(x['note'])

        return x['note']
