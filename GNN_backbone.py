from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.nn.conv import DirGNNConv
import torch

from torch.nn import Dropout

from config import DEVICE


class HeteroGNN(torch.nn.Module):
    def __init__(self, num_input_features, hidden_channels, out_channels, normalize=False):
        super(HeteroGNN, self).__init__()
        self.dropout = Dropout(0.15)
        gcn_conv = GCNConv(num_input_features, hidden_channels, normalize)
        self.conv_1 = HeteroConv({
            ('note', 'forward', 'note'): gcn_conv,
            ('note', 'onset', 'note'): gcn_conv,
            ('note', 'sustain', 'note'): gcn_conv,
            ('note', 'rest', 'note'): gcn_conv,
        }, aggr='sum')
        self.norm_1 = LayerNorm(hidden_channels)

        gcn_conv_2 = GCNConv(hidden_channels, hidden_channels, normalize)
        self.conv_2 = HeteroConv({
            ('note', 'forward', 'note'): gcn_conv_2,
            ('note', 'onset', 'note'): gcn_conv_2,
            ('note', 'sustain', 'note'): gcn_conv_2,
            ('note', 'rest', 'note'): gcn_conv_2,
        }, aggr='sum')
        self.norm_2 = LayerNorm(hidden_channels)

    def forward(self, x, edge_index_dict, attribute_dict, flatten=False):
        x = self.conv_1(x, edge_index_dict, attribute_dict)
        x['note'] = self.norm_1(x['note'])
        x['note'] = self.dropout(x['note'])
        x = self.conv_2(x, edge_index_dict, attribute_dict)
        x['note'] = self.norm_2(x['note'])

        return x['note']

