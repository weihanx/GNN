
import torch_utils
from math import ceil
from torch_geometric.nn.norm import LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn.conv import DirGNNConv
from torch_geometric.nn.conv import MessagePassing
import torch

from torch.nn import Dropout
from torch_geometric.nn import global_mean_pool


class HeteroGNN(torch.nn.Module):
    # hidden channels is the output channels
    def __init__(self, num_input_features, hidden_channels, out_channels, normalize=False):
        super(HeteroGNN, self).__init__()
        self.dropout = Dropout(0.15)
        # Define a specific convolution layer instance with predefined in_channels and out_channels
        gcn_conv = GCNConv(num_input_features, hidden_channels, normalize)  # Adjust in_channels and out_channels as needed
        # sage_conv = SAGEConv(num_input_features, hidden_channels, normalize=normalize)
        self.conv_1 = HeteroConv({
            ('note', 'forward', 'note'): DirGNNConv(gcn_conv),
            ('note', 'onset', 'note'): DirGNNConv(gcn_conv),
            ('note', 'sustain', 'note'): DirGNNConv(gcn_conv),
            ('note', 'rest', 'note'): DirGNNConv(gcn_conv),
        }, aggr='sum')
        
        self.norm_1 = LayerNorm(hidden_channels)
        
        # Define another specific convolution layer for the second layer or reuse the first one as per your design
        gcn_conv_2 = GCNConv(hidden_channels, hidden_channels, normalize) # Adjust in_channels and out_channels as needed
        # sage_conv = SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
        self.conv_2 = HeteroConv({
            ('note', 'forward', 'note'): DirGNNConv(gcn_conv_2),
            ('note', 'onset', 'note'): DirGNNConv(gcn_conv_2),
            ('note', 'sustain', 'note'): DirGNNConv(gcn_conv_2),
            ('note', 'rest', 'note'): DirGNNConv(gcn_conv_2),
        }, aggr='sum')
        self.norm_2 = LayerNorm(hidden_channels)

    def forward(self, x_dict, edge_index_dict, flatten=False):
        # deal with missing edge
        for edge_type in edge_index_dict:
            if edge_index_dict[edge_type].numel() == 0:
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long).to("cuda")
        x_dict = self.conv_1(x_dict, edge_index_dict)
        x_dict['note'] = self.norm_1(x_dict['note'])
        x_dict['note'] = self.dropout(x_dict['note'])
        x_dict = self.conv_2(x_dict, edge_index_dict)  # should return embedding, remove fully connected
        x_dict['note'] = self.norm_2(x_dict['note'])

        return x_dict['note']

