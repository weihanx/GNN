"""
5 features: note_in_chord, note_grace_note, note_measure_num, note_pitch, note_duration
4 edge type: forward, onset, sustain, rest
Model Structure: 
"""

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
from GNN_backbone import HeteroGNN
from GNN_cluster import GNN_Cluster
"""
Model Definition
# # """



"""
Differntial Pool on cluster
"""


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=111, embedding_dim = 32, hidden_dim = 200, num_classes=15, device="cuda"):
        super(GroupMat, self).__init__()
        self.device = device

        self.embed = torch.nn.Linear(num_feature, embedding_dim)
        # self.gnn1_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden_channel, output_channel
        # self.custom_weight_init(self.gnn1_embed)
        self.gnn2_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden, output
        # self.custom_weight_init(self.gnn2_embed)

        # self.linear = torch.nn.Linear(hidden_dim, 1).to(device) 

        # torch.nn.init.xavier_uniform_(self.linear.weight) # avoid all zero or all
        self.gnn_cluster1 = GNN_Cluster(embedding_dim, hidden_dim, num_classes)
        self.gnn_cluster2 = GNN_Cluster(embedding_dim, hidden_dim, num_classes)
        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def init_weights(self, m):
        if isinstance(m, SAGEConv):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def custom_weight_init(self, model):
        for m in model.modules():
            if isinstance(m, HeteroConv):
                for _, conv in m.convs.items():  # m.convs is the dictionary holding the convolutions
                    self.init_weights(conv)

    # def coord_to_adj(self, edge_index_dict, num_nodes):
    #     """
    #     change to adj matrix for pooling
    #     """
    #     adj = {}
    #     # print(f"num nodes = {num_nodes}")  # should be the same as the x shape
    #     for edge, mat in edge_index_dict.items():
    #         adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
    #         for source, target in mat.t():
    #             adj_matrix[source, target] = 1
    #             adj[edge] = adj_matrix
    #     return adj


    # def adj_to_coord(self, adj_matrix):
    #     """
    #     change back too coordinates for gnn model
    #     """
    #     coords = {}
    #     for edge, adj in adj_matrix.items():
    #         edge_index = adj.nonzero().t()
    #         coords[edge] = edge_index

    #     return coords

    # def euclidean_distance_matrix(self, x):
    #     num_node, num_feat = x.shape
    #     dist_list = []
    #     # print(f"num_ed node = {num_node}")
    #     for i in range(num_node):
    #         for j in range(i, num_node):
    #             # print(f"comb = {i}, comb = {j}")
    #             diff = x[i] - x[j]
    #             res = diff * diff
    #             dist_list.append(res)
    #     distance_mat = torch.stack(dist_list, dim=0) # add new dimension
    #     # print(f"shape of distance mat = {distance_mat.shape}")
        
    #     return distance_mat

    def sigmoid_and_threshold(self, tensor, threshold=0.5):
        # Apply sigmoid function to each element
        sigmoid_tensor = torch.sigmoid(tensor)
        # Convert to binary tensor based on the threshold
        binary_tensor = (sigmoid_tensor > threshold).int()

        return binary_tensor

    def forward(self, x, edge_index_dict, batch):
        """
        Schen: should replace s
        """
        # use two layers for now.

        num_nodes = x['note'].shape[0]
        # print(f"num nodes = {num_nodes}")

        x['note'] = x['note'].float()
        x['note'] = self.embed(x['note'])

        x,edge_dict_1, S_1 =  self.gnn_cluster1(x, edge_index_dict)
        
        #--------------Layer 2------------------

        x, edge_dict_2, S_2 = self.gnn_cluster2(x, edge_dict_1)

        z_2 = self.gnn2_embed(x, edge_dict_2).float()
        # print(f"z2  = {z_2.shape}")
        # x_2 = global_mean_pool(z_2, batch)

        # x = self.classifier(x_2)
        # x = F.log_softmax(x, dim=-1)
        # print(f"shape of s1 = {S_1.shape}, shape of s2 = {S_2.shape}")
        return z_2, S_1, S_2


