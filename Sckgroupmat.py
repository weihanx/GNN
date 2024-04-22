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
"""
Model Definition
# # """


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


"""
Differntial Pool on cluster
"""


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=111, embedding_dim = 32, hidden_dim = 200, num_classes=15, device="cuda"):
        super(GroupMat, self).__init__()
        self.device = device

        self.embed = torch.nn.Linear(num_feature, embedding_dim)
        self.gnn1_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden_channel, output_channel
        self.custom_weight_init(self.gnn1_embed)
        self.gnn2_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden, output
        self.custom_weight_init(self.gnn2_embed)

        self.linear = torch.nn.Linear(hidden_dim, 1).to(device) 

        torch.nn.init.xavier_uniform_(self.linear.weight) # avoid all zero or all

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

    def coord_to_adj(self, edge_index_dict, num_nodes):
        """
        change to adj matrix for pooling
        """
        adj = {}
        # print(f"num nodes = {num_nodes}")  # should be the same as the x shape
        for edge, mat in edge_index_dict.items():
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
            for source, target in mat.t():
                adj_matrix[source, target] = 1
                adj[edge] = adj_matrix
        return adj


    def adj_to_coord(self, adj_matrix):
        """
        change back too coordinates for gnn model
        """
        coords = {}
        for edge, adj in adj_matrix.items():
            edge_index = adj.nonzero().t()
            coords[edge] = edge_index

        return coords

    def euclidean_distance_matrix(self, x):
        num_node, num_feat = x.shape
        dist_list = []
        # print(f"num_ed node = {num_node}")
        for i in range(num_node):
            for j in range(i, num_node):
                # print(f"comb = {i}, comb = {j}")
                diff = x[i] - x[j]
                res = diff * diff
                dist_list.append(res)
        distance_mat = torch.stack(dist_list, dim=0) # add new dimension
        # print(f"shape of distance mat = {distance_mat.shape}")
        
        return distance_mat

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

        z_0 = self.gnn1_embed(x, edge_index_dict).float()  # 70, 16

        # print(f"z_0 = {z_0.shape}") # contain nan
        for edge_type, _ in edge_index_dict.items():
            if edge_index_dict[edge_type].numel() == 0:
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long).to(self.device)

        M = self.euclidean_distance_matrix(z_0) # (51*25, 200)
        # print(f"before linear = {M}")
        M_G = self.linear(M).to(self.device).float()

        M_G = torch.sigmoid(M_G) # shape is (51*25, 1)
        M_G = M_G.squeeze()
        # print(f"shape of M_G = {M_G.shape}")
        # resize to upper triangular matrix
        tri_M_G = torch.zeros((num_nodes, num_nodes),device = M_G.device)
        row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=0)
        tri_M_G[row_indices, col_indices] = M_G
        tri_M_G.requires_grad_(True)
        # ---- use package -- # 
        tri_M_G = tri_M_G.unsqueeze(0)  # only need for build-in function
        # print(f" S1: M_G = {new_M_G}")
        conv_sqrt = torch_utils.MPA_Lya.apply(tri_M_G)
        S_1 = conv_sqrt.squeeze().float()  

        adj = self.coord_to_adj(edge_index_dict, num_nodes)
        adj_1 = {}
        x['note'] = torch.matmul(S_1, x['note'])
        # print(f"pooling matrix = {S}")
        num_nodes = x['note'].shape[0]
        for edge_type, A in adj.items():
            result = torch.matmul(torch.matmul(S_1, A),S_1).float()
            adj_1[edge_type] = result

        edge_dict_1 = self.adj_to_coord(adj_1)

        #--------------Layer 2------------------#
        z_1 = self.gnn2_embed(x, edge_dict_1).float()
        # print(f"shape of z_1 = {z_1.shape}")
        for edge_type, _ in edge_dict_1.items():
            if edge_dict_1[edge_type].numel() == 0:
                edge_dict_1[edge_type] = torch.empty((2, 0), dtype=torch.long).to(self.device)
        M = self.euclidean_distance_matrix(z_1) # (51*25, 200)

        M_G = self.linear(M).to(self.device).float()

        M_G = torch.sigmoid(M_G) # shape is (51*25, 1)
        M_G = M_G.squeeze()
        # print(f"shape of M_G 2= {M_G.shape}")

        tri_M_G = torch.zeros((num_nodes, num_nodes),device = M_G.device)
        row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=0)
        tri_M_G[row_indices, col_indices] = M_G
        tri_M_G.requires_grad_(True)
        # ---- use package -- # 
        tri_M_G = tri_M_G.unsqueeze(0)  # only need for build-in function
        # print(f" S1: M_G = {new_M_G}")
        conv_sqrt = torch_utils.MPA_Lya.apply(tri_M_G)
        S_2 = conv_sqrt.squeeze().float()

        adj = self.coord_to_adj(edge_dict_1, num_nodes)
        adj_2 = {}
        x['note'] = torch.matmul(S_2, x['note'])
        # # print(f"pooling matrix = {S}")
        num_nodes = x['note'].shape[0]
        for edge_type, A in adj.items():
            result = torch.matmul(torch.matmul(S_2, A), S_2).float()
            adj_2[edge_type] = result

        edge_dict_2 = self.adj_to_coord(adj_2)
        z_2 = self.gnn2_embed(x, edge_dict_2).float()
        # print(f"z2  = {z_2.shape}")
        x_2 = global_mean_pool(z_2, batch)

        x = self.classifier(x_2)
        x = F.log_softmax(x, dim=-1)
        # print(f"shape of s1 = {S_1.shape}, shape of s2 = {S_2.shape}")
        return x_2, x, S_1, S_2


