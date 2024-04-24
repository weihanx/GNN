# This is used for cluster layer


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

class GNN_Cluster(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, device = "cuda"):
        super(GNN_Cluster, self).__init__()
        self.device = device

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
    
    def custom_weight_init(self, model):
        for m in model.modules():
            if isinstance(m, HeteroConv):
                for _, conv in m.convs.items():  # m.convs is the dictionary holding the convolutions
                    self.init_weights(conv)

    def forward(self, x, edge_index_dict):
        num_nodes = x['note'].shape[0]
        z_0 = self.gnn1_embed(x, edge_index_dict).float()  # 70, 16

        # print(f"z_0 = {z_0.shape}") # contain nan
        for edge_type, _ in edge_index_dict.items():
            if edge_index_dict[edge_type].numel() == 0:
                edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long).to(self.device)

        dist_vec = self.euclidean_distance_matrix(z_0) # (51*25, 200)
        # print(f"before linear = {M}")
        dist_vec = self.linear(dist_vec).to(self.device).float()

        dist_vec = torch.sigmoid(dist_vec) # shape is (51*25, 1)
        dist_vec = dist_vec.squeeze()
        # print(f"shape of M_G = {M_G.shape}")
        # resize to upper triangular matrix
        tri_M_G = torch.zeros((num_nodes, num_nodes),device = dist_vec.device)
        row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=0)
        # flip 
        tri_M_G[row_indices, col_indices] = dist_vec
        tri_M_G_T = tri_M_G.t()
        tri_M_G_T = tri_M_G + tri_M_G_T
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
        return x, edge_dict_1, S_1