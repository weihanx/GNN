import torch_utils
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch
import torch.nn.functional as F

from GNN_backbone import HeteroGNN
from config import DEVICE


class GNN_Cluster(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes, device=DEVICE):
        super(GNN_Cluster, self).__init__()
        self.device = device

        self.linear = torch.nn.Linear(hidden_dim, 1).to(device)

        torch.nn.init.xavier_uniform_(self.linear.weight)  # avoid all zero or all

        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def euclidean_distance_matrix(self, x):
        num_node, num_feat = x.shape
        dist_list = []
        for i in range(num_node):
            for j in range(num_node):
                diff = x[i] - x[j]
                res = diff * diff
                dist_list.append(res)
        distance_mat = torch.stack(dist_list, dim=0)

        return distance_mat

    def adj_to_coord(self, adj_matrix):
        """
        change back too coordinates for gnn model
        """
        coords = {}
        for edge, adj in adj_matrix.items():
            edge_index = adj.nonzero().t()
            coords[edge] = edge_index

        return coords

    def forward(self, x, adjacency_matrices):
        num_nodes = x['note'].shape[0]

        distance_matrix = self.euclidean_distance_matrix(x['note'])
        distance_vector = self.linear(distance_matrix).to(self.device).float()
        grouping_vector = torch.sigmoid(distance_vector)

        grouping_matrix = grouping_vector.reshape((num_nodes, num_nodes))
        grouping_matrix.requires_grad_(True)
        grouping_matrix = grouping_matrix.unsqueeze(0)

        conv_sqrt = torch_utils.MPA_Lya.apply(grouping_matrix)
        clustering_matrix = conv_sqrt.squeeze().float()
        clustering_matrix = F.softmax(clustering_matrix, dim=1)

        x['note'] = torch.matmul(clustering_matrix, x['note'])

        for edge_type, A in adjacency_matrices.items():
            result = torch.matmul(torch.matmul(clustering_matrix, A), clustering_matrix).float()
            adjacency_matrices[edge_type] = result

        return x, adjacency_matrices, clustering_matrix
