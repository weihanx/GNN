from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch

from GNN_backbone import HeteroGNN
from GNN_cluster import GNN_Cluster
from config import DEVICE


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=111, embedding_dim=32, hidden_dim=256, num_classes=15, device=DEVICE):
        super(GroupMat, self).__init__()
        self.device = device

        self.linear_embed = torch.nn.Linear(num_feature, embedding_dim)



        self.gnn_cluster1 = GNN_Cluster(embedding_dim, hidden_dim, num_classes, self.device)
        # self.gnn_cluster2 = GNN_Cluster(hidden_dim, num_classes)

        self.gnn2_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def coord_to_adj(self, edge_index_dict, num_nodes):
        """
        Translate coordinates of edge_index_dict into adjacency matrices
        """
        adj = {}
        for edge, mat in edge_index_dict.items():
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
            for source, target in mat.t():
                adj_matrix[source, target] = 1
                adj[edge] = adj_matrix
        return adj

    def forward(self, data):
        x = data.x_dict
        edge_index_dict = data.edge_index_dict
        attribute_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}

        x['note'] = x['note'].float()
        num_nodes = x['note'].shape[0]

        x['note'] = self.linear_embed(x['note'])

        x, edge_dict, attribute_dict, S_1 = self.gnn_cluster1(x, edge_index_dict, attribute_dict)

        # x = self.gnn2_embed(x, adjacency_matrices).float()
        # x_2 = global_mean_pool(z_2, batch)

        # x = self.classifier(x_2)
        # x = F.log_softmax(x, dim=-1)
        return x, S_1
