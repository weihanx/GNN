from torch_geometric.nn import Linear
import torch

from model.layers.GNN_backbone import HeteroGNN
from model.layers.GNN_cluster import GNN_Cluster
from model.layers.CatGCN import CatEmbedder, one_hot_to_indices
from config import DEVICE, EMBEDDING_METHOD


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=111, embedding_dim=32, hidden_dim=64, num_classes=15, device=DEVICE):
        super(GroupMat, self).__init__()
        self.device = device

        self.cat_embed = CatEmbedder(num_feature, 3, embedding_dim, 1, 1, 0.5, 0.5)
        self.linear_embed = torch.nn.Linear(num_feature, embedding_dim)

        self.gnn_cluster1 = GNN_Cluster(embedding_dim, hidden_dim, num_classes, self.device)

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

    def forward(self, data, grouping_matrix_true, embedding_method=EMBEDDING_METHOD, mixed=True):

        x = data.x_dict
        edge_index_dict = data.edge_index_dict
        attribute_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}
        grouping_matrix_true = grouping_matrix_true.to(self.device)

        if embedding_method == "cat":
            x['note'] = x['note'].float()
            # Number of numeric features; we assume the node embedding is given as (cat, ..., cat, num, ..., num)
            n_nums = 3
            num_features = x['note'][:, -n_nums:]
            cat_features = x['note'][:, :-n_nums]
            cat_indices = [one_hot_to_indices(cat_features[i]) for i in range(cat_features.shape[0])]

            if mixed:
                cat_embedding = self.cat_embed(torch.stack(cat_indices), num_features=num_features)
                x['note'] = cat_embedding
            else:
                cat_embedding = self.cat_embed(torch.stack(cat_indices))
                x['note'] = torch.cat((cat_embedding, torch.unsqueeze(num_features, dim=1)), 1)

        elif embedding_method == "linear":
            x['note'] = self.linear_embed(x['note'])

        x, edge_dict, attribute_dict, S_1, grouping_loss_1, grouping_matrix_pred_1 = self.gnn_cluster1(x, edge_index_dict, attribute_dict, grouping_matrix_true)

        return x, S_1, grouping_loss_1, grouping_matrix_pred_1
