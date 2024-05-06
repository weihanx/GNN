from torch_geometric.nn import Linear
import torch

from model.layers.GNN_backbone import HeteroGNN
from model.layers.GNN_cluster import GNN_Cluster
from config import DEVICE, EMB_DIM, HIDDEN_DIM, NUM_FEAT, NUM_CLUSTERING_LAYERS


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=NUM_FEAT, embedding_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_clustering_layers=NUM_CLUSTERING_LAYERS, device=DEVICE):
        super(GroupMat, self).__init__()
        self.device = device

        self.linear_embed = torch.nn.Linear(num_feature, embedding_dim)

        self.cluster_layers = torch.nn.ModuleList()
        for _ in range(NUM_CLUSTERING_LAYERS):
            self.cluster_layers.append(
                GNN_Cluster(embedding_dim, hidden_dim, self.device)
            )

    def forward(self, data, grouping_matrices_true):
        x = data.x_dict
        edge_index_dict = data.edge_index_dict
        attribute_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}

        x['note'] = x['note'].float()

        x['note'] = self.linear_embed(x['note'])

        cluster_results = {
            'clustering_matrices': [],
            'grouping_losses': [],
            'grouping_matrix_preds': []
        }
        for i, cluster_layer in enumerate(self.cluster_layers):
            x, edge_dict, attribute_dict, clustering_matrix, grouping_loss, grouping_matrix_pred = cluster_layer(
                x, edge_index_dict, attribute_dict, grouping_matrices_true[i]
            )
            cluster_results['clustering_matrices'].append(clustering_matrix)
            cluster_results['grouping_losses'].append(grouping_loss)
            cluster_results['grouping_matrix_preds'].append(grouping_matrix_pred)

        return x, cluster_results
