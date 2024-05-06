import torch

from model.layers.GNN_cluster import GNN_Cluster
from model.layers.CatGCN import CatEmbedder, one_hot_to_indices
from config import DEVICE, EMB_DIM, HIDDEN_DIM, NUM_FEAT, NUM_CLUSTERING_LAYERS, EMBEDDING_METHOD


class GroupMat(torch.nn.Module):
    def __init__(self,
                 num_feature=NUM_FEAT,
                 embedding_dim=EMB_DIM,
                 hidden_dim=HIDDEN_DIM,
                 num_clustering_layers=NUM_CLUSTERING_LAYERS,
                 device=DEVICE):
        super(GroupMat, self).__init__()
        self.device = device

        self.cat_embed = CatEmbedder(num_feature, 3, embedding_dim, 1, 1, 0.5, 0.5)
        self.linear_embed = torch.nn.Linear(num_feature, embedding_dim)

        self.cluster_layers = torch.nn.ModuleList()
        for _ in range(num_clustering_layers):
            self.cluster_layers.append(
                GNN_Cluster(embedding_dim, hidden_dim, self.device)
            )

    def forward(self, data, grouping_matrices_true, embedding_method=EMBEDDING_METHOD, mixed=True):
        x = data.x_dict
        edge_index_dict = data.edge_index_dict
        attribute_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}
        grouping_matrices_true = [m.to(self.device) for m in grouping_matrices_true]

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
