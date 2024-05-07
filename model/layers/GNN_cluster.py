from torch_geometric.nn import HeteroConv, SAGEConv
import torch
from torch.linalg import multi_dot

from model.layers.GNN_backbone import HeteroGNN
from config import DEVICE, GROUPING_CRITERION, LAMBDA_CLAMP_MIN


class GNN_Cluster(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, device=DEVICE):
        super(GNN_Cluster, self).__init__()
        self.device = device

        self.linear = torch.nn.Linear(hidden_dim, 1).to(device)
        self.gnn_embed = HeteroGNN(embedding_dim, hidden_dim)  # hidden_channel, output_channel

        self.threshold = torch.nn.Threshold(0.1, 0)

        self.custom_weight_init(self.gnn_embed)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # avoid all zero or all

    def custom_weight_init(self, model):
        for m in model.modules():
            if isinstance(m, HeteroConv):
                for _, conv in m.convs.items():  # m.convs is the dictionary holding the convolutions
                    self.init_weights(conv)

    def init_weights(self, m):
        if isinstance(m, SAGEConv):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

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

    def adj_to_coord(self, adj_matrix: torch.tensor):
        """
        change back too coordinates for gnn model
        """
        edge_dict = {}
        edge_attributes = {}
        for edge, adj in adj_matrix.items():
            indices = adj.nonzero(as_tuple=True)
            edge_index = torch.stack(indices).t()
            weights = adj[indices]

            edge_dict[edge] = edge_index.t()
            edge_attributes[edge] = weights

        return edge_dict, edge_attributes

    def coord_to_adj(self, edge_index_dict, attribute_dict, num_nodes):
        adj = {}
        for edge, mat in edge_index_dict.items():
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
            for idx, (source, target) in enumerate(mat.t()):
                adj_matrix[source, target] = attribute_dict[edge][idx]
                adj[edge] = adj_matrix
        return adj

    def forward(self, x, edge_index_dict, attribute_dict, grouping_matrix_true, cluster_results):
        num_nodes = x['note'].shape[0]

        x['note'] = self.gnn_embed(x, edge_index_dict, attribute_dict).float()

        distance_matrix = self.euclidean_distance_matrix(x['note'])
        distance_vector = self.linear(distance_matrix).to(self.device).float()
        grouping_vector = torch.sigmoid(distance_vector)

        grouping_matrix_pred = grouping_vector.reshape((num_nodes, num_nodes))
        grouping_matrix_pred = grouping_matrix_pred.unsqueeze(0)

        # Eigen Decomposition
        eigen_values, eigen_vectors = torch.linalg.eigh(grouping_matrix_pred)
        eigen_values = torch.clamp(eigen_values, min=LAMBDA_CLAMP_MIN)
        sqrt_e_values = torch.sqrt(torch.diag(eigen_values.squeeze()))

        clustering_matrix = torch.matmul(eigen_vectors, sqrt_e_values)
        clustering_matrix = clustering_matrix.squeeze().float()
        clustering_matrix = self.threshold(clustering_matrix)
        clustering_matrix = torch.div(clustering_matrix, torch.sum(clustering_matrix, dim=1).unsqueeze(1))


        # Remove 0 columns
        non_empty_mask = clustering_matrix.abs().sum(dim=0).bool()
        clustering_matrix = clustering_matrix[:, non_empty_mask]

        if len(cluster_results["clustering_matrices"]) > 0:
            grouping_matrix_pred = multi_dot(cluster_results["clustering_matrices"] + [clustering_matrix])
            grouping_matrix_pred = torch.matmul(grouping_matrix_pred, grouping_matrix_pred.t())

        grouping_loss = GROUPING_CRITERION(grouping_matrix_pred.squeeze(), grouping_matrix_true)

        x['note'] = torch.matmul(clustering_matrix.t(), x['note'])
        adjacency_matrices = self.coord_to_adj(edge_index_dict, attribute_dict, num_nodes)
        for edge_type, A in adjacency_matrices.items():
            result = torch.matmul(torch.matmul(clustering_matrix.t(), A), clustering_matrix).float()
            adjacency_matrices[edge_type] = result

        edge_dict, attribute_dict = self.adj_to_coord(adjacency_matrices)
        return x, edge_dict, attribute_dict, clustering_matrix, grouping_loss, grouping_matrix_pred
