import torch_utils
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch
import torch.nn.functional as F

from model.layers.GNN_backbone import HeteroGNN
from config import DEVICE, GROUPING_CRITERION, LAMBDA_CLAMP_MIN

class Clusterer(torch.nn.Module):

    def __init__(self):
        super(Clusterer, self).__init__()

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
    
    def pairwise_distance_matrix(self, x):
        pass

    def adj_to_coord(self, adj_matrix):
        """
        change back too coordinates for gnn model
        """
        coords = {}
        edge_attributes = {}
        for edge, adj in adj_matrix.items():
            indices = adj.nonzero(as_tuple=True)
            edge_index = torch.stack(indices).t()
            weights = adj[indices]

            coords[edge] = edge_index.t()
            edge_attributes[edge] = weights

        return coords, edge_attributes

    def coord_to_adj(self, edge_index_dict, attribute_dict, num_nodes):
        adj = {}
        for edge, mat in edge_index_dict.items():
            adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=self.device)
            for idx, (source, target) in enumerate(mat.t()):
                adj_matrix[source, target] = attribute_dict[edge][idx]
                adj[edge] = adj_matrix
        return adj

class GNN_Cluster(Clusterer):
    def __init__(self, embedding_dim, hidden_dim, num_classes, device=DEVICE):
        super(GNN_Cluster, self).__init__()
        self.device = device

        self.linear = torch.nn.Linear(hidden_dim, 1).to(device)
        self.gnn_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden_channel, output_channel

        self.custom_weight_init(self.gnn_embed)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # avoid all zero or all

        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index_dict, attribute_dict, grouping_matrix_true):
        num_nodes = x['note'].shape[0]

        x['note'] = self.gnn_embed(x, edge_index_dict, attribute_dict).float()

        distance_matrix = self.euclidean_distance_matrix(x['note'])
        distance_vector = self.linear(distance_matrix).to(self.device).float()
        grouping_vector = torch.sigmoid(distance_vector)

        grouping_matrix = grouping_vector.reshape((num_nodes, num_nodes))
        grouping_loss = GROUPING_CRITERION(grouping_matrix, grouping_matrix_true)
        grouping_matrix = grouping_matrix.unsqueeze(0)

        # Eigen Decomposition
        # clustering_matrix = torch_utils.MPA_Lya.apply(grouping_matrix)
        eigen_values, eigen_vectors = torch.linalg.eigh(grouping_matrix)
        eigen_values = torch.clamp(eigen_values, min=LAMBDA_CLAMP_MIN)
        sqrt_e_values = torch.sqrt(torch.diag(eigen_values.squeeze()))

        clustering_matrix = torch.matmul(eigen_vectors, sqrt_e_values)
        clustering_matrix = clustering_matrix.squeeze().float()
        # clustering_matrix = F.softmax(clustering_matrix, dim=1)

        x['note'] = torch.matmul(clustering_matrix, x['note'])

        adjacency_matrices = self.coord_to_adj(edge_index_dict, attribute_dict, num_nodes)
        for edge_type, A in adjacency_matrices.items():
            result = torch.matmul(torch.matmul(clustering_matrix, A), clustering_matrix).float()
            adjacency_matrices[edge_type] = result
        edge_dict, attribute_dict = self.adj_to_coord(adjacency_matrices)
        return x, edge_dict, attribute_dict, clustering_matrix, grouping_loss, grouping_matrix

# Graph partitioning based on the Fiedler method
class FiedlerClusterer(Clusterer):

    def __init__(self, embedding_dim, hidden_dim, num_classes, fielder_threshold=0.1, device=DEVICE):
        super(FiedlerClusterer, self).__init__()
        self.device = device
        self.fieldler_threshold = fielder_threshold

        self.linear = torch.nn.Linear(hidden_dim, 1).to(device)
        self.gnn_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)  # hidden_channel, output_channel

        self.custom_weight_init(self.gnn_embed)
        torch.nn.init.xavier_uniform_(self.linear.weight)  # avoid all zero or all

        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.split_classifier = torch.nn.Linear(hidden_dim, 1).to(device)

    def forward(self, x, edge_index_dict, attribute_dict, grouping_matrix_true,
                similarity_graph="base", K=3, epsilon=0.5, split_with_classifier=False):
        
        num_nodes = x['note'].shape[0]

        distance_matrix = self.get_distance_matrix(x, edge_index_dict, attribute_dict)
        fiedler_value, fiedler_vector = self.compute_fiedler(distance_matrix, similarity_graph, K, epsilon)
        
        # Perform a BFS on the computed clusters
        total_clusters = 0
        cluster_queue = [(grouping_matrix, x, edge_index_dict, attribute_dict, fiedler_value, fiedler_vector)]
        while cluster_queue:

            # Get the next 'depth' of clusters to explore
            grouping_matrix, x, edge_index_dict, attribute_dict, fiedler_value, fiedler_vector = cluster_queue.pop(0)

            # Skip exploring this cluster if the Fiedler value is below a fixed threshold
            if not split_with_classifier and fiedler_value <= self.fieldler_threshold:
                continue
            # if split_with_classifier:
            #     split = self.split_classifier(grouping_matrix)

            # Compute clusters based on Fiedler partition
            x, edge_dict, attribute_dict, clustering_matrix = self.fiedler_to_clusters(x, edge_index_dict, attribute_dict, fiedler_vector)
            total_clusters += 1

            # Compute the 2 subgraphs generated from the Fiedler partition
            subgraph_group1 = distance_matrix[clustering_matrix[:, 0]][:, clustering_matrix[:, 0]]
            subgraph_group2 = distance_matrix[clustering_matrix[:, 1]][:, clustering_matrix[:, 1]]
            # Compute associated fiedler values and vectors
            fiedler_value1, fiedler_vector1 = self.compute_fiedler(subgraph_group1, similarity_graph, K, epsilon)
            fiedler_value2, fiedler_vector2 = self.compute_fiedler(subgraph_group2, similarity_graph, K, epsilon)

            # Push each subgraph onto the queue
            cluster_queue.append((subgraph_group1, x, edge_index_dict, attribute_dict, fiedler_value1, fiedler_vector1))
            cluster_queue.append((subgraph_group2, x, edge_index_dict, attribute_dict, fiedler_value2, fiedler_vector2))

        final_grouping = torch.zeros((num_nodes, total_clusters))
        
        grouping_loss = GROUPING_CRITERION(final_grouping, grouping_matrix_true)

        return x, edge_dict, attribute_dict, clustering_matrix, grouping_loss, grouping_matrix

    # Compute pairwise Euclidean distance matrix in embedding space
    def get_distance_matrix(self, x, edge_index_dict, attribute_dict):
        # Compute embeddings of notes
        x['note'] = self.gnn_embed(x, edge_index_dict, attribute_dict).float()
        
        n = x['note'].shape[0]
        distance_matrix = torch.zeros((n, n))
        # Iterate over each pair of vectors
        for i in range(n):
            for j in range(n):
                # Compute the Euclidean distance between vector i and vector j
                diff = x['note'][i] - x['note'][j]
                distance = torch.sqrt(torch.sum(diff ** 2))
                distance_matrix[i, j] = distance
        return distance_matrix

    def compute_fiedler(self, distance_matrix, similarity_graph, K, epsilon):
        # We treat the grouping matrix as a adjacency/affinity matrix between nodes
        # and then compute a weighted adjancecy matrix W based on it
        if similarity_graph == "base":
            # Treat grouping matrix as weighted adjacency matrix
            W = distance_matrix
        elif similarity_graph == "knn":
            # Connect nodes to their K nearest neighbors
            W = torch.zeros_like(distance_matrix)
            # Sort the adjacency matrix by rows and record the indices
            _, adj_sort = torch.sort(distance_matrix, dim=1)
            # Set the weight (i, j) to 1 when either i or j is within the k-nearest neighbors of each other
            for i in range(adj_sort.shape[0]):
                W[i, adj_sort[i, :(K + 1)]] = 1
        elif similarity_graph == "mutual_knn":
            # Connect nodes iff they are each in the other's K nearest neighbors
            W1 = torch.zeros_like(distance_matrix)
            # Sort the adjacency matrix by rows and record the indices
            _, adj_sort = torch.sort(distance_matrix, dim=1)
            # Set the weight W1[i, j] to 0.5 when either i or j is within the k-nearest neighbors of each other (Flag)
            # Set the weight W1[i, j] to 1 when both i and j are within the k-nearest neighbors of each other
            for i in range(distance_matrix.shape[0]):
                for j in adj_sort[i, :(K + 1)]:
                    if i == j:
                        W1[i, i] = 1
                    elif W1[i, j] == 0 and W1[j, i] == 0:
                        W1[i, j] = 0.5
                    else:
                        W1[i, j] = W1[j, i] = 1
            W = (W1 > 0.5).float()
        elif similarity_graph == "eps_filtration":
            # Form an edge between any two nodes with an grouping value greater than epsilon
            W = (distance_matrix <= epsilon).float()

        # Compute the degree matrix D
        degree = torch.sum(W, dim=1)
        D = torch.diag(degree)
        # Graph laplacian, note that this is guaranteed to be psd
        L = D - W

        # Eigen Decomposition
        eigen_values, eigen_vectors = torch.linalg.eigh(L)
        # Sort the eigenvalues in ascending order and get the corresponding indices
        sorted_indices = torch.argsort(eigen_values).squeeze()

        # Get the fielder value/vector, the second smallest eigenvalue/vector
        fiedler_value = eigen_values[:, sorted_indices[1]]
        fiedler_vector = eigen_vectors[:, sorted_indices[1], :]
        return fiedler_value, fiedler_vector

    # TODO: Explore using a softmax to get soft assignment instead of hard clustering
    def fiedler_to_clusters(self, x, edge_index_dict, attribute_dict, fiedler_vector):

        num_nodes = x['note'].shape[0]
        positive_indices = (fiedler_vector > 0).nonzero()[:, 1]
        negative_indices = (fiedler_vector < 0).nonzero()[:, 1]
        
        clustering_matrix = torch.zeros((num_nodes, 2)).to(self.device)
        clustering_matrix[negative_indices, 0] = 1
        clustering_matrix[positive_indices, 1] = 1

        clustering_matrix = clustering_matrix.squeeze().float()
        x['note'] = torch.matmul(clustering_matrix.t(), x['note'])

        adjacency_matrices = self.coord_to_adj(edge_index_dict, attribute_dict, num_nodes)
        for edge_type, A in adjacency_matrices.items():
            result = torch.matmul(torch.matmul(clustering_matrix.t(), A), clustering_matrix).float()
            adjacency_matrices[edge_type] = result
        edge_dict, attribute_dict = self.adj_to_coord(adjacency_matrices)
        return x, edge_dict, attribute_dict, clustering_matrix