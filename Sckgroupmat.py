from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch

from GNN_backbone import HeteroGNN
from GNN_cluster import GNN_Cluster
from config import DEVICE


class GroupMat(torch.nn.Module):
    def __init__(self, num_feature=111, embedding_dim=32, hidden_dim=200, num_classes=15, device=DEVICE):
        super(GroupMat, self).__init__()
        self.device = device

        self.embed = torch.nn.Linear(num_feature, embedding_dim)

        self.gnn_cluster1 = GNN_Cluster(embedding_dim, hidden_dim, num_classes)
        self.gnn_cluster2 = GNN_Cluster(embedding_dim, hidden_dim, num_classes)

        self.gnn2_embed = HeteroGNN(embedding_dim, hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index_dict, batch):
        x['note'] = x['note'].float()
        x['note'] = self.embed(x['note'])

        x, edge_dict_1, S_1 = self.gnn_cluster1(x, edge_index_dict)
        x, edge_dict_2, S_2 = self.gnn_cluster2(x, edge_dict_1)

        z_2 = self.gnn2_embed(x, edge_dict_2).float()
        # x_2 = global_mean_pool(z_2, batch)

        # x = self.classifier(x_2)
        # x = F.log_softmax(x, dim=-1)
        return z_2, S_1, S_2
