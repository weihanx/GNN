
import torch
import torch_utils
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear

from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn.conv import DirGNNConv
from torch_geometric.nn.conv import MessagePassing

class CatEmbedder(nn.Module):

    def __init__(self):
        super(CatEmbedder, self).__init__()