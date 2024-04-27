import torch.nn
from torch_geometric.transforms.pad import NodeTypePadding, EdgeTypePadding
import torch.nn.functional as F
import torch
import torch.nn as nn
# ROOT_PATH = "C:\\Users\\88ste\\PycharmProjects\\forks\\gnn-music-analysis" #Stephen
# ROOT_PATH = "/home/users/wx83/GNN_baseline/gnn-music-analysis/final_code" #Weihan
ROOT_PATH = "/usr/xtmp/yz705/GNN"

class MatrixNormLoss(nn.Module):
    def __init__(self, ord='fro'):
        super(MatrixNormLoss, self).__init__()
        self.ord = ord

    def forward(self, input, target):
        """
        Compute the matrix norm of the difference between input and target.
        """
        diff = input - target
        loss = torch.linalg.matrix_norm(diff, ord=self.ord)
        return loss

class WeightedNLLLoss(nn.Module):
    def __init__(self):
        super(WeightedNLLLoss, self).__init__()

    def forward(self, input, target):
        """
        Compute the weighted negative log likelihood loss.
        The target contains the probability of each class being the correct class.
        The input is the raw logits from the previous layer.
        """
        # Step 1: Convert logits to log probabilities
        log_probs = F.log_softmax(input, dim=1)

        weighted_nll_loss = -torch.sum(target * log_probs, dim=1)  # Sum across classes
        
        # Step 3: Return the average loss: average loss on this sample
        return torch.mean(weighted_nll_loss)


CLASSIFICATION_CRITERION = torch.nn.NLLLoss()

SIM_CRITERION = MatrixNormLoss()
WEIGHTED_SIM_CRITERION = WeightedNLLLoss()

