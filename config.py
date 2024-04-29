import torch.nn
import torch.nn.functional as F
import torch
import torch.nn as nn
# ROOT_PATH = "C:\\Users\\88ste\\PycharmProjects\\forks\\gnn-music-analysis" #Stephen
ROOT_PATH = "/home/users/wx83/GNN_baseline/gnn-music-analysis/final_code" #Weihan

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

class CircleLoss(torch.nn.Module):
    def __init__(self, margin=0.4, gamma=80, similarity_threshold=0.1):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.similarity_threshold = similarity_threshold

    def prepare_pairs(self, similarity_matrix, labels):
        n = similarity_matrix.size(0)
        # Compute similarity between labels
        label_similarity = torch.matmul(labels, labels.t())

        # Determine positive and negative pairs based on similarity threshold
        positive_mask = label_similarity > self.similarity_threshold
        negative_mask = label_similarity <= self.similarity_threshold

        # Extract similarities of pairs
        sp = similarity_matrix[positive_mask]
        sn = similarity_matrix[negative_mask]

        return sp, sn
    
    def forward(self, similarity_matrix, labels):
        sp, sn = self.prepare_pairs(similarity_matrix, labels)
        if sp.nelement() == 0 or sn.nelement() == 0:
            return torch.tensor(0.0, device=similarity_matrix.device)  # Handle empty tensors
        
        # Calculate Circle Loss components
        ap = torch.clamp_min(-sp.detach() + 1 + self.margin, min=0.)
        an = torch.clamp_min(sn.detach() + self.margin, min=0.)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = F.softplus(torch.logsumexp(logit_p, dim=0)) + F.softplus(torch.logsumexp(logit_n, dim=0))
        return loss.mean()




# similarity_matrix = torch.randn(num_nodes, num_nodes)  # Random similarity scores
# labels = torch.rand(num_nodes, num_classes)  # Random soft labels for each node, should be one hot

CLASSIFICATION_CRITERION = torch.nn.NLLLoss()

SIM_CRITERION = MatrixNormLoss()
WEIGHTED_SIM_CRITERION = WeightedNLLLoss()

CIRCLELOSS = CircleLoss()
