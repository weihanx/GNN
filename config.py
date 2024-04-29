import torch
from criterion import *
# ROOT_PATH = "C:\\Users\\88ste\\PycharmProjects\\forks\\gnn-music-analysis" #Stephen
# ROOT_PATH = "/home/users/wx83/GNN_baseline/gnn-music-analysis/final_code" #Weihan

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRINT_MATRICES = False
TRAIN_NAMES = "train-names.txt"
SAVE_FOLDER = "hetergraph0422_4feature/"
NUM_FEAT = 111
EMB_DIM = 32
HIDDEN_DIM = 200
NUM_CLASS = 15

CLASSIFICATION_CRITERION = torch.nn.NLLLoss()
# SIM_CRITERION = MatrixNormLoss()
SIM_CRITERION = nn.CosineEmbeddingLoss()
WEIGHTED_SIM_CRITERION = WeightedNLLLoss()

