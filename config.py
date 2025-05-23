from criterion import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_PERCENT = 0.8

BATCH_SIZE = 1
LEARNING_RATE = 0.001
SCHEDULER_GAMMA = 0.1
MAX_GRAD_NORM = 0.1

PRINT_MATRICES = False
PRINT_LOSS = False
PRINT_EVERY = 400

TRAIN_NAMES = "train-names.txt"
SAVE_FOLDER = "processed_data"
INTERVAL_EDGES = [2, 3, 4, 5]
LAMBDA_CLAMP_MIN = 0
SHARE_BACKBONE_WEIGHTS = False
EMBEDDING_METHOD = 'linear'

NUM_FEAT = 44
EMB_DIM = 32
HIDDEN_DIM = 128
NUM_CLASS = 15

NUM_EPOCHS = 10

CLASSIFICATION_CRITERION = torch.nn.NLLLoss()
SIM_CRITERION = nn.CosineEmbeddingLoss()
GROUPING_CRITERION = nn.BCELoss()

