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
SAVE_FOLDER = "hetergraph0422_4feature/"
SHARE_BACKBONE_WEIGHTS = False
NUM_FEAT = 111
EMB_DIM = 32
HIDDEN_DIM = 200
NUM_CLASS = 15

NUM_EPOCHS = 50

CLASSIFICATION_CRITERION = torch.nn.NLLLoss()
SIM_CRITERION = nn.CosineEmbeddingLoss()
GROUPING_CRITERION = nn.BCELoss()

