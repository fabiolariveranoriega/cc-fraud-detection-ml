import os
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "cuda:0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

NUM_TRIALS = 20
HIDDEN_DIM = ("hidden_dim", 32, 128)
BATCH_SIZE = ('batch_size', 32, 128)
NUM_EPOCHS = ('num_epochs', 10, 50)
NUM_LAYERS = ("num_layers", 2, 5)
DROPOUT_RATE = ("dropout_rate", 0.0, 0.1)
LR = ("lr", 1e-3, 1e-1)
ACTIVATION_FN = nn.Tanh()

NUM_FOLDS = 5
SEED = 42