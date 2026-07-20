import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random Seed
SEED = 42

# Dataset
DATA_DIR = "data"

# Training
BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

# SNN
NUM_INPUTS = 784
NUM_HIDDEN = 100
NUM_OUTPUTS = 10
NUM_STEPS = 50
BETA = 0.9

# Directories
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"