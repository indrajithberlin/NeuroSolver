"""
utils.py

Utility functions for NeuroSolver.
"""

import os
import random
import numpy as np
import torch


# ======================================================
# Set Random Seed
# ======================================================

def set_seed(seed):
    """
    Makes experiments reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to {seed}")


# ======================================================
# Create Required Directories
# ======================================================

def create_directories(paths):
    """
    Creates directories if they don't exist.
    """

    for path in paths:
        os.makedirs(path, exist_ok=True)

    print("Directories verified.")


# ======================================================
# Save Model Checkpoint
# ======================================================

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saves the training state.
    """

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved: {filepath}")


# ======================================================
# Load Model Checkpoint
# ======================================================

def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads a saved checkpoint.
    """

    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded: {filepath}")

    return checkpoint