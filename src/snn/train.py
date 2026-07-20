"""
train.py

Main training script for NeuroSolver.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SNNModel
from trainer import fit

from metrics import (
    plot_loss_curve,
    plot_accuracy_curve
)

from utils import (
    create_directories,
    set_seed
)

from config import *


def main():

    # -------------------------
    # Reproducibility
    # -------------------------
    set_seed(SEED)

    # -------------------------
    # Create output folders
    # -------------------------
    create_directories([
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        LOG_DIR
    ])

    # -------------------------
    # Dataset
    # -------------------------
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = SNNModel().to(DEVICE)

    # -------------------------
    # Loss & Optimizer
    # -------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -------------------------
    # Train
    # -------------------------
    (
        train_loss,
        train_acc,
        val_loss,
        val_acc
    ) = fit(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_steps=NUM_STEPS,
        num_epochs=NUM_EPOCHS,
        checkpoint_path=f"{CHECKPOINT_DIR}/best_model.pth"
    )

    # -------------------------
    # Plots
    # -------------------------
    plot_loss_curve(
        train_loss,
        f"{OUTPUT_DIR}/loss_curve.png"
    )

    plot_accuracy_curve(
        train_acc,
        f"{OUTPUT_DIR}/accuracy_curve.png"
    )

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    main()