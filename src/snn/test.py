"""
test.py

Evaluate a trained NeuroSolver model.
"""

import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SNNModel
from trainer import validate

from utils import load_checkpoint

from metrics import (
    plot_confusion_matrix,
    print_classification_report,
    print_per_class_accuracy,
    plot_sample_predictions
)

from config import *


def main():

    print("=" * 60)
    print("NeuroSolver Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SNNModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------

    load_checkpoint(
        model=model,
        optimizer=optimizer,
        filepath=f"{CHECKPOINT_DIR}/best_model.pth"
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    (
        test_loss,
        test_accuracy,
        labels,
        predictions,
        images
    ) = validate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE,
        num_steps=NUM_STEPS
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nTest Results")
    print("-" * 60)

    print(f"Loss     : {test_loss:.4f}")
    print(f"Accuracy : {test_accuracy*100:.2f}%")

    # --------------------------------------------------
    # Reports
    # --------------------------------------------------

    print_classification_report(
        labels,
        predictions
    )

    print_per_class_accuracy(
        labels,
        predictions
    )

    # --------------------------------------------------
    # Save Confusion Matrix
    # --------------------------------------------------

    plot_confusion_matrix(
        labels,
        predictions,
        f"{OUTPUT_DIR}/confusion_matrix.png"
    )

    print("\nConfusion Matrix saved.")

    # --------------------------------------------------
    # Save Sample Predictions
    # --------------------------------------------------

    plot_sample_predictions(
        images,
        labels,
        predictions,
        f"{OUTPUT_DIR}/sample_predictions.png"
    )

    print("Sample predictions saved.")

    print("\nEvaluation Complete!")


if __name__ == "__main__":
    main()