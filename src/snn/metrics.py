"""
metrics.py

Evaluation and visualization utilities for NeuroSolver.
"""

import os

import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ======================================================
# Accuracy
# ======================================================

def calculate_accuracy(spike_counts, labels):
    """
    Calculate classification accuracy.

    Args:
        spike_counts: Tensor of shape (batch_size, num_classes)
        labels: Ground truth labels

    Returns:
        Accuracy (float)
    """

    predictions = torch.argmax(spike_counts, dim=1)

    correct = (predictions == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


# ======================================================
# Plot Loss Curve
# ======================================================

def plot_loss_curve(loss_history, save_path):
    """
    Plot training loss.
    """

    plt.figure(figsize=(8,5))

    plt.plot(loss_history, linewidth=2)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ======================================================
# Plot Accuracy Curve
# ======================================================

def plot_accuracy_curve(accuracy_history, save_path):
    """
    Plot training accuracy.
    """

    plt.figure(figsize=(8,5))

    plt.plot(accuracy_history, linewidth=2)

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ======================================================
# Confusion Matrix
# ======================================================

def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    save_path
):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(ax=ax, colorbar=False)

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ======================================================
# Create Output Folder
# ======================================================

def ensure_output_directory(path):
    """
    Create output directory if needed.
    """

    os.makedirs(path, exist_ok=True)