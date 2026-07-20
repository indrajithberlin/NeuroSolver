"""
metrics.py

Evaluation and visualization utilities for NeuroSolver.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ==========================================================
# Accuracy
# ==========================================================

def calculate_accuracy(spike_counts, labels):
    """
    Calculate classification accuracy.

    Args:
        spike_counts: Tensor of shape (batch_size, num_classes)
        labels: Ground truth labels

    Returns:
        float
    """

    predictions = torch.argmax(spike_counts, dim=1)

    correct = (predictions == labels).sum().item()

    accuracy = correct / labels.size(0)

    return accuracy


# ==========================================================
# Loss Curve
# ==========================================================

def plot_loss_curve(loss_history, save_path):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history, linewidth=2)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Accuracy Curve
# ==========================================================

def plot_accuracy_curve(accuracy_history, save_path):

    plt.figure(figsize=(8,5))

    plt.plot(accuracy_history, linewidth=2)

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(
    true_labels,
    predicted_labels,
    save_path
):

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax, colorbar=False)

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Classification Report
# ==========================================================

def print_classification_report(
    true_labels,
    predicted_labels
):
    """
    Print Precision, Recall and F1 Score.
    """

    report = classification_report(
        true_labels,
        predicted_labels
    )

    print("\nClassification Report")
    print("-" * 60)
    print(report)


# ==========================================================
# Per-Class Accuracy
# ==========================================================

def print_per_class_accuracy(
    true_labels,
    predicted_labels
):

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print("\nPer-Class Accuracy")
    print("-" * 60)

    for i in range(len(cm)):

        accuracy = cm[i, i] / cm[i].sum()

        print(f"Digit {i}: {accuracy*100:.2f}%")


# ==========================================================
# Sample Predictions
# ==========================================================

def plot_sample_predictions(
    images,
    labels,
    predictions,
    save_path,
    rows=2,
    cols=5
):

    plt.figure(figsize=(12,5))

    total = rows * cols

    for i in range(total):

        plt.subplot(rows, cols, i + 1)

        plt.imshow(
            images[i].squeeze(),
            cmap="gray"
        )

        title = f"P:{predictions[i]}\nT:{labels[i]}"

        plt.title(title)

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


# ==========================================================
# Ensure Output Folder
# ==========================================================

def ensure_output_directory(path):

    os.makedirs(path, exist_ok=True)