"""
trainer.py

Training and validation utilities for NeuroSolver.
"""

import torch
from snntorch import spikegen

from metrics import calculate_accuracy
from utils import save_checkpoint


# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    num_steps
):
    """
    Train the model for one epoch.

    Returns:
        (loss, accuracy)
    """

    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        images = images.view(images.size(0), -1)

        spike_data = spikegen.rate(
            images,
            num_steps=num_steps
        )

        spk_rec, _ = model(spike_data)

        spike_counts = spk_rec.sum(dim=0)

        loss = criterion(
            spike_counts,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        accuracy = calculate_accuracy(
            spike_counts,
            labels
        )

        running_loss += loss.item()
        running_accuracy += accuracy

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_loss, epoch_accuracy


# ==========================================================
# Validation
# ==========================================================

def validate(
    model,
    dataloader,
    criterion,
    device,
    num_steps
):
    """
    Validate the model.

    Returns:
        loss,
        accuracy,
        labels,
        predictions,
        images
    """

    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    all_labels = []
    all_predictions = []
    sample_images = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            # Save original images before flattening
            sample_images.extend(images.cpu())

            images = images.view(images.size(0), -1)

            spike_data = spikegen.rate(
                images,
                num_steps=num_steps
            )

            spk_rec, _ = model(spike_data)

            spike_counts = spk_rec.sum(dim=0)

            loss = criterion(
                spike_counts,
                labels
            )

            accuracy = calculate_accuracy(
                spike_counts,
                labels
            )

            predictions = torch.argmax(
                spike_counts,
                dim=1
            )

            running_loss += loss.item()
            running_accuracy += accuracy

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return (
        epoch_loss,
        epoch_accuracy,
        all_labels,
        all_predictions,
        sample_images
    )


# ==========================================================
# Fit
# ==========================================================

def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_steps,
    num_epochs,
    checkpoint_path
):
    """
    Train the model for multiple epochs.
    """

    train_loss_history = []
    train_accuracy_history = []

    val_loss_history = []
    val_accuracy_history = []

    best_accuracy = 0.0

    for epoch in range(num_epochs):

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_steps
        )

        (
            val_loss,
            val_accuracy,
            _,
            _,
            _
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
            num_steps
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=val_loss,
                filepath=checkpoint_path
            )

            print("Best model saved.")

    return (
        train_loss_history,
        train_accuracy_history,
        val_loss_history,
        val_accuracy_history
    )