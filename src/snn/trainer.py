"""
trainer.py

Training utilities for NeuroSolver.
"""

import torch
from snntorch import spikegen
from utils import save_checkpoint
from metrics import calculate_accuracy


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

    Args:
        model: SNN model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: CPU or CUDA
        num_steps: Number of simulation time steps

    Returns:
        tuple:
            (average_loss, average_accuracy)
    """

    # Set model to training mode
    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in dataloader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Flatten images
        images = images.view(images.size(0), -1)

        # Convert images to spike trains
        spike_data = spikegen.rate(
            images,
            num_steps=num_steps
        )

        # Forward pass
        spk_rec, _ = model(spike_data)

        # Decode output by counting spikes
        spike_counts = spk_rec.sum(dim=0)

        # Compute loss
        loss = criterion(
            spike_counts,
            labels
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        accuracy = calculate_accuracy(
            spike_counts,
            labels
        )

        # Accumulate statistics
        running_loss += loss.item()
        running_accuracy += accuracy

    # Average metrics over all batches
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_loss, epoch_accuracy

def validate(
    model,
    dataloader,
    criterion,
    device,
    num_steps
):
    """
    Validate the model for one epoch.

    Args:
        model: SNN model
        dataloader: Validation/Test DataLoader
        criterion: Loss function
        device: CPU or CUDA
        num_steps: Number of simulation time steps

    Returns:
        tuple:
            (average_loss, average_accuracy)
    """

    # Set model to evaluation mode
    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    # Disable gradient computation
    with torch.no_grad():

        for images, labels in dataloader:

            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Flatten images
            images = images.view(images.size(0), -1)

            # Convert images to spike trains
            spike_data = spikegen.rate(
                images,
                num_steps=num_steps
            )

            # Forward pass
            spk_rec, _ = model(spike_data)

            # Decode output
            spike_counts = spk_rec.sum(dim=0)

            # Compute loss
            loss = criterion(
                spike_counts,
                labels
            )

            # Compute accuracy
            accuracy = calculate_accuracy(
                spike_counts,
                labels
            )

            # Accumulate metrics
            running_loss += loss.item()
            running_accuracy += accuracy

    # Average metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)

    return epoch_loss, epoch_accuracy

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

    Returns:
        train_loss_history
        train_accuracy_history
        val_loss_history
        val_accuracy_history
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

        val_loss, val_accuracy = validate(
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