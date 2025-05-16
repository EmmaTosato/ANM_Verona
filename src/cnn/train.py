# train.py
import torch
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track cumulative loss

    for x_batch, y_batch in train_loader:
        # Move inputs and labels to the selected device (CPU or GPU)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()  # Reset gradients to zero before each step

        outputs = model(x_batch)  # Forward pass: compute predictions
        loss = criterion(outputs, y_batch)  # Compute loss between predictions and labels
        loss.backward()  # Backpropagation: compute gradients
        optimizer.step()  # Update model weights using gradients

        # Accumulate the loss weighted by batch size
        running_loss += loss.item() * x_batch.size(0)

    # Compute average loss for the epoch
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0  # Count of correct predictions

    with torch.no_grad():  # Disable gradient computation for memory/speed
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            outputs = model(x_val)  # Forward pass
            loss = criterion(outputs, y_val)  # Compute loss
            running_loss += loss.item() * x_val.size(0)  # Accumulate loss

            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            correct += (predicted == y_val).sum().item()  # Count correct predictions

    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)  # Compute accuracy
    return val_loss, val_accuracy

def plot_losses(train_losses, val_losses, val_accuracies=None, save_path=None):
    plt.figure(figsize=(8, 5))

    # Plot training and validation loss
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Val Loss', marker='s', color='orange')

    # Plot validation accuracy if provided
    if val_accuracies is not None:
        plt.plot(val_accuracies, label='Val Accuracy', marker='^', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()