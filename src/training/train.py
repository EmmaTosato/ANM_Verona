# train.py
import torch
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    """
        Perform one epoch of training.

        - Sets the model in training mode.
        - Iterates through training data batches.
        - Computes the forward pass and loss.
        - Performs backpropagation and optimizer step.
        - Accumulates total training loss to return the average at the end.
    """
    # Set the model to training mode
    model.train()
    # Track cumulative loss
    running_loss = 0.0

    for batch in train_loader:
        x_train, y_train = batch['X'].to(device), batch['y'].to(device)

        # Reset gradients to zero before each step
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(x_train)
        # Compute loss between predictions and labels
        loss = criterion(outputs, y_train)
        # Backpropagation: compute gradients
        loss.backward()
        # Update model weights using gradients
        optimizer.step()

        # Accumulate the loss weighted by batch size
        running_loss += loss.item() * x_train.size(0)

    # Compute average loss for the epoch
    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def validate(model, val_loader, criterion, device):
    """
        Evaluate the model on the validation set.

        - Sets the model in evaluation mode.
        - Disables gradient computation.
        - Computes forward pass and accumulates loss and accuracy.
        - Returns average loss and overall accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0  # Count of correct predictions

    # Disable gradient computation for memory/speed
    with torch.no_grad():
        for batch in val_loader:
            x_val, y_val = batch['X'].to(device), batch['y'].to(device)

            # Forward pass
            outputs = model(x_val)
            # Compute loss
            loss = criterion(outputs, y_val)
            # Accumulate loss
            running_loss += loss.item() * x_val.size(0)

            # Get predicted class indices
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            correct += (predicted == y_val).sum().item()


    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)  # Compute accuracy

    return val_loss, val_accuracy

def plot_losses(train_losses, val_losses, val_accuracies=None, save_path=None, title = None):
    """
        Plot training/validation loss and optionally validation accuracy across epochs.
    """
    plt.figure(figsize=(8, 5))
    # Plot training and validation loss
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Val Loss', marker='s', color='orange')

    # Plot validation accuracy if provided
    if val_accuracies is not None:
        plt.plot(val_accuracies, label='Val Accuracy', marker='^', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()