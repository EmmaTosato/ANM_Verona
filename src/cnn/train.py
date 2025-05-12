# train.py
import torch
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    # Enable training mode
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_batch)

        # Compute loss
        loss = criterion(outputs, y_batch)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    return train_loss

def validate(model, val_loader, criterion, device):
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0

    # Disable gradient computation
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            loss = criterion(outputs, y_val)
            running_loss += loss.item() * x_val.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_val).sum().item()

    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    return val_loss, val_accuracy


def save_best_model(model, path):
    torch.save(model.state_dict(), path)

def plot_losses(train_losses, val_losses, val_accuracies=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Val Loss', marker='s', color='orange')
    if val_accuracies is not None:
        plt.plot(val_accuracies, label='Val Accuracy', marker='^', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.show()
