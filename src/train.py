import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, criterion, optimizer, batch_size, epochs, device, task):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_running_loss += loss.item() * x_val.size(0)

                preds = torch.argmax(outputs, dim=1) if task == 'classification' else outputs.squeeze()

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_val.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if task == 'classification':
            acc = accuracy_score(all_targets, all_preds)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {acc:.4f}")
        else:
            r2 = r2_score(all_targets, all_preds)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f} - Val R²: {r2:.4f}")

    return model, train_losses, val_losses
