import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model, loader, device):
    model.eval()  # Set the model to evaluation mode (no dropout, etc.)
    true_labels, pred_labels = [], []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for x, y in loader:
            # Move data to device (CPU or GPU)
            x, y = x.to(device), y.to(device)

            # Forward pass
            outputs = model(x)

            # Get predicted class index (argmax over output logits)
            preds = torch.argmax(outputs, dim=1)

            # Store ground-truth and predictions
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return np.array(true_labels), np.array(pred_labels)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix
    }
    return metrics


def print_metrics(metrics):
    print("\n--- Test Metrics ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])