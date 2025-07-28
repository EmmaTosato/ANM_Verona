#test.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

def evaluate(model, loader, device):
    """
        Evaluate a trained model on a test set.

        - Switches the model to evaluation mode.
        - Iterates through the dataloader without computing gradients.
        - Applies the model to each batch and stores predicted and true labels.
        - Returns two NumPy arrays: ground-truth labels and predicted labels.
    """
    model.eval()  # Set the model to evaluation mode (no dropout, etc.)
    true_labels, pred_labels = [], []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in loader:
            x, y = batch['X'].to(device), batch['y'].to(device)

            # Forward pass
            outputs = model(x)

            # Get predicted class index (argmax over output logits)
            preds = torch.argmax(outputs, dim=1)

            # Store ground-truth and predictions
            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return np.array(true_labels), np.array(pred_labels)


def compute_metrics(y_true, y_pred):
    """
        Compute classification metrics from true and predicted labels.

        Returns a dictionary containing:
        - accuracy
        - precision
        - recall
        - F1 score
        - confusion matrix
    """
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


def plot_confusion_matrix(conf_matrix, class_names, save_path=None, title= None):
    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor='black', ax=ax,
        annot_kws={"size": 20, "weight": "bold"},
    )

    # Axis labels with spacing
    ax.set_xlabel("Predicted", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel("True", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)

    # Enlarge colorbar tick labels
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')


    # Tick labels styling
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticklabels(class_names, fontsize=18, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=18, fontweight='bold', rotation=0)

    # Add THICK outer border as a rectangle
    rows, cols = conf_matrix.shape
    rect = patches.Rectangle(
        (0, 0), cols, rows,
        linewidth=4.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


    plt.savefig(save_path, dpi=300)
    plt.close()


def print_metrics(metrics):
    print("\n--- Test Metrics ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])