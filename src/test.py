import torch
import numpy as np

def evaluate_model(model, test_loader, task, device):
    model.eval()
    true_labels, pred_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)

            if task == 'classification':
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs.squeeze()

            true_labels.extend(y.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return np.array(true_labels), np.array(pred_labels)
