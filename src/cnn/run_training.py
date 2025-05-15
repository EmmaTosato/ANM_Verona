# run_training.py
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D
from train import train, validate
from test import evaluate, compute_metrics, print_metrics

def main(params=None):
    # Configuration dictionary
    if params is None:
        raise ValueError("Parameters must be provided when calling main().")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label file
    df_labels = pd.read_csv(params['labels_path'])
    labels = df_labels[params['label_column']].values
    subjects = df_labels['ID'].values

    # Split dataset into train/val/test
    train_subj, test_subj = train_test_split(subjects, test_size=params['test_size'], stratify=labels, random_state=42)
    train_labels = df_labels[df_labels['ID'].isin(train_subj)].reset_index(drop=True)
    test_labels = df_labels[df_labels['ID'].isin(test_subj)].reset_index(drop=True)

    val_subj, final_train_subj = train_test_split(train_subj, test_size=(1 - params['val_size']), stratify=labels[train_labels.index], random_state=42)
    val_labels = df_labels[df_labels['ID'].isin(val_subj)].reset_index(drop=True)
    final_train_labels = df_labels[df_labels['ID'].isin(final_train_subj)].reset_index(drop=True)

    # Dataloaders
    train_dataset = AugmentedFCDataset(params['data_dir_augmented'], df_train, params['label_column'],task='classification')
    val_dataset = FCDataset(params['data_dir'], df_val, params['label_column'], task='classification')
    test_dataset = FCDataset(params['data_dir'], test_labels, params['label_column'], task='classification')

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    # Model selection
    if params['model_type'] == 'resnet':
        model = ResNet3D(n_classes=2).to(device)
    elif params['model_type'] == 'densenet':
        model = DenseNet3D(n_classes=2).to(device)

    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_accuracy = -float('inf')
    best_epoch = -1
    train_losses, val_losses, val_accuracies = [], [], []

    # Training loop
    for epoch in range(params['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), params['checkpoint_path'])

    print(f"\nBest model saved from epoch {best_epoch} with validation accuracy: {best_accuracy:.4f}")

    # Test best model
    model.load_state_dict(torch.load(params['checkpoint_path'], map_location=device))
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

if __name__ == '__main__':

# Configuration dictionary
    params = {
        'data_dir_augmented': '/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed',
        'data_dir': '/data/users/etosato/ANM_Verona/data/FC_maps',
        'labels_path': '/data/users/etosato/ANM_Verona/data/labels.csv',
        'label_column': 'Group',
        'model_type': 'resnet',
        'epochs': 2,
        'batch_size': 4,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'val_size': 0.2,
        'test_size': 0.2,
        'checkpoint_path': 'best_model.pt'
    }

    main()
