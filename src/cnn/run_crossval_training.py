import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D
from train import train, validate
from test import evaluate, compute_metrics, print_metrics

def main(params=None):
    if params is None:
        raise ValueError("Parameters must be provided to main()")

    os.makedirs(params['checkpoints_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_labels = pd.read_csv(params['labels_path'])

    # Temporary exclusion of problematic subjects
    to_exclude = ['3_S_5003', '4_S_5003', '4_S_5005', '4_S_5007', '4_S_5008']
    df_labels = df_labels[~df_labels['ID'].isin(to_exclude)].reset_index(drop=True)

    df_labels = df_labels[df_labels['Group'].isin([params['group1'], params['group2']])].reset_index(drop=True)

    subjects = df_labels['ID'].values
    labels = df_labels[params['label_column']].values

    skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=42)
    best_fold_info = {'accuracy': -float('inf')}

    for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
        print(f"\n--- Fold {fold + 1}/{params['n_folds']} ---")

        train_ids = subjects[train_idx]
        val_ids = subjects[val_idx]

        df_train = df_labels[df_labels['ID'].isin(train_ids)].reset_index(drop=True)
        df_val = df_labels[df_labels['ID'].isin(val_ids)].reset_index(drop=True)

        train_dataset = AugmentedFCDataset(params['data_dir_augmented'], df_train, params['label_column'], task='classification')
        val_dataset = FCDataset(params['data_dir'], df_val, params['label_column'], task='classification')

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        if params['model_type'] == 'resnet':
            model = ResNet3D(n_classes=2).to(device)
        elif params['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=2).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        best_accuracy = -float('inf')
        best_epoch = -1
        best_model_path = os.path.join(params['checkpoints_dir'], f"best_model_fold{fold+1}.pt")

        train_losses = []
        val_losses = []
        val_accuracies = []

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
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'epoch': best_epoch,
                    'metrics': {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'val_accuracies': val_accuracies
                    }
                }, best_model_path)

        print(f"Best model for fold {fold+1} saved with val accuracy {best_accuracy:.4f} at epoch {best_epoch}")

        if best_accuracy > best_fold_info['accuracy']:
            best_fold_info = {
                'fold': fold + 1,
                'accuracy': best_accuracy,
                'model_path': best_model_path
            }

    print("\n=== Best Fold Summary ===")
    print(f"Fold: {best_fold_info['fold']}")
    print(f"Accuracy: {best_fold_info['accuracy']:.4f}")
    print(f"Model path: {best_fold_info['model_path']}")

if __name__ == '__main__':
    params = {
    'data_dir_augmented': '/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed',
    'data_dir': '/data/users/etosato/ANM_Verona/data/FC_maps',
    'labels_path': '/data/users/etosato/ANM_Verona/data/labels.csv',
    'label_column': 'Group',
    'group1': 'ADNI',
    'group2': 'PSP',
    'model_type': 'resnet',
    'epochs': 2,
    'batch_size': 4,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'n_folds': 2,
    'checkpoints_dir': '/data/users/etosato/ANM_Verona/src/cnn/checkpoints'
    }

    main(params)
