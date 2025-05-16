# run.py
import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D
from train import train, validate, plot_losses
from test import evaluate, compute_metrics, print_metrics


def save_best_model(model, path):
    torch.save(model.state_dict(), path)


def run_epochs(model, train_loader, val_loader, criterion, optimizer, params):
    best_accuracy = -float('inf')
    best_epoch = -1
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(params['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, params['device'])
        val_loss, val_accuracy = validate(model, val_loader, criterion, params['device'])

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
            }, params['checkpoint_path'])

    print(f"\nBest model saved with val accuracy {best_accuracy:.4f} at epoch {best_epoch}")

    if params['plot_path']:
        plot_losses(train_losses, val_losses, val_accuracies)
        plt.savefig(params['plot_path'])
        plt.close()

    return best_accuracy, best_epoch, train_losses, val_losses, val_accuracies

def main_worker(params):
    os.makedirs(params['checkpoints_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['device'] = device

    df = pd.read_csv(params['split_csv'])
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True) if 'val' in df['split'].unique() else None
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    if params['evaluation_flag']:
        test_dataset = FCDataset(params['data_dir'], test_df, params['label_column'], task='classification')
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        if params['model_type'] == 'resnet':
            model = ResNet3D(n_classes=2)
        elif params['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=2)
        else:
            raise ValueError("Unsupported model type")

        checkpoint = torch.load(params['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        model.to(device)

        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(metrics)
        return

    if params['crossval_flag']:
        subjects = train_df['ID'].values
        labels = train_df[params['label_column']].values

        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['seed'])
        best_fold_info = {'accuracy': -float('inf')}
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
            print(f"\n--- Fold {fold + 1}/{params['n_folds']} ---")

            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            train_dataset = AugmentedFCDataset(params['data_dir_augmented'], fold_train_df, params['label_column'], task='classification')
            val_dataset = FCDataset(params['data_dir'], fold_val_df, params['label_column'], task='classification')

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            if params['model_type'] == 'resnet':
                model = ResNet3D(n_classes=2).to(device)
            elif params['model_type'] == 'densenet':
                model = DenseNet3D(n_classes=2).to(device)
            else:
                raise ValueError("Unsupported model type")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            params['checkpoint_path'] = os.path.join(params['checkpoints_dir'], f"best_model_fold{fold+1}.pt")
            best_accuracy, _, _, _, _ = (run_epochs
                                         (model, train_loader, val_loader, criterion, optimizer, params))
            fold_accuracies.append(best_accuracy)

            if best_accuracy > best_fold_info['accuracy']:
                best_fold_info = {
                    'fold': fold + 1,
                    'accuracy': best_accuracy,
                    'model_path': params['checkpoint_path']
                }

        print("\n=== Cross-Validation Summary ===")
        print(f"Best fold: {best_fold_info['fold']}")
        print(f"Best accuracy: {best_fold_info['accuracy']:.4f}")
        print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")

        # Copy best model on all folds
        shutil.copy(best_fold_info['model_path'], os.path.join(params['checkpoints_dir'], "best_model_overall.pt"))

    else:
        train_dataset = AugmentedFCDataset(params['data_dir_augmented'], train_df, params['label_column'], task='classification')
        val_dataset = FCDataset(params['data_dir'], val_df, params['label_column'], task='classification') if val_df is not None else None

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False) if val_df is not None else None

        if params['model_type'] == 'resnet':
            model = ResNet3D(n_classes=2).to(device)
        elif params['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=2).to(device)
        else:
            raise ValueError("Unsupported model type")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        run_epochs(model, train_loader, val_loader, criterion, optimizer, params)


if __name__ == '__main__':
    args = {
        'data_dir_augmented': '/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed',
        'data_dir': '/data/users/etosato/ANM_Verona/data/FC_maps_processsed',
        'label_column': 'Group',
        'model_type': 'resnet',
        'epochs': 2,
        'batch_size': 4,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'n_folds': 2,
        'seed': 42,

        'checkpoints_dir': '/data/users/etosato/ANM_Verona/src/cnn/checkpoints',
        'checkpoint_path': '/data/users/etosato/ANM_Verona/src/cnn/checkpoints/best_model.pt',
        'plot_path': '/data/users/etosato/ANM_Verona/src/cnn/output/',

        'split_csv': '/data/users/etosato/ANM_Verona/data/ADNI_PSP_splitted.csv',
        'group1': 'ADNI',
        'group2': 'PSP',

        'crossval_flag': True,
        'evaluation_flag': False
    }

    main_worker(args)
