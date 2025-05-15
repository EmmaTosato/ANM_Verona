# run.py
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D
from train import train, validate
from test import evaluate, compute_metrics, print_metrics


def run_epochs(model, train_loader, val_loader, criterion, optimizer, device, epochs, checkpoint_path):
    best_accuracy = -float('inf')
    best_epoch = -1
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

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
            }, checkpoint_path)

    print(f"Best model saved with val accuracy {best_accuracy:.4f} at epoch {best_epoch}")

    return best_accuracy, best_epoch, train_losses, val_losses, val_accuracies


def main_worker(params, crossval=True, evaluate=False):
    os.makedirs(params['checkpoints_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels and apply exclusions
    df_labels = pd.read_csv(params['labels_path'])
    to_exclude = params.get('to_exclude', [])
    if to_exclude:
        df_labels = df_labels[~df_labels['ID'].isin(to_exclude)].reset_index(drop=True)

    df_pair = df_labels[df_labels['Group'].isin([params['group1'], params['group2']])].reset_index(drop=True)

    # Explicit train/test split
    subjects = df_pair['ID'].values
    labels = df_pair[params['label_column']].values

    train_subj, test_subj = train_test_split(
        subjects,
        stratify=labels,
        test_size=params.get('test_size', 0.2),
        random_state=42
    )

    train_df = df_pair[df_pair['ID'].isin(train_subj)].reset_index(drop=True)
    test_df = df_pair[df_pair['ID'].isin(test_subj)].reset_index(drop=True)

    if evaluate:
        # Evaluation on test set
        test_dataset = FCDataset(params['data_dir'], test_df, params['label_column'], task='classification')
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        if params['model_type'] == 'resnet':
            model = ResNet3D(n_classes=len(df_pair[params['label_column']].unique()))
        elif params['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=len(df_pair[params['label_column']].unique()))
        else:
            raise ValueError("Unsupported model type")

        checkpoint = torch.load(params['checkpoint_path'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        model.to(device)

        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(metrics)
        return

    # Cross-validation on training set only
    train_subjects = train_df['ID'].values
    train_labels = train_df[params['label_column']].values

    if crossval:
        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=42)
        best_fold_info = {'accuracy': -float('inf')}

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_subjects, train_labels)):
            print(f"\n--- Fold {fold + 1}/{params['n_folds']} ---")

            fold_train_ids = train_subjects[train_idx]
            fold_val_ids = train_subjects[val_idx]

            fold_train_df = train_df[train_df['ID'].isin(fold_train_ids)].reset_index(drop=True)
            fold_val_df = train_df[train_df['ID'].isin(fold_val_ids)].reset_index(drop=True)

            train_dataset = AugmentedFCDataset(params['data_dir_augmented'], fold_train_df, params['label_column'], task='classification')
            val_dataset = FCDataset(params['data_dir'], fold_val_df, params['label_column'], task='classification')

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            if params['model_type'] == 'resnet':
                model = ResNet3D(n_classes=len(df_pair[params['label_column']].unique())).to(device)
            elif params['model_type'] == 'densenet':
                model = DenseNet3D(n_classes=len(df_pair[params['label_column']].unique())).to(device)
            else:
                raise ValueError("Unsupported model type")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

            best_model_path = os.path.join(params['checkpoints_dir'], f"best_model_fold{fold+1}.pt")

            best_accuracy, best_epoch, train_losses, val_losses, val_accuracies = run_epochs(
                model, train_loader, val_loader, criterion, optimizer, device,
                epochs=params['epochs'], checkpoint_path=best_model_path
            )

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

    main_worker(params, crossval=True, evaluate=False)
