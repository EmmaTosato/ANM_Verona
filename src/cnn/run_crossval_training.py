# run_crossval_training.py
import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, VGG3D, AlexNet3D
from train import train, validate
from test import evaluate, compute_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--label_column', type=str, default='Group')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vgg', 'alexnet'], default='resnet')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    df_labels = pd.read_csv(args.labels_path)
    subjects = df_labels['ID'].values
    labels = df_labels[args.label_column].values

    # For k-fold cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    best_fold_info = {'accuracy': -float('inf')}

    # Loop over the k-fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")

        train_ids = subjects[train_idx]
        val_ids = subjects[val_idx]

        df_train = df_labels[df_labels['ID'].isin(train_ids)].reset_index(drop=True)
        df_val = df_labels[df_labels['ID'].isin(val_ids)].reset_index(drop=True)

        # Create Datasets
        train_dataset = AugmentedFCDataset(args.data_dir, df_train, args.label_column, task='classification')
        val_dataset = FCDataset(args.data_dir, df_val, args.label_column, task='classification')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create the model and optimizer
        if args.model_type == 'resnet':
            model = ResNet3D(n_classes=2).to(device)
        elif args.model_type == 'vgg':
            model = VGG3D(n_classes=2).to(device)
        elif args.model_type == 'alexnet':
            model = AlexNet3D(n_classes=2).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Variables
        best_accuracy = -float('inf')
        best_epoch = -1
        best_model_path = os.path.join(args.checkpoints_dir, f"best_model_fold{fold+1}.pt")

        # Loop over epochs
        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch + 1
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'epoch': best_epoch
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
    main()
