# run_training.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

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
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    df_labels = pd.read_csv(args.labels_path)
    labels = df_labels[args.label_column].values
    subjects = df_labels['ID'].values

    # Train/Val/Test split
    train_subj, test_subj = train_test_split(subjects, test_size=args.test_size, stratify=labels, random_state=42)
    train_labels = df_labels[df_labels['ID'].isin(train_subj)].reset_index(drop=True)
    test_labels = df_labels[df_labels['ID'].isin(test_subj)].reset_index(drop=True)

    val_subj, final_train_subj = train_test_split(train_subj, test_size=(1 - args.val_size), stratify=labels[train_labels.index], random_state=42)
    val_labels = df_labels[df_labels['ID'].isin(val_subj)].reset_index(drop=True)
    final_train_labels = df_labels[df_labels['ID'].isin(final_train_subj)].reset_index(drop=True)

    # Datasets and Loaders
    train_dataset = AugmentedFCDataset(args.data_dir, final_train_labels, args.label_column, task='classification')
    val_dataset = FCDataset(args.data_dir, val_labels, args.label_column, task='classification')
    test_dataset = FCDataset(args.data_dir, test_labels, args.label_column, task='classification')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model selection
    if args.model_type == 'resnet':
        model = ResNet3D(n_classes=2).to(device)
    elif args.model_type == 'vgg':
        model = VGG3D(n_classes=2).to(device)
    elif args.model_type == 'alexnet':
        model = AlexNet3D(n_classes=2).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training Loop
    best_accuracy = -float('inf')
    best_epoch = -1

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.checkpoint_path)

    print(f"\nBest model saved from epoch {best_epoch} with validation accuracy: {best_accuracy:.4f}")

    # Test best model
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics)

if __name__ == '__main__':
    main()
