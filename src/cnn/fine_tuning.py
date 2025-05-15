# fine_tuning.py
import os
import itertools
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D
from train import train, validate


class CNNGridSearch:
    def __init__(self, df_labels, data_dir, label_column, param_grid, n_folds=5, output_dir='tuning_results'):
        self.df_labels = df_labels
        self.data_dir = data_dir
        self.label_column = label_column
        self.param_grid = list(itertools.product(*param_grid.values()))
        self.param_names = list(param_grid.keys())
        self.n_folds = n_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.subjects = df_labels['ID'].values
        self.labels = df_labels[label_column].values

    def build_model(self, model_type, n_classes):
        if param_grid['model_type'] == 'resnet':
            model = ResNet3D(n_classes=2).to(self.device)
        elif param_grid['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=2).to(self.device)
        else:
            raise ValueError("Invalid model type")
        return model.to(self.device)

    def run_fold(self, model, df_train, df_val, params):
        train_dataset = AugmentedFCDataset(self.data_dir_augmented, df_train, self.label_column, task='classification')
        val_dataset = FCDataset(self.data_dir, df_val, self.label_column, task='classification')

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(params['optimizer'], model.parameters(), params['lr'], params['weight_decay'])

        best_acc = -np.inf
        best_epoch = -1
        train_losses, val_losses, val_accuracies = [], [], []

        for epoch in range(params['epochs']):
            train_loss = train(model, train_loader, criterion, optimizer, self.device)
            val_loss, val_acc = validate(model, val_loader, criterion, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': best_acc,
            'best_epoch': best_epoch
        }, model

    def get_optimizer(self, name, parameters, lr, weight_decay):
        if name == 'adam':
            return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        elif name == 'sgd':
            return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer")

    def plot_losses(self, train_losses, val_losses, save_path):
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        summary_records = []

        for config_id, values in enumerate(tqdm(self.param_grid, desc="Grid Search")):
            config = dict(zip(self.param_names, values))
            config_name = f"config_{config_id:04d}"
            config_path = os.path.join(self.output_dir, config_name)
            os.makedirs(config_path, exist_ok=True)

            fold_accuracies = []
            fold_train_losses = []
            fold_val_losses = []
            fold_val_accuracies = []
            log_rows = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(self.subjects, self.labels)):
                train_ids = self.subjects[train_idx]
                val_ids = self.subjects[val_idx]

                df_train = self.df_labels[self.df_labels['ID'].isin(train_ids)].reset_index(drop=True)
                df_val = self.df_labels[self.df_labels['ID'].isin(val_ids)].reset_index(drop=True)

                model = self.build_model(config['model_type'], n_classes=2)
                fold_result, trained_model = self.run_fold(model, df_train, df_val, config)

                # Save model with metrics inside
                model_path = os.path.join(config_path, f"model_fold{fold+1}.pt")
                torch.save({
                    'state_dict': trained_model.state_dict(),
                    'metrics': fold_result,
                    'fold': fold + 1,
                    'params': config
                }, model_path)

                # Accumulate logs
                fold_accuracies.append(fold_result['best_accuracy'])
                fold_train_losses.append(np.mean(fold_result['train_losses']))
                fold_val_losses.append(np.mean(fold_result['val_losses']))
                fold_val_accuracies.append(np.mean(fold_result['val_accuracies']))

                for epoch, (tr, va, acc) in enumerate(zip(fold_result['train_losses'], fold_result['val_losses'], fold_result['val_accuracies'])):
                    log_rows.append({
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'train_loss': tr,
                        'val_loss': va,
                        'val_accuracy': acc
                    })

                # Final plot for this fold
                self.plot_losses(fold_result['train_losses'], fold_result['val_losses'], os.path.join(config_path, f"loss_plot_fold{fold+1}.png"))

            # Save log.csv
            df_log = pd.DataFrame(log_rows)
            df_log.to_csv(os.path.join(config_path, "log.csv"), index=False)

            # Save summary.json (one per combination)
            combination_summary = {
                'config_id': config_name,
                'params': config,
                'fold_accuracies': fold_accuracies,
                'fold_train_losses': fold_train_losses,
                'fold_val_losses': fold_val_losses,
                'fold_val_accuracies': fold_val_accuracies
            }
            with open(os.path.join(config_path, "combination_summary.json"), 'w') as f:
                json.dump(combination_summary, f, indent=4)

            # Add to global summary
            summary_records.append({
                'config_id': config_name,
                **config,
                'mean_train_loss': np.mean(fold_train_losses),
                'mean_val_loss': np.mean(fold_val_losses),
                'mean_val_accuracy': np.mean(fold_val_accuracies),
                'std_val_accuracy': np.std(fold_val_accuracies),
                'path': config_path
            })

        # Save global grid summary
        df_summary = pd.DataFrame(summary_records)
        df_summary.to_csv(os.path.join(self.output_dir, "grid_summary.csv"), index=False)
        print("\nGrid search completed. Summary saved.")


if __name__ == '__main__':
    import argparse
    df = pd.read_csv('path/to/labels.csv')
    data_dir = 'path/to/fcmaps'
    data_dir_augmented = 'path/to/augmented_fcmaps'

    param_grid = {
        'model_type': ['resnet', 'densenet', 'medmamba'],
        'lr': [1e-3, 1e-4],
        'batch_size': [8, 16],
        'optimizer': ['adam', 'sgd'],
        'weight_decay': [1e-5, 1e-4],
        'epochs': [20],
        'channels': [8, 16, 32]
    }

    grid_search = CNNGridSearch(
        df_labels=df,
        data_dir=data_dir,
        label_column='Group',
        param_grid=param_grid,
        n_folds=5
    )

    grid_search.run()
