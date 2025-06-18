# run.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import json
import random
import sys

from datasets import FCDataset, AugmentedFCDataset
from models import ResNet3D, DenseNet3D, VGG16_3D
from train import train, validate, plot_losses
from test import evaluate, compute_metrics, plot_confusion_matrix

from utils import (
    create_training_summary,
    create_testing_summary,
    resolve_split_csv_path
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Trains the model and evaluates on validation set at each epoch.
def run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold):
    best_accuracy = -float('inf')
    best_val_loss = float('inf')
    best_epoch = -1
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(params['epochs']):
        # Training and validation step
        train_loss = train(model, train_loader, criterion, optimizer, params['device'])
        val_loss, val_accuracy = validate(model, val_loader, criterion, params['device'])

        print(f"Epoch {epoch+1}/{params['epochs']} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_accuracy:.3f}")

        # Store losses and accuracy for later analysis
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save best epoch in a checkpoint based on validation accuracy
        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_val_loss):
            best_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_train_loss = train_loss
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'epoch': best_epoch,
                'best_train_loss': best_train_loss,
                'best_val_loss': best_val_loss,
                'fold': fold
            }, params['ckpt_path_evaluation'])

    print(f"\nBest model saved with val accuracy {best_accuracy:.3} at epoch {best_epoch}\n")

    # Save learning curves
    if params['plot']:
        title = f"Training curves - {params['group1'].upper()} vs {params['group2'].upper()} ({params['model_type'].upper()} - Fold {fold})"
        filename_base = f"{params['model_type']}_{params['group1']}_vs_{params['group2']}_fold_{fold}"
        temp_dir_out= os.path.join(params['actual_run_dir'], "plots")
        os.makedirs(temp_dir_out, exist_ok=True)

        # Plot without accuracy
        save_path = os.path.join(temp_dir_out, filename_base + "_loss.png")
        plot_losses(train_losses, val_losses, save_path=save_path, title=title)

        # Plot with accuracy
        title_acc = title + " accuracy"
        save_path_acc = os.path.join(temp_dir_out, filename_base + "_loss_acc.png")
        plot_losses(train_losses, val_losses, val_accuracies, save_path=save_path_acc, title=title_acc)

    if params.get('training_csv', False):
        # Save per-epoch training results for this fold in Excel
        df_fold = pd.DataFrame({
            'Epoch': list(range(1, params['epochs'] + 1)),
            'Train Loss': train_losses,
            'Val Loss': val_losses,
            'Val Accuracy': val_accuracies
        })

        excel_path = os.path.join(params['actual_run_dir'], "training_folds.xlsx")

        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_fold.to_excel(writer, index=False, sheet_name=f"Fold_{fold}")
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_fold.to_excel(writer, index=False, sheet_name=f"Fold_{fold}")


    return best_accuracy, best_train_loss, best_val_loss, best_epoch


def main_worker(params):
    # Handle run subdirectory
    if params.get("tuning_flag", False):
        ckpt_dir = params["runs_dir"]
    else:
        ckpt_dir = os.path.join(params["runs_dir"], f"run{params['run_id']}")

    os.makedirs(ckpt_dir, exist_ok=True)
    params["actual_run_dir"] = ckpt_dir

    # Re-direct the output
    if params.get("tuning_flag", False):
        run_id = params["run_id"]
        config_id = params["config_id"]
        log_filename = f"log_train_run{run_id}_config{config_id}"
    elif params['crossval_flag'] and not params['evaluation_flag']:
        log_filename = f"log_train{params['run_id']}"
    elif not params['crossval_flag'] and params['evaluation_flag']:
        log_filename = f"log_test{params['run_id']}"
    elif params['crossval_flag'] and params['evaluation_flag']:
        log_filename = f"log_total{params['run_id']}"
    else:
        log_filename = f"log_misc{params['run_id']}"

    log_path = os.path.join(ckpt_dir, log_filename)
    sys.stdout = open(log_path, "w")
    sys.stderr = sys.stdout
    sys.stdout.reconfigure(line_buffering=True)

    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['device'] = device

    # Set reproducibility
    set_seed(params['seed'])

    # Load precomputed train/val/test split
    split_csv_path = resolve_split_csv_path(params['split_dir'], params['group1'], params['group2'])
    df = pd.read_csv(split_csv_path)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True) if 'val' in df['split'].unique() else None
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # --- Training with Cross Validation mode ---
    if params['crossval_flag']:
        params["ckpt_path_evaluation"] = None

        # Extract subject IDs and labels
        subjects = train_df['ID'].values
        labels = train_df[params['label_column']].values

        # Stratified K-Fold setup
        skf = StratifiedKFold(n_splits=params['n_folds'], shuffle=True, random_state=params['seed'])

        # Variables
        best_fold_info = {
            'accuracy': -float('inf'),
            'val_loss': float('inf')
        }
        fold_accuracies = []
        fold_train_losses = []
        fold_val_losses = []
        fold_infos = []

        # Training
        print("================================")
        print("========== TRAINING ============")
        print("================================")
        for fold, (train_idx, val_idx) in enumerate(skf.split(subjects, labels)):
            print(f"\n--- Fold {fold + 1}/{params['n_folds']} ---")

            # Subset the dataframes
            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            # Datasets and loaders
            train_dataset = AugmentedFCDataset(params['data_dir_augmented'], fold_train_df, params['label_column'], task='classification')
            val_dataset = FCDataset(params['data_dir'], fold_val_df, params['label_column'], task='classification')

            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

            # Model, loss, optimizer
            if params['model_type'] == 'resnet':
                model = ResNet3D(n_classes=2).to(device)
            elif params['model_type'] == 'densenet':
                model = DenseNet3D(n_classes=2).to(device)
            elif params['model_type'] == 'vgg16':
                model = VGG16_3D(num_classes=2, input_channels=1).to(device)
            else:
                raise ValueError("Unsupported model type")

            # Optimizer
            criterion = torch.nn.CrossEntropyLoss()
            if params['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            elif params['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                            momentum=params.get('momentum', 0.9))
            else:
                raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

            # Run epochs
            params['ckpt_path_evaluation'] = os.path.join(params['actual_run_dir'],f"best_model_fold{fold + 1}.pt")
            best_accuracy, best_train_loss, best_val_loss, best_epoch = run_epochs(model, train_loader, val_loader, criterion, optimizer, params, fold + 1 )

            # Save and track best fold
            fold_accuracies.append(best_accuracy)
            fold_train_losses.append(best_train_loss)
            fold_val_losses.append(best_val_loss)
            fold_infos.append({
                'fold': fold + 1,
                'accuracy': best_accuracy,
                'val_loss': best_val_loss,
                'model_path': params['ckpt_path_evaluation'],
                'epoch': best_epoch
            })

        # Decide best fold after all folds are completed
        for info in fold_infos:
            if (
                    info['accuracy'] > best_fold_info['accuracy']
                    or (
                    info['accuracy'] == best_fold_info['accuracy'] and info['val_loss'] < best_fold_info['val_loss'])
            ):
                best_fold_info = info

        params['ckpt_path_evaluation'] = best_fold_info['model_path']

        print("=================================")
        print("=== CROSS VALIDATION SUMMARY ====")
        print("=================================")
        print(f"Run number            : {params['run_id']}")
        print(f"Group                 : {params['group1']} vs {params['group2']}")
        print(f"Best fold             : {best_fold_info['fold']}")
        print(f"Best epoch            : {best_fold_info['epoch']}")
        print(f"Best accuracy         : {best_fold_info['accuracy']:.3f}")
        print(f"Best validation loss  : {best_fold_info['val_loss']:.3f}")
        print(f"Average accuracy      : {np.mean(fold_accuracies):.3f}")
        print(f"Average training loss : {np.mean(fold_train_losses):.3f}")
        print(f"Average val loss      : {np.mean(fold_val_losses):.3f}")
        print(f"Best model path       : {best_fold_info['model_path']}\n")

        print("NETWORK INFORMATION: ")
        print(f"Model type            : {params['model_type']}")
        print(f"Epochs                : {params['epochs']}")
        print(f"Batch size            : {params['batch_size']}")
        print(f"Optimizer             : {params['optimizer']}")
        print(f"Learning rate         : {params['lr']}")
        print(f"Weight decay          : {params['weight_decay']}")

        if not params.get("tuning_flag", False):
            summary_path = os.path.join(params['runs_dir'], "all_training_results.csv")
            row_summary = create_training_summary(params, best_fold_info, fold_accuracies, fold_val_losses, fold_train_losses)

            df_summary = pd.DataFrame([row_summary])

            # Create or append
            if os.path.exists(summary_path):
                df_summary.to_csv(summary_path, mode='a', header=False, index=False)
            else:
                df_summary.to_csv(summary_path, index=False)

        # Return results for fine-tuning
        if params['tuning_flag']:
            return {
                'best_fold': best_fold_info['fold'],
                'best_accuracy': best_fold_info['accuracy'],
                'avg_accuracy': float(np.mean(fold_accuracies)),
                'avg_train_loss': float(np.mean(fold_train_losses)),
                'avg_val_loss': float(np.mean(fold_val_losses))
            }


    # --- Evaluation mode ---
    if params['evaluation_flag']:
        # Prepare test dataloader
        test_dataset = FCDataset(params['data_dir'], test_df, params['label_column'], task='classification')
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

        # Load model and weights
        if params['model_type'] == 'resnet':
            model = ResNet3D(n_classes=2).to(device)
        elif params['model_type'] == 'densenet':
            model = DenseNet3D(n_classes=2).to(device)
        elif params['model_type'] == 'vgg16':
            model = VGG16_3D(num_classes=2, input_channels=1).to(device)
        else:
            raise ValueError("Unsupported model type")

        # Load checkpoint
        checkpoint = torch.load(params['ckpt_path_evaluation'], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        model.to(device)

        # Run evaluation
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        # Compute group counts in test set
        group_counts = test_df[params['label_column']].value_counts().to_dict()

        print("===========================")
        print("=== EVALUATION SUMMARY ====")
        print("===========================")
        print(f"Model path: {params['ckpt_path_evaluation']}\n")
        print(f"Model type: {params['model_type']}")
        print(f"Best fold: {checkpoint.get('fold', '-')}")
        print(f"Best epoch: {checkpoint.get('epoch', '-')}\n")
        print(f"Test set size: {len(test_df)}")
        print(f"{params['group1']}: {group_counts.get(params['group1'], 0)} subjects")
        print(f"{params['group2']}: {group_counts.get(params['group2'], 0)} subjects\n")
        print("Metrics on test set:")
        metrics_main = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        max_key_len = max(len(k) for k in metrics_main)
        for k, v in metrics_main.items():
            print(f"{k:<{max_key_len}} : {v:.3f}")

        # CSV summary
        results_path = os.path.join(params['runs_dir'], "all_testing_results.csv")
        row = create_testing_summary(params, metrics)
        df = pd.DataFrame([row])

        # Create or append
        if os.path.exists(results_path):
            df.to_csv(results_path, mode='a', header=False, index=False, float_format="%.3f")
        else:
            df.to_csv(results_path, index=False, float_format="%.3f")

        # Save confusion matrix
        if params.get('plot'):
            title = f"Confusion Matrix - {params['group1'].upper()} vs {params['group2'].upper()} ({params['model_type'].upper()})"
            filename = f"{params['model_type']}_{params['group1']}_vs_{params['group2']}_conf_matrix.png"
            save_path = os.path.join(params['actual_run_dir'], filename)
            class_names = sorted(test_df[params['label_column']].unique())
            plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path= save_path, title = title)

    return None


if __name__ == '__main__':
    # Load json file
    config_path = "parameters/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    args = {**config["paths"], **config["training"], **config["fixed"], **config["experiment"]}

    # Some checks
    if args['crossval_flag'] and args['tuning_flag'] and args['evaluation_flag']:
        raise ValueError("Invalid config: Cannot run training + evaluation with tuning_flag=True")

    main_worker(args)

