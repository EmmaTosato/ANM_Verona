import numpy as np
import os

# Create a summary of the cross-validation training process
def create_training_summary(params, best_fold_info, fold_accuracies, fold_val_losses, fold_train_losses):
    return {
        'run id': f"run{params['run_id']}",
        'group': f"{params['group1']} vs {params['group2']}",
        'seed': params['seed'],
        'threshold': params.get("threshold", "unspecified"),
        'best fold': best_fold_info['fold'],
        'best epoch': best_fold_info['epoch'],
        'best accuracy': round(best_fold_info['accuracy'], 3),
        'best validation loss': round(best_fold_info['val_loss'], 3),
        'average accuracy': round(float(np.mean(fold_accuracies)), 3),
        'average training loss': round(float(np.mean(fold_train_losses)), 3),
        'average validation loss': round(float(np.mean(fold_val_losses)), 3),
        'model_type': params['model_type'],
        'optimizer': params['optimizer'],
        'lr': round(params['lr'], 3),
        'batch_size': params['batch_size'],
        'weight_decay': round(params['weight_decay'], 3),
        'epochs': params['epochs'],
        'test size': params['test_size']
    }

# Create a summary row for a single hyperparameter tuning configuration
def create_tuning_summary(config_id, params, metrics):
    return {
        'config': f"config{config_id}",
        'group': f"{params['group1']} vs {params['group2']}",
        'threshold': params.get("threshold", "unspecified"),
        'best_fold': metrics['best_fold'],
        'best_accuracy': round(metrics['best_accuracy'], 3),
        'avg_accuracy': round(metrics['avg_accuracy'], 3),
        'avg_train_loss': round(metrics['avg_train_loss'], 3),
        'avg_val_loss': round(metrics['avg_val_loss'], 3),
        'optimizer': params['optimizer'],
        'batch_size': params['batch_size'],
        'lr': round(params['lr'], 3),
        'weight_decay': round(params['weight_decay'], 3),
        'model_type': params['model_type'],
        'epochs': params['epochs'],
        'test size': params['test_size']
    }

# Create a summary of model performance on the held-out test set
def create_testing_summary(params, metrics):
    summary = {
        'run_id': f"run{params['run_id']}",
        'group': f"{params['group1']} vs {params['group2']}",
        "seed": round(params['seed'])
    }
    metrics_rounded = {k: round(v, 3) for k, v in metrics.items() if k != "confusion_matrix"}
    summary.update(metrics_rounded)
    return summary


def resolve_split_csv_path(split_dir, group1, group2):
    fname1 = f"{group1}_{group2}_splitted.csv"
    fname2 = f"{group2}_{group1}_splitted.csv"

    path1 = os.path.join(split_dir, fname1)
    path2 = os.path.join(split_dir, fname2)

    if os.path.exists(path1):
        print(f"\nUsing split file: {path1}\n")
        return path1
    elif os.path.exists(path2):
        print(f"\nUsing split file: {path2}\n")
        return path2
    else:
        raise FileNotFoundError(f"No split CSV found for groups {group1} and {group2} in {split_dir}\n")
