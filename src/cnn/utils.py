import numpy as np

# Create a summary of the cross-validation training process
def create_training_summary(params, best_fold_info, fold_accuracies, fold_val_losses):
    return {
        'run id': f"run{params['run_id']}",
        'group': f"{params['group1']} vs {params['group2']}",
        'threshold': params.get("threshold", "unspecified"),
        'best fold': best_fold_info['fold'],
        'best epoch': best_fold_info['epoch'],
        'best accuracy': round(best_fold_info['accuracy'], 4),
        'average accuracy': round(float(np.mean(fold_accuracies)), 4),
        'average validation loss': round(float(np.mean(fold_val_losses)), 4),
        'model_type': params['model_type'],
        'optimizer': params['optimizer'],
        'lr': params['lr'],
        'batch_size': params['batch_size'],
        'weight_decay': params['weight_decay'],
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
        'best_accuracy': metrics['best_accuracy'],
        'avg_accuracy': metrics['avg_accuracy'],
        'avg_train_loss': metrics['avg_train_loss'],
        'avg_val_loss': metrics['avg_val_loss'],
        'optimizer': params['optimizer'],
        'batch_size': params['batch_size'],
        'lr': params['lr'],
        'weight_decay': params['weight_decay'],
        'model_type': params['model_type'],
        'epochs': params['epochs'],
        'test size': params['test_size']
    }

# Create a summary of model performance on the held-out test set
def create_testing_summary(params, metrics):
    summary = {
        'run id': f"run{params['run_id']}",
        'group': f"{params['group1']} vs {params['group2']}",
        'threshold': params.get("threshold", "unspecified"),
        "test size": round(params['test_size'], 1)
    }
    metrics_rounded = {k: round(v, 3) for k, v in metrics.items() if k != "confusion_matrix"}
    summary.update(metrics_rounded)
    return summary
