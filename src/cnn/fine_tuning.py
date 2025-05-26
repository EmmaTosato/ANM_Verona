# fine_tuning.py

import json
import os
from itertools import product
from run import main_worker

# Define the hyperparameter grid
grid = {
    'lr': [1e-4, 1e-3],
    'batch_size': [4, 8],
    'weight_decay': [1e-5, 1e-4],
    'model_type': ['resnet', 'densenet']
}

# Fixed parameters
fixed_params = {
    'data_dir_augmented': '/data/users/etosato/ANM_Verona/data/FCmaps_augmented_processed',
    'data_dir': '/data/users/etosato/ANM_Verona/data/FC_maps_processsed',
    'label_column': 'Group',
    'epochs': 2,
    'n_folds': 2,
    'seed': 42,
    'crossval_flag': True,
    'evaluation_flag': False,
    'plot': False,
    'split_csv': '/data/users/etosato/ANM_Verona/data/ADNI_PSP_splitted.csv',
    'group1': 'ADNI',
    'group2': 'PSP',
    'checkpoints_dir': '/data/users/etosato/ANM_Verona/src/cnn/checkpoints',
    'plot_dir': '/data/users/etosato/ANM_Verona/src/cnn/output'
}

# Output path for results
output_json = os.path.join(fixed_params['plot_dir'], 'grid_results.json')
os.makedirs(fixed_params['plot_dir'], exist_ok=True)

# Convert grid to list of combinations
keys = list(grid.keys())
combinations = list(product(*grid.values()))

# Storage for results
all_results = []

# Iterate over each combination
for i, combo in enumerate(combinations):
    print(f"\nRunning combination {i+1}/{len(combinations)}")

    # Create param dictionary
    combo_params = dict(zip(keys, combo))
    params = {**fixed_params, **combo_params}

    # Set checkpoint path for this run
    ckpt_name = f"{params['model_type']}_lr{params['lr']}_bs{params['batch_size']}_wd{params['weight_decay']}".replace('.', '')
    params['checkpoint_path'] = os.path.join(params['checkpoints_dir'], f"{ckpt_name}_best_model.pt")

    # Run training with CV
    result = main_worker(params)

    # Attach hyperparameters to results
    result.update(combo_params)
    all_results.append(result)

# Save all results to JSON
with open(output_json, 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\nSaved all results to {output_json}")
