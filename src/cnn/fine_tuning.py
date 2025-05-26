# fine_tuning.py

import json
import os
import pandas as pd
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
    'plot': True,
    'split_csv': '/data/users/etosato/ANM_Verona/data/ADNI_PSP_splitted.csv',
    'group1': 'ADNI',
    'group2': 'PSP'
}

# Folder where each config's results will be saved
tuning_root = '/data/users/etosato/ANM_Verona/src/cnn/tuning_results'
os.makedirs(tuning_root, exist_ok=True)

# Prepare combinations
keys = list(grid.keys())
combinations = list(product(*grid.values()))

# Storage for results
all_results = []

# Iterate over each combination
for i, combo in enumerate(combinations):
    config_name = f"config{i+1}"
    print(f"\nRunning {config_name}...")

    # Build specific parameter set
    combo_params = dict(zip(keys, combo))
    params = {**fixed_params, **combo_params}

    # Create output directory for this configuration
    config_dir = os.path.join(tuning_root, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Override save paths
    params['checkpoints_dir'] = config_dir
    params['plot_dir'] = config_dir
    params['checkpoint_path'] = os.path.join(config_dir, "best_model.pt")
    params['fine_tuning_flag'] = True

    # Run training
    result = main_worker(params)

    # Add metadata to results
    result.update(combo_params)
    result['config'] = config_name
    all_results.append(result)

# Save results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(tuning_root, 'grid_results.csv'), index=False)

print(f"\nSaved all results to: {os.path.join(tuning_root, 'grid_results.csv')}")
