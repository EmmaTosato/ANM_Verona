# fine_tuning.py

# TODO: controlla che salvi tutto giusto
import json
import os
import pandas as pd
from itertools import product
from run import main_worker

# Define the hyperparameter grid
with open("parameters/grid.json", "r") as f:
    grid = json.load(f)

# Fixed parameters
with open("parameters/fixed_finetuning.json", "r") as f:
    fixed_params = json.load(f)


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

    # Create folder
    params['checkpoint_path'] = os.path.join(config_dir, "best_model.pt")

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
