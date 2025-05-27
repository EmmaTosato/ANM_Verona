# fine_tuning.py

import json
import os
import pandas as pd
from itertools import product
from run import main_worker

# Define the hyperparameter and parameters
with open("/data/users/etosato/ANM_Verona/src/cnn/parameters/grid.json", "r") as f:
    grid = json.load(f)

with open("/data/users/etosato/ANM_Verona/src/cnn/parameters/config_tuning.json", "r") as f:
    fixed_params = json.load(f)

# Prepare combinations and storage for results
keys = list(grid.keys())
combinations = list(product(*grid.values()))
all_results = []

# Iterate over each combination
for i, combo in enumerate(combinations):
    config_name = f"config{i+1}"
    print(f"\nRunning {config_name}...")

    # Build specific parameter set
    combo_params = dict(zip(keys, combo))
    params = {**fixed_params, **combo_params}

    # Create output directory for this configuration
    config_dir = os.path.join(fixed_params['tuning_results_dir'], config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Create folder
    params['checkpoint_path'] = os.path.join(config_dir, "best_model_overall.pt")
    params['checkpoints_dir'] = config_dir
    params['plot_dir'] = config_dir

    # Run training
    result = main_worker(params)

    # Add metadata to results
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(combo_params, f, indent=4)
    result['config'] = config_name
    all_results.append(result)

# Save results to CSV
results_df = pd.DataFrame(all_results)

# Reorder columns to put 'config' first
columns = ['config'] + [col for col in results_df.columns if col != 'config']
results_df = results_df[columns]
results_df.to_csv(os.path.join(fixed_params['tuning_results_dir'], 'grid_results.csv'), index=False)

print(f"\n\n")
print(f"----------------------------------------------------------------")
print(f"Fine tuning completed")
print(f"----------------------------------------------------------------")
