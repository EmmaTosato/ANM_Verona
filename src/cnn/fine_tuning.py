# fine_tuning.py

import json
import os
import pandas as pd
from itertools import product
from run import main_worker

def tuning(fixed_params, grid):

    # Prepare combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    all_results = []

    for i, combo in enumerate(combinations):
        config_name = f"config{i+1}"
        print(f"\nRunning {config_name}...")

        # Merge fixed and grid parameters
        combo_params = dict(zip(keys, combo))
        params = {**fixed_params, **combo_params}

        # Set up output directories
        config_dir = os.path.join(fixed_params['tuning_results_dir'], config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Overwrite
        params['checkpoint_id'] = f"tuning{i + 1}"
        params['checkpoints_dir'] = config_dir
        params['plot_dir'] = config_dir

        # Train and validate
        result = main_worker(params)

        # Save configuration and collect results
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(combo_params, f, indent=4)
        result['config'] = config_name
        result.update(combo_params)
        all_results.append(result)

    # Save all results
    results_df = pd.DataFrame(all_results)
    columns = ['config'] + [col for col in results_df.columns if col != 'config']
    results_df = results_df[columns]
    results_df.to_csv(os.path.join(fixed_params['tuning_results_dir'], 'grid_results.csv'), index=False)

    print(f"\n\n----------------------------------------------------------------")
    print(f"Fine tuning completed")
    print(f"----------------------------------------------------------------")


if __name__ == '__main__':
    config_path = "/data/users/etosato/ANM_Verona/src/cnn/parameters/config_tuning.json"
    grid_path = "/data/users/etosato/ANM_Verona/src/cnn/parameters/grid.json"

    with open(grid_path, "r") as f:
        grid = json.load(f)
    with open(config_path, "r") as f:
        fixed_params = json.load(f)

    tuning(fixed_params, grid)
