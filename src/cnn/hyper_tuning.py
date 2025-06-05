import json
import os
import pandas as pd
from itertools import product
from copy import deepcopy
from run import main_worker

def tuning(base_args_path, grid_path):
    # Load fixed config and hyperparameter grid
    with open(base_args_path, "r") as f:
        args = json.load(f)

    with open(grid_path, "r") as f:
        grid = json.load(f)

    # Set the running directory
    run_id = args["run_id"]  # fixed ID for this entire tuning execution
    tuning_results_dir = args["tuning_results_dir"]
    run_dir = os.path.join(tuning_results_dir, f"tuning{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    all_results = []

    for i, combo in enumerate(combinations):
        config_id = i + 1
        print(f"\nRunning tuning{run_id} - config{config_id}...")

        # Prepare parameters for this config
        params = deepcopy(args)
        combo_params = dict(zip(keys, combo))
        params.update(combo_params)

        # Inject tracking info
        params["config_id"] = config_id
        params["tuning_flag"] = True

        # Set output paths
        config_dir = os.path.join(run_dir, f"config{config_id}")
        os.makedirs(config_dir, exist_ok=True)
        params["checkpoints_dir"] = config_dir
        params["plot_dir"] = config_dir

        # Run training
        result = main_worker(params)

        # Collect results for CSV
        result["config"] = f"config{config_id}"  # chiave unica, prima colonna
        all_results.append(result)

    # Save full grid results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(run_dir, "grid_results.csv"), index=False)

    print("\n\n----------------------------------------------------------------")
    print(f"Fine tuning run {run_id} completed")
    print("----------------------------------------------------------------")

if __name__ == '__main__':
    base_args_path = "/data/users/etosato/ANM_Verona/src/cnn/parameters/config.json"
    grid_path = "/data/users/etosato/ANM_Verona/src/cnn/parameters/grid.json"
    tuning(base_args_path, grid_path)
