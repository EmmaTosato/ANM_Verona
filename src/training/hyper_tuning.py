# hyper_tuning.py

import json
import os
import pandas as pd
from itertools import product
from copy import deepcopy
from src.training.run import main_worker
from resources import create_tuning_summary

def is_valid_combo(params):
    """
        Check whether a given hyperparameter combination is valid.
        In particular, enforce constraints on batch size depending on the model type,
        to avoid GPU memory issues
    """

    model = params["model_type"]
    batch = params["batch_size"]

    if model == "vgg16" and batch not in [4, 8, 16]:
        return False
    if model in ["resnet", "densenet"] and batch not in [4, 16]:
        return False
    return True

def tuning(base_args_path, grid_path):
    """
        Perform hyperparameter tuning for CNN models using grid search.
        For each valid combination of hyperparameters:
        - Create a configuration dictionary
        - Launch training with k-fold cross-validation
        - Save performance metrics and configuration info
        - Store the summary in a cumulative CSV
        """
    
    # Load fixed config and hyperparameter grid
    with open(base_args_path, "r") as f:
        config = json.load(f)

    base_args = {**config["paths"], **config["training"], **config["fixed"], **config["experiment"]}

    with open(grid_path, "r") as f:
        tuning_args = json.load(f)

    grid = tuning_args["grid"]
    overrides = tuning_args.get("experiment", {})
    base_args.update(overrides)

    # Set the running directory
    run_id = base_args["run_id"]  # fixed ID for this entire tuning execution
    tuning_results_dir = base_args["tuning_results_dir"]
    run_dir = os.path.join(tuning_results_dir, f"tuning{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Prepare combinations
    keys = list(grid.keys())
    combinations = list(product(*grid.values()))
    all_results = []

    for combo in combinations:
        # Prepare parameters
        params = deepcopy(base_args)
        combo_params = dict(zip(keys, combo))
        params.update(combo_params)

        # Skip invalid combo
        if not is_valid_combo(params):
            continue

        # Assign variables
        config_id = len(all_results) + 1
        params["config_id"] = config_id
        params["tuning_flag"] = True

        # Output folders
        config_dir = os.path.join(run_dir, f"config{config_id}")
        os.makedirs(config_dir, exist_ok=True)
        params["runs_dir"] = config_dir
        params["plot_dir"] = config_dir

        # Train and collect results
        result = main_worker(params, config_id)
        summary_row = create_tuning_summary(config_id, params, result)
        all_results.append(summary_row)


    # Save full grid results to CSV
    results_df = pd.DataFrame(all_results)
    first_cols = ["config", "group", "threshold"]
    remaining_cols = [col for col in results_df.columns if col not in first_cols]
    cols = first_cols + remaining_cols
    results_df = results_df[cols]
    results_df.to_csv(os.path.join(run_dir, "grid_results.csv"), index=False)

if __name__ == '__main__':
    base_args_path = "/src/cnn/parameters/config.json"
    grid_path = "/src/cnn/parameters/grid.json"
    tuning(base_args_path, grid_path)
