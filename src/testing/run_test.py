import os
import json
import pandas as pd
import subprocess

import os
import subprocess

env = os.environ.copy()
env["MKL_THREADING_LAYER"] = "GNU"


# === CONFIGURAZIONE ===
csv_path = "/data/users/etosato/ANM_Verona/results/runs/all_training_results.csv"
base_config_path = "/data/users/etosato/ANM_Verona/src/config/config.json"
config_save_path = base_config_path
run_script_path = "/data/users/etosato/ANM_Verona/src/training/run.py"
runs_dir = "/data/users/etosato/ANM_Verona/results/runs"
results_path = os.path.join(runs_dir, "all_testing_results.csv")

# === CARICA CSV INPUT ===
df = pd.read_csv(csv_path)
df.rename(columns={"best fold": "best_fold"}, inplace=True)

# === CARICA CONFIG BASE ===
with open(base_config_path) as f:
    base_config = json.load(f)

# === IDENTIFICA RUN GIÀ COMPLETATE ===
if os.path.exists(results_path):
    df_results = pd.read_csv(results_path)
    completed_run_ids = set(df_results["run_id"].astype(str).str.replace("run", "", regex=False).astype(int))
else:
    completed_run_ids = set()

# === LOOP SU OGNI CONFIGURAZIONE ===
for _, row in df.iterrows():
    run_id = int(row["run_id"])
    if run_id in completed_run_ids:
        print(f" Skipping run{run_id}: already completed.")
        continue

    best_fold = int(row["best_fold"])
    group1, group2 = row["group"].split(" vs ")
    seed = int(row["seed_x"])
    threshold = row["threshold"]

    ckpt_path = os.path.join(runs_dir, f"run{run_id}", f"best_model_fold{best_fold}.pt")

    # === COPIA E MODIFICA CONFIG ===
    config = base_config.copy()
    config["training"]["model_type"] = row["model_type"]
    config["training"]["epochs"] = int(row["epochs"])
    config["training"]["batch_size"] = int(row["batch_size"])
    config["training"]["lr"] = float(row["lr"])
    config["training"]["weight_decay"] = float(row["weight_decay"])
    config["training"]["optimizer"] = row["optimizer"]
    config["training"]["seed"] = seed

    config["fixed"]["threshold"] = threshold
    config["fixed"]["tuning_flag"] = False
    config["fixed"]["training_csv"] = True
    config["fixed"]["plot"] = True
    config["fixed"]["test_size"] = float(row["test size"])

    config["experiment"]["group1"] = group1
    config["experiment"]["group2"] = group2
    config["experiment"]["run_id"] = run_id
    config["experiment"]["crossval_flag"] = False
    config["experiment"]["evaluation_flag"] = True
    config["experiment"]["ckpt_path_evaluation"] = ckpt_path
    config["experiment"]["seed"] = seed

    # === SALVA CONFIG E LANCIA ===
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nLaunching run.py for RUN {run_id} | {group1} vs {group2} | Fold {best_fold}")
    subprocess.run(["python", run_script_path], env=env)
