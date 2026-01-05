import os
import json
import pandas as pd
import subprocess

# === CONFIGURATION ===
base_config_path = "/data/users/etosato/ANM_Verona/src/config/cnn_config.json"
run_script_path = "/src/cnn/training/run.py"
config_save_path = base_config_path
runs_dir = "/src/cnn/runs"
results_path = os.path.join(runs_dir, "all_training_results.csv")
seeds = [42, 123, 2023, 31415, 98765]
completed_runs = []

# === GET INITIAL RUN ID ===
existing_runs = [d for d in os.listdir(runs_dir) if d.startswith("run") and os.path.isdir(os.path.join(runs_dir, d))]
run_ids = [int(d.replace("run", "")) for d in existing_runs]
next_run_id = max(run_ids) + 1 if run_ids else 1

# === LOAD BASE CONFIG ===
with open(base_config_path) as f:
    base_config = json.load(f)

# === LOOP OVER SEEDS ===
for seed in seeds:
    config = base_config.copy()
    config["experiment"]["run_id"] = next_run_id
    config["training"]["seed"] = seed
    config["experiment"]["seed"] = seed  # to ensure compatibility with create_training_summary
    config["experiment"]["crossval_flag"] = True
    config["experiment"]["evaluation_flag"] = False
    config["fixed"]["tuning_flag"] = False
    config["fixed"]["training_csv"] = True
    config["fixed"]["plot"] = True

    # Save config
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Launching training run {next_run_id} with seed {seed} ===")
    subprocess.run(["python", run_script_path])

    completed_runs.append({
        "run_id": next_run_id,
        "seed": seed
    })

    next_run_id += 1

# === UPDATE FINAL CSV WITH SEEDS ===
df_results = pd.read_csv(results_path)
df_results.rename(columns={"run id": "run_id"}, inplace=True)
df_results["run_id"] = df_results["run_id"].astype(str).str.replace("run", "", regex=False).astype(int)

df_completed = pd.DataFrame(completed_runs)
df_completed["run_id"] = df_completed["run_id"].astype(int)

# Merge and save
df_merged = pd.merge(df_results, df_completed, on="run_id", how="left")
print(df_merged.columns)
df_merged.to_csv(results_path, index=False, float_format="%.3f")
