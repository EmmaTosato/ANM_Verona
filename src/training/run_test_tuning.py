import os
import json
import pandas as pd
import subprocess

# === CONFIGURATION ===
csv_path = "/src/cnn/tuning/adni_psp.csv"
base_config_path = "/src/cnn/parameters/config.json"
tuning_results_dir = "/src/cnn/tuning_results"
config_save_path = "/src/cnn/parameters/config.json"
run_script_path = "/src/cnn/training/run.py"
completed_runs = []

# === LOAD CSV ===
df = pd.read_csv(csv_path)

# You can filter if needed, e.g., only top 5 accuracy
# Load the CSV
df = pd.read_csv(csv_path)

# Filter by specific best_accuracy values
#allowed_accuracies = [0.81, 0.857, 0.905]
#df = df[df["best_accuracy"].isin(allowed_accuracies)]

# === GET INITIAL RUN ID ===
runs_dir = "/src/cnn/runs"
existing_runs = [d for d in os.listdir(runs_dir) if d.startswith("run") and os.path.isdir(os.path.join(runs_dir, d))]
run_ids = [int(d.replace("run", "")) for d in existing_runs]
next_run_id = max(run_ids) + 1 if run_ids else 1

# === LOAD BASE CONFIG ===
with open(base_config_path) as f:
    base_config = json.load(f)

# === LOOP OVER CONFIGURATIONS ===
for _, row in df.iterrows():
    tuning = row["tuning"]
    config_name = row["config"]
    best_fold = int(row["best_fold"])
    model_type = row["model_type"]

    # Build checkpoint path
    ckpt_path = f"/src/cnn/tuning/tuning{tuning}/{config_name}/best_model_fold{best_fold}.pt"

    # Update config
    config = base_config.copy()
    config["training"]["model_type"] = model_type
    config["training"]["epochs"] = int(row["epochs"])
    config["training"]["batch_size"] = int(row["batch_size"])
    config["training"]["lr"] = float(row["lr"])
    config["training"]["weight_decay"] = float(row["weight_decay"])
    config["training"]["optimizer"] = row["optimizer"]

    config["fixed"]["threshold"] = row["threshold"]
    config["fixed"]["tuning_flag"] = False
    config["fixed"]["training_csv"] = True
    config["fixed"]["plot"] = True

    config["experiment"]["group1"], config["experiment"]["group2"] = row["group"].split(" vs ")
    config["experiment"]["run_id"] = next_run_id
    config["experiment"]["crossval_flag"] = False
    config["experiment"]["evaluation_flag"] = True
    config["experiment"]["ckpt_path_evaluation"] = ckpt_path

    # Save updated config
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Launching run {next_run_id} for {config_name} , tuning {tuning} ===")
    subprocess.run(["python", run_script_path])

    completed_runs.append({
        "run_id": next_run_id,
        "tuning": int(tuning),
        "config": str(config_name)
    })

    next_run_id += 1

results_path = "/src/cnn/runs/all_testing_results.csv"

# Load
df_results = pd.read_csv(results_path)
df_results.rename(columns={"run id": "run_id"}, inplace=True)

# Clean run_id format
df_results["run_id"] = df_results["run_id"].astype(str).str.replace("run", "", regex=False).astype(int)

# Prepare df_completed
df_completed = pd.DataFrame(completed_runs)
df_completed["run_id"] = df_completed["run_id"].astype(int)

# Debug: print shapes and preview
print(f"df_results: {df_results.shape}, df_completed: {df_completed.shape}")
print(df_results.head(3))
print(df_completed.head(3))

# Merge and save
df_merged = pd.merge(df_results, df_completed, on="run_id", how="left")
print(df_merged.columns)
df_merged.to_csv(results_path, index=False, float_format="%.3f")
