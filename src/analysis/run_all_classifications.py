import json
import subprocess

# Percorsi
config_path = "src/config/ml_config.json"
script_path = "/Users/emmatosato/Documents/PhD/ANM_Verona/src/analysis/classification.py"

# Combinazioni da eseguire
configs = [
    # VOXEL
    {
        "dataset_type": "voxel", "umap": True, "umap_all": True,
        "group1": "ADNI", "group2": "PSP",
        "RandomForest": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt", "min_samples_split": 10},
        "GradientBoosting": {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 3, "subsample": 0.8},
        "KNN": {"n_neighbors": 7, "weights": "uniform", "metric": "euclidean"}
    },
    {
        "dataset_type": "voxel", "umap": True, "umap_all": True,
        "group1": "ADNI", "group2": "CBS",
        "RandomForest": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "min_samples_split": 10},
        "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 5, "subsample": 1.0},
        "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "manhattan"}
    },
    {
        "dataset_type": "voxel", "umap": True, "umap_all": True,
        "group1": "CBS", "group2": "PSP",
        "RandomForest": {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "min_samples_split": 2},
        "GradientBoosting": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
        "KNN": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean"}
    },
    # NETWORK
    {
        "dataset_type": "networks", "umap": False, "umap_all": False,
        "group1": "ADNI", "group2": "PSP",
        "RandomForest": {"n_estimators": 100, "max_depth": None, "max_features": "sqrt", "min_samples_split": 5},
        "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 3, "subsample": 0.8},
        "KNN": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"}
    },
    {
        "dataset_type": "networks", "umap": False, "umap_all": False,
        "group1": "ADNI", "group2": "CBS",
        "RandomForest": {"n_estimators": 200, "max_depth": None, "max_features": "log2", "min_samples_split": 10},
        "GradientBoosting": {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 3, "subsample": 1.0},
        "KNN": {"n_neighbors": 9, "weights": "uniform", "metric": "manhattan"}
    },
    {
        "dataset_type": "networks", "umap": False, "umap_all": False,
        "group1": "PSP", "group2": "CBS",
        "RandomForest": {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_split": 5},
        "GradientBoosting": {"n_estimators": 200, "learning_rate": 0.01, "max_depth": 3, "subsample": 0.8},
        "KNN": {"n_neighbors": 3, "weights": "distance", "metric": "euclidean"}
    }
]

# Loop su ciascuna configurazione
for cfg in configs:
    # Carica config.json
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Modifica la sezione "job"
    full_config["job"]["dataset_type"] = cfg["dataset_type"]
    full_config["job"]["task_type"] = "classification"
    full_config["job"]["threshold"] = False
    full_config["job"]["umap"] = cfg["umap"]

    # Modifica la sezione "classification"
    full_config["classification"] = {
        "tuning": False,
        "umap_all": cfg["umap_all"],
        "permutation_test": True,
        "n_permutations": 100,
        "perm_cv": 5,
        "group1": cfg["group1"],
        "group2": cfg["group2"],
        "seeds": [42, 123, 2023, 31415, 98765],
        "n_folds": 5,
        "RandomForest": cfg["RandomForest"],
        "GradientBoosting": cfg["GradientBoosting"],
        "KNN": cfg["KNN"]
    }

    # Salva nuova configurazione
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    # Esegui lo script classification.py
    print(f"\n>>> Running classification.py for {cfg['group1']} vs {cfg['group2']} | {cfg['dataset_type'].upper()}")
    subprocess.run(["python", script_path])
