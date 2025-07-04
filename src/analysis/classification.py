import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
from preprocessing.config import ConfigLoader
from analysis.utils import run_umap, log_to_file, reset_stdout, resolve_split_csv_path, build_output_path
from analysis.plotting import plot_confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_split_and_prepare(df_input, split_path):
    """
    Loads the split CSV and merges it with input features.
    Returns:
    - x: features (voxel-level)
    - y: labels (group)
    - splits: train/test indicator
    - df_merged: merged DataFrame with metadata + features
    """
    meta_columns = ["ID", "Group", "CDR_SB", "MMSE", "split"]
    df_split = pd.read_csv(split_path)[meta_columns]
    df_classification = df_split.merge(df_input, on="ID", how="left")

    # Feature and target
    x = df_classification.drop(columns=meta_columns).to_numpy()
    y = df_classification["Group"].to_numpy()
    splits = df_classification["split"].to_numpy()

    return x, y, splits


def evaluate_metrics(y_true, y_pred, y_proba=None):
    """
    Computes standard classification metrics.
    Optionally includes AUC if probability scores are available.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None
    return metrics


def classification_pipeline(x_train_cv, y_train_cv, x_test, y_test, params, le):
    """
    Performs classification using GridSearchCV with cross-validation,
    selects the best model for each classifier, and evaluates it on the test set.
    Saves CV results, test metrics, and confusion matrix.
    """
    seed = params["seed"]
    results_dir = params["path_umap_class_seed"]
    n_folds = params["n_folds"]
    param_grids = {
        "SVM": params["SVM"],
        "RandomForest": params["RandomForest"]
    }

    all_results = []

    # Loop over each classifier
    for model_name in ["SVM", "RandomForest"]:
        print(f"\nRunning GridSearchCV for {model_name}")

        # Define base model
        if model_name == "SVM":
            base_model = SVC(probability=True, random_state=seed)
        else:
            base_model = RandomForestClassifier(random_state=seed)

        # Define grid search with stratified k-fold cross-validation
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[model_name],
            scoring="accuracy",
            cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed),
            n_jobs=-1,
            refit=True,
            verbose=0,
            return_train_score=False
        )

        # Fit grid search on training data
        grid.fit(x_train_cv, y_train_cv)

        # Save all CV results for all parameter combinations
        df_grid = pd.DataFrame(grid.cv_results_)

        rename_map = {
            col: col.replace("test_score", "accuracy")
            for col in df_grid.columns
            if col.startswith("split") and "test_score" in col
        }
        rename_map["mean_test_score"] = "mean_accuracy"

        df_grid = df_grid.rename(columns=rename_map)

        keep_cols = ["params"] + list(rename_map.values()) + ["rank_test_score"]
        df_grid = df_grid[keep_cols]
        df_grid = df_grid.round(3)
        df_grid.to_csv(os.path.join(results_dir, f"cv_grid_{model_name}.csv"), index=False)

        # Extract best model and its parameters
        best_params = grid.best_params_
        best_model = grid.best_estimator_

        # Evaluate best model on the test set
        y_test_pred = best_model.predict(x_test)
        try:
            y_test_proba = best_model.predict_proba(x_test)
        except:
            y_test_proba = None

        test_metrics = evaluate_metrics(y_test, y_test_pred, y_test_proba)
        test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

        # Plot and save confusion matrix
        plot_confusion_matrix(
            y_test, y_test_pred,
            class_names=le.classes_,
            title=f"{model_name} | Seed {seed} | Test Confusion",
            save_path=os.path.join(results_dir, f"conf_matrix_test_{model_name}.png")
        )

        # Save summary with best params and test metrics
        df_cv_summary = pd.DataFrame([{
            "fold": "cv_best",
            "model": model_name,
            "params": str(best_params),
            **{k: v for k, v in test_metrics.items() if k.startswith("test_")}
        }])
        df_cv_summary.to_csv(os.path.join(results_dir, f"cv_metrics_{model_name}.csv"), index=False)

        # Collect overall results for final summary
        result = {
            "model": model_name,
            "seed": seed,
            "best_params": str(best_params),
            **test_metrics
        }
        for k, v in test_metrics.items():
            result[k] = round(v, 3) if isinstance(v, float) else v
        all_results.append(result)

    # Return summary DataFrame with results for all models
    return pd.DataFrame(all_results)


def main_classification(params, df_input):
    """
    Runs classification pipeline across multiple seeds.
    Saves cross-validation results for each seed and prints summary.
    """
    # Define and create main output directory
    group_dirname = f"{params['group1'].lower()}_{params['group2'].lower()}"
    output_dir = os.path.join(
        build_output_path(params['output_dir'], params['task_type'], params['dataset_type'], params['umap']),
        group_dirname
    )

    os.makedirs(output_dir, exist_ok=True)
    params["path_umap_classification"] = output_dir

    # Logging
    log_path = os.path.join(output_dir, "log.txt")
    log_to_file(log_path)

    # Load and preprocess data
    split_path = resolve_split_csv_path(params["dir_split"], params["group1"], params["group2"])
    x_all, y_all, splits = load_split_and_prepare(df_input, split_path)

    # Encode labels once
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    # Split into train/test
    train_mask = (splits == "train")
    test_mask = (splits == "test")
    x_train_raw, y_train_cv = x_all[train_mask], y_encoded[train_mask]
    x_test_raw, y_test = x_all[test_mask], y_encoded[test_mask]

    # Apply UMAP once (fit on training only)
    if params.get("umap", False):
        x_train_cv, x_test = run_umap(x_train_raw, x_test_raw)
        print("UMAP applied (fit on training only).\n")
    else:
        x_train_cv = x_train_raw
        x_test = x_test_raw
        print("UMAP not applied, using original features.\n")

    all_seeds_results = []

    # Loop over seeds
    for seed in params["seeds"]:
        print(f"\nSEED {seed} - Running 5-fold CV + Test on holdout set: {params['group1']} vs {params['group2']}")
        params["seed"] = seed
        set_seed(seed)

        # Create subfolder for this seed
        params["path_umap_class_seed"] = os.path.join(params["path_umap_classification"], f"seed_{seed}")
        os.makedirs(params["path_umap_class_seed"], exist_ok=True)

        # Run classification
        df_summary = classification_pipeline(x_train_cv, y_train_cv, x_test, y_test, params, le)
        df_summary = df_summary.round(3)
        df_summary["seed"] = seed

        # Reorder columns
        df_summary = df_summary[["model", "seed"] + [c for c in df_summary.columns if c not in ["model", "seed"]]]

        all_seeds_results.append(df_summary)

    # Combine all results into one summary file
    df_all = pd.concat(all_seeds_results).reset_index(drop=True)
    df_all = df_all.round(3)
    df_all.to_csv(os.path.join(output_dir, "summary_all_seeds.csv"), index=False)

    reset_stdout()


if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_classification(args, input_dataframe)
