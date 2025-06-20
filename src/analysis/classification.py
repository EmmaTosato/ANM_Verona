# classification.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from preprocessing.processflat import x_features_return
from umap_run import run_umap
from preprocessing.config import ConfigLoader
from analysis.utils import ensure_dir, threshold_prefix
import argparse


# ---------------------------------------------------------------
# Return the correct split file path for the given pair of groups
# ---------------------------------------------------------------
def resolve_split_csv_path(split_dir, group1, group2):
    fname1 = f"{group1}_{group2}_splitted.csv"
    fname2 = f"{group2}_{group1}_splitted.csv"
    path1 = os.path.join(split_dir, fname1)
    path2 = os.path.join(split_dir, fname2)
    if os.path.exists(path1):
        print(f"\nUsing split file: {path1}\n")
        return path1
    elif os.path.exists(path2):
        print(f"\nUsing split file: {path2}\n")
        return path2
    else:
        raise FileNotFoundError(f"No split CSV found for {group1}, {group2} in {split_dir}")

# ---------------------------------------------------------------
# Load dataframe for classification, metadata and split
# ---------------------------------------------------------------
def load_data(paths, classification, args):
    df_key = classification["df"]
    df_gm = pd.read_pickle(args[df_key])
    df_meta = pd.read_csv(paths["df_meta"])
    split_path = resolve_split_csv_path(paths["split_dir"], classification["group1"], classification["group2"])
    df_split = pd.read_csv(split_path)

    return df_gm, df_meta, df_split

# ---------------------------------------------------------------
# Filter and align X, y, split arrays from merged dataframe
# ---------------------------------------------------------------
def prepare_data(df_gm, df_meta, df_split, group1, group2):
    df_merged, X = x_features_return(df_gm, df_meta)
    df_merged = df_merged.merge(df_split[["ID", "split"]], on="ID", how="inner")
    df_pair = df_merged[df_merged["Group"].isin([group1, group2])].copy()
    X_pair = X.loc[df_pair.index].values
    y_pair = df_pair["Group"].values
    splits = df_pair["split"].values
    return X_pair, y_pair, splits, df_pair["ID"].values, df_pair

# ---------------------------------------------------------------
# Train a model on a fold and return metrics
# ---------------------------------------------------------------
def evaluate_model(X_train, y_train, X_val, y_val, classifier_name, classifier, le, results_dir, seed, fold):
    X_umap_train = run_umap(X_train, plot_flag=False, title=None)
    X_umap_val = run_umap(X_val, plot_flag=False, title=None)

    classifier.fit(X_umap_train, y_train)
    y_pred = classifier.predict(X_umap_val)

    metrics = {
        "model": classifier_name,
        "seed": seed,
        "fold": fold + 1,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="macro"),
        "recall": recall_score(y_val, y_pred, average="macro"),
        "f1": f1_score(y_val, y_pred, average="macro")
    }

    try:
        metrics["auc_roc"] = roc_auc_score(y_val, classifier.predict_proba(X_umap_val)[:, 1])
    except Exception:
        metrics["auc_roc"] = None

    # Save metrics and confusion matrix
    ensure_dir(results_dir)
    path_json = os.path.join(results_dir, f"val_metrics_{classifier_name}_seed{seed}_fold{fold+1}.json")
    with open(path_json, "w") as f:
        json.dump(metrics, f, indent=2)

    conf = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf, annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{classifier_name} - Fold {fold+1} - Seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"conf_matrix_{classifier_name}_seed{seed}_fold{fold+1}.png"))
    plt.close()

    return metrics

# ---------------------------------------------------------------
# Main routine: fixed train/test split + internal CV on train set
# ---------------------------------------------------------------
def run_umap_classification(args, classification, paths):
    df_gm, df_meta, df_split = load_data(paths, classification, args)
    group1, group2 = classification["group1"], classification["group2"]
    seeds = classification["seeds"]
    n_folds = classification.get("n_folds", 5)
    results_dir = paths["path_umap_classification"]

    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    all_results = []

    # Prepare data
    X_all, y_all, split_array, _, df_pair = prepare_data(df_gm, df_meta, df_split, group1, group2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    # Split by predefined split column: test = split==0
    test_mask = (split_array == 0)
    train_mask = ~test_mask
    X_train_full, y_train_full = X_all[train_mask], y_encoded[train_mask]

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

            for name, clf in classifiers.items():
                metrics = evaluate_model(X_train, y_train, X_val, y_val,
                                         classifier_name=name, classifier=clf, le=le,
                                         results_dir=results_dir, seed=seed, fold=fold)
                all_results.append(metrics)

    return pd.DataFrame(all_results)

# ----------------------
# Script entry point
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run classification pipeline")
    parser.add_argument("--config", default="src/parameters/config.json", help="Path to config file")
    parser.add_argument("--paths", default="src/parameters/paths.json", help="Path to paths file")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    loader = ConfigLoader(cli_args.config, cli_args.paths)
    args, config, paths = loader.load()

    # Extract classification block
    classification = config["classification"]

    # Run classification
    results_df = run_umap_classification(args, classification, paths)
    print(results_df)

if __name__ == "__main__":
    main()
