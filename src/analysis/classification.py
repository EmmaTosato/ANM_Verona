# classification.py (refactored)

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing.loading import load_args_and_data
from preprocessing.processflat import x_features_return
from analysis.umap_run import run_umap

np.random.seed(42)

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

def load_split_and_prepare(df_input, df_meta, split_path, group1, group2):
    df_split = pd.read_csv(split_path)
    df_merged, X = x_features_return(df_input, df_meta)
    df_merged = df_merged.merge(df_split[["ID", "split"]], on="ID", how="inner")
    df_pair = df_merged[df_merged["Group"].isin([group1, group2])].copy()
    X_pair = X.loc[df_pair.index].values
    y_pair = df_pair["Group"].values
    splits = df_pair["split"].values
    return X_pair, y_pair, splits, df_pair

def evaluate_model(X_train, y_train, X_val, y_val, classifier_name, classifier, le, results_dir, seed, fold):
    X_umap_train = run_umap(X_train, plot_flag=False)
    X_umap_val = run_umap(X_val, plot_flag=False)

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

    os.makedirs(results_dir, exist_ok=True)
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

def classification_pipeline(df_input, df_meta, split_path, args, classification):
    group1, group2 = classification["group1"], classification["group2"]
    seeds = classification["seeds"]
    n_folds = classification.get("n_folds", 5)
    results_dir = args["path_umap_classification"]

    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    X_all, y_all, split_array, df_pair = load_split_and_prepare(df_input, df_meta, split_path, group1, group2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    test_mask = (split_array == 0)
    train_mask = ~test_mask
    X_train_full, y_train_full = X_all[train_mask], y_encoded[train_mask]

    all_results = []
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

def main():
    from preprocessing.loading import load_args_resolved

    args, config, paths = load_args_resolved()
    classification = config["classification"]

    df_input = pd.read_pickle(args["df_path"])
    df_meta = pd.read_csv(args["df_meta"])

    split_path = os.path.join(paths["split_dir"], f"{classification['group1']}_{classification['group2']}_splitted.csv")
    if not os.path.exists(split_path):
        split_path = os.path.join(paths["split_dir"], f"{classification['group2']}_{classification['group1']}_splitted.csv")

    if not os.path.exists(split_path):
        raise FileNotFoundError("Split file not found for selected group pair.")

    df_results = classification_pipeline(df_input, df_meta, split_path, args, classification)
    print(df_results)

if __name__ == "__main__":
    main()
