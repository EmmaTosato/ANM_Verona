# classification.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from preprocessing.loading import ConfigLoader
from preprocessing.processflat import x_features_return
from analysis.utils import run_umap, log_to_file, reset_stdout

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

def evaluate_model(X_train, y_train, X_val, y_val, classifier_name, classifier, le, seed_dir, seed):
    X_umap_train = run_umap(X_train)
    X_umap_val = run_umap(X_val)

    classifier.fit(X_umap_train, y_train)
    y_pred = classifier.predict(X_umap_val)

    metrics = {
        "model": classifier_name,
        "seed": seed,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average="macro"),
        "recall": recall_score(y_val, y_pred, average="macro"),
        "f1": f1_score(y_val, y_pred, average="macro")
    }

    try:
        metrics["auc_roc"] = roc_auc_score(y_val, classifier.predict_proba(X_umap_val)[:, 1])
    except Exception:
        metrics["auc_roc"] = None

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap="Blues", cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{classifier_name} - Seed {seed}")
    plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, f"conf_matrix_{classifier_name}.png"))
    plt.close()

    return metrics

def classification_pipeline(df_input, df_meta, split_path, args):
    group1, group2 = args["group1"], args["group2"]
    seeds = args["seeds"]
    results_dir = args["path_umap_classification"]

    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    X_all, y_all, split_array, df_pair = load_split_and_prepare(df_input, df_meta, split_path, group1, group2)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)

    train_mask = (split_array != 0)
    X_train_full, y_train_full = X_all[train_mask], y_encoded[train_mask]

    all_results = []

    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        val_idx = np.random.RandomState(seed).choice(len(X_train_full), size=int(0.2 * len(X_train_full)), replace=False)
        train_idx = np.setdiff1d(np.arange(len(X_train_full)), val_idx)

        X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        for name, clf in classifiers.items():
            metrics = evaluate_model(
                X_train, y_train, X_val, y_val,
                classifier_name=name,
                classifier=clf,
                le=le,
                seed_dir=seed_dir,
                seed=seed
            )
            all_results.append(metrics)

    return pd.DataFrame(all_results)

def main_classification(params, df_input, df_meta):
    output_dir = params["path_umap_classification"]
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "log.txt")
    log_to_file(log_path)

    print(f"\nUMAP-BASED CLASSIFICATION: {params['group1']} vs {params['group2']}\n")

    split_path = resolve_split_csv_path(params["dir_split"], params["group1"], params["group2"])
    df_results = classification_pipeline(df_input, df_meta, split_path, params)

    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    summary = df_results.groupby("model")[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]

    print("\n====== SUMMARY ACROSS SEEDS ======")
    print(summary.reset_index().to_string(index=False))

    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary.reset_index().to_csv(summary_path, index=False)

    reset_stdout()

if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_classification(args, input_dataframe, metadata_dataframe)
