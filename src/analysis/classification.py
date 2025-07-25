import os
import json
import numpy as np
import pandas as pd
import random
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
from preprocessing.config import ConfigLoader
from analysis.utils import run_umap, log_to_file, reset_stdout, resolve_split_csv_path, build_output_path
from analysis.plotting import plot_confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DataSplit:
    def __init__(self, df_input: pd.DataFrame, split_path: str, label_col: str = "Group", use_full_input: bool = False):
        self.df_split = pd.read_csv(split_path)

        if use_full_input:
            # Mantieni tutto df_input e fai il merge più tardi
            self.df_full = df_input.copy()  # salva per umap_all
            self.df = None  # sarà creato manualmente dopo
        else:
            # Subset basato su split (default)
            self.df = self.df_split.merge(df_input, on="ID", how="left")

        self.label_col = label_col
        self.meta_columns = ["ID", "Group", "Sex", "Age", "Education", "CDR_SB", "MMSE", "split"]

        self.x_all = None
        self.y_all = None
        self.y_encoded = None
        self.splits = None

        self.le = LabelEncoder()
        self.x_train, self.x_test = None, None
        self.y_train, self.y_test = None, None

    def insert_umap(self, x_umap: np.ndarray):
        """After running UMAP externally, reinsert embedding + ID and merge with split."""
        df_umap = pd.DataFrame(x_umap, columns=[f"UMAP{i+1}" for i in range(x_umap.shape[1])])
        df_umap["ID"] = self.df_full["ID"].values
        self.df = self.df_split.merge(df_umap, on="ID", how="left")

    def prepare_features(self):
        self.x_all = self.df.drop(columns=self.meta_columns).to_numpy()
        self.y_all = self.df[self.label_col].to_numpy()
        self.y_encoded = self.le.fit_transform(self.y_all)
        self.splits = self.df["split"].to_numpy()

    def apply_split(self):
        self.x_train = self.x_all[self.splits == "train"]
        self.y_train = self.y_encoded[self.splits == "train"]
        self.x_test = self.x_all[self.splits == "test"]
        self.y_test = self.y_encoded[self.splits == "test"]


def get_model_map(seed, tuning=False):
    model_map = {
        "RandomForest": RandomForestClassifier(random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "KNN": KNeighborsClassifier()
    }
    if tuning:
        model_map["SVM"] = SVC(probability=True, random_state=seed)
    return model_map


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def run_permutation_test(model_class, param_dict, X, y, model_name, seed, save_dir, n_permutations=1000, cv_folds=5):
    """
    Run permutation test on a fitted model and save results incrementally to a single CSV.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model = model_class(**param_dict)

    score, perm_scores, pvalue = permutation_test_score(
        estimator=model,
        X=X,
        y=y,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        n_jobs=-1,
        random_state=seed
    )

    result = {
        "model": model_name,
        "seed": seed,
        "accuracy": round(score, 3),
        "pvalue": round(pvalue, 5)
    }

    # Path to cumulative CSV
    cumulative_path = os.path.join(save_dir, "permutation_results.csv")
    df_result = pd.DataFrame([result])

    # Append or write with header
    if os.path.exists(cumulative_path):
        df_result.to_csv(cumulative_path, mode="a", header=False, index=False)
    else:
        df_result.to_csv(cumulative_path, mode="w", header=True, index=False)

    # Save histogram
    plt.figure()
    plt.hist(perm_scores, bins=20, density=True, alpha=0.7, label="Permuted scores")
    plt.axvline(score, color="red", linestyle="--", label=f"Observed score = {score:.2f}")
    plt.title(f"Permutation Test - {model_name} (seed {seed})\np = {pvalue:.4f}")
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"perm_hist_{model_name}.png"))
    plt.close()

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics. Include AUC if probabilities are available."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_roc": None
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            pass
    return metrics

def train_and_evaluate_model(base_model, model_name, param_dict, data: DataSplit, params: dict):
    """Train model with or without hyperparameter tuning and evaluate on test data."""
    seed = params["seed"]

    if params["tuning"]:
        skf = StratifiedKFold(n_splits=params["n_folds"], shuffle=True, random_state=seed)
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_dict,
            scoring="accuracy",
            cv=skf,
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        grid.fit(data.x_train, data.y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        df_grid = pd.DataFrame(grid.cv_results_)
        rename_map = {col: col.replace("test_score", "accuracy") for col in df_grid.columns if col.startswith("split") and "test_score" in col}
        rename_map["mean_test_score"] = "mean_accuracy"
        df_grid = df_grid.rename(columns=rename_map)
        keep_cols = ["params"] + list(rename_map.values()) + ["rank_test_score"]

        tuning_dir = os.path.join(params["path_umap_class_seed"], "tuning")
        os.makedirs(tuning_dir, exist_ok=True)
        df_grid[keep_cols].round(3).to_csv(os.path.join(tuning_dir, f"cv_grid_{model_name}.csv"), index=False)

        # Skips test evaluation
        return None, best_params, None, None

    else:
        base_model.set_params(**param_dict)
        base_model.fit(data.x_train, data.y_train)
        best_model = base_model
        best_params = param_dict

        y_pred = best_model.predict(data.x_test)
        try:
            y_proba = best_model.predict_proba(data.x_test)
        except:
            y_proba = None

        plot_confusion_matrix(
            data.y_test, y_pred, class_names=data.le.classes_,
            title=f"{model_name} | Seed {seed} | Test Confusion",
            save_path=os.path.join(params["path_umap_class_seed"], f"conf_matrix_test_{model_name}.png")
        )

        # Run permutation test if specified
        if params.get("permutation_test", False):
            run_permutation_test(
                model_class=type(base_model),
                param_dict=param_dict,
                X=data.x_test,
                y=data.y_test,
                model_name=model_name,
                seed=seed,
                save_dir=params["path_umap_class_seed"],
                n_permutations=params.get("n_permutations", 1000),
                cv_folds=params.get("perm_cv", 5)
            )

        # Get test subject IDs
        test_ids = data.df[data.splits == "test"]["ID"].values

        # Save predictions with subject IDs
        df_preds = pd.DataFrame({
            "ID": test_ids,
            "Seed": params["seed"],
            "Model": model_name,
            "TrueLabel": data.le.inverse_transform(data.y_test),
            "PredLabel": data.le.inverse_transform(y_pred)
        })
        return best_model, best_params, evaluate_metrics(data.y_test, y_pred, y_proba), y_pred


def classification_pipeline(data: DataSplit, params: dict):
    seed = params["seed"]

    if params["tuning"]:
        with open("/Users/emmatosato/Documents/PhD/ANM_Verona/src/parameters/grid.json", "r") as f:
            param_grids = json.load(f)
    else:
        param_grids = {
            "RandomForest": params["RandomForest"],
            "GradientBoosting": params["GradientBoosting"],
            "KNN": params["KNN"]
        }

    model_map = get_model_map(seed, tuning=params["tuning"])

    results = []
    all_predictions = []

    for model_name, base_model in model_map.items():
        print(f"\nRunning {'GridSearchCV' if params['tuning'] else 'direct training'} for {model_name}")
        best_model, best_params, metrics, y_pred = train_and_evaluate_model(
            base_model, model_name, param_grids[model_name], data, params
        )

        # Skip saving test metrics if tuning
        if not params["tuning"]:
            result = {"model": model_name, "seed": seed, "best_params": str(best_params)}
            result.update({f"test_{k}": round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
            results.append(result)

            # Get test IDs and labels
            test_ids = data.df[data.splits == "test"]["ID"].values
            true_labels = data.le.inverse_transform(data.y_test)
            pred_labels = data.le.inverse_transform(y_pred)

            df_preds = pd.DataFrame({
                "ID": test_ids,
                "Seed": seed,
                "Model": model_name,
                "TrueLabel": true_labels,
                "PredLabel": pred_labels
            })
            all_predictions.append(df_preds)

    df_results = pd.DataFrame(results) if results else None
    df_preds_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else None
    return df_results, df_preds_all

def main_classification(params, df_input):
    group_dir = f"{params['group1'].lower()}_{params['group2'].lower()}"
    output_dir = os.path.join(
        build_output_path(params['output_dir'], params['task_type'], params['dataset_type'], params['umap'], params.get('umap_all', False)),
        group_dir)
    os.makedirs(output_dir, exist_ok=True)
    params["path_umap_classification"] = output_dir

    log_path = os.path.join(output_dir, "log.txt")
    log_to_file(log_path)

    # Path to the unified predictions file
    out_preds = os.path.join(output_dir, "all_test_predictions.csv")
    # Overwrite if exists from previous runs
    if os.path.exists(out_preds):
        os.remove(out_preds)
    first_write = True  # flag to control header inclusion

    split_path = resolve_split_csv_path(params["dir_split"], params["group1"], params["group2"])
    data = DataSplit(df_input, split_path, use_full_input=params.get("umap_all", False))

    if params.get("umap_all", False):
        print("UMAP applied to entire dataset before split.\n")
        x_all = data.df_full.drop(columns=["ID"]).to_numpy()
        x_all_umap = run_umap(x_all)
        data.insert_umap(x_all_umap)
        data.prepare_features()
        data.apply_split()

    elif params.get("umap", False):
        data.prepare_features()
        data.apply_split()
        print("UMAP applied only on training set and transformed test.\n")
        x_train_umap, x_test_umap = run_umap(data.x_train, data.x_test)
        data.x_train = x_train_umap
        data.x_test = x_test_umap

    else:
        data.prepare_features()
        data.apply_split()
        print("UMAP not applied, using original features.\n")

    all_results = []

    for seed in params["seeds"]:
        print(f"\nSEED {seed} - Running classification")
        params["seed"] = seed
        set_seed(seed)
        params["path_umap_class_seed"] = os.path.join(output_dir, f"seed_{seed}")
        os.makedirs(params["path_umap_class_seed"], exist_ok=True)

        df_summary, df_preds = classification_pipeline(data, params)

        if df_summary is not None:
            all_results.append(df_summary)

        if df_preds is not None:
            df_preds.to_csv(out_preds, mode='a', header=first_write, index=False)
            first_write = False  # only include header once

    if all_results:
        pd.concat(all_results).reset_index(drop=True).to_csv(
            os.path.join(output_dir, "summary_all_seeds.csv"), index=False
        )

    reset_stdout()

if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_classification(args, input_dataframe)
