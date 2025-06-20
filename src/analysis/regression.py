# regression.py

import os
import sys
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing.loading import load_args_and_data
from preprocessing.processflat import x_features_return
from analysis.umap_run import run_umap
from preprocessing.processflat import x_features_return
from preprocessing.config import ConfigLoader
from analysis.utils import threshold_prefix, ensure_dir
from analysis.plotting import plot_ols_diagnostics, plot_actual_vs_predicted

warnings.filterwarnings("ignore")
np.random.seed(42)

def group_value_to_str(value):
    if pd.isna(value): return "nan"
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)

def remove_missing_values(raw_df, meta_df, target_col):
    ids_nan = meta_df[meta_df[target_col].isna()]['ID'].tolist()
    return raw_df[~raw_df['ID'].isin(ids_nan)].reset_index(drop=True)

def build_design_matrix(df_merged, x_umap, covariates=None):
    x = pd.DataFrame(x_umap, columns=['UMAP1', 'UMAP2'])
    if covariates:
        covar = pd.get_dummies(df_merged[covariates], drop_first=True)
        x = pd.concat([x, covar], axis=1)
    return x.astype(float)

def fit_ols_model(input_data, target):
    input_const = sm.add_constant(input_data)
    model = sm.OLS(target, input_const).fit()
    preds = model.predict(input_const)
    residuals = target - preds
    return model, preds, residuals

def shuffling_regression(input_data, target, n_iter=100):
    input_const = sm.add_constant(input_data)
    r2_real = sm.OLS(target, input_const).fit().rsquared
    r2_shuffled = [sm.OLS(target.sample(frac=1).reset_index(drop=True), input_const).fit().rsquared for _ in range(n_iter)]
    p_value = np.mean([r >= r2_real for r in r2_shuffled])
    return r2_real, r2_shuffled, p_value

def compute_rmse_stats(df_merged, y_pred, residuals):
    rmse_vals = np.sqrt(residuals ** 2)
    df_err = df_merged[['ID', 'Group', 'CDR_SB']].copy()
    df_err['Predicted CDR_SB'] = y_pred
    df_err['RMSE'] = rmse_vals
    stats = df_err.groupby('Group')['RMSE'].agg(Mean_RMSE='mean', Std_RMSE='std', N='count').round(2)
    return df_err.sort_values('RMSE'), stats

def regression_pipeline(df_input, df_meta, args):
    df_input = remove_missing_values(df_input, df_meta, args['target_variable'])
    df_merged, x = x_features_return(df_input, df_meta)
    y = np.log1p(df_merged[args['target_variable']]) if args['y_log_transform'] else df_merged[args['target_variable']]
    x_umap = run_umap(x, plot_flag=False)
    x_ols = build_design_matrix(df_merged, x_umap, args['covariates'])

    model, y_pred, residuals = fit_ols_model(x_ols, y)
    r2_real, r2_shuffled, p_value = shuffling_regression(x_ols, y)
    df_sorted, rmse_stats = compute_rmse_stats(df_merged, y_pred, residuals)

    if args['plot_regression']:
        group_labels = df_merged[args['group_name']]
        plot_ols_diagnostics(y, y_pred, residuals, args['prefix'], args['output_dir'], True, args['color_by_group'], group_labels)
        plot_actual_vs_predicted(y, y_pred, args['prefix'], args['output_dir'], True)

    print("OLS REGRESSION SUMMARY")
    print(model.summary())
    print("\nSHUFFLING REGRESSION")
    print(f"R^2 real: {r2_real:.4f} | shuffled mean: {np.mean(r2_shuffled):.4f} | p-value: {p_value:.4f}")
    print("\nRMSE BY GROUP")
    print(rmse_stats)
    print("\nMAE:", round(mean_absolute_error(y, y_pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 4))
    print("\nSUBJECTS SORTED BY RMSE")
    print(df_sorted.to_string(index=False))

    print("\n\n" + "-" * 80)
    print("SHUFFLING REGRESSION")
    print("-" * 80)
    print(f"R² real: {r2_real:.4f}")
    print(f"R² shuffled: {float(np.mean(r2_shuffled)):.4f}")
    print(f"Empirical p:  {p_value:.4f}")

    print("\n\n" + "-" * 80)
    print("RMSE BY DIAGNOSTIC GROUP AND OVERALL METRICS")
    print("-" * 80)
    print(group_rmse_stats)
    print("\n")
    print(f"MAE:  {round(mean_absolute_error(y, y_pred), 4)}")
    print(f"RMSE: {round(np.sqrt(mean_squared_error(y, y_pred)), 4)}")

    print("\n\n" + "-" * 80)
    print("SUBJECTS RANKED BY RMSE (BEST TO WORST)")
    print("-" * 80)
    print(subject_errors_sorted.to_string(index=False))

def main():
    args, df_input, df_meta = load_args_and_data()
    base_out = os.path.join(args["path_umap_regression"], args["target_variable"])
    os.makedirs(base_out, exist_ok=True)

    args['log'] = f"log_{args['threshold']}_threshold" if args['threshold'] in [0.1, 0.2] else "log_no_threshold"
    args['prefix'] = f"{args['threshold']} Threshold" if args['threshold'] in [0.1, 0.2] else "No Threshold"



import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run regression analysis")
    parser.add_argument("--config", default="src/parameters/config.json", help="Path to config file")
    parser.add_argument("--paths", default="src/parameters/paths.json", help="Path to paths file")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    loader = ConfigLoader(cli_args.config, cli_args.paths)
    args, _, _ = loader.load()

    # Load configuration and metadata
    df_masked_raw = pd.read_pickle(args['df_path'])
    df_metadata = pd.read_csv(args['df_meta'])

    # Set output directory
    output_dir = ensure_dir(os.path.join(str(args["path_umap_regression"]), str(args["target_variable"])))

    # Check if threshold is set
    args['prefix_regression'], args['log'] = threshold_prefix(args.get("threshold"))

    # Check if covariates are present and modify titles and path
    if args['flag_covariates']:
        args['log'] = f"{args['log']}"
        args['prefix_regression'] = f"{args['prefix_regression']} - Covariates"
        output_dir = ensure_dir(os.path.join(output_dir, "covariates"))
    else:
        args['covariates'] = None
        output_dir = ensure_dir(os.path.join(output_dir, "no_covariates"))
    if args.get("group_regression", False):
        group_col = args['group_col']

        # Crea directory principale per il group_col
        output_dir = ensure_dir(os.path.join(output_dir, re.sub(r'[\s\-]+', '_', group_col.strip().lower())))

        # Estrai i gruppi unici (senza NaN)
        unique_groups = df_metadata[group_col].dropna().unique()

        for group_id in sorted(unique_groups):
            group_id_str = group_value_to_str(group_id)

            output_dir = ensure_dir(os.path.join(output_dir, group_id_str))
            args['output_dir'] = output_dir

            # Log file specifico per il gruppo
            log_name = f'{args["log"]}_{group_col}_{group_id_str}.txt'.lower()
            sys.stdout = open(os.path.join(output_dir, log_name), "w")

            print(f"\n=== Group by {group_col} - {group_id_str} ===")

            # Filtra i dati per gruppo
            ids = df_metadata[df_metadata[group_col] == group_id]["ID"]
            df_meta_cluster = df_metadata[df_metadata["ID"].isin(ids)].reset_index(drop=True)
            df_cluster = df_masked_raw[df_masked_raw["ID"].isin(ids)].reset_index(drop=True)

            # Esegui la regressione
            main_regression(df_cluster, df_meta_cluster, args)
    else:
        # Modify output directory
        output_dir = ensure_dir(os.path.join(output_dir, 'all'))
        args['output_dir'] = output_dir

        # Redirect stdout
        sys.stdout = open(os.path.join(output_dir, args['log']), "w")

        # Run regression
        main_regression(df_masked_raw, df_metadata, args)
        
    sys.stdout = sys.__stdout__

    # Reset stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__
