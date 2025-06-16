# regression.py

import re
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings("ignore")
import sys
from analysis.dimensionality_reduction import x_features_return, run_umap
from analysis.plotting import plot_ols_diagnostics, plot_actual_vs_predicted


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def group_value_to_str(value):
    if pd.isna(value):
        return "nan"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

def clean_title_string(title):
    title = re.sub(r'\bcovariates\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[\s\-]+', '_', title)
    title = re.sub(r'_+', '_', title)
    title = title.strip('_')

    return title.lower()


# ------------------------------------------------------------
# Removing subjects without target values
# ------------------------------------------------------------
def remove_missing_values(raw_df, meta_df, target_col):
    subjects_nan = meta_df[meta_df[target_col].isna()]['ID'].tolist()
    df = raw_df[~raw_df['ID'].isin(subjects_nan)].reset_index(drop=True)
    return df

# ------------------------------------------------------------
# UMAP features + optional covariates
# ------------------------------------------------------------
def build_design_matrix(df_merged, x_umap, covariates=None):
    x = pd.DataFrame(x_umap, columns=['UMAP1', 'UMAP2'])
    if covariates:
        covar = df_merged[covariates]
        covar = pd.get_dummies(covar, drop_first=True)
        x = pd.concat([x, covar], axis=1)
    return x.astype(float)

# ------------------------------------------------------------
# Fit OLS regression model
# ------------------------------------------------------------
def fit_ols_model(input_data, target):
    # Input and target
    input_constants = sm.add_constant(input_data)
    target = target.astype(float)

    # Fit OLS model
    model_ols = sm.OLS(target, input_constants).fit()

    # Predictions and residuals
    predictions = model_ols.predict(input_constants)
    residuals = target - predictions
    return model_ols, predictions, residuals

# ------------------------------------------------------------
# Compute RMSE per subject
# ------------------------------------------------------------
def compute_rmse_per_subject(df_merged, y_pred, residuals):
    rmse_subject = np.sqrt(residuals ** 2)
    subject_errors = df_merged[['ID', 'Group', 'CDR_SB']].copy()
    subject_errors['Predicted CDR_SB'] = y_pred
    subject_errors['RMSE'] = rmse_subject

    group_rmse_stats = subject_errors.groupby('Group')['RMSE'].agg(
        Mean_RMSE='mean',
        Std_RMSE='std',
        N='count'
    ).round(2).sort_values('Mean_RMSE', ascending=False)

    return subject_errors, group_rmse_stats

# ------------------------------------------------------------
# Compute Shuffling Regression
# ------------------------------------------------------------
def shuffling_regression(input_data, target, n_iterations=100):
    # Input and target
    input_constants = sm.add_constant(input_data)
    target = target.astype(float)

    # Fit OLS
    model_real = sm.OLS(target, input_constants).fit()
    r2_real = model_real.rsquared

    # Perform shuffling
    r2_shuffled = []
    for _ in range(n_iterations):
        y_shuffled = target.sample(frac=1, replace=False).reset_index(drop=True)
        model_shuffled = sm.OLS(y_shuffled, input_constants).fit()
        r2_shuffled.append(model_shuffled.rsquared)

    # Compute empirical p-value
    p_value = np.mean([r >= r2_real for r in r2_shuffled])

    return r2_real, r2_shuffled, p_value


# ------------------------------------------------------------
# Main regression pipeline
# ------------------------------------------------------------
def main_regression(df_masked, df_meta, params):
    # Variable definition
    target_variable = params['target_variable']
    plot_flag = params['plot_regression']
    save_path = params['output_dir']
    title_prefix = params['prefix_regression']

    # Remove subjects without target value
    df_masked = remove_missing_values(df_masked, df_meta, target_variable)

    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Target variable
    y = df_merged[target_variable]

    if params['y_log_transform']:
        y = np.log1p(y)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=False, save_path = None, title = None)

    # Regression with OLS
    x_ols = build_design_matrix(df_merged, x_umap, params['covariates'])

    # Fit OLS model
    model, y_pred, residuals = fit_ols_model(x_ols, y)

    # Shuffling regression
    r2_real, r2_shuffled, p_value = shuffling_regression(x_ols, y)

    # Statistics
    subject_errors, group_rmse_stats = compute_rmse_per_subject(df_merged, y_pred, residuals)
    subject_errors_sorted = subject_errors.sort_values(by='RMSE').reset_index(drop=True)

    # Plot diagnostics if requested

    if plot_flag or save_path:
        group_labels = df_merged[params['group_name']]
        plot_ols_diagnostics(y, y_pred, residuals, title=title_prefix ,save_path=save_path, plot_flag=plot_flag, color_by_group=params['color_by_group'], group_labels=group_labels)
        plot_actual_vs_predicted(y, y_pred, title=title_prefix, save_path=save_path, plot_flag=plot_flag)


    # Print results
    print("OLS REGRESSION SUMMARY")
    print(model.summary())

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


if __name__ == "__main__":
    # Load json
    with open("src/parameters/config.json", "r") as f:
        config = json.load(f)

    with open("src/parameters/run.json", "r") as f:
        run = json.load(f)

    config.update(run)

    # Load configuration and metadata
    df_masked_raw = pd.read_pickle(config['df_masked'])
    df_metadata = pd.read_csv(config['df_meta'])

    # Set output directory
    output_dir = os.path.join(str(config["path_umap_regression"]), str(config["target_variable"]))
    os.makedirs(output_dir, exist_ok=True)

    # Check if threshold is set
    if config.get("threshold") in [0.1, 0.2]:
        config['log'] = f"log_{config['threshold']}_threshold"
        config['prefix_regression'] = f"{config['threshold']} Threshold"
    else:
        config['log'] = "log_no_threshold"
        config['prefix_regression'] = "No Threshold"

    # Check if covariates are present and modify titles and path
    if config['flag_covariates']:
        config['log'] = f"{config['log']}"
        config['prefix_regression'] = f"{config['prefix_regression']} - Covariates"
        output_dir = os.path.join(output_dir, "covariates")
        os.makedirs(output_dir, exist_ok=True)
    else:
        config['covariates'] = None
        output_dir = os.path.join(output_dir, "no_covariates")
        os.makedirs(output_dir, exist_ok=True)

    if config['group_regression']:
        group_col = config['group_col']

        # Crea directory principale per il group_col
        output_dir = os.path.join(output_dir, re.sub(r'[\s\-]+', '_', group_col.strip().lower()))
        os.makedirs(output_dir, exist_ok=True)

        # Estrai i gruppi unici (senza NaN)
        unique_groups = df_metadata[group_col].dropna().unique()

        for group_id in sorted(unique_groups):
            group_id_str = group_value_to_str(group_id)

            output_dir = os.path.join(output_dir, group_id_str)
            os.makedirs( output_dir, exist_ok=True)
            config['output_dir'] = output_dir

            # Log file specifico per il gruppo
            log_name = f'{config["log"]}_{group_col}_{group_id_str}.txt'.lower()
            sys.stdout = open(os.path.join(output_dir, log_name), "w")

            print(f"\n=== Group by {group_col} - {group_id_str} ===")

            # Filtra i dati per gruppo
            ids = df_metadata[df_metadata[group_col] == group_id]["ID"]
            df_meta_cluster = df_metadata[df_metadata["ID"].isin(ids)].reset_index(drop=True)
            df_cluster = df_masked_raw[df_masked_raw["ID"].isin(ids)].reset_index(drop=True)

            # Esegui la regressione
            main_regression(df_cluster, df_meta_cluster, config)
    else:
        # Modify output directory
        output_dir = os.path.join(output_dir, 'all')
        os.makedirs(output_dir, exist_ok=True)
        config['output_dir'] = output_dir

        # Redirect stdout
        sys.stdout = open(os.path.join(output_dir, config['log']), "w")

        # Run regression
        main_regression(df_masked_raw,df_metadata, config)



    # Reset stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__