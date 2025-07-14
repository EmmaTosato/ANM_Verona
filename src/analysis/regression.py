# regression.py
import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing.config import ConfigLoader
from preprocessing.processflat import x_features_return
from analysis.plotting import plot_ols_diagnostics, plot_actual_vs_predicted, plot_umap_embedding
from analysis.utils import log_to_file, reset_stdout, run_umap, build_output_path



# Suppress all warnings to keep output clean
warnings.filterwarnings("ignore")
np.random.seed(42)

def group_value_to_str(value):
    """
    Converts a group value to a string for directory naming.
    - NaN values are converted to "nan"
    - Float integers are cast to int and then to string
    - Other values are returned as strings
    """
    if pd.isna(value): return "nan"
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)

def remove_missing_values(raw_df, meta_df, target_col):
    """
    Removes rows from raw_df if the corresponding subject has a missing target variable.
    """
    ids_nan = meta_df[meta_df[target_col].isna()]['ID'].tolist()
    return raw_df[~raw_df['ID'].isin(ids_nan)].reset_index(drop=True)

def build_design_matrix(df_merged, x_input, covariates=None):
    """
    Builds the design matrix for regression:
    - Uses projected features (e.g., UMAP or original features)
    - Adds dummy-coded covariates if provided
    """
    # Use x_input directly
    x = x_input.copy()

    # Optional renaming for 2D case
    if x.shape[1] == 2 and list(x.columns) == [0, 1]:
        x.columns = ['UMAP1', 'UMAP2']

    # Add covariates if provided
    if covariates:
        cov_df = df_merged[covariates]
        cat_cols = cov_df.select_dtypes(include=['object', 'category']).columns
        num_cols = cov_df.select_dtypes(include=['int64', 'float64']).columns
        covar_cat = pd.get_dummies(cov_df[cat_cols], drop_first=True)
        covar = pd.concat([cov_df[num_cols], covar_cat], axis=1)

        if covar.shape[1] > 0:
            x = pd.concat([x, covar], axis=1)
        else:
            print("Warning: no covariates added.")

    return x.astype(float)


def fit_ols_model(input_data, target):
    """
    Fits an OLS model using the input data and target.
    Returns the fitted model, predictions, and residuals.
    """
    input_const = sm.add_constant(input_data)
    model = sm.OLS(target, input_const).fit()
    preds = model.predict(input_const)
    residuals = target - preds
    return model, preds, residuals

def shuffling_regression(input_data, target, n_iter=100):
    """
    Performs shuffling-based regression to evaluate significance:
    - Compares true R^2 to distribution of shuffled R^2 values
    - Returns true R^2, shuffled R^2 values, and empirical p-value
    """
    input_const = sm.add_constant(input_data)
    r2_real = sm.OLS(target, input_const).fit().rsquared
    r2_shuffled = [sm.OLS(target.sample(frac=1).reset_index(drop=True), input_const).fit().rsquared for _ in range(n_iter)]
    p_value = np.mean([r >= r2_real for r in r2_shuffled])
    return r2_real, r2_shuffled, p_value

def compute_rmse_stats(df_merged, y_pred, residuals):
    """
    Computes RMSE statistics for each group in the dataset:
    - Returns per-subject RMSE values and grouped summary stats
    """
    rmse_vals = np.sqrt(residuals ** 2)
    df_err = df_merged[['ID', 'Group', 'CDR_SB']].copy()
    df_err['Predicted CDR_SB'] = y_pred
    df_err['RMSE'] = rmse_vals
    stats = df_err.groupby('Group')['RMSE'].agg(Mean_RMSE='mean', Std_RMSE='std', N='count').round(2)
    return df_err.sort_values('RMSE'), stats

def regression_pipeline(df_input, df_meta, args):
    """
    Executes the full regression pipeline:
    - Cleans data
    - Generates UMAP embedding and design matrix
    - Runs OLS and shuffling regression
    - Computes and prints evaluation metrics
    - Generates diagnostic plots
    """
    # Remove missing values based on target variable
    df_input = remove_missing_values(df_input, df_meta, args['target_variable'])
    # Merge input features with metadata
    df_merged, x_feature = x_features_return(df_input, df_meta)

    # Target variable handling
    y = np.log1p(df_merged[args['target_variable']]) if args['y_log_transform'] else df_merged[args['target_variable']]
    # Feature projection
    if args["umap"]:
        x = run_umap(x_feature)
    else:
        x = x_feature

    x_ols = build_design_matrix(df_merged, x, args["covariates"])

    # Fit OLS model
    model, y_pred, residuals = fit_ols_model(x_ols, y)
    # Perform shuffling regression
    r2_real, r2_shuffled, p_value = shuffling_regression(x_ols, y)
    # Compute RMSE statistics
    df_sorted, rmse_stats = compute_rmse_stats(df_merged, y_pred, residuals)

    # Plotting
    group_labels = df_merged[args['group_name']]
    plot_ols_diagnostics(y, y_pred, residuals,args['prefix'], args['output_dir'], args['plot_regression'], args['save_flag'],args['color_by_group'], group_labels)
    plot_actual_vs_predicted(y, y_pred, args['prefix'], args['output_dir'], args['plot_regression'], args['save_flag'])

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

def main_regression(params, df_input, df_meta):
    """
    Main entry point for launching the regression pipeline.
    - Handles split by group or global
    - Manages output structure and logging
    """
    base_out_temp = build_output_path(params['output_dir'], params['task_type'], params['dataset_type'], params['umap'])
    base_out = os.path.join(base_out_temp, params["target_variable"])
    os.makedirs(base_out, exist_ok=True)

    params['log'] = f"log_{params['threshold']}_threshold" if params['threshold'] in [0.1, 0.2] else "log_no_threshold"
    params['prefix'] = f"{params['threshold']} Threshold" if params['threshold'] in [0.1, 0.2] else "No Threshold"

    if params['flag_covariates']:
        params['prefix'] += " - Covariates"
        base_out = os.path.join(base_out, "covariates")
    else:
        params['covariates'] = None
        base_out = os.path.join(base_out, "no_covariates")
    os.makedirs(base_out, exist_ok=True)

    if params.get("group_regression", False):
        group_col = params['group_col']
        for group_val in sorted(df_meta[group_col].dropna().unique()):
            group_str = group_value_to_str(group_val)
            params['output_dir'] = os.path.join(base_out, group_col.lower(), group_str)
            os.makedirs(params['output_dir'], exist_ok=True)

            log_path = os.path.join(params['output_dir'], f"{params['log']}_{group_col}_{group_str}.txt")
            log_to_file(log_path)

            ids = df_meta[df_meta[group_col] == group_val]['ID']
            df_group = df_input[df_input['ID'].isin(ids)].reset_index(drop=True)
            df_meta_group = df_meta[df_meta['ID'].isin(ids)].reset_index(drop=True)
            regression_pipeline(df_group, df_meta_group, params)

            reset_stdout()
    else:
        params['output_dir'] = os.path.join(base_out, "all")
        os.makedirs(params['output_dir'], exist_ok=True)

        log_path = os.path.join(params['output_dir'], params['log'])
        log_to_file(log_path)

        regression_pipeline(df_input, df_meta, params)

        reset_stdout()

if __name__ == "__main__":
    loader = ConfigLoader()
    args, input_dataframe, metadata_dataframe = loader.load_all()
    main_regression(args, input_dataframe, metadata_dataframe)
