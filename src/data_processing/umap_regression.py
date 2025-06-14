# umap_regression.py

import re
import json
from tokenize import group

import numpy as np
import pandas as pd
import statsmodels.api as sm
from umap_run import x_features_return, run_umap
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
import sys
import matplotlib.collections as mcoll


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

    return title


# ------------------------------------------------------------
# Removing subjects without target values
# ------------------------------------------------------------
def remove_missing_values(raw_df, df_meta, target_col):
    subjects_nan = df_meta[df_meta[target_col].isna()]['ID'].tolist()
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
# Plot diagnostics: True vs Predicted and Residuals vs Fitted
# ------------------------------------------------------------
def plot_ols_diagnostics(target, predictions, residuals, title, save_path=None, plot_flag=True, color_by_group=False, group_labels=None):

    if color_by_group and group_labels is not None:
        # Prepare dataframe
        df_plot = pd.DataFrame({
            'target': target,
            'predictions': predictions,
            'residuals': residuals,
            'group': group_labels
        })

        # Compute symmetric axis limits
        x_min, x_max = df_plot['target'].min(), df_plot['target'].max()
        y_min, y_max = df_plot['predictions'].min(), df_plot['predictions'].max()
        axis_min = min(x_min, y_min)
        axis_max = max(x_max, y_max)

        # lmplot with confidence intervals
        g = sns.lmplot(
            data=df_plot,
            x='target',
            y='predictions',
            hue='group',
            palette="Set2",
            height=8,
            aspect=1.2,
            scatter_kws=dict(s=180, alpha=0.6, edgecolor="black", linewidths=1.4),
            line_kws=dict(linewidth=2.5),
            ci=95
        )

        # Make the confidence band more visible
        for ax in g.axes.flat:
            for coll in ax.collections:
                if isinstance(coll, mcoll.PolyCollection):
                    coll.set_alpha(0.2)

        # Axis limits
        g.set(xlim=(axis_min - 1, axis_max + 1), ylim=(axis_min - 1, axis_max + 1))
        g.set_axis_labels("True", "Predicted", fontsize=18, weight="bold")
        g.fig.suptitle(f"{title} - OLS True vs Predicted", fontsize=20, weight="bold", y=1.02)

        for ax in g.axes.flat:
            ax.tick_params(labelsize=14)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_linewidth(2)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

        g._legend.set_title("Group")
        g._legend.set_bbox_to_anchor((-0.15, 1.05))
        legf = g._legend.get_frame()
        legf.set_facecolor("white")
        legf.set_edgecolor("black")
        legf.set_linewidth(1.2)

        if save_path:
            clean_title = clean_title_string(title)
            filename = f"{clean_title}_diagnostics_diagnosis.png"
            g.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')

        if plot_flag:
            plt.show()
        plt.close()

    # --- Two-panel plot: True vs Predicted and Residuals ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=target, y=predictions, ax=axes[0], color='#61bdcd',
                    edgecolor='black', alpha=0.8, s=70, linewidth=0.9)
    axes[0].plot([target.min(), target.max()], [target.min(), target.max()],
                 '--', color='gray')
    axes[0].set_title(f"{title} - OLS True vs Predicted")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")

    sns.scatterplot(x=predictions, y=residuals, ax=axes[1], color='#61bdcd',
                    edgecolor='black', alpha=0.8, s=70, linewidth=0.9)
    axes[1].axhline(0, linestyle='--', color='gray')
    axes[1].set_title(f"{title} - OLS Residuals vs Fitted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")

    plt.tight_layout()

    if save_path:
        clean_title= clean_title_string(title)
        filename = f"{clean_title}_diagnostics.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()

    plt.close()


# ------------------------------------------------------------
# Plot histograms of Actual and Predicted values
# ------------------------------------------------------------
def plot_actual_vs_predicted(target, predictions, title, save_path=None, plot_flag=True):
    bins = np.arange(min(target), max(target) + 0.5, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].hist(target, bins=bins, color='#61bdcd', edgecolor='black', alpha=0.85)
    axs[0].set_title(f'{title} - Actual Distribution')

    axs[1].hist(predictions, bins=bins, color='#95d6bb', edgecolor='black', alpha=0.85)
    axs[1].set_title(f'{title} - Predicted Distribution')

    plt.tight_layout()

    if save_path:
        clean_title = clean_title_string(title)
        filename = f"{clean_title}_distribution.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()


# ------------------------------------------------------------
# Main regression pipeline
# ------------------------------------------------------------
def main_regression(df_masked, df_meta, target_variable="CDR_SB", covariates=None,
                    y_log_transform=False, plot_flag=True, save_path=None, title_prefix=None):

    # Remove subjects without target value
    df_masked = remove_missing_values(df_masked, df_meta, target_variable)

    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Target variable
    y = df_merged[target_variable]

    if y_log_transform:
        y = np.log1p(y)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=False, save_path = None, title = None)

    # Regression with OLS
    x_ols = build_design_matrix(df_merged, x_umap, covariates)

    # Fit OLS model
    model, y_pred, residuals = fit_ols_model(x_ols, y)

    # Shuffling regression
    r2_real, r2_shuffled, p_value = shuffling_regression(x_ols, y)

    # Statistics
    subject_errors, group_rmse_stats = compute_rmse_per_subject(df_merged, y_pred, residuals)
    subject_errors_sorted = subject_errors.sort_values(by='RMSE').reset_index(drop=True)

    # Plot diagnostics if requested
    if plot_flag or save_path:
        plot_ols_diagnostics(y, y_pred, residuals, title=title_prefix ,save_path=save_path, plot_flag=plot_flag, color_by_group=config['color_by_group'], group_labels=df_merged['Group'])
        plot_actual_vs_predicted(y, y_pred, title=title_prefix, save_path=save_path, plot_flag=plot_flag)

    # Print results
    print("OLS REGRESSION SUMMARY")
    print(model.summary())

    print("\n\n" + "-" * 80)
    print("SHUFFLING REGRESSION")
    print("-" * 80)
    print(f"R² real:      {round(r2_real, 4)}")
    print(f"R² shuffled:  {np.mean(r2_shuffled):.4f}")
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

    return model, y_pred, residuals, subject_errors, group_rmse_stats

if __name__ == "__main__":
    # Load json
    with open("src/data_processing/config.json", "r") as f:
        config = json.load(f)

    with open("src/data_processing/run.json", "r") as f:
        run = json.load(f)

    config.update(run)

    # Load configuration and metadata
    df_masked = pd.read_pickle(config['df_masked'])
    df_meta = pd.read_csv(config['df_meta'])

    # Set output directory
    output_dir = os.path.join(config["path_umap_regression"], config["target_variable"])
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
        unique_groups = df_meta[group_col].dropna().unique()

        for group_id in sorted(unique_groups):
            group_id_str = group_value_to_str(group_id)

            # Crea sottocartella per ogni gruppo
            output_dir_group = os.path.join(output_dir, group_id_str)
            os.makedirs(output_dir_group, exist_ok=True)

            # Log file specifico per il gruppo
            log_name = f'{config["log"]}_{group_col}_{group_id_str}.txt'.lower()
            sys.stdout = open(os.path.join(output_dir_group, log_name), "w")

            print(f"\n=== Group by {group_col} - {group_id_str} ===")

            # Filtra i dati per gruppo
            ids = df_meta[df_meta[group_col] == group_id]["ID"]
            df_meta_cluster = df_meta[df_meta["ID"].isin(ids)].reset_index(drop=True)
            df_cluster = df_masked[df_masked["ID"].isin(ids)].reset_index(drop=True)

            # Esegui la regressione
            model, y_pred, residuals, subject_errors, group_rmse_stats = main_regression(
                df_masked=df_cluster,
                df_meta=df_meta_cluster,
                target_variable=config["target_variable"],
                covariates=config['covariates'],
                y_log_transform=config.get('y_log_transform', False),
                plot_flag=config["plot_regression"],
                save_path=output_dir_group,
                title_prefix=f"{config['prefix_regression']}"
            )
    else:
        # Modify output directory
        output_dir = os.path.join(output_dir, 'all')
        os.makedirs(output_dir, exist_ok=True)

        # Redirect stdout
        sys.stdout = open(os.path.join(output_dir, config['log']), "w")

        # Run regression
        model, y_pred, residuals, subject_errors, group_rmse_stats = main_regression(
            df_masked=df_masked,
            df_meta=df_meta,
            target_variable= config["target_variable"],
            covariates=config['covariates'],
            y_log_transform=config.get('y_log_transform', False),
            plot_flag=config["plot_regression"],
            save_path= output_dir,
            title_prefix= config['prefix_regression']
        )



    # Reset stdout
    sys.stdout.close()
    sys.stdout = sys.__stdout__