import numpy as np
import pandas as pd
import statsmodels.api as sm
from umap_run import x_features_return, run_umap
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    # Fit the model with the true y
    model_real = sm.OLS(target, input_data).fit()
    r2_real = model_real.rsquared

    # Perform shuffling
    r2_shuffled = []
    for _ in range(n_iterations):
        y_shuffled = target.sample(frac=1, replace=False).reset_index(drop=True)
        model_shuffled = sm.OLS(y_shuffled, input_data).fit()
        r2_shuffled.append(model_shuffled.rsquared)

    # Compute empirical p-value
    p_value = np.mean([r >= r2_real for r in r2_shuffled])

    return r2_real, r2_shuffled, p_value


# ------------------------------------------------------------
# Plot diagnostics: True vs Predicted and Residuals vs Fitted
# ------------------------------------------------------------
def plot_ols_diagnostics(target, predictions, residuals, title, save_path=None, plot_flag=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x=target, y=predictions, ax=axes[0], color='#61bdcd',
                    edgecolor='black', alpha=0.8, s=50)
    axes[0].plot([target.min(), target.max()], [target.min(), target.max()],
                 '--', color='gray')
    axes[0].set_title(f"{title} - True vs Predicted")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")

    sns.scatterplot(x=predictions, y=residuals, ax=axes[1], color='#61bdcd',
                    edgecolor='black', alpha=0.8, s=50)
    axes[1].axhline(0, linestyle='--', color='gray')
    axes[1].set_title(f"{title} - Residuals vs Fitted")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residuals")

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}_diagnostics.png"), dpi=300)

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
        plt.savefig(os.path.join(save_path, f"{title}_distribution.png"), dpi=300)

    if plot_flag:
        plt.show()
    plt.close()

# ------------------------------------------------------------
# Main regression pipeline
# ------------------------------------------------------------
def main_regression(df_masked, df_meta, target_variable="CDR_SB", covariates=None,
                    y_log_transform=False, plot_flag=True, save_path=None, title_prefix="OLS"):
    # Merge voxel and metadata
    df_merged, x = x_features_return(df_masked, df_meta)

    # Target variable
    y = df_merged[target_variable]

    if y_log_transform:
        y = np.log1p(y)

    # Reduce dimensionality with UMAP
    x_umap = run_umap(x, plot_flag=False, save_path = None, title = '')

    # Regression with OLS
    x_ols = build_design_matrix(df_merged, x_umap, covariates)

    # Fit OLS model
    model, y_pred, residuals = fit_ols_model(x_ols, y)

    # Statistics
    subject_errors, group_rmse_stats = compute_rmse_per_subject(df_merged, y_pred, residuals)

    # Shuffling regression
    r2_real, r2_shuffled, p_value = shuffling_regression(x_umap, y)

    # Plot diagnostics if requested
    if plot_flag or save_path:
        plot_ols_diagnostics(y, y_pred, residuals, title=title_prefix, save_path=save_path, plot_flag=plot_flag)
        plot_actual_vs_predicted(y, y_pred, title=title_prefix, save_path=save_path, plot_flag=plot_flag)

    # Print results
    print(model.summary())
    print("\nRMSE by diagnostic group:")
    print(group_rmse_stats)

    subject_errors_sorted = subject_errors.sort_values(by='RMSE').reset_index(drop=True)
    print("\nSubjects ranked by RMSE (best to worst):")
    print(subject_errors_sorted.to_string(index=False))

    print("\nMAE:", round(mean_absolute_error(y, y_pred), 4),
          "RMSE:", round(np.sqrt(mean_squared_error(y, y_pred)), 4))

    print("\nShuffling Regression")
    print("R² real", round(r2_real, 4))
    print(f"R² shuffled: {np.mean(r2_shuffled):.4f}")
    print(f"Empirical p-value: {p_value:.4f}")

    return model, y_pred, residuals, subject_errors, group_rmse_stats
