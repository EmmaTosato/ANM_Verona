# plotting.py

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.font_manager import FontProperties
import matplotlib.collections as mcoll
from sklearn.metrics import confusion_matrix

# === Utility Functions ===
def clean_title_string(title):
    title = re.sub(r'\bcovariates\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[\s\-]+', '_', title)
    title = re.sub(r'_+', '_', title)
    return title.strip('_').lower()

# === Regression & Diagnostic Plots ===
def plot_ols_diagnostics(target, predictions, residuals,
                         title, save_path=None,
                         plot_flag=True, save_flag=False,
                         color_by_group=False, group_labels=None):
    title_font = FontProperties(family="DejaVu Sans", weight='bold', size=14)
    label_font = FontProperties(family="DejaVu Sans", weight='bold', size=12)

    if color_by_group and group_labels is not None:
        df_plot = pd.DataFrame({
            'target': target,
            'predictions': predictions,
            'residuals': residuals,
            'group': group_labels
        })

        x_min, x_max = df_plot['target'].min(), df_plot['target'].max()
        y_min, y_max = df_plot['predictions'].min(), df_plot['predictions'].max()
        axis_min = min(x_min, y_min)
        axis_max = max(x_max, y_max)

        g = sns.lmplot(
            data=df_plot,
            x='target',
            y='predictions',
            hue='group',
            palette="Set2",
            height=6,
            aspect=1,
            scatter_kws=dict(s=100, alpha=0.6, edgecolor="black", linewidths=1),
            line_kws=dict(linewidth=2.2),
            ci=95
        )

        g.set(xlim=(axis_min - 1, axis_max + 1), ylim=(axis_min - 1, axis_max + 1))
        for ax in g.axes.flat:
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=11)
            ax.grid(False)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('normal')
            ax.set_xlabel("True", fontproperties=label_font)
            ax.set_ylabel("Predicted", fontproperties=label_font)

            for coll in ax.collections:
                if isinstance(coll, mcoll.PolyCollection):
                    coll.set_alpha(0.2)

        g.fig.suptitle("OLS True vs Predicted", fontproperties=title_font, y=1.02)

        leg = g.ax.get_legend()
        if leg is not None:
            leg.set_title("Group")
            leg.set_bbox_to_anchor((1.01, 1.02))
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(1)

        if save_path and save_flag:
            filename = f"{clean_title_string(title)}_diagnostics_labelled.png"
            g.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight', pad_inches=0.05)

        if plot_flag:
            plt.show()

        plt.close()

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(x=target, y=predictions, ax=ax, color='#61bdcd',
                    edgecolor='black', alpha=0.8, s=70, linewidth=0.9)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], '--', color='gray')
    ax.set_title("OLS True vs Predicted", fontproperties=title_font)
    ax.set_xlabel("True", fontproperties=label_font)
    ax.set_ylabel("Predicted", fontproperties=label_font)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    plt.tight_layout()

    if save_path:
        filename = f"{clean_title_string(title)}_diagnostics.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()

    plt.close()

def plot_actual_vs_predicted(target, predictions,
                             title, save_path=None,
                             plot_flag=False, save_flag = False):
    title_font = FontProperties(family="DejaVu Sans", weight='bold', size=14)
    label_font = FontProperties(family="DejaVu Sans", weight='bold', size=12)

    bins = np.arange(min(target), max(target) + 0.5, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].hist(target, bins=bins, color='#61bdcd', edgecolor='black', alpha=0.85)
    axs[0].set_title('Actual Distribution', fontproperties=title_font)
    axs[0].set_xlabel("Value", fontproperties=label_font)
    axs[0].set_ylabel("Count", fontproperties=label_font)

    axs[1].hist(predictions, bins=bins, color='#95d6bb', edgecolor='black', alpha=0.85)
    axs[1].set_title('Predicted Distribution', fontproperties=title_font)
    axs[1].set_xlabel("Value", fontproperties=label_font)
    axs[1].set_ylabel("Count", fontproperties=label_font)

    for ax in axs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('normal')

    plt.tight_layout()

    if save_path and save_flag:
        filename = f"{clean_title_string(title)}_distribution.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()

    plt.close()

# === UMAP & Clustering Plots ===
def plot_clusters_vs_groups(x_umap, labels_dict, group_column,
                            save_path, title_prefix,
                            margin=2.0,
                            plot_flag=True, save_flag=False, title_flag=False,
                            colors_gmm=False):
    """
    Plots clustering results side-by-side with group labels using UMAP coordinates.
    One row per clustering method.
    """
    n = len(labels_dict)
    fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))
    if n == 1:
        axes = [axes]

    x1, x2 = x_umap[:, 0], x_umap[:, 1]
    min_val = min(x1.min(), x2.min()) - margin
    max_val = max(x1.max(), x2.max()) + margin

    left_colors = ['#E24141', '#74c476', '#7BD3EA', '#fd8d3c', '#37659e', '#fbbabd', '#ffdb24',
                   '#413d7b', '#9dd569', '#e84a9b', '#056c39', '#6788ee']
    right_colors = sns.color_palette("Set2")[2:] if colors_gmm else sns.color_palette("Set2")

    for i, (name, labels) in enumerate(labels_dict.items()):
        df_plot = pd.DataFrame({'X1': x1, 'X2': x2, 'cluster': labels, 'label': group_column}).dropna(subset=['label'])

        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='cluster', palette=left_colors,
            s=50, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][0]
        )
        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='label', palette=right_colors,
            s=50, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][1]
        )

        for ax in axes[i]:
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlabel("UMAP 1", fontsize=12, fontweight='bold')
            ax.set_ylabel("UMAP 2", fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_edgecolor('black')
            ax.spines['left'].set_edgecolor('black')
            ax.grid(False)
            ax.tick_params(labelsize=11)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('normal')

        if title_flag:
            axes[i][0].set_title(name, fontsize=14, fontweight='bold')
            axes[i][1].set_title(f"{name} - Labeling by {group_column.name}", fontsize=14, fontweight='bold')

    if title_flag:
        fig.suptitle("Clustering Results", fontsize=22, fontweight='bold')
        fig.text(0.5, 0.92, title_prefix, fontsize=16, ha='center')

    if save_path and save_flag:
        os.makedirs(save_path, exist_ok=True)
        fname = clean_title_string(title_prefix) + "_clustering.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')

    if plot_flag:
        plt.show()

    plt.close(fig)


def plot_umap_embedding(labeling_umap,
                        title=None,
                        save_path=None,
                        plot_flag=True,
                        save_flag=False,
                        title_flag=False,
                        dot_color="#d74c4c",
                        color_by_group=False,
                        group_column="group",
                        palette=None):
    """
    Plots 2D UMAP embedding using the 'labeling_umap' DataFrame.
    If color_by_group is True, points are colored according to 'group_column'.
    """
    clean_title = re.sub(r'[\s\-]+', '_', title.strip().lower()) if title else "umap"

    # --- FIRST PLOT: standard embedding ---
    plt.figure(figsize=(6, 4))
    plt.scatter(
        labeling_umap["X1"], labeling_umap["X2"],
        s=50, alpha=0.9,
        color=dot_color,
        edgecolor='black',
        linewidth=0.5
    )
    if title_flag is not False:
        plt.title(f'UMAP Embedding - {title}', fontsize=14, fontweight='bold')
    plt.xlabel("UMAP 1", fontsize=12, fontweight='bold')
    plt.ylabel("UMAP 2", fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.grid(False)
    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    if save_path and save_flag:
        file_standard = os.path.join(save_path, f"{clean_title}_embedding.png")
        plt.savefig(file_standard, dpi=300, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close()

    # --- SECOND PLOT: color by group ---
    if color_by_group and group_column in labeling_umap.columns:
        plt.figure(figsize=(6, 4))

        if palette is None:
            palette = sns.color_palette("Set2", labeling_umap[group_column].nunique())

        sns.scatterplot(
            data=labeling_umap,
            x="X1", y="X2",
            hue=group_column,
            palette=palette,
            s=50,
            edgecolor='black',
            linewidth=0.5
        )
        plt.legend(title=group_column, bbox_to_anchor=(1.02, 1), loc='upper left')
        if title_flag is not False:
            plt.title(f'UMAP Embedding - {title} (by group)', fontsize=14, fontweight='bold')
        plt.xlabel("UMAP 1", fontsize=12, fontweight='bold')
        plt.ylabel("UMAP 2", fontsize=12, fontweight='bold')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.grid(False)
        ax.tick_params(labelsize=11)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('normal')

        if save_path and save_flag:
            file_colored = os.path.join(save_path, f"{clean_title}_embedding_label.png")
            plt.savefig(file_colored, dpi=300, bbox_inches='tight')
        if plot_flag:
            plt.show()
        plt.close()



# === Classification Evaluation ===
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Plots and saves a confusion matrix.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True, fmt='d', cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
