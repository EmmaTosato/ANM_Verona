import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.collections as mcoll
import re
from matplotlib.font_manager import FontProperties

def clean_title_string(title):
    title = re.sub(r'\bcovariates\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'[\s\-]+', '_', title)
    title = re.sub(r'_+', '_', title)
    return title.strip('_').lower()

def plot_ols_diagnostics(target, predictions, residuals, title, save_path=None, plot_flag=True, color_by_group=False, group_labels=None):
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

        if save_path:
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

def plot_actual_vs_predicted(target, predictions, title, save_path=None, plot_flag=True):
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

    if save_path:
        filename = f"{clean_title_string(title)}_distribution.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300)

    if plot_flag:
        plt.show()

    plt.close()

def plot_clusters_vs_groups(x_umap, labels_dict, group_column, save_path, title_prefix, margin=2.0, plot_flag=True, colors_gmm=False):
    n = len(labels_dict)
    fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))
    if n == 1:
        axes = [axes]
    x1, x2 = x_umap[:, 0], x_umap[:, 1]
    min_val, max_val = min(x1.min(), x2.min()) - margin, max(x1.max(), x2.max()) + margin

    left_colors = ['#74c476', '#f44f39', '#7BD3EA', '#fd8d3c', '#37659e','#fbbabd', '#ffdb24', '#413d7b', '#9dd569', '#e84a9b','#056c39', '#6788ee']
    right_colors = sns.color_palette("Set2")[2:] if colors_gmm else sns.color_palette("Set2")

    for i, (name, labels) in enumerate(labels_dict.items()):
        df_plot = pd.DataFrame({'X1': x1, 'X2': x2, 'cluster': labels, 'label': group_column}).dropna(subset=['label'])
        sns.scatterplot(data=df_plot, x='X1', y='X2', hue='cluster', palette=left_colors, s=50, ax=axes[i][0])
        sns.scatterplot(data=df_plot, x='X1', y='X2', hue='label', palette=right_colors, s=50, ax=axes[i][1])

        axes[i][0].set_title(name)
        axes[i][1].set_title(f"{name} - Labeling according to {group_column.name}")
        for ax in axes[i]:
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

    fig.suptitle("Clustering Results", fontsize=22, fontweight='bold')
    fig.text(0.5, 0.92, title_prefix, fontsize=16, ha='center')

    if save_path:
        fname = re.sub(r'\s+', '_', title_prefix.strip().lower()) + "_clustering.png"
        fig.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
    if plot_flag:
        plt.show()
    plt.close(fig)

# plotting.py
import os
import re
import matplotlib.pyplot as plt

def plot_umap_embedding(x_umap, title=None, save_path=None, plot_flag=True, dot_color="#d74c4c"):
    """
    Plots 2D UMAP embedding with optional save and display.
    """
    plt.figure(figsize=(6, 4))

    plt.scatter(
        x_umap[:, 0], x_umap[:, 1],
        s=50,
        alpha=0.9,
        color=dot_color,
        edgecolor='black',
        linewidth=0.5
    )

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

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    if save_path:
        clean_title = re.sub(r'[\s\-]+', '_', title.strip().lower())
        save_file = os.path.join(save_path, f"{clean_title}_embedding.png")
        plt.savefig(save_file, dpi=300)

    if plot_flag:
        plt.show()

    plt.close()
