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
                            colors_gmm=False, separated=False):
    """
    Plots clustering results side-by-side with group labels using UMAP coordinates.
    Also generates separate plots for cluster and group if 'separated' is True.
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

        # Left subplot: by cluster
        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='cluster', palette=left_colors,
            s=80, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][0]
        )
        # Right subplot: by group
        sns.scatterplot(
            data=df_plot, x='X1', y='X2', hue='label', palette=right_colors,
            s=80, alpha=0.9, edgecolor='black', linewidth=0.5, ax=axes[i][1]
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

        # ---------- Separate plots ----------
        if separated:
            fig_c, ax_c = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=df_plot, x="X1", y="X2", hue="cluster", palette=left_colors,
                s=110, alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax_c
            )
            ax_c.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
            ax_c.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
            ax_c.spines['top'].set_visible(False)
            ax_c.spines['right'].set_visible(False)
            ax_c.spines['bottom'].set_linewidth(1.5)
            ax_c.spines['left'].set_linewidth(1.5)
            ax_c.spines['bottom'].set_edgecolor('black')
            ax_c.spines['left'].set_edgecolor('black')
            ax_c.tick_params(labelsize=12)
            ax_c.grid(False)

            for label in ax_c.get_xticklabels() + ax_c.get_yticklabels():
                label.set_fontweight('bold')
            if title_flag:
                ax_c.set_title(f"{name} - Cluster view", fontsize=14, fontweight='bold')
            if save_flag and save_path:
                fname = clean_title_string(f"{title_prefix}_{name}_cluster") + ".png"
                fig_c.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
            if plot_flag:
                plt.show()
            plt.close(fig_c)

            fig_g, ax_g = plt.subplots(figsize=(8, 5))
            sns.scatterplot(
                data=df_plot, x="X1", y="X2", hue="label", palette=right_colors,
                s=110, alpha=0.9, edgecolor='black', linewidth=0.6, ax=ax_g
            )

            ax_g.set_xlabel("UMAP 1", fontsize=14, fontweight='bold')
            ax_g.set_ylabel("UMAP 2", fontsize=14, fontweight='bold')
            ax_g.spines['top'].set_visible(False)
            ax_g.spines['right'].set_visible(False)
            ax_g.spines['bottom'].set_linewidth(1.5)
            ax_g.spines['left'].set_linewidth(1.5)
            ax_g.spines['bottom'].set_edgecolor('black')
            ax_g.spines['left'].set_edgecolor('black')
            ax_g.tick_params(labelsize=12)
            ax_g.grid(False)

            for label in ax_g.get_xticklabels() + ax_g.get_yticklabels():
                label.set_fontweight('bold')
            if title_flag:
                ax_g.set_title(f"{name} - Group view", fontsize=14, fontweight='bold')
            if save_flag and save_path:
                fname = clean_title_string(f"{title_prefix}") + ".png"
                fig_g.savefig(os.path.join(save_path, fname), dpi=300, bbox_inches='tight')
            if plot_flag:
                plt.show()
            plt.close(fig_g)

    # ----- Final combined plot -----
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
                        ):
    """
    Plots 2D UMAP embedding using the 'labeling_umap' DataFrame.
    """
    clean_title = re.sub(r'[\s\-]+', '_', title.strip().lower()) if title else "umap"

    plt.figure(figsize=(6, 4))
    plt.scatter(
        labeling_umap["X1"], labeling_umap["X2"],
        s=80, alpha=0.9,
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


# === Classification Evaluation ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Plots a stylized confusion matrix with enhanced readability and thick external border.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = sns.heatmap(
        cm, annot=True, fmt='d', cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1, linecolor='black', ax=ax,
        annot_kws={"size": 20, "weight": "bold"},
    )

    # Axis labels with spacing
    ax.set_xlabel("Predicted", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_ylabel("True", fontsize=20, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=22, fontweight='bold')

    # Enlarge colorbar tick labels
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')


    # Tick labels styling
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticklabels(class_names, fontsize=18, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=18, fontweight='bold', rotation=0)

    # Add THICK outer border as a rectangle
    rows, cols = cm.shape
    rect = patches.Rectangle(
        (0, 0), cols, rows,
        linewidth=4.5, edgecolor='black', facecolor='none'
    )
    ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

    # To save the figure instead of showing it, uncomment the line below:
    # plt.savefig(save_path, dpi=300)
    # plt.close()





