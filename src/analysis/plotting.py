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
