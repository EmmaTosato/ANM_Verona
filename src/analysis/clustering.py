# clustering.py
import os
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import hdbscan

from preprocessing.loading import load_args_and_data
from preprocessing.processflat import x_features_return
from analysis.umap_run import run_umap
from analysis.clustering_evaluation import evaluate_kmeans, evaluate_gmm, evaluate_hdbscan, evaluate_consensus

warnings.filterwarnings("ignore")
np.random.seed(42)

def group_value_to_str(value):
    if pd.isna(value): return "nan"
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)

def run_clustering(x_umap):
    return {
        "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(x_umap),
        "K-Means": KMeans(n_clusters=3, random_state=42).fit_predict(x_umap)
    }

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

def clustering_pipeline(df_input, df_meta, args):
    df_merged, x = x_features_return(df_input, df_meta)
    x_umap = run_umap(x, plot_flag=args['plot_cluster'], save_path=args['path_umap'], title=args['prefix'])

    if args.get("do_evaluation"):
        clean_title = re.sub(r'[\s\-]+', '_', args['prefix'].strip().lower())
        evaluate_kmeans(x_umap, save_path=args['path_opt_cluster'], prefix=clean_title, plot_flag=args['plot_cluster'])
        evaluate_gmm(x_umap, save_path=args['path_opt_cluster'], prefix=clean_title, plot_flag=args['plot_cluster'])
        evaluate_consensus(x_umap, save_path=args['path_opt_cluster'], prefix=clean_title, plot_flag=args['plot_cluster'])
        evaluate_hdbscan(x_umap)

    labels_dict = run_clustering(x_umap)
    labeling_umap = pd.DataFrame({
        'ID': df_merged['ID'],
        'group': df_merged['Group'],
        'X1': x_umap[:, 0],
        'X2': x_umap[:, 1],
        'labels_hdb': labels_dict['HDBSCAN'],
        'labels_km': labels_dict['K-Means'],
        'labels_gmm_cdr': df_merged['labels_gmm_cdr']
    })

    if args['plot_cluster'] or args['path_cluster']:
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['group'], args['path_cluster'], args['prefix'] + " - Group label")
        plot_clusters_vs_groups(x_umap, labels_dict, labeling_umap['labels_gmm_cdr'], args['path_cluster'], args['prefix'] + " - GMM label", colors_gmm=True)

    if args.get("threshold") in [0.1, 0.2]:
        suffix = f"_thr{str(args['threshold']).replace('.', '')}"
        labeling_umap = labeling_umap.rename(columns={"labels_km": f"labels_km{suffix}", "labels_hdb": f"labels_hdb{suffix}"})

    km_col = [col for col in labeling_umap.columns if col.startswith("labels_km")][-1]
    hdb_col = [col for col in labeling_umap.columns if col.startswith("labels_hdb")][-1]

    if km_col not in df_meta.columns and hdb_col not in df_meta.columns:
        df_meta = df_meta.merge(labeling_umap[['ID', km_col, hdb_col]], on='ID', how='left')
    df_meta['labels_gmm_cdr'] = df_meta['labels_gmm_cdr'].astype('Int64')
    df_meta.to_csv(args['df_meta'], index=False)
    return labeling_umap

def main():
    args, df_input, df_meta = load_args_and_data()
    args['prefix'] = f"{args['threshold']} Threshold" if args['threshold'] in [0.1, 0.2] else "No Threshold"
    clustering_pipeline(df_input, df_meta, args)

if __name__ == "__main__":
    main()