import os
import pandas as pd
import matplotlib.pyplot as plt
import umap
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ---------------------------
# Merge voxel data with metadata
# ---------------------------
def x_features_return(df_voxel, df_labels):
    # Meta data columns
    meta_columns = list(df_labels.columns)

    # Merging datasets
    dataframe_merge = pd.merge(df_voxel, df_labels, on='ID', how='left', validate='one_to_one')

    # Ordering
    ordered_cols = meta_columns + [col for col in dataframe_merge.columns if col not in meta_columns]
    dataframe_merge = dataframe_merge[ordered_cols]
    assert (dataframe_merge['ID'].values == df_voxel['ID'].values).all(), "Row order mismatch after merge"

    # Features data
    x = dataframe_merge.drop(columns=meta_columns)

    print("Dataframe shape after merge:", dataframe_merge.shape)
    print("Meta columns:", len(meta_columns))
    print("Feature matrix shape:", x.shape, "\n")

    return dataframe_merge, x

# --------------------------------------
# Run UMAP and optionally save/show plot
# --------------------------------------
def run_umap(x_input, plot_flag=True, save_path=None, title=None):
    reducer = umap.UMAP(
        n_neighbors=15, n_components=2, metric='euclidean', n_epochs=1000,
        learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0,
        low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1,
        repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0,
        random_state=42
    )
    print("Running UMAP...\n")
    x_umap = reducer.fit_transform(x_input)

    plt.figure(figsize=(6, 4))

    dot_color = "#d74c4c"

    plt.scatter(
        x_umap[:, 0], x_umap[:, 1],
        s=50,
        alpha=0.9,
        color=dot_color,
        edgecolor='black',
        linewidth=0.5
    )

    plt.title(f' UMAP Embedding - {title}', fontsize=14, fontweight='bold')
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


    return x_umap
