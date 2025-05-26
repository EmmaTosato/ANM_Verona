import os
import pandas as pd
import matplotlib.pyplot as plt
import umap

# ---------------------------
# Merge voxel data with metadata
# ---------------------------
def x_features_return(df_voxel, df_labels):
    meta_columns = list(df_labels.columns)

    dataframe_merge = pd.merge(df_voxel, df_labels, on='ID', how='left', validate='one_to_one')

    ordered_cols = meta_columns + [col for col in dataframe_merge.columns if col not in meta_columns]
    dataframe_merge = dataframe_merge[ordered_cols]

    assert (dataframe_merge['ID'].values == df_voxel['ID'].values).all(), "Row order mismatch after merge"

    x = dataframe_merge.drop(columns=meta_columns)

    return dataframe_merge, x

# --------------------------------------
# Run UMAP and optionally save/show plot
# --------------------------------------
def run_umap(x_input, plot_flag=True, save_path=None, title="UMAP_Embedding"):
    reducer = umap.UMAP(
        n_neighbors=15, n_components=2, metric='euclidean', n_epochs=1000,
        learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0,
        low_memory=False, set_op_mix_ratio=1.0, local_connectivity=1,
        repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0,
        random_state=42
    )

    x_umap = reducer.fit_transform(x_input)

    if plot_flag or save_path:
        plt.figure(figsize=(6, 4))
        plt.scatter(x_umap[:, 0], x_umap[:, 1], s=10, alpha=0.6)
        plt.title(title)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.grid(True)

        if save_path:
            save_file = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
            plt.savefig(save_file, dpi=300)
            print(f"Embeddings plot saved to: {save_file}")

        if plot_flag:
            plt.show()

        plt.close()

    return x_umap
