#%% md
# ### Input Data
#%% md
# For the further analysis we will remove the ID labels columns, but the order is mantained.
#
# Possible dataset from mean maps:
# - `df_thr01_gm_masked`
# - `df_thr02_gm_masked`
# - `df_thr01_har_masked`
# - `df_thr02_har_masked`
# - `df_gm_masked`
# - `df_har_masked`
#
# Possible dataset from mean network:
# - `df_networks_no_thr`
# - `df_networks_thr01`
# - `df_networks_thr02`
#
# Here we go with one example
#%%
def features_merging(df_voxel, df_labels):
    # Meta columns
    meta_columns = list(df_labels.columns)

    # Merge based on subject ID
    dataframe_merge = pd.merge(df_voxel, df_labels, on='ID', how='left', validate='one_to_one')

    # Reorder columns: metadata first, then voxel features
    ordered_cols = meta_columns + [col for col in dataframe_merge.columns if col not in meta_columns]
    dataframe_merge = dataframe_merge[ordered_cols]

    # Sanity check to ensure row alignment
    assert (dataframe_merge['ID'].values == df_voxel['ID'].values).all(), "Row order mismatch after merge"

    # Extract features only (drop metadata)
    x = dataframe_merge.drop(columns=meta_columns)

    return dataframe_merge, x
#%%
# Function for skipping the preprocessing step above
def open_dataframes(path_data, path_metadata):
    df_masked = pd.read_pickle(path_data)
    df_meta = pd.read_csv(path_metadata)

    return df_masked, df_meta
#%%
# Load dataframes
df_gm, df_meta = open_dataframes(path_df_gm, path_df_meta)

# Merge with metadata
df_merged, X = features_merging(df_gm, df_meta)
#%% md
# ## Dimensionality Reduction
#%% md
# UMAP is not changing the order of the rows so each row correspond to the previous subject in the not reduced dataset.
#%%
np.random.seed(42)
#%%
umap_params = {'n_neighbors': 15, 'n_components': 2, 'metric': 'euclidean', 'n_epochs': 1000, 'learning_rate': 1.0, 'init': 'spectral', 'min_dist': 0.1, 'spread': 1.0, 'low_memory': False, 'set_op_mix_ratio': 1.0, 'local_connectivity': 1, 'repulsion_strength': 1.0, 'negative_sample_rate': 5, 'transform_queue_size': 4.0, 'a': None, 'b': None, 'random_state': 42, 'metric_kwds': None, 'angular_rp_forest': False, 'target_n_neighbors': -1, 'transform_seed': 42, 'verbose': False, 'unique': False}
#%%
# Parameters
reducer = umap.UMAP(**umap_params)

# Fit the model
X_umap = reducer.fit_transform(X)
#%%
plt.figure(figsize=(6, 4))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, alpha=0.6)
plt.title("UMAP Embedding")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.show()
#%% md
# ## Unsupervised Clustering
#%% md
# ### Fit models
#%%
# HDBSCAN
cluster_hdb = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdb = cluster_hdb.fit_predict(X_umap)

# DBSCAN
clusterer_db = DBSCAN(eps=0.5, min_samples=5)
labels_db = clusterer_db.fit_predict(X_umap)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(X_umap)

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
labels_gmm = gmm.fit_predict(X_umap)
#%% md
# Result collection
#%%
labels_dict = {
    "HDBSCAN": labels_hdb,
    #"DBSCAN": labels_db,
    "K-Means": labels_km,
    #"GMM": labels_gmm
}
#%%
labels_df = pd.DataFrame({
    'ID': df_merged['ID'],
    'Group': df_merged['Group'],
    'CDR_SB': df_merged['CDR_SB'],
    'MMSE': df_merged['MMSE'],
    'labels_gmm_cdr': df_merged['labels_gmm_cdr'],
    'HDBSCAN': labels_hdb,
    'DBSCAN': labels_db,
    'KMeans': labels_km,
    'GMM': labels_gmm
})

#labels_df.to_csv("/Users/emmatosato/Documents/PhD/ANM_Verona/utils/clustering_labels_by_ID.csv", index=False)
#%%
# HDBSCAN optimal clusters
labels_hdb, n_hdb = evaluate_hdbscan(X_umap, min_cluster_size=5)
#%% md
# ### Optimal number of clusters
#%%
# K-Means optimal clusters
#inertias, sil_scores = evaluate_kmeans(X_umap,K_range=range(2, 11),save_path=None,prefix= "gm",plot_flag=True)

# GMM optimal clusters
#aic, bic = evaluate_gmm(X_umap, K_range=range(2, 11), save_path=None, prefix="gm",plot_flag=True)

# Consenus clustering
#stability_scores = evaluate_consensus(X_umap,K_range=range(2, 11),n_runs=100,save_path=None,prefix='gm',plot_flag=True)
#%% md
# ### Plotting
#%%
def plot_clusters_vs_labels(x_umap, labels_dictionary, label_source_df, label_column, figsize=(16, 24), margin=5):
    n = len(labels_dictionary)
    n_cols = 2
    n_rows = n

    x_min, x_max = x_umap[:, 0].min() - margin, x_umap[:, 0].max() + margin
    y_min, y_max = x_umap[:, 1].min() - margin, x_umap[:, 1].max() + margin

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, (title, labels) in enumerate(labels_dictionary.items()):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        plot_df = pd.DataFrame({
            'X1': x_umap[:, 0],
            'X2': x_umap[:, 1],
            'cluster': labels,
            'label': label_source_df[label_column].reset_index(drop=True)
        })

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='cluster', palette='Set1', s=50, ax=ax_left, legend='full')
        ax_left.set_title(f'{title} - Clustering')
        ax_left.set_xlim(x_min, x_max)
        ax_left.set_ylim(y_min, y_max)
        ax_left.set_xlabel("UMAP 1")
        ax_left.set_ylabel("UMAP 2")

        sns.scatterplot(data=plot_df, x='X1', y='X2', hue='label', palette='Set2', s=50, ax=ax_right, legend='full')
        ax_right.set_title(f'{title} - {label_column}')
        ax_right.set_xlim(x_min, x_max)
        ax_right.set_ylim(y_min, y_max)
        ax_right.set_xlabel("UMAP 1")
        ax_right.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.show()
#%%
#plot_clusters_vs_labels(X_umap, labels_dict, df_merged, label_column='labels_gmm_cdr', margin=1.5)
#%%
#plot_clusters_vs_labels(X_umap, labels_dict, df_merged, label_column='Group', margin=2.5)
#%%
plot_clusters_vs_groups(X_umap, labels_dict, labels_df['Group'], save_path = None, title_prefix ='Prova', margin = 2.0, plot_flag=True)
#%%
plot_clusters_vs_groups(X_umap, labels_dict, labels_df['labels_gmm_cdr'], save_path = None, title_prefix =None, margin= 2.0, plot_flag=True, colors_gmm= True)
#%% md
# ### Statistical Evaluation
#%% md
# ##### Do clusters separate diagnoses?
#%%
diagnoses = ['ADNI', 'CBS', 'PSP']

for diag in diagnoses:
    print(f"\n=== Chi-squared test for {diag} ===")

    # Crea colonna binaria con etichette leggibili
    labels_df[f'{diag}_label'] = labels_df['Group'].apply(lambda x: diag if x == diag else 'other')

    # Tabella di contingenza: Cluster vs (diag vs other)
    contingency = pd.crosstab(labels_df['KMeans'], labels_df[f'{diag}_label'])
    print("Contingency Table:")
    print(contingency)

    # Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    # Output
    print(f"\nChi² = {chi2:.4f}")
    print(f"p-value = {p_value:.4f}")
    print(f"Degrees of Freedom = {dof}")