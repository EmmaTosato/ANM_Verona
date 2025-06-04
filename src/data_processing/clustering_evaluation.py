import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
import hdbscan
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------
# K-Means Evaluation: Elbow + Silhouette
# ---------------------------------------
def evaluate_kmeans(X, K_range=range(2, 11), save_path=None, prefix='', plot_flag=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    inertias, sil_scores = [], []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, kmeans.labels_))

    if plot_flag or save_path:
        plt.figure()
        plt.plot(K_range, inertias, 'o-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title(f'{prefix} K-Means Elbow')
        if save_path:
            plt.savefig(os.path.join(save_path, f"{prefix}_kmeans_elbow.png"))
        if plot_flag:
            plt.show()
        plt.close()

        plt.figure()
        plt.plot(K_range, sil_scores, 'o-')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title(f'{prefix} K-Means Silhouette')
        if save_path:
            plt.savefig(os.path.join(save_path, f"{prefix}_kmeans_silhouette.png"))
        if plot_flag:
            plt.show()
        plt.close()

    return inertias, sil_scores

# ---------------------------
# GMM Evaluation: AIC + BIC
# ---------------------------
def evaluate_gmm(X, K_range=range(2, 11), save_path=None, prefix='', plot_flag=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    aic, bic = [], []
    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
        aic.append(gmm.aic(X))
        bic.append(gmm.bic(X))

    if plot_flag or save_path:
        plt.figure()
        plt.plot(K_range, aic, marker='o', label='AIC')
        plt.plot(K_range, bic, marker='o', label='BIC')
        plt.xlabel('Components')
        plt.ylabel('Score')
        plt.title(f'{prefix} GMM AIC/BIC')
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, f"{prefix}_gmm_aic_bic.png"))
        if plot_flag:
            plt.show()
        plt.close()

    return aic, bic


def evaluate_consensus(X, K_range=range(2, 11), n_runs=100, save_path=None, prefix='', plot_flag=True):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    stability_scores = []
    for n_clusters in K_range:
        co_assoc_matrix = np.zeros((X.shape[0], X.shape[0]))
        for _ in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=None)
            labels = kmeans.fit_predict(X)
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        co_assoc_matrix[i, j] += 1
                        co_assoc_matrix[j, i] += 1
        co_assoc_matrix /= n_runs
        distance_matrix = 1 - co_assoc_matrix
        Z = linkage(distance_matrix, method='average')
        consensus_labels = fcluster(Z, n_clusters, criterion='maxclust')
        score = silhouette_score(X, consensus_labels)
        stability_scores.append((n_clusters, score))

    ks, scores = zip(*stability_scores)
    if plot_flag or save_path:
        plt.figure()
        plt.plot(ks, scores, marker='o')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title(f'{prefix} Consensus Clustering Stability')
        if save_path:
            plt.savefig(os.path.join(save_path, f"{prefix}_consensus_stability.png"))
        if plot_flag:
            plt.show()
        plt.close()

    return stability_scores


# ----------------------------------------
# HDBSCAN Evaluation: Number of clusters
# -----------------------------------------
def evaluate_hdbscan(X, min_cluster_size=5):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = cluster.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"HDBSCAN found {n_clusters} clusters (excluding noise)\n")
    return labels, n_clusters

