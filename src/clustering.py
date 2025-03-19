from collections import defaultdict
import random

import numpy as np

import hdbscan
import sklearn.preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances

def cluster(features, algo='dbscan', kwargs={}):

	if algo == 'dbscan':
		return dbscan(features, kwargs)
	elif algo =='gmm':
		return gmm(features, kwargs)
	else:
		raise Exception('algo must be either dbscan or gmm')


def dbscan(features, min_samples, min_cluster_size, cluster_selection_method, cluster_selection_epsilon, alpha, normalize=True):

	# Normalize the features (optional)
	if normalize:
		features = sklearn.preprocessing.normalize(features)
	
	cosine_distance_matrix = cosine_distances(features)

	# Define HDBSCAN with 'precomputed' metric
	hdbscan_clusterer = hdbscan.HDBSCAN(
		metric='precomputed', min_samples=min_samples, 
		min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method, 
		cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)

	# Fit the model
	cluster_labels = hdbscan_clusterer.fit_predict(cosine_distance_matrix)

	return cluster_labels


def dbscan_precomp(cosine_distance_matrix, min_samples, min_cluster_size, cluster_selection_method, cluster_selection_epsilon, alpha):

    # Define HDBSCAN with 'precomputed' metric
    hdbscan_clusterer = hdbscan.HDBSCAN(
        metric='precomputed', min_samples=min_samples, 
        min_cluster_size=min_cluster_size, cluster_selection_method=cluster_selection_method, 
        cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)

    # Fit the model
    cluster_labels = hdbscan_clusterer.fit_predict(cosine_distance_matrix)

    return cluster_labels


def gmm(features, normalize=True, n_components=10, covariance_type='full', percentile=50, random_state=42):

	if normalize:
		features = sklearn.preprocessing.normalize(features)

	# Define GMM
	gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)

	# Fit GMM model
	gmm.fit(features)

	# Predict the likelihood of each observation under the model
	log_probabilities = gmm.score_samples(features)

	# Set a threshold for excluding low-probability points
	threshold = np.percentile(log_probabilities, percentile)  # Keep top 90% of points

	thresh_ix = np.where(log_probabilities > threshold)[0]

	cluster_labels = gmm.predict(features)

	cluster_labels[thresh_ix] = -1

	return cluster_labels


def silhouette_score(ix, clus_labels, distances_dict):
    """
    Compute the average Silhouette Score for the clustering.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    
    Returns:
    - avg_silhouette_score: The average silhouette score for all points.
    """
    
    # Group points by cluster
    clusters = defaultdict(list)
    for i, label in zip(ix, clus_labels):
        clusters[label].append(i)
    
    silhouette_scores = []
    
    # Loop over each point
    for i, label in zip(ix, clus_labels):
        # Within-cluster distances
        own_cluster = clusters[label]
        if len(own_cluster) > 1:
            a_i = sum(distances_dict[i][j] for j in own_cluster if j != i) / (len(own_cluster) - 1)
        else:
            a_i = 0  # If only one point in the cluster
        
        # Nearest other cluster (between-cluster distance)
        b_i = float('inf')
        for other_label, other_cluster in clusters.items():
            if other_label != label:
                avg_dist = sum(distances_dict[i][j] for j in other_cluster) / len(other_cluster)
                b_i = min(b_i, avg_dist)
        
        # Silhouette score for point i
        s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_scores.append(s_i)
    
    # Average Silhouette score
    avg_silhouette_score = sum(silhouette_scores) / len(silhouette_scores)
    
    return avg_silhouette_score


def calinski_harabasz_index(ix, clus_labels, distances_dict):
    """
    Compute the Calinski-Harabasz index for the clustering.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    
    Returns:
    - ch_index: The Calinski-Harabasz index.
    """
    
    # Group points by cluster
    clusters = defaultdict(list)
    for i, label in zip(ix, clus_labels):
        clusters[label].append(i)
    
    k = len(clusters)  # Number of clusters
    n = len(ix)  # Total number of points
    
    # Compute the centroid for all points (using index 0 as a proxy for the centroid)
    overall_centroid = ix[0]
    
    # Compute within-cluster dispersion W_k
    W_k = 0
    for label, points in clusters.items():
        for i in points:
            W_k += sum(distances_dict[i][j] for j in points) / (len(points) - 1)
    
    # Compute between-cluster dispersion B_k
    B_k = 0
    for label, points in clusters.items():
        for i in points:
            B_k += distances_dict[i][overall_centroid] * len(points)
    
    # Calinski-Harabasz Index
    ch_index = (B_k / (k - 1)) / (W_k / (n - k))
    
    return ch_index


def davies_bouldin_index(ix, clus_labels, distances_dict):
    """
    Compute the Davies-Bouldin index for the clustering.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    
    Returns:
    - db_index: The Davies-Bouldin index.
    """
    
    # Group points by cluster
    clusters = defaultdict(list)
    for i, label in zip(ix, clus_labels):
        clusters[label].append(i)
    
    # Calculate the scatter (average within-cluster distance) for each cluster
    scatter = {}
    for label, points in clusters.items():
        if len(points) > 1:
            s_i = sum(distances_dict[i][j] for i in points for j in points if i != j) / (len(points) * (len(points) - 1))
            scatter[label] = s_i
        else:
            scatter[label] = 0
    
    # Calculate the Davies-Bouldin index
    db_ratios = []
    
    cluster_labels = list(clusters.keys())
    for i in range(len(cluster_labels)):
        max_ratio = -np.inf
        for j in range(len(cluster_labels)):
            if i != j:
                label_i = cluster_labels[i]
                label_j = cluster_labels[j]
                centroid_dist = sum(distances_dict[a][b] for a in clusters[label_i] for b in clusters[label_j]) / (len(clusters[label_i]) * len(clusters[label_j]))
                ratio = (scatter[label_i] + scatter[label_j]) / centroid_dist
                max_ratio = max(max_ratio, ratio)
        db_ratios.append(max_ratio)
    
    db_index = np.mean(db_ratios)
    
    return db_index


def gap_statistic(ix, clus_labels, distances_dict, n_refs=10):
    """
    Compute the Gap Statistic for the clustering.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    - n_refs: Number of reference datasets to generate.
    
    Returns:
    - gap_value: The gap statistic value.
    """
    
    def within_cluster_dispersion(ix, clus_labels, distances_dict):
        """ Compute the within-cluster dispersion for a given clustering. """
        clusters = defaultdict(list)
        for i, label in zip(ix, clus_labels):
            clusters[label].append(i)
        W_k = 0
        for label, points in clusters.items():
            for i in points:
                W_k += sum(distances_dict[i][j] for j in points) / (len(points) - 1)
        return W_k
    
    # Step 1: Compute the within-cluster dispersion for the actual clustering
    W_k = within_cluster_dispersion(ix, clus_labels, distances_dict)
    
    # Step 2: Generate reference datasets and compute dispersions for those
    ref_disp = []
    for _ in range(n_refs):
        random_labels = [random.choice(list(set(clus_labels))) for _ in ix]
        ref_disp.append(within_cluster_dispersion(ix, random_labels, distances_dict))
    
    # Step 3: Compute the gap statistic
    log_Wk = np.log(W_k)
    log_Wk_ref = np.mean(np.log(ref_disp))
    
    gap_value = log_Wk_ref - log_Wk
    
    return gap_value
