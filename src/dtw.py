import numpy as np

def average_dtw_distance(indices, distance_dict):
    """
    Compute the average DTW distance between all pairs of time series in the list.
    
    Args:
    - time_series_list: List of NumPy arrays, where each array is a time series.
    
    Returns:
    - average_dtw: The average DTW distance between all pairs of time series.
    """
    n = len(indices)
    if n < 2:
        raise ValueError("Need at least two time series to compute DTW distances.")
    
    # Initialize distance accumulator
    total_dtw_distance = 0
    count = 0
    
    # Compute DTW distance between each pair of time series
    for i in range(n):
        for j in range(i + 1, n):

            i1 = indices[i]
            j1 = indices[j]
            try:
                distance = distance_dict[i1][j1]
            except:
                distance = distance_dict[j1][i1]

            # Accumulate the distance
            total_dtw_distance += distance
            count += 1
    
    # Compute the average DTW distance
    average_dtw = total_dtw_distance / count
    return average_dtw


def within_cluster_distances(ix, clus_labels, distances_dict):
    """
    Compute the within-cluster distances using precomputed distances.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    
    Returns:
    - within_cluster_distances: A dictionary with the within-cluster average distances.
    """
    # Step 1: Group indices by cluster labels
    clusters = {}
    for i, label in zip(ix, clus_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Step 2: Initialize a dictionary to store within-cluster distances
    within_cluster_distances = {}
    
    # Step 3: Compute the average pairwise distance within each cluster
    for label, points in clusters.items():
        if len(points) < 2:
            within_cluster_distances[label] = 0  # No distance to compute if the cluster has only one point
            continue
        
        total_distance = 0
        count = 0
        
        # Compute the pairwise distances between all points within the same cluster
        for i in range(len(points)):
            for j in range(i + 1, len(points)):  # Only consider pairs once
                p1 = points[i]
                p2 = points[j]
                total_distance += distances_dict[p1][p2]
                count += 1
        
        # Compute the average distance within the cluster
        average_distance = total_distance / count
        within_cluster_distances[label] = average_distance
    
    return within_cluster_distances


def between_cluster_distances(ix, clus_labels, distances_dict):
    """
    Compute the between-cluster distances using precomputed distances.
    
    Args:
    - ix: List of indices representing the time series.
    - clus_labels: List of cluster labels corresponding to the indices in ix.
    - distances_dict: Precomputed distance dictionary, where distances_dict[i][j]
                      gives the distance between observation i and observation j.
    
    Returns:
    - cluster_distances: A dictionary with the pairwise between-cluster distances (average).
    """
    # Step 1: Group indices by cluster labels
    clusters = {}
    for i, label in zip(ix, clus_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    # Step 2: Initialize a dictionary to store between-cluster distances
    cluster_distances = {}
    
    # Step 3: Compute the average pairwise distance between clusters
    cluster_labels = list(clusters.keys())
    
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            cluster_1 = cluster_labels[i]
            cluster_2 = cluster_labels[j]
            
            # Get all the points in cluster_1 and cluster_2
            points_1 = clusters[cluster_1]
            points_2 = clusters[cluster_2]
            
            # Compute the average distance between the points in the two clusters
            total_distance = 0
            count = 0
            for p1 in points_1:
                for p2 in points_2:
                    total_distance += distances_dict[p1][p2]
                    count += 1
            
            # Compute the average distance between the two clusters
            average_distance = total_distance / count
            cluster_distances[(cluster_1, cluster_2)] = average_distance
    
    return cluster_distances


def evaluate_clustering(ix, clus_labels, distance_dict):

    all_bcd = between_cluster_distances(ix, clus_labels, distance_dict)
    bcd = np.mean([v for k,v in all_bcd.items() if -1 not in k])

    all_wcd = within_cluster_distances(ix, clus_labels, distance_dict)
    wcd = np.mean([v for k,v in all_bcd.items() if k != -1])

    return wcd/bcd


def silhouette_score(ix, clus_labels, distance_dict):

    all_bcd = between_cluster_distances(ix, clus_labels, distance_dict)
    bcd = np.min([v for k,v in all_bcd.items() if -1 not in k])

    all_wcd = within_cluster_distances(ix, clus_labels, distance_dict)
    wcd = np.mean([v for k,v in all_bcd.items() if k != -1])

    return wcd/bcd


