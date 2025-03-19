import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, combine_pvalues, kruskal

def cramers_v(cluster_labels, feature):
    contingency_table = pd.crosstab(cluster_labels, feature)
    
    # Perform Chi-Square test of independence
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()  # total number of observations
    r, k = contingency_table.shape
    cramers_v_value = np.sqrt(chi2 / (n * (min(k-1, r-1))))
    
    return cramers_v_value, p


def combine_p_values(p_values):
    # Use Fisher's method to combine p-values
    stat, combined_p = combine_pvalues(p_values, method='fisher')
    return combined_p


def kruskal_with_epsilon_squared(categorical, continuous):
    # Ensure inputs are numpy arrays
    categorical = np.array(categorical)
    continuous = np.array(continuous)
    
    # Get unique clusters
    unique_clusters = np.unique(categorical)
    
    # Group continuous by clusters
    groups = [continuous[categorical == cluster] for cluster in unique_clusters]
    
    # Perform Kruskal-Wallis test
    h_stat, p_value = kruskal(*groups)
    
    # Calculate Epsilon-Squared (ε²)
    n = len(continuous)
    epsilon_squared = (h_stat - len(unique_clusters) + 1) / (n - len(unique_clusters))
    
    return h_stat, p_value, epsilon_squared

# Example usage:
# categorical = [0, 1, 1, 0, 2, 2, 1]  # Cluster labels
# durations = [10, 20, 15, 10, 30, 25, 20]  # Durations or continuous variable
# h_stat, p_value, epsilon_squared = kruskal_with_epsilon_squared(clus_labels, durations)
# print(f"Kruskal-Wallis H-statistic: {h_stat}, p-value: {p_value}, Epsilon-Squared: {epsilon_squared}")


# Example usage:
# h_stat, p_value, epsilon_squared = kruskal_with_epsilon_squared(df, 'height', 'group')
# print(f"Kruskal-Wallis result: H-stat={h_stat}, p-value={p_value}")
# print(f"Epsilon-Squared (Effect size): {epsilon_squared}")


