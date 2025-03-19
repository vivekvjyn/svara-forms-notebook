from collections import defaultdict, Counter
import numpy as np
from scipy.stats import chi2_contingency
from math import sqrt

import pandas as pd

def get_contiguous_context_sequences(df, character, k, gap):
    """
    Given a DataFrame of sequence annotations, find all occurrences of a specified character
    and return a list of surrounding contexts (as strings) with the character capitalized.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'start_sec', 'end_sec', and 'label' columns.
    - character (str): The character to find in the sequences.
    - k (int): The number of contiguous characters to include on each side of the specified character.

    Returns:
    - List of strings, each representing a sequence with the central character capitalized.
    """
    # Sort the DataFrame by start_sec to ensure proper sequential processing
    df = df.sort_values(by='start_sec').reset_index(drop=True)

    # List to store the contextual sequences
    contextual_sequences = []

    # Iterate through the DataFrame to find all occurrences of the specified character
    for i, row in df.iterrows():
        if row['label'] == character:
            # Collect context around the central character
            context = []

            # Collect preceding characters within gap seconds
            preceding_context = []
            previous_end_time = df.loc[i, 'start_sec']  # Start contiguity check from the central character
            for j in range(i - 1, max(i - k - 1, -1), -1):
                if previous_end_time - df.loc[j, 'end_sec'] <= gap:
                    preceding_context.insert(0, df.loc[j, 'label'][0].lower())
                    previous_end_time = df.loc[j, 'start_sec']  # Update contiguity reference
                else:
                    preceding_context.insert(0, 'z')
                    break  # Stop if the character is not contiguous

            # Add the central character capitalized
            context.append(df.loc[i, 'label'][0].upper())

            # Collect succeeding characters within gap seconds
            succeeding_context = []
            next_start_time = df.loc[i, 'end_sec']  # Start contiguity check from the central character
            for j in range(i + 1, min(i + k + 1, len(df))):
                if df.loc[j, 'start_sec'] - next_start_time <= gap:
                    succeeding_context.append(df.loc[j, 'label'][0].lower())
                    next_start_time = df.loc[j, 'end_sec']  # Update contiguity reference
                else:
                    succeeding_context.append('z')
                    break  # Stop if the character is not contiguous

            # Combine preceding, central, and succeeding parts into one string
            context_string = ''.join(preceding_context + context + succeeding_context)
            contextual_sequences.append(context_string)

    return contextual_sequences



def compute_observed_frequencies(sequences):
    """
    Compute observed frequencies of sequences leading to the specified last character.
    
    Parameters:
    - sequences: list of strings, the sequences to analyze
    - last_char: the character that sequences must end with
    
    Returns:
    - observed_counts: dictionary of observed frequencies leading to `last_char`
    """
    observed_counts = Counter()
    
    for seq in sequences:
        observed_counts[(seq,)] += 1
                
    return observed_counts


def normalise_counts(counts):
    total = sum(counts.values())
    return {k:v/total for k,v in counts.items()}


def compute_expected_frequencies(observed_counts, transition_matrix, total, last_char):
    """
    Compute expected frequencies of sequences leading to the specified last character
    based on the transition matrix.
    
    Parameters:
    - observed_counts: Counter of observed frequencies
    - transition_matrix: dict of transition probabilities
    - last_char: the character that sequences must end with
    
    Returns:
    - expected_counts: dictionary of expected frequencies leading to `last_char`
    """
    expected_counts = defaultdict(float)

    # Iterate over each context in the transition matrix
    for context, next_chars in transition_matrix.items():
        expected_counts[context] = next_chars.get(last_char, 1e-10)
    
    total_exp = sum(expected_counts.values())
    
    expected_counts = {k:v*total/total_exp for k,v in expected_counts.items()}
    
    return expected_counts


def chi_squared_test(observed_counts, expected_counts):
    """
    Perform a Chi-squared test on the observed and expected frequencies.
    
    Parameters:
    - observed_counts: Counter of observed frequencies
    - expected_counts: Counter of expected frequencies
    
    Returns:
    - chi2: float, Chi-squared statistic
    - p_value: float, p-value for the Chi-squared test
    - effect_size: float, Cramér's V indicating effect size
    """
    observed_counts = normalise_counts(observed_counts)
    expected_counts = normalise_counts(expected_counts)

    # Create arrays for observed and expected counts
    observed = []
    expected = []
    
    # Collect all contexts
    all_contexts = set(observed_counts.keys()).union(expected_counts.keys())
    
    for context in all_contexts:
        observed_count = observed_counts.get(context, 0)
        expected_count = expected_counts.get(context, 0)
        
        observed.append(observed_count)
        expected.append(expected_count)
    
    # Perform Chi-squared test
    chi2, p_value = chi2_contingency([observed, expected])[:2]
    
    # Calculate effect size (Cramér's V)
    n = sum(observed)  # Total number of transitions in observed data
    min_dim = min(len(observed), len(expected)) - 1
    if min_dim != 0:
        effect_size = sqrt(chi2 / (n * min_dim))  # Cramér's V formula
    else:
        effect_size = 0

    return chi2, p_value, effect_size


def permutation_test(observed_counts, expected_counts, num_permutations=1000):
    real_chi2_stat, real_p_value, real_effect_size = chi_squared_test(observed_counts, expected_counts)
    permuted_effect_sizes = []
    permuted_p_values = []

    expected_keys = list(expected_counts.keys())
    expected_values = list(expected_counts.values())
    expected_values = [x/sum(expected_values) for x in expected_values]

    total = sum(observed_counts.values())
    for _ in range(num_permutations):
        permuted_expected = {}
        
        for i in range(total):
            key = np.random.choice(range(len(expected_keys)), replace=True, p=expected_values)
            key = expected_keys[key]

            if key not in permuted_expected:
                permuted_expected[key] = 1
            else:
                permuted_expected[key] += 1

        chi2_stat, p_value, effect_size = chi_squared_test(observed_counts, permuted_expected)
        permuted_effect_sizes.append(effect_size)
        permuted_p_values.append(p_value)

    # Calculate empirical p-value and effect size significance
    empirical_p_value = np.mean([1 if g >= real_chi2_stat else 0 for g in permuted_effect_sizes])
    effect_size_mean = np.mean(permuted_effect_sizes)

    return {
        "real_chi2_stat": real_chi2_stat,
        "real_p_value": real_p_value,
        "real_effect_size": real_effect_size,
        "empirical_p_value": empirical_p_value,
        "mean_random_effect_size": effect_size_mean
    }



# 2. Extract Varying-Length Context Sequences
def extract_context_sequences(group, context_len=1):
    pre_capital, post_capital = [], []
    
    for seq in group:
        cap_index = next((i for i, c in enumerate(seq) if c.isupper()), None)
        
        if cap_index is not None:
            # Preceding context
            if cap_index >= context_len:
                pre_capital.append(seq[cap_index - context_len:cap_index])
            else:
                pre_capital.append(seq[:cap_index])
            
            # Succeeding context
            if cap_index + 1 + context_len <= len(seq):
                post_capital.append(seq[cap_index + 1:cap_index + 1 + context_len])
            else:
                post_capital.append(seq[cap_index + 1:])
    
    return pre_capital, post_capital




def prec_permutation_test(group, all_groups, k, num_permutations=1000):
    # context counts for group
    context = extract_context_sequences(group, k)
    pc = context[0]
    
    pc = [p for p in pc]
    
    pc = [x for x in pc if len(x)>=k]
    
    if len(pc) == 0:
        return {
            "real_chi2_stat": np.nan,
            "real_p_value": np.nan,
            "real_effect_size": np.nan,
            "empirical_p_value": np.nan,
            "mean_random_effect_size": np.nan,
            "n": np.nan
        }

    # context counts for all groups
    all_context = extract_context_sequences(all_groups, k)
    all_pc = all_context[0]

    all_pc = [p for p in all_pc]

    all_pc = [x for x in all_pc if len(x)>=k]

    # transition matrices
    observed_counts = compute_observed_frequencies(pc)
    expected_counts = compute_observed_frequencies(all_pc)
    
    # chi2
    #res = permutation_test(observed_counts, expected_counts, num_permutations=num_permutations)
    #_, p_value, effect_size = chi_squared_test(observed_counts, expected_counts)
    all_sequences = list(expected_counts.keys())
    alpha_expected = np.array([expected_counts.get(seq, 0) + 1 for seq in all_sequences])

    observed_js_divergence, p_value = permutation_test(observed_counts, alpha_expected, all_sequences, n_samples=100, n_permutations=1000)
    # chi2
    #res = permutation_test(observed_counts, expected_counts, num_permutations=num_permutations)
    ##_, p_value, effect_size = chi_squared_test(observed_counts, expected_counts)
    #res['n'] = len(sc)

    res['n'] = len(pc)
    res['real_effect_size'] = observed_js_divergence
    res['empirical_p_value'] = p_value

    return res

    
def succ_permutation_test(group, all_groups, k, num_permutations=1000):
    # context counts for group
    context = extract_context_sequences(group, k)
    sc = context[1]
    
    sc = [s for s in sc]
    sc = [s for s in sc if len(s)>=k]
    
    if len(sc) == 0:
        return {
            "real_chi2_stat": np.nan,
            "real_p_value": np.nan,
            "real_effect_size": np.nan,
            "empirical_p_value": np.nan,
            "mean_random_effect_size": np.nan,
            "n": np.nan
        }

    # context counts for all groups
    all_context = extract_context_sequences(all_groups, k)
    all_sc = all_context[1]

    all_sc = [s for s in all_sc]
    all_sc = [s for s in all_sc if len(s)>=k]

    # transition matrices
    observed_counts = compute_observed_frequencies(sc)
    expected_counts = compute_observed_frequencies(all_sc)
    





    all_sequences = list(expected_counts.keys())
    alpha_expected = np.array([expected_counts.get(seq, 0) + 1 for seq in all_sequences])

    observed_js_divergence, p_value = permutation_test(observed_counts, alpha_expected, all_sequences, n_samples=100, n_permutations=1000)
    # chi2
    #res = permutation_test(observed_counts, expected_counts, num_permutations=num_permutations)
    ##_, p_value, effect_size = chi_squared_test(observed_counts, expected_counts)
    #res['n'] = len(sc)

    res['n'] = len(sc)
    res['real_effect_size'] = observed_js_divergence
    res['empirical_p_value'] = p_value

    return res


def both_permutation_test(group, all_groups, k, num_permutations=1000):
    # context counts for group
    context = extract_context_sequences(group, k)
    pc = context[0]
    sc = context[1]
    
    bc = [f'{p} {s}' for p,s in zip(pc,sc)] # space to indicate where current svara is
    bc = [b for b in bc if len(bc)>=k*2+1]

    if len(bc) == 0:
        return {
            "real_chi2_stat": np.nan,
            "real_p_value": np.nan,
            "real_effect_size": np.nan,
            "empirical_p_value": np.nan,
            "mean_random_effect_size": np.nan,
            "n": np.nan
        }
    # context counts for all groups
    all_context = extract_context_sequences(all_groups, k)
    all_pc = all_context[0]
    all_sc = all_context[1]

    all_bc = [f'{p} {s}' for p,s in zip(all_pc, all_sc)] # space to indicate where current svara is
    
    all_bc = [x for x in all_bc if len(x)>=k*2+1]

    # transition matrices
    observed_counts = compute_observed_frequencies(bc)
    expected_counts = compute_observed_frequencies(all_bc)
    
    # chi2
    #res = permutation_test(observed_counts, expected_counts, num_permutations=num_permutations)
    #_, p_value, effect_size = chi_squared_test(observed_counts, expected_counts)
    
    all_sequences = list(expected_counts.keys())
    alpha_expected = np.array([expected_counts.get(seq, 0) + 1 for seq in all_sequences])

    observed_js_divergence, p_value = permutation_test(observed_counts, alpha_expected, all_sequences, n_samples=100, n_permutations=1000)
    # chi2
    #res = permutation_test(observed_counts, expected_counts, num_permutations=num_permutations)
    ##_, p_value, effect_size = chi_squared_test(observed_counts, expected_counts)
    #res['n'] = len(sc)

    res['n'] = len(bc)
    res['real_effect_size'] = observed_js_divergence
    res['empirical_p_value'] = p_value

    return res
