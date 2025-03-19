from collections import defaultdict, Counter, defaultdict
import re

from sklearn.metrics import mutual_info_score


def longest_common_subsequence(str1, str2):
    """
    Finds the longest common subsequence between two strings using dynamic programming.
    
    Parameters:
    - str1: First string.
    - str2: Second string.
    
    Returns:
    - LCS: The longest common subsequence string.
    """
    m, n = len(str1), len(str2)
    # Create a 2D DP array to store the lengths of LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build the dp array from bottom-up
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Reconstruct the LCS from the dp array
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))


def longest_common_subsequence_in_list(strings):
    """
    Finds the longest common subsequence in a list of strings.
    
    Parameters:
    - strings: List of strings.
    
    Returns:
    - LCS: The longest common subsequence string for all strings in the list.
    """
    if not strings:
        return ""
    
    # Start with the first string as the LCS
    lcs = strings[0]
    
    # Iteratively find the LCS between the current LCS and the next string
    for i in range(1, len(strings)):
        lcs = longest_common_subsequence(lcs, strings[i])
        if not lcs:  # If the LCS becomes empty, stop early
            break
    
    return lcs


def positional_frequency(strings, window=1):
    """
    Analyzes the frequency of letters around the capital letter in each string.
    
    Parameters:
    - strings: List of strings (each containing one capital letter).
    - window: Number of positions to the left and right of the capital letter to analyze.
    
    Returns:
    - freq_dict: A dictionary showing the frequency of neighboring letters around the capital letter.
    """
    freq_dict = defaultdict(lambda: defaultdict(int))
    
    for seq in strings:
        # Find the capital letter and its position
        cap_index = next(i for i, c in enumerate(seq) if c.isupper())
        capital_letter = seq[cap_index]
        
        # Analyze neighbors within the window
        for i in range(-window, window + 1):
            if i == 0 or not (0 <= cap_index + i < len(seq)):
                continue
            neighbor = seq[cap_index + i]
            position = 'left' if i < 0 else 'right'
            freq_dict[position][neighbor] += 1
    
    return dict(freq_dict)


def ngram_frequency(strings, n=3):
    """
    Analyzes the frequency of n-grams around the capital letter in each string.
    
    Parameters:
    - strings: List of strings (each containing one capital letter).
    - n: Size of the n-gram to analyze (e.g., 3 for trigrams).
    
    Returns:
    - ngram_freq: A dictionary showing the frequency of n-grams around the capital letter.
    """
    ngram_freq = Counter()

    for seq in strings:
        # Find the capital letter and its position
        cap_index = next(i for i, c in enumerate(seq) if c.isupper())
        
        # Extract the n-gram centered on the capital letter
        start = max(0, cap_index - (n // 2))
        end = min(len(seq), cap_index + (n // 2) + 1)
        ngram = seq[start:end]
        if len(ngram) == n:
            ngram_freq[ngram] += 1
    
    return ngram_freq


def markov_chain(strings):
    """
    Builds a Markov chain model for the letters surrounding the capital letter in each string.
    
    Parameters:
    - strings: List of strings (each containing one capital letter).
    
    Returns:
    - transition_prob: A dictionary showing transition probabilities between letters around the capital letter.
    """
    transitions = defaultdict(lambda: defaultdict(int))
    total_transitions = defaultdict(int)
    
    for seq in strings:
        cap_index = next(i for i, c in enumerate(seq) if c.isupper())
        
        # Analyze transitions around the capital letter
        for i in range(1, len(seq)):
            if i == cap_index:
                continue
            prev_char = seq[i - 1] if i - 1 >= 0 else None
            current_char = seq[i]
            
            if prev_char:
                transitions[prev_char][current_char] += 1
                total_transitions[prev_char] += 1

    # Convert counts to probabilities
    transition_prob = {}
    for prev_char, next_chars in transitions.items():
        transition_prob[prev_char] = {char: count / total_transitions[prev_char] for char, count in next_chars.items()}
    
    return transition_prob


#def mutual_information(strings):
#    """
#    Calculates mutual information between the capital letter and its neighboring letters.
#    
#    Parameters:
#    - strings: List of strings (each containing one capital letter).
#    
#    Returns:
#    - mutual_info_dict: Mutual information score between capital letter and neighboring letters.
#    """
#    capital_letters = []
#    neighbors = []

#    for seq in strings:
#        # Find the capital letter and its position
#        cap_index = next(i for i, c in enumerate(seq) if c.isupper())
#        capital_letters.append(seq[cap_index])
#        
#        # Collect neighbors within a small window (e.g., 1 character on each side)
#        window_neighbors = []
#        if cap_index > 0:
#            window_neighbors.append(seq[cap_index - 1])
#        if cap_index < len(seq) - 1:
#            window_neighbors.append(seq[cap_index + 1])
#        
#        neighbors.append("".join(window_neighbors))

#    # Calculate mutual information
#    mi_score = mutual_info_score(capital_letters, neighbors)
#    
#    return mi_score


def regex_pattern(strings, pattern="..A.."):
    """
    Finds and analyzes patterns around the capital letter using regular expressions.
    
    Parameters:
    - strings: List of strings (each containing one capital letter).
    - pattern: Regular expression pattern to match surrounding letters (e.g., "..A.." matches two letters before and after "A").
    
    Returns:
    - pattern_matches: A dictionary of patterns found and their frequencies.
    """
    pattern_matches = Counter()

    for seq in strings:
        match = re.search(pattern, seq)
        if match:
            pattern_matches[match.group()] += 1
    
    return pattern_matches


def group_strings(strings):
    groups = defaultdict(list)
    used = set()  # To track used strings
    substr_length_range = range(2, 6)  # Substring lengths from 2 to 5

    # Iterate through each string
    for s in strings:
        capital_letter = next((c for c in s if c.isupper()), None)
        
        # Generate all relevant substrings containing the capital letter
        for length in substr_length_range:
            for i in range(len(s) - length + 1):
                substring = s[i:i + length]
                if capital_letter in substring:
                    groups[substring].append(s)

    # Prepare final output while ensuring all strings are grouped once
    result = {}
    
    for substring, members in sorted(groups.items(), key=lambda x: len(x[0])):  # Sort by substring length
        if all(member not in used for member in members):  # Check if any member is already used
            result[substring] = members
            used.update(members)  # Mark members as used
    
    return result