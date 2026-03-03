"""
Extract Binary Semantic Attributes from PMI Matrix

Based on the paper "On the Emergence of Linear Analogies in Word Embeddings"
by Korchinski et al. (2025)

The key insight: eigenvectors of the PMI matrix directly encode binary semantic
attributes of words. The k-th eigenvector v_k gives the k-th attribute for all words.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def extract_attributes(
    PMI: np.ndarray,
    words: List[str],
    d: Optional[int] = None,
    threshold_factor: float = 0.1,
    auto_detect_threshold: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract binary semantic attributes from a PMI matrix.

    According to the model, words are characterized by d binary attributes, and
    the eigenvectors of log M (PMI) directly encode these attributes:
        α_i^(k) = sign(v_k(i))

    Parameters
    ----------
    PMI : np.ndarray
        m × m matrix of pointwise mutual information values
    words : List[str]
        List of m word strings corresponding to rows/columns of PMI
    d : Optional[int]
        Number of semantic attributes to extract. If None, will auto-detect
        from eigenvalue spectrum.
    threshold_factor : float
        Binarization threshold as a fraction of standard deviation.
        Higher values = more conservative (more 0s in output).
    auto_detect_threshold : float
        For auto-detection: eigenvalue ratio threshold to identify the "elbow"
        in the spectrum. Default 10.0 means look for 10x drops.

    Returns
    -------
    attributes : np.ndarray
        m × d matrix where attributes[i, k] ∈ {-1, 0, +1} gives the value
        of attribute k for word i. 0 indicates uncertain/neutral.
    eigenvalues : np.ndarray
        The top d+1 eigenvalues (including the constant mode λ_0)
    eigenvectors : np.ndarray
        The top d+1 eigenvectors (m × (d+1) matrix)

    Examples
    --------
    >>> # Assume you have computed a PMI matrix
    >>> attributes, eigenvals, eigenvecs = extract_attributes(PMI, word_list)
    >>> # Find words with +1 for first attribute
    >>> pos_words = [word_list[i] for i in range(len(word_list))
    ...              if attributes[i, 0] == 1]
    """
    m = len(words)
    assert PMI.shape == (m, m), "PMI must be square matrix matching word count"

    # Step 1: Eigendecompose the PMI matrix
    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = np.linalg.eigh(PMI)

    # Sort by eigenvalue magnitude (descending)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 2: Auto-detect d if not provided
    if d is None:
        d = _auto_detect_d(eigenvalues, auto_detect_threshold)
        print(f"Auto-detected d = {d} semantic attributes")
    else:
        print(f"Using d = {d} semantic attributes")

    # Step 3: Extract attribute eigenvectors (skip v_0, the constant mode)
    attribute_vecs = eigenvectors[:, 1:d+1]

    # Step 4: Binarize with threshold
    # Use per-attribute thresholds based on standard deviation
    if threshold_factor == 0:
        # If threshold is 0, use only signs (no neutral values)
        attributes = np.sign(attribute_vecs)
        # Handle exact zeros (should be rare with float eigenvectors)
        # Assign them to +1 arbitrarily
        attributes[attributes == 0] = 1
    else:
        thresholds = threshold_factor * np.std(attribute_vecs, axis=0)
        attributes = np.where(
            np.abs(attribute_vecs) > thresholds,
            np.sign(attribute_vecs),
            0
        )

    # Return top d+1 eigenvalues and eigenvectors for inspection
    return attributes, eigenvalues[:d+1], eigenvectors[:, :d+1]


def _auto_detect_d(eigenvalues: np.ndarray, threshold: float = 10.0) -> int:
    """
    Auto-detect number of semantic attributes from eigenvalue spectrum.

    Looks for the "elbow" where eigenvalues drop significantly, indicating
    the transition from semantic modes to noise/zero modes.
    """
    # Compute ratios of consecutive eigenvalues
    ratios = np.abs(eigenvalues[:-1]) / (np.abs(eigenvalues[1:]) + 1e-10)

    # Find first significant drop (ratio > threshold)
    drops = np.where(ratios > threshold)[0]

    if len(drops) > 0:
        # Found a clear elbow - subtract 1 because we skip v_0
        d = max(1, drops[0] - 1)
    else:
        # No clear elbow - use heuristic based on positive eigenvalues
        # Keep eigenvectors with eigenvalues > 1% of max
        significant = np.abs(eigenvalues) > 0.01 * np.abs(eigenvalues[0])
        d = max(1, np.sum(significant) - 1)

    return int(d)


def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    d: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the eigenvalue spectrum to visualize semantic vs. noise modes.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (typically from extract_attributes output)
    d : Optional[int]
        If provided, draw a vertical line at d+1 to show the cutoff
    figsize : Tuple[int, int]
        Figure size for matplotlib
    """
    plt.figure(figsize=figsize)
    plt.plot(np.abs(eigenvalues), 'o-', markersize=4)
    plt.yscale('log')
    plt.xlabel('Eigenvalue index', fontsize=12)
    plt.ylabel('|Eigenvalue|', fontsize=12)
    plt.title('PMI Matrix Eigenvalue Spectrum', fontsize=14)
    plt.grid(True, alpha=0.3)

    if d is not None:
        plt.axvline(x=d, color='r', linestyle='--',
                   label=f'd+1 = {d+1} (cutoff)')
        plt.legend()

    plt.tight_layout()
    plt.show()


def interpret_attributes(
    attributes: np.ndarray,
    words: List[str],
    eigenvalues: Optional[np.ndarray] = None,
    top_n: int = 15
) -> None:
    """
    Print human-readable interpretation of extracted attributes.

    For each attribute, shows the top words with +1 and -1 values,
    which helps identify what semantic dimension it represents.

    Parameters
    ----------
    attributes : np.ndarray
        m × d matrix from extract_attributes
    words : List[str]
        List of word strings
    eigenvalues : Optional[np.ndarray]
        Eigenvalues to display (helps judge importance)
    top_n : int
        Number of example words to show for each polarity
    """
    d = attributes.shape[1]

    for k in range(d):
        print(f"\n{'='*70}")
        if eigenvalues is not None and k+1 < len(eigenvalues):
            print(f"Attribute {k} (eigenvalue λ_{k+1} = {eigenvalues[k+1]:.2f})")
        else:
            print(f"Attribute {k}")
        print('='*70)

        # Find words with clear +1 or -1 values
        pos_idx = np.where(attributes[:, k] == 1)[0]
        neg_idx = np.where(attributes[:, k] == -1)[0]
        neutral_idx = np.where(attributes[:, k] == 0)[0]

        print(f"\n+1 polarity ({len(pos_idx)} words):")
        if len(pos_idx) > 0:
            print("  " + ", ".join([words[i] for i in pos_idx[:top_n]]))
            if len(pos_idx) > top_n:
                print(f"  ... and {len(pos_idx) - top_n} more")

        print(f"\n-1 polarity ({len(neg_idx)} words):")
        if len(neg_idx) > 0:
            print("  " + ", ".join([words[i] for i in neg_idx[:top_n]]))
            if len(neg_idx) > top_n:
                print(f"  ... and {len(neg_idx) - top_n} more")

        print(f"\nNeutral/Uncertain: {len(neutral_idx)} words")


def compute_pmi(cooccurrence_matrix: np.ndarray,
                marginal_counts: np.ndarray,
                total_count: float) -> np.ndarray:
    """
    Compute PMI matrix from co-occurrence counts.

    PMI(i,j) = log(P(i,j) / (P(i)P(j)))
             = log(count(i,j) * N / (count(i) * count(j)))

    Parameters
    ----------
    cooccurrence_matrix : np.ndarray
        m × m matrix where entry (i,j) is count of i and j co-occurring
    marginal_counts : np.ndarray
        Length m array where entry i is total count of word i
    total_count : float
        Total number of co-occurrence observations

    Returns
    -------
    PMI : np.ndarray
        m × m PMI matrix
    """
    m = len(marginal_counts)
    P_ij = cooccurrence_matrix / total_count
    P_i = marginal_counts / total_count

    # Compute P(i)P(j) for all pairs
    P_i_P_j = np.outer(P_i, P_i)

    # PMI = log(P(i,j) / (P(i)P(j)))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    PMI = np.log((P_ij + epsilon) / (P_i_P_j + epsilon))

    return PMI


# Example usage
if __name__ == "__main__":
    # This is a toy example - in practice you'd load real PMI data
    print("Toy Example: Extracting attributes from synthetic data\n")

    # Create a simple example with d=2 attributes
    # 4 words: man, woman, king, queen
    # Attribute 1: gender (-1=masc, +1=fem)
    # Attribute 2: royalty (-1=common, +1=royal)

    words = ["man", "woman", "king", "queen"]

    # Synthetic PMI matrix following the model structure
    # In reality, you'd compute this from corpus co-occurrence counts
    np.random.seed(42)
    d_true = 2

    # Create attribute matrix
    A = np.array([
        [-1, -1],  # man: masculine, commoner
        [+1, -1],  # woman: feminine, commoner
        [-1, +1],  # king: masculine, royal
        [+1, +1],  # queen: feminine, royal
    ])

    # Create synthetic PMI following log M = δ·11^T + ADA^T
    delta = 2.0
    D = np.diag([1.5, 1.2])  # semantic strengths
    PMI_synthetic = delta * np.ones((4, 4)) + A @ D @ A.T

    # Add small noise
    PMI_synthetic += 0.1 * np.random.randn(4, 4)
    PMI_synthetic = (PMI_synthetic + PMI_synthetic.T) / 2  # symmetrize

    print("Synthetic PMI matrix:")
    print(PMI_synthetic)
    print()

    # Extract attributes
    attributes, eigenvals, eigenvecs = extract_attributes(
        PMI_synthetic, words, d=2, threshold_factor=0.01
    )

    print("\nExtracted attributes:")
    print(attributes)
    print("\nTrue attributes:")
    print(A)

    print("\nEigenvalues:")
    print(eigenvals)

    # Interpret
    interpret_attributes(attributes, words, eigenvals)

    print("\n" + "="*70)
    print("Note: With real data, attributes may have opposite sign")
    print("(e.g., +1/-1 swapped), but the semantic contrast is preserved.")
    print("="*70)
