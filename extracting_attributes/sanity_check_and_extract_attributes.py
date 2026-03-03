"""
Extract PMI Matrix and Binary Semantic Attributes from Skip-gram Vectors

This script:
1. Loads target (W) and context (C) vectors from trained Skip-gram model
2. Optionally filters vocabulary by frequency to reduce memory usage
3. Performs sanity check by showing nearest neighbors for test words
4. Computes PMI approximation as W·C^T (in memory-efficient batches)
5. Extracts binary semantic attributes via eigendecomposition
6. Saves and interprets the results

Usage:
    Load from model file (RECOMMENDED - avoids file duplication):
        python sanity_check_and_extract_attributes.py --from-model ./output/dutch_skipgram_model.bin --min-count 50

    Load from .npy files (if model file not available):
        python sanity_check_and_extract_attributes.py --min-count 50

    With custom test words:
        python sanity_check_and_extract_attributes.py --from-model ./output/dutch_skipgram_model.bin --test-words koning,koningin,stad

    Skip sanity check:
        python sanity_check_and_extract_attributes.py --from-model ./output/dutch_skipgram_model.bin --skip-sanity-check

Input files (from skip-gram training):
    Option 1 - Model file (recommended):
        - ./output/dutch_skipgram_model.bin

    Option 2 - Separate files:
        - ./output/target_vectors.npy
        - ./output/context_vectors.npy
        - ./output/word_counts.npy
        - ./output/vocab.txt

Output files:
    - ./output/pmi_approximation.npy    (PMI matrix)
    - ./output/filtered_vocab.txt       (vocabulary for PMI rows/columns and attributes)
    - ./output/attributes.npy           (binary semantic attributes)
    - ./output/eigenvalues.npy          (eigenvalues)
    - ./output/eigenvectors.npy         (eigenvectors)

Memory optimization:
    For large vocabularies (e.g., 100k+ words), the PMI matrix can require
    40GB+ of RAM. Use --min-count to filter to frequent words only:
    - --min-count 50: Only words occurring 50+ times
    - --min-count 100: Only words occurring 100+ times
    This can reduce memory from 40GB to <1GB while keeping meaningful words.

Required words preservation:
    When using --min-count, words from ./online_testing_all.csv (column: test_sentence)
    are automatically included even if they don't meet the frequency threshold.
    This ensures important test words are always present in the vocabulary.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from extract_attributes import extract_attributes, interpret_attributes, plot_eigenvalue_spectrum

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


def extract_words_from_csv(
    csv_path: str,
    column_name: str = "test_sentence"
) -> Set[str]:
    """
    Extract unique words from a column in a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    column_name : str
        Name of column containing sentences

    Returns
    -------
    words : Set[str]
        Set of unique words found in the sentences
    """
    if not Path(csv_path).exists():
        print(f"⚠️  Warning: CSV file not found: {csv_path}")
        return set()

    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        print(f"⚠️  Warning: Column '{column_name}' not found in CSV")
        print(f"   Available columns: {', '.join(df.columns)}")
        return set()

    # Extract all words from sentences
    words = set()
    for sentence in df[column_name].dropna():
        # Simple tokenization: lowercase and split on whitespace
        tokens = str(sentence).lower().split()
        for token in tokens:
            # Remove common punctuation
            token = token.strip('.,!?;:()[]{}"\'-')
            if len(token) > 0 and not token.isdigit():
                words.add(token)

    print(f"✓ Extracted {len(words)} unique words from {csv_path}")
    return words


def load_vectors(
    target_path: str = "./output/target_vectors.npy",
    context_path: str = "./output/context_vectors.npy",
    vocab_path: str = "./output/vocab.txt",
    counts_path: str = "./output/word_counts.npy",
    min_count: Optional[int] = None,
    required_words: Optional[Set[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load target vectors, context vectors, vocabulary, and optionally filter by frequency.

    Parameters
    ----------
    target_path : str
        Path to target vectors .npy file
    context_path : str
        Path to context vectors .npy file
    vocab_path : str
        Path to vocabulary .txt file
    counts_path : str
        Path to word counts .npy file
    min_count : Optional[int]
        Minimum word frequency threshold. If provided, only keeps words with
        count >= min_count. This drastically reduces memory usage.
    required_words : Optional[Set[str]]
        Set of words that must be included even if they don't meet min_count threshold

    Returns
    -------
    target_vectors : np.ndarray
        Target (input) vectors, shape (vocab_size, vector_size)
    context_vectors : np.ndarray
        Context (output) vectors, shape (vocab_size, vector_size)
    vocab : List[str]
        Ordered list of words
    word_counts : np.ndarray
        Word frequency counts, shape (vocab_size,)
    """
    print("="*70)
    print("LOADING VECTORS")
    print("="*70)

    # Check files exist
    required_paths = [target_path, context_path, vocab_path, counts_path]
    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    # Load vectors
    print(f"Loading target vectors from {target_path}...")
    target_vectors = np.load(target_path)
    print(f"✓ Target vectors shape: {target_vectors.shape}")

    print(f"Loading context vectors from {context_path}...")
    context_vectors = np.load(context_path)
    print(f"✓ Context vectors shape: {context_vectors.shape}")

    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    print(f"✓ Vocabulary size: {len(vocab)}")

    # Load word counts
    print(f"Loading word counts from {counts_path}...")
    word_counts = np.load(counts_path)
    print(f"✓ Word counts shape: {word_counts.shape}")
    print(f"  Count range: {word_counts.min()} - {word_counts.max()}")

    # Verify shapes match
    assert target_vectors.shape[0] == len(vocab), \
        f"Target vectors ({target_vectors.shape[0]}) doesn't match vocab size ({len(vocab)})"
    assert context_vectors.shape[0] == len(vocab), \
        f"Context vectors ({context_vectors.shape[0]}) doesn't match vocab size ({len(vocab)})"
    assert word_counts.shape[0] == len(vocab), \
        f"Word counts ({word_counts.shape[0]}) doesn't match vocab size ({len(vocab)})"

    # Filter by frequency if requested
    if min_count is not None and min_count > 0:
        print(f"\n" + "="*70)
        print(f"FILTERING BY FREQUENCY (min_count >= {min_count})")
        print("="*70)

        # Find indices of words meeting threshold
        keep_indices_bool = word_counts >= min_count

        # Add required words even if they don't meet threshold
        if required_words is not None and len(required_words) > 0:
            print(f"Required words: {required_words}")
            print(f"Ensuring {len(required_words)} required words are included...")
            n_forced = 0
            for i, word in enumerate(vocab):
                if word in required_words and not keep_indices_bool[i]:
                    keep_indices_bool[i] = True
                    n_forced += 1
            if n_forced > 0:
                print(f"✓ Added {n_forced} required words that were below threshold")

        keep_indices = np.where(keep_indices_bool)[0]
        n_kept = len(keep_indices)
        n_removed = len(vocab) - n_kept

        print(f"Words kept: {n_kept:,} ({100*n_kept/len(vocab):.1f}%)")
        print(f"Words removed: {n_removed:,} ({100*n_removed/len(vocab):.1f}%)")

        # Filter all arrays
        target_vectors = target_vectors[keep_indices]
        context_vectors = context_vectors[keep_indices]
        vocab = [vocab[i] for i in keep_indices]
        word_counts = word_counts[keep_indices]

        print(f"\n✓ Filtered vocabulary size: {len(vocab):,}")
        print(f"✓ New count range: {word_counts.min()} - {word_counts.max()}")

        # Estimate memory savings
        original_pmi_memory = (len(vocab) + n_removed) ** 2 * 4 / (1024**3)
        filtered_pmi_memory = len(vocab) ** 2 * 4 / (1024**3)
        savings = original_pmi_memory - filtered_pmi_memory

        print(f"\nMemory estimate for PMI matrix:")
        print(f"  Without filtering: {original_pmi_memory:.2f} GB")
        print(f"  With filtering: {filtered_pmi_memory:.2f} GB")
        print(f"  Savings: {savings:.2f} GB ({100*savings/original_pmi_memory:.1f}%)")

    return target_vectors, context_vectors, vocab, word_counts


def load_from_model(
    model_path: str,
    min_count: Optional[int] = None,
    required_words: Optional[Set[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Load vectors directly from a saved Word2Vec model file.

    This avoids loading separate .npy files and uses the model as single source of truth.

    Parameters
    ----------
    model_path : str
        Path to saved Word2Vec model (.bin file)
    min_count : Optional[int]
        Minimum word frequency threshold for filtering
    required_words : Optional[Set[str]]
        Set of words that must be included even if they don't meet min_count threshold

    Returns
    -------
    target_vectors : np.ndarray
        Target (input) vectors
    context_vectors : np.ndarray
        Context (output) vectors
    vocab : List[str]
        Vocabulary list
    word_counts : np.ndarray
        Word frequency counts
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "Gensim is required to load from model file. "
            "Install it with: pip install gensim\n"
            "Or use separate .npy files instead."
        )

    print("="*70)
    print("LOADING FROM MODEL FILE")
    print("="*70)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = Word2Vec.load(model_path)
    print(f"✓ Model loaded")

    # Extract vectors
    target_vectors = model.wv.vectors
    vocab = model.wv.index_to_key
    word_counts = np.array([model.wv.get_vecattr(word, "count") for word in vocab])

    # Extract context vectors
    if hasattr(model, 'syn1neg'):
        context_vectors = model.syn1neg
        print("✓ Using negative sampling context vectors (syn1neg)")
    elif hasattr(model, 'syn1'):
        context_vectors = model.syn1
        print("✓ Using hierarchical softmax context vectors (syn1)")
    else:
        raise ValueError(
            "Model does not have context vectors (syn1neg or syn1). "
            "Was it trained with negative sampling or hierarchical softmax?"
        )

    print(f"\nExtracted from model:")
    print(f"  Target vectors shape: {target_vectors.shape}")
    print(f"  Context vectors shape: {context_vectors.shape}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Word counts range: {word_counts.min()} - {word_counts.max()}")

    # Filter by frequency if requested
    if min_count is not None and min_count > 0:
        print(f"\n" + "="*70)
        print(f"FILTERING BY FREQUENCY (min_count >= {min_count})")
        print("="*70)

        # Find indices of words meeting threshold
        keep_indices_bool = word_counts >= min_count

        # Add required words even if they don't meet threshold
        if required_words is not None and len(required_words) > 0:
            print(f"Ensuring {len(required_words)} required words are included...")
            n_forced = 0
            for i, word in enumerate(vocab):
                if word in required_words and not keep_indices_bool[i]:
                    keep_indices_bool[i] = True
                    n_forced += 1
            if n_forced > 0:
                print(f"✓ Added {n_forced} required words that were below threshold")

        keep_indices = np.where(keep_indices_bool)[0]
        n_kept = len(keep_indices)
        n_removed = len(vocab) - n_kept

        print(f"Words kept: {n_kept:,} ({100*n_kept/len(vocab):.1f}%)")
        print(f"Words removed: {n_removed:,} ({100*n_removed/len(vocab):.1f}%)")

        # Filter all arrays
        target_vectors = target_vectors[keep_indices]
        context_vectors = context_vectors[keep_indices]
        vocab = [vocab[i] for i in keep_indices]
        word_counts = word_counts[keep_indices]

        print(f"\n✓ Filtered vocabulary size: {len(vocab):,}")
        print(f"✓ New count range: {word_counts.min()} - {word_counts.max()}")

        # Estimate memory savings
        original_pmi_memory = (len(vocab) + n_removed) ** 2 * 4 / (1024**3)
        filtered_pmi_memory = len(vocab) ** 2 * 4 / (1024**3)
        savings = original_pmi_memory - filtered_pmi_memory

        print(f"\nMemory estimate for PMI matrix:")
        print(f"  Without filtering: {original_pmi_memory:.2f} GB")
        print(f"  With filtering: {filtered_pmi_memory:.2f} GB")
        print(f"  Savings: {savings:.2f} GB ({100*savings/original_pmi_memory:.1f}%)")

    return target_vectors, context_vectors, vocab, word_counts


def find_nearest_neighbors(
    word: str,
    target_vectors: np.ndarray,
    vocab: List[str],
    k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find k nearest neighbors for a given word using cosine similarity.

    Parameters
    ----------
    word : str
        Query word
    target_vectors : np.ndarray
        Target vectors (vocab_size, vector_size)
    vocab : List[str]
        Vocabulary list
    k : int
        Number of nearest neighbors to return

    Returns
    -------
    neighbors : List[Tuple[str, float]]
        List of (word, similarity) tuples, sorted by similarity (descending)
    """
    # Find word index
    if word not in vocab:
        return []

    word_idx = vocab.index(word)
    word_vec = target_vectors[word_idx]

    # Normalize vectors for cosine similarity
    word_vec_norm = word_vec / (np.linalg.norm(word_vec) + 1e-10)
    target_vectors_norm = target_vectors / (np.linalg.norm(target_vectors, axis=1, keepdims=True) + 1e-10)

    # Compute cosine similarities
    similarities = target_vectors_norm @ word_vec_norm

    # Get top k+1 (including the word itself)
    top_indices = np.argsort(-similarities)[:k+1]

    # Build result list, excluding the query word itself
    neighbors = []
    for idx in top_indices:
        if idx != word_idx:
            neighbors.append((vocab[idx], similarities[idx]))

    return neighbors[:k]


def sanity_check_vectors(
    target_vectors: np.ndarray,
    vocab: List[str],
    test_words: Optional[List[str]] = None,
    k: int = 10
) -> None:
    """
    Sanity check: verify vector quality by showing nearest neighbors for test words.

    This helps verify that the loaded vectors produce semantically meaningful
    nearest neighbors, which indicates the Skip-gram training was successful.

    Parameters
    ----------
    target_vectors : np.ndarray
        Target vectors (vocab_size, vector_size)
    vocab : List[str]
        Vocabulary list
    test_words : Optional[List[str]]
        List of words to test. If None, will auto-select common words.
    k : int
        Number of nearest neighbors to show per word
    """
    print("\n" + "="*70)
    print("SANITY CHECK: Target Vector Quality")
    print("="*70)
    print("\nShowing nearest neighbors to verify semantic coherence...")

    # If no test words provided, try to auto-select common ones
    if test_words is None:
        # Common Dutch words to test (if they exist in vocab)
        candidate_words = [
            'koning', 'koningin', 'man', 'vrouw', 'jongen', 'meisje',
            'stad', 'land', 'water', 'huis', 'groot', 'klein',
            'goed', 'slecht', 'dag', 'nacht', 'zomer', 'winter',
            'rood', 'blauw', 'een', 'twee'
        ]
        test_words = [w for w in candidate_words if w in vocab][:10]

        if len(test_words) < 5:
            # If very few Dutch words found, just pick first 10 from vocab
            print("(Auto-selecting words from vocabulary)")
            test_words = vocab[:10]

    print(f"\nTesting {len(test_words)} words:\n")

    # Find and display neighbors for each test word
    for i, word in enumerate(test_words, 1):
        neighbors = find_nearest_neighbors(word, target_vectors, vocab, k=k)

        if not neighbors:
            print(f"{i}. '{word}' - NOT FOUND IN VOCABULARY")
            continue

        print(f"{i}. '{word}'")
        print(f"   Nearest neighbors:")

        for j, (neighbor, similarity) in enumerate(neighbors, 1):
            print(f"      {j:2d}. {neighbor:20s} (similarity: {similarity:.4f})")

        print()

    print("="*70)
    print("If neighbors are semantically coherent, vectors are good!")
    print("="*70)


def compute_pmi_approximation(
    target_vectors: np.ndarray,
    context_vectors: np.ndarray,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute PMI approximation as W·C^T in batches to save memory.

    According to Levy & Goldberg (2014), for Word2Vec with negative sampling:
        W·C^T ≈ PMI - log(k)
    where k is the number of negative samples.

    Instead of computing the full matrix multiplication at once (which would be
    too memory-intensive for large vocabularies), we compute it row-by-row in
    batches using matrix-vector multiplications.

    Parameters
    ----------
    target_vectors : np.ndarray
        Target vectors (vocab_size, vector_size)
    context_vectors : np.ndarray
        Context vectors (vocab_size, vector_size)
    batch_size : int
        Number of rows to process at once (default: 1000)

    Returns
    -------
    pmi_approx : np.ndarray
        Approximation of PMI matrix (vocab_size, vocab_size)
    """
    print("\n" + "="*70)
    print("COMPUTING PMI APPROXIMATION")
    print("="*70)

    vocab_size = target_vectors.shape[0]
    vector_dim = target_vectors.shape[1]

    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Vector dimension: {vector_dim}")
    print(f"Full matrix size: {vocab_size:,} × {vocab_size:,}")

    # Estimate memory requirement
    memory_gb = (vocab_size * vocab_size * 4) / (1024**3)  # 4 bytes per float32
    print(f"Estimated memory: {memory_gb:.2f} GB")

    if memory_gb > 10:
        print(f"⚠️  Warning: This will require {memory_gb:.2f} GB of RAM!")
        print(f"Computing in batches of {batch_size} rows to reduce memory usage...")
    else:
        print(f"Computing in batches of {batch_size} rows...")

    # Pre-allocate result matrix
    pmi_approx = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    # Compute in batches
    num_batches = (vocab_size + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, vocab_size)

        # Get batch of target vectors
        target_batch = target_vectors[start_idx:end_idx]

        # Compute this batch: target_batch @ context_vectors.T
        # Shape: (batch_size, vector_dim) @ (vector_dim, vocab_size) = (batch_size, vocab_size)
        pmi_approx[start_idx:end_idx, :] = target_batch @ context_vectors.T

        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            progress = ((i + 1) / num_batches) * 100
            print(f"\rProgress: {progress:.1f}% ({i+1}/{num_batches} batches)", end='')

    print()  # New line after progress
    print(f"✓ PMI approximation computed")
    print(f"  Shape: {pmi_approx.shape}")
    print(f"  Min value: {pmi_approx.min():.2f}")
    print(f"  Max value: {pmi_approx.max():.2f}")
    print(f"  Mean value: {pmi_approx.mean():.2f}")

    return pmi_approx


def extract_and_save_attributes(
    pmi_matrix: np.ndarray,
    vocab: List[str],
    d: Optional[int] = None,
    threshold_factor: float = 0.1,
    auto_detect_threshold: float = 10.0,
    output_dir: str = "./output"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract binary semantic attributes and save results.

    Parameters
    ----------
    pmi_matrix : np.ndarray
        PMI matrix (vocab_size, vocab_size)
    vocab : List[str]
        Vocabulary list
    d : Optional[int]
        Number of attributes to extract (None = auto-detect)
    threshold_factor : float
        Binarization threshold factor
    auto_detect_threshold : float
        Threshold for auto-detection
    output_dir : str
        Directory to save outputs

    Returns
    -------
    attributes : np.ndarray
        Binary attribute matrix (vocab_size, d)
    eigenvalues : np.ndarray
        Top eigenvalues
    eigenvectors : np.ndarray
        Top eigenvectors
    """
    print("\n" + "="*70)
    print("EXTRACTING SEMANTIC ATTRIBUTES")
    print("="*70)

    attributes, eigenvals, eigenvecs = extract_attributes(
        pmi_matrix,
        vocab,
        d=d,
        threshold_factor=threshold_factor,
        auto_detect_threshold=auto_detect_threshold
    )

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    np.save(output_path / "attributes.npy", attributes)
    print(f"✓ Saved attributes to {output_path / 'attributes.npy'}")
    print(f"  Shape: {attributes.shape}")

    np.save(output_path / "eigenvalues.npy", eigenvals)
    print(f"✓ Saved eigenvalues to {output_path / 'eigenvalues.npy'}")

    np.save(output_path / "eigenvectors.npy", eigenvecs)
    print(f"✓ Saved eigenvectors to {output_path / 'eigenvectors.npy'}")

    return attributes, eigenvals, eigenvecs


def main(
    target_vectors_path: str = "./output/target_vectors.npy",
    context_vectors_path: str = "./output/context_vectors.npy",
    vocab_path: str = "./output/vocab.txt",
    counts_path: str = "./output/word_counts.npy",
    model_path: Optional[str] = None,
    output_dir: str = "./output",
    d: Optional[int] = None,
    threshold_factor: float = 0.1,
    show_interpretation: bool = True,
    plot_spectrum: bool = False,
    skip_sanity_check: bool = False,
    test_words: Optional[List[str]] = None,
    batch_size: int = 1000,
    min_count: Optional[int] = None
):
    """
    Main pipeline: Load vectors → Filter → Sanity check → Compute PMI → Extract attributes

    Parameters
    ----------
    target_vectors_path : str
        Path to target vectors file (ignored if model_path is provided)
    context_vectors_path : str
        Path to context vectors file (ignored if model_path is provided)
    vocab_path : str
        Path to vocabulary file (ignored if model_path is provided)
    counts_path : str
        Path to word counts file (ignored if model_path is provided)
    model_path : Optional[str]
        Path to Word2Vec model file. If provided, loads from model instead of .npy files.
    output_dir : str
        Directory for output files
    d : Optional[int]
        Number of attributes (None = auto-detect)
    threshold_factor : float
        Binarization threshold
    show_interpretation : bool
        Whether to print attribute interpretation
    plot_spectrum : bool
        Whether to plot eigenvalue spectrum
    skip_sanity_check : bool
        Whether to skip the vector quality sanity check
    test_words : Optional[List[str]]
        Words to use for sanity check (None = auto-select)
    batch_size : int
        Batch size for PMI computation (default: 1000)
    min_count : Optional[int]
        Minimum word frequency (None = use all words). Setting this to e.g. 50
        will only process words with count >= 50, drastically reducing memory.
    """
    print("="*70)
    print("PMI MATRIX & SEMANTIC ATTRIBUTES EXTRACTION PIPELINE")
    print("="*70)

    # Step 0: Load required words from CSV if min_count filtering is active
    required_words = None
    if min_count is not None and min_count > 0:
        csv_path = "./online_testing_all.csv"
        if Path(csv_path).exists():
            print(f"\nLoading required words from {csv_path}...")
            required_words = extract_words_from_csv(csv_path, column_name="test_sentence")
        else:
            print(f"\n⚠️  CSV file not found at {csv_path}, proceeding without required words")

    # Step 1: Load vectors (from model file or separate .npy files)
    if model_path is not None:
        target_vectors, context_vectors, vocab, word_counts = load_from_model(
            model_path,
            min_count=min_count,
            required_words=required_words
        )
    else:
        target_vectors, context_vectors, vocab, word_counts = load_vectors(
            target_vectors_path,
            context_vectors_path,
            vocab_path,
            counts_path,
            min_count=min_count,
            required_words=required_words
        )

    # Step 2: Sanity check - verify vector quality
    if not skip_sanity_check:
        sanity_check_vectors(target_vectors, vocab, test_words=test_words, k=10)

    # Step 3: Compute PMI approximation
    pmi_matrix = compute_pmi_approximation(target_vectors, context_vectors, batch_size=batch_size)

    # Save PMI matrix and vocabulary
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    np.save(output_path / "pmi_approximation.npy", pmi_matrix)
    print(f"\n✓ Saved PMI matrix to {output_path / 'pmi_approximation.npy'}")

    # Save the vocabulary (filtered, if applicable)
    with open(output_path / "filtered_vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")
    print(f"✓ Saved filtered vocabulary to {output_path / 'filtered_vocab.txt'}")
    print(f"  (This vocabulary corresponds to PMI matrix rows/columns)")

    # Free up memory
    del target_vectors
    del context_vectors

    # Step 4: Extract attributes
    attributes, eigenvals, eigenvecs = extract_and_save_attributes(
        pmi_matrix,
        vocab,
        d=d,
        threshold_factor=threshold_factor,
        output_dir=output_dir
    )

    # Step 5: Interpret results
    if show_interpretation:
        print("\n" + "="*70)
        print("INTERPRETING ATTRIBUTES")
        print("="*70)

        interpret_attributes(
            attributes,
            vocab,
            eigenvalues=eigenvals,
            top_n=20
        )

    # Step 6: Plot eigenvalue spectrum
    if plot_spectrum:
        print("\nPlotting eigenvalue spectrum...")
        # Load all eigenvalues for plotting (not just top d+1)
        from extract_attributes import _auto_detect_d
        all_eigenvals, _ = np.linalg.eigh(pmi_matrix)
        all_eigenvals = all_eigenvals[np.argsort(-np.abs(all_eigenvals))]

        detected_d = _auto_detect_d(all_eigenvals) if d is None else d
        plot_eigenvalue_spectrum(all_eigenvals, d=detected_d)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nExtracted {attributes.shape[1]} semantic attributes")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"\nOutput files in {output_dir}:")
    print("  - pmi_approximation.npy     (PMI matrix)")
    print("  - filtered_vocab.txt        (vocabulary for PMI rows/columns)")
    print("  - attributes.npy            (binary semantic attributes)")
    print("  - eigenvalues.npy           (eigenvalues)")
    print("  - eigenvectors.npy          (eigenvectors)")

    return attributes, eigenvals, vocab


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    show_help = "--help" in sys.argv or "-h" in sys.argv

    if show_help:
        print(__doc__)
        print("\nOptions:")
        print("  --from-model PATH      Load from Word2Vec model file (avoids .npy files)")
        print("  --plot                 Plot eigenvalue spectrum")
        print("  --no-interpret         Skip attribute interpretation")
        print("  --skip-sanity-check    Skip vector quality sanity check")
        print("  --d N                  Extract N attributes (default: auto-detect)")
        print("  --threshold F          Binarization threshold factor (default: 0.1)")
        print("  --test-words W1,W2,... Comma-separated list of words for sanity check")
        print("  --batch-size N         Batch size for PMI computation (default: 1000)")
        print("  --min-count N          Only use words with count >= N (reduces memory)")
        print("  --help, -h             Show this help message")
        print("\nExamples:")
        print("  # Load from model file with frequency filtering")
        print("  python sanity_check_and_extract_attributes.py --from-model ./output/dutch_skipgram_model.bin --min-count 50")
        print()
        print("  # Load from .npy files (default)")
        print("  python sanity_check_and_extract_attributes.py --min-count 50")
        sys.exit(0)

    # Parse options
    plot = "--plot" in sys.argv
    show_interpret = "--no-interpret" not in sys.argv
    skip_sanity = "--skip-sanity-check" in sys.argv

    model_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--from-model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]

    d = None
    for i, arg in enumerate(sys.argv):
        if arg == "--d" and i + 1 < len(sys.argv):
            d = int(sys.argv[i + 1])

    threshold = 0.1
    for i, arg in enumerate(sys.argv):
        if arg == "--threshold" and i + 1 < len(sys.argv):
            threshold = float(sys.argv[i + 1])

    test_words = None
    for i, arg in enumerate(sys.argv):
        if arg == "--test-words" and i + 1 < len(sys.argv):
            test_words = sys.argv[i + 1].split(',')

    batch_size = 1000
    for i, arg in enumerate(sys.argv):
        if arg == "--batch-size" and i + 1 < len(sys.argv):
            batch_size = int(sys.argv[i + 1])

    min_count = None
    for i, arg in enumerate(sys.argv):
        if arg == "--min-count" and i + 1 < len(sys.argv):
            min_count = int(sys.argv[i + 1])

    # Run pipeline
    main(
        model_path=model_path,
        d=d,
        threshold_factor=threshold,
        show_interpretation=show_interpret,
        plot_spectrum=plot,
        skip_sanity_check=skip_sanity,
        test_words=test_words,
        batch_size=batch_size,
        min_count=min_count
    )
