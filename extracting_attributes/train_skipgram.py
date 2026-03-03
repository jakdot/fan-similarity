"""
Train Skip-gram model and extract both target and context vectors

This script trains a Word2Vec Skip-gram model and saves:
1. Target vectors (W) - the input embeddings
2. Context vectors (C) - the output embeddings

These can be used to compute W·C^T ≈ PMI (pointwise mutual information)
"""

import numpy as np
import logging
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from pathlib import Path

# Set up logging for gensim to show training progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_skipgram(sentences, vector_size=100, window=5, min_count=1,
                   negative=5, epochs=5, sg=1):
    """
    Train Skip-gram model with specified parameters.

    Parameters
    ----------
    sentences : iterable of list of str
        Corpus as list of tokenized sentences
    vector_size : int
        Dimensionality of word vectors
    window : int
        Context window size
    min_count : int
        Ignore words with frequency below this
    negative : int
        Number of negative samples (for negative sampling)
    epochs : int
        Number of training epochs
    sg : int
        1 for skip-gram, 0 for CBOW

    Returns
    -------
    model : Word2Vec
        Trained model
    """
    print("Training Skip-gram model...")
    print(f"  Vector size: {vector_size}")
    print(f"  Window size: {window}")
    print(f"  Min count: {min_count}")
    print(f"  Negative samples: {negative}")
    print(f"  Epochs: {epochs}")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,  # 1 = skip-gram
        negative=negative,  # negative sampling
        epochs=epochs,
        seed=42
    )

    print(f"✓ Training complete!")
    print(f"  Vocabulary size: {len(model.wv)}")

    return model


def extract_vectors(model):
    """
    Extract both target and context vectors from trained model.

    Parameters
    ----------
    model : Word2Vec
        Trained Word2Vec model

    Returns
    -------
    target_vectors : np.ndarray
        Target (input) vectors, shape (vocab_size, vector_size)
    context_vectors : np.ndarray
        Context (output) vectors, shape (vocab_size, vector_size)
    vocab : list of str
        Ordered list of words
    word_counts : np.ndarray
        Word frequency counts, shape (vocab_size,)
    """
    # Target vectors (input embeddings)
    target_vectors = model.wv.vectors
    vocab = model.wv.index_to_key

    # Extract word counts
    word_counts = np.array([model.wv.get_vecattr(word, "count") for word in vocab])

    # Context vectors (output embeddings)
    # These are stored in syn1neg (for negative sampling) or syn1 (for hierarchical softmax)
    if hasattr(model, 'syn1neg'):
        context_vectors = model.syn1neg
        print("✓ Using negative sampling context vectors (syn1neg)")
    elif hasattr(model, 'syn1'):
        context_vectors = model.syn1
        print("✓ Using hierarchical softmax context vectors (syn1)")
    else:
        print("⚠️  Warning: Context vectors not found!")
        print("    Model may not have been trained with negative sampling or hierarchical softmax")
        context_vectors = None

    print(f"\nVector shapes:")
    print(f"  Target vectors: {target_vectors.shape}")
    if context_vectors is not None:
        print(f"  Context vectors: {context_vectors.shape}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Word counts range: {word_counts.min()} - {word_counts.max()}")

    return target_vectors, context_vectors, vocab, word_counts


def compute_pmi_approximation(target_vectors, context_vectors):
    """
    Compute PMI approximation as W·C^T.

    According to Levy & Goldberg (2014), for Word2Vec with negative sampling:
        W·C^T ≈ PMI - log(k)
    where k is the number of negative samples.

    Parameters
    ----------
    target_vectors : np.ndarray
        Target vectors (vocab_size, vector_size)
    context_vectors : np.ndarray
        Context vectors (vocab_size, vector_size)

    Returns
    -------
    pmi_approx : np.ndarray
        Approximation of PMI matrix (vocab_size, vocab_size)
    """
    if context_vectors is None:
        print("⚠️  Cannot compute PMI approximation without context vectors")
        return None

    print("\nComputing PMI approximation (W·C^T)...")
    pmi_approx = target_vectors @ context_vectors.T

    print(f"✓ PMI approximation computed")
    print(f"  Shape: {pmi_approx.shape}")
    print(f"  Min value: {pmi_approx.min():.2f}")
    print(f"  Max value: {pmi_approx.max():.2f}")
    print(f"  Mean value: {pmi_approx.mean():.2f}")

    return pmi_approx


def save_vectors(target_vectors, context_vectors, vocab, word_counts, output_dir="./output"):
    """
    Save vectors, vocabulary, and word counts to files.

    Parameters
    ----------
    target_vectors : np.ndarray
        Target vectors
    context_vectors : np.ndarray
        Context vectors
    vocab : list of str
        Vocabulary
    word_counts : np.ndarray
        Word frequency counts
    output_dir : str
        Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "target_vectors.npy", target_vectors)
    print(f"✓ Saved target vectors to {output_dir / 'target_vectors.npy'}")

    if context_vectors is not None:
        np.save(output_dir / "context_vectors.npy", context_vectors)
        print(f"✓ Saved context vectors to {output_dir / 'context_vectors.npy'}")

    np.save(output_dir / "word_counts.npy", word_counts)
    print(f"✓ Saved word counts to {output_dir / 'word_counts.npy'}")

    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")
    print(f"✓ Saved vocabulary to {output_dir / 'vocab.txt'}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("SKIP-GRAM TRAINING PIPELINE")
    print("="*70)

    # Load corpus (currently using common_texts example)
    # TODO: Replace with your Dutch corpus
    print("\nLoading corpus...")
    sentences = common_texts
    print(f"  Number of sentences: {len(sentences)}")
    print(f"  Example sentence: {sentences[0]}")

    # Train model
    print("\n" + "="*70)
    model = train_skipgram(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        negative=5,
        epochs=5
    )

    # Extract vectors
    print("\n" + "="*70)
    print("EXTRACTING VECTORS")
    print("="*70)
    target_vectors, context_vectors, vocab, word_counts = extract_vectors(model)

    # Save everything
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)
    save_vectors(target_vectors, context_vectors, vocab, word_counts)

    # Compute PMI approximation
    print("\n" + "="*70)
    print("PMI APPROXIMATION")
    print("="*70)
    pmi_approx = compute_pmi_approximation(target_vectors, context_vectors)

    if pmi_approx is not None:
        np.save("./output/pmi_approximation.npy", pmi_approx)
        print(f"✓ Saved PMI approximation to ./output/pmi_approximation.npy")

    # Save the full model too
    model.save("./output/skipgram_model.bin")
    print(f"✓ Saved full model to ./output/skipgram_model.bin")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Replace common_texts with your Dutch corpus")
    print("  2. Use pmi_approximation.npy with extract_attributes.py")
    print("  3. Experiment with hyperparameters (vector_size, window, etc.)")


if __name__ == "__main__":
    main()
