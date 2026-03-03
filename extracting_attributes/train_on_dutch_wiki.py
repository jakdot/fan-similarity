"""
Train Skip-gram on Dutch Wikipedia

This script:
1. Loads Dutch Wikipedia corpus (prepared by prepare_dutch_wikipedia.py)
2. Trains Skip-gram model
3. Extracts and saves target and context vectors

Usage:
    Standard mode (loads all data at once):
        python train_on_dutch_wiki.py

    Incremental mode (memory efficient, trains on each part separately):
        python train_on_dutch_wiki.py --incremental

The script automatically detects split corpus files (part1 and part2) created by
prepare_dutch_wikipedia.py. In incremental mode, it builds vocabulary from both
parts but trains on each part separately to reduce memory usage.

Next step:
    After running this script, use sanity_check_and_extract_attributes.py to
    verify vector quality and extract semantic attributes from the saved vectors.
"""

from prepare_dutch_wikipedia import load_corpus, load_corpus_part, main as prepare_corpus
from train_skipgram import train_skipgram, extract_vectors, save_vectors
import numpy as np
from pathlib import Path


def main(incremental_training=False):
    """Training pipeline: corpus → skip-gram → vectors

    Parameters
    ----------
    incremental_training : bool
        If True, train on each corpus part separately to save memory.
        If False, load all parts and train together (may use more memory but faster).
    """

    print("="*70)
    print("DUTCH WIKIPEDIA SKIP-GRAM TRAINING PIPELINE")
    print("="*70)

    if incremental_training:
        print("Mode: INCREMENTAL TRAINING (memory efficient)")
    else:
        print("Mode: STANDARD TRAINING (loads all data)")

    # Step 1: Load or prepare corpus
    corpus_prefix = "./data/dutch_wiki_sentences"
    corpus_file = "./data/dutch_wiki_sentences.pkl"
    part1_file = f"{corpus_prefix}_part1.pkl"
    part2_file = f"{corpus_prefix}_part2.pkl"

    # Check if split files exist
    sentences = None
    use_split_files = False

    if Path(part1_file).exists() and Path(part2_file).exists():
        use_split_files = True
        print("\n✓ Found split corpus files.")

        if not incremental_training:
            # Load and combine all parts
            print("Loading in parts for memory efficiency...")

            # Load part 1
            print("\nLoading part 1...")
            sentences_part1 = load_corpus_part(1, corpus_prefix)

            # Load part 2
            print("\nLoading part 2...")
            sentences_part2 = load_corpus_part(2, corpus_prefix)

            # Combine for training
            print("\nCombining corpus parts...")
            sentences = sentences_part1 + sentences_part2

            # Clear individual parts to save memory
            del sentences_part1
            del sentences_part2

            print(f"✓ Total sentences loaded: {len(sentences):,}")
        else:
            print("Will train incrementally on each part to save memory.")

    elif Path(corpus_file).exists():
        print("\n✓ Loading existing corpus (old single-file format)...")
        sentences = load_corpus(corpus_file)

    else:
        print("\nCorpus not found. Preparing Dutch Wikipedia...")
        print("This will download ~600MB and may take 30-60 minutes...")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Run prepare_dutch_wikipedia.py separately first.")
            return

        prepare_corpus(max_articles=None)  # Process all articles
        use_split_files = True

        if not incremental_training:
            # After preparation, load the split files
            print("\nLoading newly prepared corpus...")
            sentences_part1 = load_corpus_part(1, corpus_prefix)
            sentences_part2 = load_corpus_part(2, corpus_prefix)
            sentences = sentences_part1 + sentences_part2
            del sentences_part1
            del sentences_part2

    # Step 2: Train Skip-gram model
    print("\n" + "="*70)
    print("TRAINING SKIP-GRAM MODEL")
    print("="*70)

    if incremental_training and use_split_files:
        # Incremental training: build vocab from both parts, train on each separately
        from gensim.models import Word2Vec

        print("Building vocabulary from all corpus parts...")

        # Load part 1 for vocab building
        print("\nLoading part 1...")
        sentences_part1 = load_corpus_part(1, corpus_prefix)

        # Initialize model with part 1
        model = Word2Vec(
            sentences=sentences_part1,
            vector_size=100,
            window=5,
            min_count=10,
            workers=4,
            sg=1,
            negative=5,
            epochs=0,  # Don't train yet, just build vocab
            seed=42
        )

        # Update vocab with part 2
        print("\nLoading part 2 to update vocabulary...")
        sentences_part2 = load_corpus_part(2, corpus_prefix)
        model.build_vocab(sentences_part2, update=True)
        print(f"✓ Vocabulary built: {len(model.wv)} words")

        # Train on part 1
        print("\nTraining on part 1...")
        model.train(sentences_part1, total_examples=len(sentences_part1), epochs=5)
        print("✓ Part 1 training complete")

        # Clear part 1 from memory
        del sentences_part1

        # Train on part 2
        print("\nTraining on part 2...")
        model.train(sentences_part2, total_examples=len(sentences_part2), epochs=5)
        print("✓ Part 2 training complete")

        # Clear part 2 from memory
        del sentences_part2

        print(f"✓ Incremental training complete!")
        print(f"  Vocabulary size: {len(model.wv)}")

    else:
        # Standard training: train on all sentences at once
        model = train_skipgram(
            sentences=sentences,
            vector_size=100,      # Embedding dimension
            window=5,             # Context window
            min_count=10,         # Ignore rare words (< 10 occurrences)
            negative=5,           # Number of negative samples
            epochs=5              # Training epochs
        )

    # Step 3: Extract and save vectors
    print("\n" + "="*70)
    print("EXTRACTING VECTORS")
    print("="*70)

    target_vectors, context_vectors, vocab, word_counts = extract_vectors(model)
    save_vectors(target_vectors, context_vectors, vocab, word_counts, output_dir="./output")

    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    model.save("./output/dutch_skipgram_model.bin")
    print("✓ Saved full model to ./output/dutch_skipgram_model.bin")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nOutputs saved to ./output/:")
    print("  - dutch_skipgram_model.bin")
    print("  - target_vectors.npy")
    print("  - context_vectors.npy")
    print("  - word_counts.npy")
    print("  - vocab.txt")
    print("\n" + "="*70)
    print("NEXT STEP")
    print("="*70)
    print("\nTo verify vectors and extract semantic attributes, run:")
    print("  python sanity_check_and_extract_attributes.py")
    print("\nThis will:")
    print("  1. Load the target and context vectors")
    print("  2. Perform sanity check (show nearest neighbors)")
    print("  3. Compute the PMI approximation matrix")
    print("  4. Extract binary semantic attributes via eigendecomposition")
    print("  5. Save and interpret the results")


if __name__ == "__main__":
    import sys

    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  --incremental   Use incremental training (memory efficient)")
        print("  --help, -h      Show this help message")
        sys.exit(0)

    # Check for command line arguments
    incremental = "--incremental" in sys.argv

    if incremental:
        print("Using incremental training mode (memory efficient)\n")

    main(incremental_training=incremental)
