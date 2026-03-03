"""
Download and prepare Dutch Wikipedia for word embedding training

This script:
1. Downloads Dutch Wikipedia dump
2. Extracts and cleans the text
3. Tokenizes sentences
4. Saves in format ready for train_skipgram.py
"""

import re
import pickle
from pathlib import Path
from typing import List, Iterator
import multiprocessing as mp


def download_wikipedia():
    """
    Download Dutch Wikipedia using gensim's built-in downloader.

    Returns
    -------
    wiki_corpus : WikiCorpus
        Gensim Wikipedia corpus object
    """
    try:
        from gensim.corpora import WikiCorpus
        import urllib.request

        print("="*70)
        print("DOWNLOADING DUTCH WIKIPEDIA")
        print("="*70)

        # Download latest Dutch Wikipedia dump
        wiki_dump_url = "https://dumps.wikimedia.org/nlwiki/latest/nlwiki-latest-pages-articles.xml.bz2"
        wiki_dump_file = "./data/nlwiki-latest-pages-articles.xml.bz2"

        Path("./data").mkdir(exist_ok=True)

        if Path(wiki_dump_file).exists():
            print(f"✓ Wikipedia dump already exists at {wiki_dump_file}")
        else:
            print(f"Downloading from {wiki_dump_url}")
            print("This may take a while (file is ~600MB compressed)...")

            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                print(f"\rProgress: {percent:.1f}%", end='')

            urllib.request.urlretrieve(wiki_dump_url, wiki_dump_file, download_progress)
            print("\n✓ Download complete!")

        # Load the corpus
        print("\nLoading Wikipedia corpus...")
        wiki_corpus = WikiCorpus(wiki_dump_file, dictionary={})
        print("✓ Corpus loaded!")

        return wiki_corpus, wiki_dump_file

    except ImportError as e:
        print(e)
        return None, None


def simple_tokenizer(text: str) -> List[str]:
    """
    Simple tokenizer for Dutch text.

    Parameters
    ----------
    text : str
        Raw text

    Returns
    -------
    tokens : List[str]
        List of tokens (words)
    """
    # Lowercase
    text = text.lower()

    # Remove special characters, keep letters, numbers, and spaces
    text = re.sub(r'[^a-zàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ0-9\s-]', ' ', text)

    # Split on whitespace
    tokens = text.split()

    # Filter out very short tokens and numbers-only
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]

    return tokens


def extract_sentences_from_wiki(wiki_corpus, max_articles=None) -> Iterator[tuple]:
    """
    Extract tokenized sentences from Wikipedia corpus.

    Parameters
    ----------
    wiki_corpus : WikiCorpus
        Gensim Wikipedia corpus
    max_articles : int, optional
        Maximum number of articles to process (for testing)

    Yields
    ------
    article_num : int
        Article number (0-indexed)
    sentence : List[str]
        Tokenized sentence
    """
    print("\n" + "="*70)
    print("EXTRACTING SENTENCES")
    print("="*70)

    article_count = 0
    sentence_count = 0

    for article_tokens in wiki_corpus.get_texts():
        # article_tokens is already tokenized by gensim (list of byte strings)
        # Punctuation is already removed, so we need to split differently

        # Convert bytes to strings
        tokens = [token.decode('utf-8') if isinstance(token, bytes) else token
                 for token in article_tokens]

        if len(tokens) < 50:
            article_count += 1
            continue
            # skip short articles

        # Split article into sentences (chunks of ~15-30 tokens as heuristic)
        # Since punctuation is gone, we use fixed-size sliding windows
        sentence_length = 20  # Average sentence length in tokens

        for i in range(0, len(tokens), sentence_length):
            sentence_tokens = tokens[i:i+sentence_length]

            # Filter and clean tokens
            clean_tokens = []
            for token in sentence_tokens:
                token = token.lower()
                # Keep only tokens with letters
                if len(token) > 1 and any(c.isalpha() for c in token):
                    clean_tokens.append(token)

            if len(clean_tokens) >= 3:  # At least 3 words
                yield (article_count, clean_tokens)
                sentence_count += 1

        article_count += 1

        if article_count % 1000 == 0:
            print(f"\rProcessed {article_count:,} articles, {sentence_count:,} sentences", end='')

        if max_articles and article_count >= max_articles:
            break

    print(f"\n✓ Extraction complete!")
    print(f"  Total articles: {article_count:,}")
    print(f"  Total sentences: {sentence_count:,}")


def save_corpus(sentences: List[List[str]], output_file: str = "./data/dutch_wiki_sentences.pkl"):
    """
    Save tokenized sentences to disk.

    Parameters
    ----------
    sentences : List[List[str]]
        List of tokenized sentences
    output_file : str
        Output file path
    """
    print("\n" + "="*70)
    print("SAVING CORPUS")
    print("="*70)

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(sentences, f)

    print(f"✓ Saved {len(sentences):,} sentences to {output_file}")

    # Print file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


def save_corpus_split(sentences_part1: List[List[str]],
                      sentences_part2: List[List[str]],
                      output_prefix: str = "./data/dutch_wiki_sentences"):
    """
    Save tokenized sentences to disk in two parts.

    Parameters
    ----------
    sentences_part1 : List[List[str]]
        List of tokenized sentences from articles 0-499,999
    sentences_part2 : List[List[str]]
        List of tokenized sentences from articles 500,000+
    output_prefix : str
        Output file path prefix (will add _part1.pkl and _part2.pkl)
    """
    print("\n" + "="*70)
    print("SAVING CORPUS (SPLIT INTO TWO FILES)")
    print("="*70)

    output_path = Path(output_prefix)
    output_path.parent.mkdir(exist_ok=True)

    # Save part 1 (articles 0-499,999)
    part1_file = f"{output_prefix}_part1.pkl"
    with open(part1_file, 'wb') as f:
        pickle.dump(sentences_part1, f)

    size1_mb = Path(part1_file).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(sentences_part1):,} sentences (articles 0-499,999) to {part1_file}")
    print(f"  File size: {size1_mb:.1f} MB")

    # Save part 2 (articles 500,000+)
    part2_file = f"{output_prefix}_part2.pkl"
    with open(part2_file, 'wb') as f:
        pickle.dump(sentences_part2, f)

    size2_mb = Path(part2_file).stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(sentences_part2):,} sentences (articles 500,000+) to {part2_file}")
    print(f"  File size: {size2_mb:.1f} MB")

    print(f"\n  Total sentences: {len(sentences_part1) + len(sentences_part2):,}")
    print(f"  Total size: {size1_mb + size2_mb:.1f} MB")


def load_corpus_part(part_num: int, corpus_prefix: str = "./data/dutch_wiki_sentences") -> List[List[str]]:
    """
    Load a specific part of the split corpus.

    Parameters
    ----------
    part_num : int
        Which part to load (1 or 2)
    corpus_prefix : str
        Prefix for the corpus files

    Returns
    -------
    sentences : List[List[str]]
        List of tokenized sentences from that part
    """
    part_file = f"{corpus_prefix}_part{part_num}.pkl"

    if not Path(part_file).exists():
        raise FileNotFoundError(f"Corpus part {part_num} not found at {part_file}")

    print(f"Loading corpus part {part_num} from {part_file}...")
    with open(part_file, 'rb') as f:
        sentences = pickle.load(f)
    print(f"✓ Loaded {len(sentences):,} sentences from part {part_num}")
    return sentences


def load_corpus(corpus_file: str = "./data/dutch_wiki_sentences.pkl") -> List[List[str]]:
    """
    Load previously saved corpus.

    Parameters
    ----------
    corpus_file : str
        Path to saved corpus file, or prefix for split files

    Returns
    -------
    sentences : List[List[str]]
        List of tokenized sentences
    """
    # Check if split files exist
    if corpus_file.endswith('.pkl'):
        corpus_prefix = corpus_file[:-4]
    else:
        corpus_prefix = corpus_file

    part1_file = f"{corpus_prefix}_part1.pkl"
    part2_file = f"{corpus_prefix}_part2.pkl"

    if Path(part1_file).exists() and Path(part2_file).exists():
        print(f"Loading split corpus from {part1_file} and {part2_file}...")
        with open(part1_file, 'rb') as f:
            sentences_part1 = pickle.load(f)
        print(f"✓ Loaded part 1: {len(sentences_part1):,} sentences")

        with open(part2_file, 'rb') as f:
            sentences_part2 = pickle.load(f)
        print(f"✓ Loaded part 2: {len(sentences_part2):,} sentences")

        sentences = sentences_part1 + sentences_part2
        print(f"✓ Total: {len(sentences):,} sentences")
        return sentences
    else:
        # Load single file
        print(f"Loading corpus from {corpus_file}...")
        with open(corpus_file, 'rb') as f:
            sentences = pickle.load(f)
        print(f"✓ Loaded {len(sentences):,} sentences")
        return sentences


def show_corpus_stats(sentences: List[List[str]]):
    """
    Display statistics about the corpus.

    Parameters
    ----------
    sentences : List[List[str]]
        List of tokenized sentences
    """
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)

    num_sentences = len(sentences)

    # Check data structure and flatten if needed
    if num_sentences > 0:
        first_item = sentences[0]
        # If first item is a tuple (article_num, sentence), extract sentences
        if isinstance(first_item, tuple):
            print("  (Detected tuple format, extracting sentences...)")
            sentences = [sent for _, sent in sentences]
            num_sentences = len(sentences)

    num_tokens = sum(len(s) for s in sentences)

    # Vocabulary
    vocab = set()
    for sent in sentences:
        # Handle both list and tuple formats
        if isinstance(sent, (list, tuple)):
            # Make sure items in sent are strings, not lists
            for item in sent:
                if isinstance(item, str):
                    vocab.add(item)
                elif isinstance(item, (list, tuple)):
                    # Nested list - flatten it
                    for subitem in item:
                        if isinstance(subitem, str):
                            vocab.add(subitem)
        else:
            print(f"  Warning: unexpected item type: {type(sent)}")

    # Average sentence length
    avg_length = num_tokens / num_sentences if num_sentences > 0 else 0

    print(f"  Number of sentences: {num_sentences:,}")
    print(f"  Total tokens: {num_tokens:,}")
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Average sentence length: {avg_length:.1f} tokens")

    # Show example sentences
    print("\n  Example sentences:")
    for i, sent in enumerate(sentences[:3], 1):
        if isinstance(sent, (list, tuple)):
            print(f"    {i}. {' '.join(sent[:15])}{'...' if len(sent) > 15 else ''}")
        else:
            print(f"    {i}. (unexpected format: {type(sent)})")


def main(max_articles=None, force_download=False, split_at=500000):
    """
    Main pipeline to prepare Dutch Wikipedia.

    Parameters
    ----------
    max_articles : int, optional
        Limit number of articles (for testing). None = process all.
    force_download : bool
        Force re-download and re-processing even if corpus exists
    split_at : int
        Article number to split corpus at (default: 500000)
    """
    corpus_prefix = "./data/dutch_wiki_sentences"

    # Check if corpus already exists
    part1_file = f"{corpus_prefix}_part1.pkl"
    part2_file = f"{corpus_prefix}_part2.pkl"
    old_corpus_file = "./data/dutch_wiki_sentences.pkl"

    if (Path(part1_file).exists() and Path(part2_file).exists()) and not force_download:
        print("="*70)
        print("CORPUS ALREADY EXISTS")
        print("="*70)
        print(f"Found existing split corpus at:")
        print(f"  {part1_file}")
        print(f"  {part2_file}")
        print("Loading...")

        sentences = load_corpus(corpus_prefix)
        show_corpus_stats(sentences)

        print("\nTo re-download and re-process, run:")
        print("  python prepare_dutch_wikipedia.py --force")
        return sentences
    elif Path(old_corpus_file).exists() and not force_download:
        print("="*70)
        print("CORPUS ALREADY EXISTS (OLD FORMAT)")
        print("="*70)
        print(f"Found existing corpus at {old_corpus_file}")
        print("Loading...")

        sentences = load_corpus(old_corpus_file)
        show_corpus_stats(sentences)

        print("\nTo re-download and re-process with split format, run:")
        print("  python prepare_dutch_wikipedia.py --force")
        return sentences

    # Download Wikipedia
    wiki_corpus, wiki_file = download_wikipedia()

    if wiki_corpus is None:
        print("Failed to download Wikipedia corpus")
        return None

    # Extract sentences
    print("\nExtracting and tokenizing sentences...")
    if max_articles:
        print(f"(Limited to first {max_articles:,} articles for testing)")
    print(f"Will split corpus at article {split_at:,}")

    # Process sentences and save incrementally to avoid memory issues
    sentences_part1 = []
    sentences_part2 = []
    part1_saved = False
    part1_count = 0
    size1_mb = 0
    part1_file = f"{corpus_prefix}_part1.pkl"
    part2_file = f"{corpus_prefix}_part2.pkl"

    for article_num, sentence in extract_sentences_from_wiki(wiki_corpus, max_articles=max_articles):
        if article_num < split_at:
            sentences_part1.append(sentence)
        else:
            # When we hit the split point, save part1 and clear it from memory
            if not part1_saved:
                print(f"\n\nReached article {split_at:,}, saving part 1...")
                Path(part1_file).parent.mkdir(exist_ok=True)
                with open(part1_file, 'wb') as f:
                    pickle.dump(sentences_part1, f)

                size1_mb = Path(part1_file).stat().st_size / (1024 * 1024)
                print(f"✓ Saved {len(sentences_part1):,} sentences (articles 0-{split_at-1:,}) to {part1_file}")
                print(f"  File size: {size1_mb:.1f} MB")

                # Get count before clearing
                part1_count = len(sentences_part1)

                # Clear part1 from memory
                sentences_part1 = []
                part1_saved = True
                print("✓ Cleared part 1 from memory, continuing with part 2...\n")

            sentences_part2.append(sentence)

    # Save part 2 (if we have any sentences in it)
    size2_mb = 0
    if sentences_part2:
        print("\n" + "="*70)
        print("SAVING PART 2")
        print("="*70)
        with open(part2_file, 'wb') as f:
            pickle.dump(sentences_part2, f)

        size2_mb = Path(part2_file).stat().st_size / (1024 * 1024)
        print(f"✓ Saved {len(sentences_part2):,} sentences (articles {split_at:,}+) to {part2_file}")
        print(f"  File size: {size2_mb:.1f} MB")

    # If part1 wasn't saved (all articles < split_at), save it now
    if not part1_saved and sentences_part1:
        print("\n" + "="*70)
        print("SAVING PART 1")
        print("="*70)
        Path(part1_file).parent.mkdir(exist_ok=True)
        with open(part1_file, 'wb') as f:
            pickle.dump(sentences_part1, f)

        size1_mb = Path(part1_file).stat().st_size / (1024 * 1024)
        print(f"✓ Saved {len(sentences_part1):,} sentences to {part1_file}")
        print(f"  File size: {size1_mb:.1f} MB")
        part1_count = len(sentences_part1)

    # Show stats by loading only one part at a time
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)

    part2_count = len(sentences_part2)
    total_sentences = part1_count + part2_count
    total_size = size1_mb + size2_mb

    print(f"  Total sentences: {total_sentences:,}")
    print(f"  Total size: {total_size:.1f} MB")
    print(f"  Part 1 sentences: {part1_count:,}")
    print(f"  Part 2 sentences: {part2_count:,}")

    # Clear part2 from memory
    sentences_part2 = []

    print("\n" + "="*70)
    print("PREPARATION COMPLETE!")
    print("="*70)
    print("\nNext step: Use this corpus with train_skipgram.py")
    print("\nExample 1 - Load all sentences (requires more memory):")
    print("  from prepare_dutch_wikipedia import load_corpus")
    print("  sentences = load_corpus()")
    print("  # Then pass 'sentences' to train_skipgram()")
    print("\nExample 2 - Load one part at a time (memory efficient):")
    print("  from prepare_dutch_wikipedia import load_corpus_part")
    print("  sentences_part1 = load_corpus_part(1)")
    print("  # Process part 1...")
    print("  sentences_part2 = load_corpus_part(2)")
    print("  # Process part 2...")
    print("\nNote: Corpus is split into two files for memory efficiency:")
    if part1_count > 0:
        print(f"  - Part 1: articles 0-{split_at-1:,} ({part1_file})")
    if part2_count > 0:
        print(f"  - Part 2: articles {split_at:,}+ ({part2_file})")

    return None


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    force = "--force" in sys.argv

    # For testing, you can limit the number of articles
    # Remove or set to None to process entire Wikipedia
    test_mode = "--test" in sys.argv
    max_articles = 100 if test_mode else None

    if test_mode:
        print("="*70)
        print("RUNNING IN TEST MODE")
        print("="*70)
        print("Processing only first 1,000 articles")
        print("Remove --test flag to process entire Wikipedia")
        print("="*70 + "\n")

    main(max_articles=max_articles, force_download=force)
