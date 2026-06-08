import logging
import os
import sys
import pickle
from pathlib import Path
from collections import Counter
import pandas as pd

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import OUTPUT_DIR, VOCAB_THRESHOLD

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vocabulary_building.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("build_vocabulary")

logger = setup_logging()

class Vocabulary:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            # Standard special tokens for captioning models
            self.special_tokens = ["<pad>", "startseq", "endseq", "<unk>"]
        else:
            self.special_tokens = special_tokens
            
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_frequencies = Counter()
        self.vocab_size = 0
        
    def build_frequencies(self, captions):
        """Count raw frequency of all words (excluding special sequence boundaries if desired)."""
        logger.info("Counting word frequencies from cleaned captions...")
        for caption in captions:
            words = str(caption).split()
            # Exclude boundary tags from raw frequencies if they are added as special tokens
            filtered_words = [w for w in words if w not in ["startseq", "endseq"]]
            self.word_frequencies.update(filtered_words)
        logger.info(f"Unique words found in captions (excluding boundaries): {len(self.word_frequencies)}")

    def build_vocabulary(self, threshold=1):
        """
        Build bidirectional index maps. 
        Only keeps words with frequency >= threshold.
        Others are implicitly mapped to <unk>.
        """
        logger.info(f"Building vocabulary index mappings (frequency threshold: {threshold})...")
        self.word_to_index = {}
        self.index_to_word = {}
        
        # 1. First add special tokens
        for idx, token in enumerate(self.special_tokens):
            self.word_to_index[token] = idx
            self.index_to_word[idx] = token
            
        # 2. Add words meeting the threshold
        current_idx = len(self.special_tokens)
        
        # Sort words alphabetically/by frequency for deterministic indexing
        sorted_words = sorted(
            [word for word, freq in self.word_frequencies.items() if freq >= threshold]
        )
        
        for word in sorted_words:
            if word not in self.word_to_index:
                self.word_to_index[word] = current_idx
                self.index_to_word[current_idx] = word
                current_idx += 1
                
        self.vocab_size = len(self.word_to_index)
        logger.info(f"Vocabulary successfully built. Total Vocab Size: {self.vocab_size} (including {len(self.special_tokens)} special tokens)")

    def get_rare_word_analysis(self) -> dict:
        """Analyze how many words fall below different thresholds (project pruning impact)."""
        thresholds = [1, 2, 3, 5, 10]
        analysis = {}
        total_unique_words = len(self.word_frequencies)
        
        for t in thresholds:
            kept_count = sum(1 for word, freq in self.word_frequencies.items() if freq >= t)
            pruned_count = total_unique_words - kept_count
            pruned_pct = (pruned_count / total_unique_words * 100) if total_unique_words > 0 else 0
            
            analysis[t] = {
                "words_kept": kept_count,
                "words_pruned": pruned_count,
                "pct_pruned": pruned_pct
            }
            
        return analysis

def save_vocab_artifacts(vocab: Vocabulary, stats: dict, analysis: dict):
    """Save word mappings to pickle and write the vocabulary analysis report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    i2w_path = OUTPUT_DIR / "index_to_word.pkl"
    report_path = OUTPUT_DIR / "vocabulary_report.txt"
    
    # Save Pickles
    try:
        with open(w2i_path, "wb") as f:
            pickle.dump(vocab.word_to_index, f)
        with open(i2w_path, "wb") as f:
            pickle.dump(vocab.index_to_word, f)
        logger.info(f"Saved word_to_index mapping to: {w2i_path}")
        logger.info(f"Saved index_to_word mapping to: {i2w_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle files: {e}")
        
    # Save report
    try:
        # Generate Top 50 table
        top_50 = vocab.word_frequencies.most_common(50)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Flickr8K Vocabulary Statistics Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- General Statistics ---\n")
            f.write(f"Total Unique Words (excluding boundaries)  : {len(vocab.word_frequencies)}\n")
            f.write(f"Active Vocabulary Size (threshold: {vocab.threshold}) : {vocab.vocab_size} (includes 4 special tokens)\n\n")
            
            f.write("--- Rare Word Pruning Projections ---\n")
            f.write(f"{'Threshold':<10} | {'Words Kept':<12} | {'Words Pruned':<14} | {'% Pruned':<10}\n")
            f.write("-" * 55 + "\n")
            for t, data in analysis.items():
                f.write(f"{t:<10} | {data['words_kept']:<12} | {data['words_pruned']:<14} | {data['pct_pruned']:.2f}%\n")
            f.write("\n")
            
            f.write("--- Top 50 Most Frequent Words Table ---\n")
            f.write(f"{'Rank':<5} | {'Word':<20} | {'Frequency':<10}\n")
            f.write("-" * 42 + "\n")
            for rank, (word, freq) in enumerate(top_50, 1):
                f.write(f"{rank:<5} | {word:<20} | {freq:<10}\n")
                
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Saved vocabulary report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write vocabulary report: {e}")

def main():
    logger.info("Starting Vocabulary Builder...")
    
    csv_input = OUTPUT_DIR / "cleaned_captions.csv"
    if not csv_input.exists():
        logger.error(f"Cleaned captions file not found at: {csv_input}. Please run clean_captions.py first.")
        sys.exit(1)
        
    try:
        # Load cleaned captions
        df = pd.read_csv(csv_input)
        if "cleaned_caption" not in df.columns:
            logger.error("Column 'cleaned_caption' not found in cleaned captions file.")
            sys.exit(1)
            
        # Instantiate vocab builder
        # Use config's VOCAB_THRESHOLD. For mock data, we could use threshold=2 to keep some words,
        # but let's adhere to config's VOCAB_THRESHOLD or fall back to 1 if the vocabulary gets empty.
        threshold = VOCAB_THRESHOLD
        
        vocab = Vocabulary()
        vocab.threshold = threshold
        
        # Build frequencies
        vocab.build_frequencies(df["cleaned_caption"])
        
        # Compute projection analysis before actual building
        analysis = vocab.get_rare_word_analysis()
        
        # If threshold is too high for the small mock dataset, adjust dynamically with warning
        kept_with_threshold = sum(1 for w, f in vocab.word_frequencies.items() if f >= threshold)
        if kept_with_threshold == 0 and threshold > 1:
            logger.warning(f"Threshold of {threshold} would result in empty vocab on this dataset. Overriding threshold to 2 for verification.")
            vocab.threshold = 2
            threshold = 2
            
        # Build actual vocabulary mapping
        vocab.build_vocabulary(threshold=threshold)
        
        # Save pickles and report
        save_vocab_artifacts(vocab, {}, analysis)
        
        logger.info("Vocabulary building task finished successfully.")
        
    except Exception as e:
        logger.critical(f"Vocabulary building crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
