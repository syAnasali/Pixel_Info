import logging
import os
import sys
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import FLICKR_DIR, OUTPUT_DIR
from src.data_processing.explore_dataset import load_dataset, validate_integrity

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "caption_cleaning.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("clean_captions")

logger = setup_logging()

def clean_text(text: str, keep_single_char: list = None) -> str:
    """
    Clean a single caption text:
    1. Lowercase
    2. Remove punctuation
    3. Remove non-alphabetic tokens
    4. Remove single-character words (except those specified in keep_single_char)
    5. Normalize whitespace
    """
    if keep_single_char is None:
        keep_single_char = ["a", "i"]
        
    # Lowercase
    text = str(text).lower()
    
    # Remove punctuation (replace with space to prevent blending words)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split by whitespace
    tokens = text.split()
    
    # Filter tokens
    cleaned_tokens = []
    for token in tokens:
        # Check if alphabetic
        if token.isalpha():
            # Keep if longer than 1 char OR is a configured meaningful single char
            if len(token) > 1 or token in keep_single_char:
                cleaned_tokens.append(token)
                
    # Join tokens and normalize spaces
    cleaned = " ".join(cleaned_tokens).strip()
    return cleaned

def process_captions(df: pd.DataFrame, keep_single_char: list = None) -> pd.DataFrame:
    """
    Apply text cleaning to the captions DataFrame and wrap in startseq / endseq.
    """
    logger.info("Cleaning captions text...")
    df_clean = df.copy()
    
    # Apply raw text cleaning
    df_clean["cleaned_caption_raw"] = df_clean["caption"].apply(
        lambda x: clean_text(x, keep_single_char=keep_single_char)
    )
    
    # Wrap with start/end tokens
    df_clean["cleaned_caption"] = df_clean["cleaned_caption_raw"].apply(
        lambda x: f"startseq {x} endseq"
    )
    
    logger.info("Caption cleaning complete.")
    return df_clean

def analyze_vocabulary(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> dict:
    """
    Compute vocabulary statistics before and after text cleaning.
    """
    logger.info("Analyzing vocabulary statistics...")
    stats = {}
    
    # Vocabulary before cleaning (case-insensitive for comparison, split by spaces)
    raw_tokens = []
    for cap in df_raw["caption"]:
        raw_tokens.extend(str(cap).lower().split())
    raw_vocab = set(raw_tokens)
    stats["vocab_size_before"] = len(raw_vocab)
    
    # Vocabulary after cleaning (includes startseq and endseq)
    cleaned_tokens = []
    for cap in df_cleaned["cleaned_caption"]:
        cleaned_tokens.extend(cap.split())
    cleaned_vocab = set(cleaned_tokens)
    stats["vocab_size_after"] = len(cleaned_vocab)
    
    # Vocabulary size excluding startseq and endseq
    cleaned_tokens_raw = []
    for cap in df_cleaned["cleaned_caption_raw"]:
        cleaned_tokens_raw.extend(cap.split())
    stats["vocab_size_after_raw"] = len(set(cleaned_tokens_raw))
    
    # Top 20 most frequent words (excluding startseq and endseq)
    counter = Counter(cleaned_tokens_raw)
    stats["top_20_words"] = counter.most_common(20)
    
    # Caption length distribution (including startseq and endseq)
    lengths = df_cleaned["cleaned_caption"].apply(lambda x: len(x.split()))
    stats["len_mean"] = float(lengths.mean())
    stats["len_min"] = int(lengths.min())
    stats["len_max"] = int(lengths.max())
    stats["len_median"] = float(lengths.median())
    stats["len_p25"] = float(np.percentile(lengths, 25))
    stats["len_p75"] = float(np.percentile(lengths, 75))
    stats["len_p90"] = float(np.percentile(lengths, 90))
    stats["len_p95"] = float(np.percentile(lengths, 95))
    
    return stats

def save_outputs(df: pd.DataFrame, stats: dict):
    """
    Save cleaned captions to CSV and report metrics to a text file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_path = OUTPUT_DIR / "cleaned_captions.csv"
    report_path = OUTPUT_DIR / "cleaning_report.txt"
    
    # Save cleaned_captions.csv
    try:
        # Save only necessary columns
        df_out = df[["image", "cleaned_caption"]]
        df_out.to_csv(csv_path, index=False)
        logger.info(f"Saved cleaned captions to: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        
    # Save cleaning_report.txt
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Flickr8K Caption Cleaning Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Vocabulary Statistics ---\n")
            f.write(f"Vocabulary Size (Before Cleaning) : {stats['vocab_size_before']}\n")
            f.write(f"Vocabulary Size (After Cleaning)  : {stats['vocab_size_after']} (includes startseq/endseq)\n")
            f.write(f"Vocabulary Size (Tokens Only)     : {stats['vocab_size_after_raw']}\n\n")
            
            f.write("--- Caption Length Distribution (including startseq/endseq) ---\n")
            f.write(f"Average Length : {stats['len_mean']:.2f} words\n")
            f.write(f"Median Length  : {stats['len_median']:.2f} words\n")
            f.write(f"Min Length     : {stats['len_min']} words\n")
            f.write(f"Max Length     : {stats['len_max']} words\n")
            f.write(f"Percentile 25  : {stats['len_p25']:.1f} words\n")
            f.write(f"Percentile 75  : {stats['len_p75']:.1f} words\n")
            f.write(f"Percentile 90  : {stats['len_p90']:.1f} words\n")
            f.write(f"Percentile 95  : {stats['len_p95']:.1f} words\n\n")
            
            f.write("--- Top 20 Most Frequent Words (excluding startseq/endseq) ---\n")
            f.write(f"{'Word':<15} | {'Frequency':<10}\n")
            f.write("-" * 30 + "\n")
            for word, freq in stats["top_20_words"]:
                f.write(f"{word:<15} | {freq:<10}\n")
                
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Saved cleaning report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaning report: {e}")

def main():
    logger.info("Initializing Caption Cleaning Preprocessor...")
    try:
        # Load dataset
        df = load_dataset(FLICKR_DIR)
        
        # Drop duplicates/missing values as pre-validation
        _, df_clean_init = validate_integrity(df)
        
        # Process and clean captions (meaningful single-char words configured as ["a", "i"])
        df_cleaned = process_captions(df_clean_init, keep_single_char=["a", "i"])
        
        # Analyze vocabulary and lengths
        stats = analyze_vocabulary(df_clean_init, df_cleaned)
        
        # Save output documents
        save_outputs(df_cleaned, stats)
        
        logger.info("Caption cleaning task finished successfully.")
        
    except FileNotFoundError as fnf:
        logger.error(f"Data directory issue: {fnf}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Preprocessors crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
