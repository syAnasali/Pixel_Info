import logging
import os
import sys
import string
from pathlib import Path
from collections import Counter
import pandas as pd

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import OUTPUT_DIR

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "caption_validation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("validate_cleaned_captions")

logger = setup_logging()

def verify_caption_format(df: pd.DataFrame) -> bool:
    """
    Verify all captions:
    1. Do not contain any punctuation.
    2. Start with 'startseq'.
    3. End with 'endseq'.
    
    Aborts execution by raising ValueError if checks fail.
    """
    logger.info("Verifying caption format integrity...")
    
    # We define punctuation character set to check
    # Excluding spaces, we check for string.punctuation
    punct_chars = set(string.punctuation)
    
    validation_passed = True
    
    for idx, row in df.iterrows():
        caption = str(row["cleaned_caption"])
        image_name = row["image"]
        
        # 1. Verify start/end tags
        if not caption.startswith("startseq"):
            logger.error(f"Format Failure: Caption for image '{image_name}' at row {idx} does not start with 'startseq'.")
            validation_passed = False
            
        if not caption.endswith("endseq"):
            logger.error(f"Format Failure: Caption for image '{image_name}' at row {idx} does not end with 'endseq'.")
            validation_passed = False
            
        # 2. Check for punctuation
        found_puncts = [char for char in caption if char in punct_chars]
        if found_puncts:
            logger.error(
                f"Punctuation Failure: Caption for image '{image_name}' at row {idx} contains punctuation: {list(set(found_puncts))}\n"
                f"  Caption: \"{caption}\""
            )
            validation_passed = False
            
    if not validation_passed:
        logger.critical("[CRITICAL] Cleaned captions validation failed. Aborting pipeline execution.")
        raise ValueError("Cleaned captions contained punctuation or invalid sequence boundary tags.")
        
    logger.info("[SUCCESS] Punctuation and sequence boundaries successfully verified for all captions.")
    return True

def run_validation_pipeline():
    logger.info("Initializing Cleaned Captions Validation Pipeline...")
    
    csv_input = OUTPUT_DIR / "cleaned_captions.csv"
    if not csv_input.exists():
        logger.error(f"Cleaned captions file not found at: {csv_input}. Please run clean_captions.py first.")
        sys.exit(1)
        
    try:
        # 1. Load captions
        df_raw = pd.read_csv(csv_input)
        initial_row_count = len(df_raw)
        logger.info(f"Loaded cleaned captions. Initial row count: {initial_row_count}")
        
        # 2. Drop rows containing missing values
        # Drop rows where 'image' or 'cleaned_caption' is null
        df_clean = df_raw.dropna(subset=["image", "cleaned_caption"]).copy()
        # Drop rows where 'image' or 'cleaned_caption' is empty/whitespace
        df_clean = df_clean[
            (df_clean["image"].astype(str).str.strip() != "") & 
            (df_clean["cleaned_caption"].astype(str).str.strip() != "")
        ].copy()
        
        # 3. Drop exact duplicate rows
        df_clean = df_clean.drop_duplicates(subset=["image", "cleaned_caption"]).copy()
        
        final_row_count = len(df_clean)
        removed_rows = initial_row_count - final_row_count
        logger.info(f"Removed {removed_rows} anomalous/duplicate rows. Final row count: {final_row_count}")
        
        # 4. Verify formatting rules (abort if punctuation or tag checks fail)
        verify_caption_format(df_clean)
        
        # Overwrite cleaned_captions.csv with the fully validated version
        df_clean.to_csv(csv_input, index=False)
        logger.info(f"Overwrote cleaned captions with fully validated data: {csv_input}")
        
        # 5. Extract statistics
        # Unique words (candidates, excluding sequence boundaries)
        vocabulary_words = []
        for cap in df_clean["cleaned_caption"]:
            tokens = cap.split()
            # Filter boundary markers
            filtered = [tok for tok in tokens if tok not in ["startseq", "endseq"]]
            vocabulary_words.extend(filtered)
            
        unique_vocab_candidates = len(set(vocabulary_words))
        
        # Top 20 words
        word_freq = Counter(vocabulary_words)
        top_20 = word_freq.most_common(20)
        
        # 6. Generate outputs/caption_validation_report.txt
        report_path = OUTPUT_DIR / "caption_validation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Cleaned Captions Validation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Pipeline Quality Checks ---\n")
            f.write(f"Punctuation Cleaned Verification : PASSED (Zero punctuation characters detected)\n")
            f.write(f"Boundary Tag format check       : PASSED (All start with startseq, end with endseq)\n\n")
            
            f.write("--- Data Cleaning Statistics ---\n")
            f.write(f"Initial Row Count               : {initial_row_count}\n")
            f.write(f"Removed Rows Count              : {removed_rows}\n")
            f.write(f"Final Clean Row Count           : {final_row_count}\n\n")
            
            f.write("--- Vocabulary Information ---\n")
            f.write(f"Unique Vocabulary Candidates    : {unique_vocab_candidates} (excluding boundary tags)\n\n")
            
            f.write("--- Top 20 Most Frequent Words ---\n")
            f.write(f"{'Word':<15} | {'Frequency':<10}\n")
            f.write("-" * 30 + "\n")
            for word, freq in top_20:
                f.write(f"{word:<15} | {freq:<10}\n")
                
            f.write("\n" + "=" * 60 + "\n")
            f.write("Validation pipeline check complete.\n")
            
        logger.info(f"Successfully generated validation report at: {report_path}")
        logger.info("Captions validation pipeline executed successfully.")
        
    except ValueError as ve:
        logger.critical(f"Validation failure: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_validation_pipeline()
