import logging
import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import OUTPUT_DIR

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sequence_generation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("generate_sequences")

logger = setup_logging()

def load_inputs():
    """Load cleaned captions CSV and word_to_index mapping pickle."""
    csv_path = OUTPUT_DIR / "cleaned_captions.csv"
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned captions not found at {csv_path}. Run clean_captions.py first.")
    if not w2i_path.exists():
        raise FileNotFoundError(f"Word index mapping not found at {w2i_path}. Run build_vocabulary.py first.")
        
    logger.info("Loading cleaned captions and vocabulary mappings...")
    df = pd.read_csv(csv_path)
    
    with open(w2i_path, "rb") as f:
        word_to_index = pickle.load(f)
        
    return df, word_to_index

def generate_training_data(df: pd.DataFrame, word_to_index: dict) -> tuple:
    """
    Convert captions to sequences, generate input-target slices, apply pre-padding.
    Returns:
        X_img: List of image names for each training sample.
        X_seq: List of padded input token index arrays of shape (num_samples, max_length).
        y: List of target token indices for each training sample.
        stats: Dictionary containing diagnostic numbers.
    """
    logger.info("Starting sequence generation processing...")
    
    pad_idx = word_to_index.get("<pad>", 0)
    unk_idx = word_to_index.get("<unk>", 3)
    
    # 1. Determine maximum sequence length (in words/tokens)
    caption_lengths = df["cleaned_caption"].apply(lambda x: len(str(x).split()))
    max_length = int(caption_lengths.max())
    logger.info(f"Maximum caption length identified in dataset: {max_length} tokens")
    
    X_img, X_seq, y = [], [], []
    
    # Diagnostic stats
    total_tokens_mapped = 0
    total_unk_mapped = 0
    sample_walkthrough = None
    
    # Process each caption
    for row_idx, row in df.iterrows():
        image_name = row["image"]
        caption = str(row["cleaned_caption"])
        words = caption.split()
        
        # Convert words to indices
        seq = []
        for word in words:
            total_tokens_mapped += 1
            if word in word_to_index:
                seq.append(word_to_index[word])
            else:
                seq.append(unk_idx)
                total_unk_mapped += 1
                
        # Split into progressive sequences
        # e.g., for [startseq, a, dog, endseq], length is 4.
        # i goes from 1 to 3:
        # i=1: input=[startseq], target=a
        # i=2: input=[startseq, a], target=dog
        # i=3: input=[startseq, a, dog], target=endseq
        for i in range(1, len(seq)):
            input_seq = seq[:i]
            target_val = seq[i]
            
            # Pre-pad sequence to max_length
            padded_input = [pad_idx] * (max_length - len(input_seq)) + input_seq
            
            X_img.append(image_name)
            X_seq.append(padded_input)
            y.append(target_val)
            
            # Save the first sequence generation for the walkthrough report
            if row_idx == 0 and sample_walkthrough is None:
                sample_walkthrough = {
                    "raw_caption": caption,
                    "tokens": words,
                    "indices": seq,
                    "steps": []
                }
            
            if sample_walkthrough is not None and row_idx == 0:
                sample_walkthrough["steps"].append({
                    "input_words": words[:i],
                    "input_indices": input_seq,
                    "padded_indices": padded_input,
                    "target_word": words[i],
                    "target_index": target_val
                })
                
    stats = {
        "max_length": max_length,
        "total_samples": len(y),
        "total_tokens_mapped": total_tokens_mapped,
        "total_unk_mapped": total_unk_mapped,
        "unk_rate_pct": (total_unk_mapped / total_tokens_mapped * 100) if total_tokens_mapped > 0 else 0,
        "sample_walkthrough": sample_walkthrough,
        "pad_index": pad_idx,
        "unk_index": unk_idx
    }
    
    logger.info(f"Generated {len(y)} training sequences from {len(df)} captions.")
    return X_img, X_seq, y, stats

def save_sequence_artifacts(X_img: list, X_seq: list, y: list, stats: dict, word_to_index: dict):
    """Save training arrays to pickle and metadata report to text file."""
    w2i_rev = {v: k for k, v in word_to_index.items()}
    
    pkl_path = OUTPUT_DIR / "training_sequences.pkl"
    report_path = OUTPUT_DIR / "sequence_report.txt"
    
    # Save Pickled Sequence Data
    try:
        # Save as a standard structured dictionary
        data_dict = {
            "image_names": X_img,
            "input_sequences": np.array(X_seq),
            "target_words": np.array(y),
            "max_length": stats["max_length"],
            "pad_index": stats["pad_index"],
            "unk_index": stats["unk_index"]
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data_dict, f)
        logger.info(f"Saved training sequences to: {pkl_path}")
    except Exception as e:
        logger.error(f"Failed to save training sequences pickle: {e}")
        
    # Save Report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Flickr8K Sequence Generation Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Metadata ---\n")
            f.write(f"Maximum Caption Length (Tokens)     : {stats['max_length']}\n")
            f.write(f"Total Training Samples Generated    : {stats['total_samples']}\n")
            f.write(f"Padding Token Index (<pad>)         : {stats['pad_index']}\n")
            f.write(f"Unknown Token Index (<unk>)         : {stats['unk_index']}\n\n")
            
            f.write("--- Vocabulary Coverage Analysis ---\n")
            f.write(f"Total Tokens Checked in Pre-Clean   : {stats['total_tokens_mapped']}\n")
            f.write(f"Tokens Mapped to <unk> (Pruned)     : {stats['total_unk_mapped']}\n")
            f.write(f"Vocabulary Out-Of-Vocabulary Rate   : {stats['unk_rate_pct']:.2f}%\n")
            f.write(f"Vocabulary Coverage Rate            : {100.0 - stats['unk_rate_pct']:.2f}%\n\n")
            
            f.write("--- Sample Sequence Generation Walk-through ---\n")
            walk = stats["sample_walkthrough"]
            if walk:
                f.write(f"Raw Cleaned Caption: \"{walk['raw_caption']}\"\n")
                f.write(f"Tokenized Sequence : {walk['tokens']}\n")
                f.write(f"Index Mapping      : {walk['indices']}\n\n")
                
                f.write("Progressive Iteration Steps:\n")
                for s_idx, step in enumerate(walk["steps"], 1):
                    f.write(f"  Step {s_idx}:\n")
                    f.write(f"    - Input Words   : {step['input_words']}\n")
                    f.write(f"    - Input Indices : {step['input_indices']}\n")
                    f.write(f"    - Padded Inputs : {step['padded_indices']}\n")
                    f.write(f"    - Target Word   : '{step['target_word']}' (Index: {step['target_index']})\n")
                    f.write("-" * 50 + "\n")
            else:
                f.write("No walkthrough samples generated.\n")
                
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Saved sequence report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save sequence report text: {e}")

def run_validations(X_seq: list, y: list, max_length: int, word_to_index: dict):
    """Run sanity checks to ensure shapes and index boundaries are valid."""
    logger.info("Running validation checks...")
    
    assert len(X_seq) == len(y), "Mismatch between inputs count and targets count."
    assert len(X_seq) > 0, "No training samples generated."
    
    # Check padding lengths
    for idx, seq in enumerate(X_seq):
        assert len(seq) == max_length, f"Sequence at index {idx} has length {len(seq)} instead of {max_length}."
        
    # Check index boundary limits
    vocab_limit = len(word_to_index)
    for idx, target in enumerate(y):
        assert 0 <= target < vocab_limit, f"Target index {target} at sample {idx} is out of vocabulary boundaries (0-{vocab_limit-1})."
        
    logger.info("[SUCCESS] Validation checks completed. Training arrays are ready.")

def main():
    logger.info("Starting Training Sequence Generator...")
    try:
        # Load inputs
        df, word_to_index = load_inputs()
        
        # Process and generate sequences
        X_img, X_seq, y, stats = generate_training_data(df, word_to_index)
        
        # Run validations
        run_validations(X_seq, y, stats["max_length"], word_to_index)
        
        # Save output documents
        save_sequence_artifacts(X_img, X_seq, y, stats, word_to_index)
        
        logger.info("Sequence generation task finished successfully.")
        
    except FileNotFoundError as fnf:
        logger.error(f"Input file issue: {fnf}")
        sys.exit(1)
    except AssertionError as ae:
        logger.error(f"[VALIDATION FAILED] {ae}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Sequence generation crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
