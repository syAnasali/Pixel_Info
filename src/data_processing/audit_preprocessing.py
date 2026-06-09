import os
import sys
import pickle
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import FLICKR_DIR, OUTPUT_DIR

def run_audit():
    # 1. Source captions file path
    possible_files = ["captions.txt", "captions.csv", "Flickr8k.token.txt"]
    captions_file = None
    for fname in possible_files:
        path = FLICKR_DIR / fname
        if path.exists():
            captions_file = path.resolve()
            break
            
    if not captions_file:
        raise FileNotFoundError(f"Source captions file not found in {FLICKR_DIR}")
        
    # Load raw captions count
    raw_df = pd.read_csv(captions_file)
    raw_rows = len(raw_df)
    
    # 2. Cleaned captions count
    clean_csv = OUTPUT_DIR / "cleaned_captions.csv"
    if not clean_csv.exists():
        raise FileNotFoundError(f"Cleaned captions file not found at {clean_csv}")
    clean_df = pd.read_csv(clean_csv)
    final_captions_count = len(clean_df)
    unique_images = clean_df["image"].nunique()
    
    # 3. Vocab sizes
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    if not w2i_path.exists():
        raise FileNotFoundError(f"word_to_index mapping not found at {w2i_path}")
    with open(w2i_path, "rb") as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    
    # Exclude boundary tokens and special tokens to get unique candidates
    special_tokens = ["<pad>", "startseq", "endseq", "<unk>"]
    # Let's count unique words in validated captions to see candidates count
    vocabulary_words = []
    for cap in clean_df["cleaned_caption"]:
        tokens = cap.split()
        filtered = [tok for tok in tokens if tok not in ["startseq", "endseq"]]
        vocabulary_words.extend(filtered)
    unique_vocab_candidates = len(set(vocabulary_words))
    
    # 4. Training sequences
    seq_path = OUTPUT_DIR / "training_sequences.pkl"
    if not seq_path.exists():
        raise FileNotFoundError(f"Training sequences not found at {seq_path}")
    with open(seq_path, "rb") as f:
        seq_data = pickle.load(f)
    total_sequences = len(seq_data["target_words"])
    
    # Determine dataset type
    # Flickr8k has 8,091 images * 5 = 40,455 captions. Mock has 21 captions.
    is_mock = (raw_rows < 100)
    dataset_type = "Mock Test Dataset" if is_mock else "Full Flickr8K Captions Dataset"
    
    # Write report
    report_path = OUTPUT_DIR / "preprocessing_audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("             Dataset Preprocessing Audit Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"1. Absolute Path of Source Captions File : {captions_file}\n")
        f.write(f"2. Total Raw Caption Rows Processed      : {raw_rows}\n")
        f.write(f"   Final Cleaned & Validated Rows        : {final_captions_count}\n")
        f.write(f"3. Total Unique Images Referenced        : {unique_images}\n")
        f.write(f"4. Total Unique Vocabulary Words         : {unique_vocab_candidates} (Excluding boundary tags)\n")
        f.write(f"   Active Vocabulary Size (thresholded)  : {vocab_size} (Including 4 special tokens)\n")
        f.write(f"5. Total Generated Training Sequences    : {total_sequences}\n\n")
        
        f.write("--- Pipeline Source Determination ---\n")
        f.write(f"The preprocessing pipeline was built from the: {dataset_type}\n")
        f.write(f"  (Reason: raw captions count is {raw_rows}, which is < 100)\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Audit report generated successfully.\n")
        
    print(f"Audit completed successfully. Report written to {report_path}")

if __name__ == "__main__":
    run_audit()
