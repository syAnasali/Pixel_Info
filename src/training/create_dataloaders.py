import logging
import os
import sys
import pickle
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import OUTPUT_DIR, BATCH_SIZE
from src.datasets.flickr_caption_dataset import FlickrCaptionDataset

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataloader_creation.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("create_dataloaders")

logger = setup_logging()

def load_preprocessed_data() -> tuple:
    """
    Load vocab index, sequence outputs, and image features dictionary.
    Supports resilient search across alternate image features directories.
    """
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    seq_path = OUTPUT_DIR / "training_sequences.pkl"
    
    # Check alternate features paths
    possible_feat_paths = [
        OUTPUT_DIR / "features" / "image_features.pkl",
        OUTPUT_DIR / "image_features.pkl"
    ]
    feat_path = None
    for p in possible_feat_paths:
        if p.exists():
            feat_path = p
            break
            
    if not w2i_path.exists():
        raise FileNotFoundError(f"Vocabulary index not found at {w2i_path}. Run build_vocabulary.py first.")
    if not seq_path.exists():
        raise FileNotFoundError(f"Training sequences not found at {seq_path}. Run generate_sequences.py first.")
    if not feat_path:
        raise FileNotFoundError(
            f"Image features dictionary not found. Checked locations: {possible_feat_paths}. Run extract_features.py first."
        )
        
    logger.info("Loading preprocessed pickle files...")
    
    with open(w2i_path, "rb") as f:
        word_to_index = pickle.load(f)
        
    with open(seq_path, "rb") as f:
        seq_data = pickle.load(f)
        
    with open(feat_path, "rb") as f:
        image_features_dict = pickle.load(f)
        
    logger.info(f"Loaded {len(word_to_index)} vocabulary items.")
    logger.info(f"Loaded {len(seq_data['target_words'])} training sequence samples.")
    logger.info(f"Loaded {len(image_features_dict)} image features from {feat_path.name}.")
    
    return word_to_index, seq_data, image_features_dict

def prepare_dataloaders(val_split: float = 0.2, batch_size: int = None, num_workers: int = 0) -> tuple:
    """
    Split dataset at the image level, instantiate Datasets, and return train/validation DataLoaders.
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
        
    word_to_index, seq_data, image_features_dict = load_preprocessed_data()
    
    image_names = seq_data["image_names"]
    input_sequences = seq_data["input_sequences"]
    target_words = seq_data["target_words"]
    max_length = seq_data["max_length"]
    
    # 1. Collect unique images and split them (deterministic split using seed)
    unique_images = sorted(list(set(image_names)))
    random.seed(42)
    random.shuffle(unique_images)
    
    split_idx = int(len(unique_images) * (1 - val_split))
    train_images = set(unique_images[:split_idx])
    val_images = set(unique_images[split_idx:])
    
    logger.info(f"Unique images split: {len(train_images)} train, {len(val_images)} validation (split ratio: {1-val_split:.1f}/{val_split:.1f})")
    
    # 2. Slice sequence indices based on image set membership
    train_indices = [idx for idx, name in enumerate(image_names) if name in train_images]
    val_indices = [idx for idx, name in enumerate(image_names) if name in val_images]
    
    # Train sequences
    train_names = [image_names[i] for i in train_indices]
    train_seqs = input_sequences[train_indices]
    train_targets = target_words[train_indices]
    
    # Val sequences
    val_names = [image_names[i] for i in val_indices]
    val_seqs = input_sequences[val_indices]
    val_targets = target_words[val_indices]
    
    logger.info(f"Sequences split: {len(train_targets)} train sequences, {len(val_targets)} validation sequences")
    
    # 3. Instantiate FlickrCaptionDataset splits
    train_dataset = FlickrCaptionDataset(train_names, train_seqs, train_targets, image_features_dict)
    val_dataset = FlickrCaptionDataset(val_names, val_seqs, val_targets, image_features_dict)
    
    # 4. Construct DataLoaders
    # Pin memory to speed up CPU-GPU transfers if CUDA is active
    pin_mem = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    stats = {
        "total_samples": len(target_words),
        "train_samples": len(train_targets),
        "validation_samples": len(val_targets),
        "batch_size": batch_size,
        "max_length": max_length,
        "vocab_size": len(word_to_index)
    }
    
    return train_loader, val_loader, stats

def run_sanity_checks(train_loader: DataLoader, stats: dict):
    """
    Fetch the first batch and perform shape consistency assertions.
    """
    logger.info("Executing DataLoader sanity checks...")
    
    # Get a single batch
    batch = next(iter(train_loader))
    
    # Assert keys exist
    assert "image_features" in batch, "Missing key 'image_features' in batch output."
    assert "input_sequence" in batch, "Missing key 'input_sequence' in batch output."
    assert "target_word" in batch, "Missing key 'target_word' in batch output."
    
    img_feats = batch["image_features"]
    input_seqs = batch["input_sequence"]
    targets = batch["target_word"]
    
    current_batch_size = img_feats.shape[0]
    
    # Shape checks
    assert img_feats.shape == (current_batch_size, 2048), f"Incorrect image features shape: {img_feats.shape} instead of ({current_batch_size}, 2048)"
    assert input_seqs.shape == (current_batch_size, stats["max_length"]), f"Incorrect input sequence shape: {input_seqs.shape} instead of ({current_batch_size}, {stats['max_length']})"
    assert targets.shape == (current_batch_size, 1), f"Incorrect target word shape: {targets.shape} instead of ({current_batch_size}, 1)"
    
    logger.info("[SUCCESS] DataLoader sanity checks completed. Tensors mapped successfully:")
    logger.info(f"  - Image Features Shape : {img_feats.shape}")
    logger.info(f"  - Input Sequence Shape : {input_seqs.shape}")
    logger.info(f"  - Target Word Shape    : {targets.shape}")

def write_report(stats: dict):
    """Write outputs/dataloader_report.txt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "dataloader_report.txt"
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             PyTorch DataLoaders Verification Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Splitting Details (Image-Level) ---\n")
            f.write("Split Method                : Image-Level (Prevents data leakage)\n")
            f.write(f"Train/Val Ratio             : 80% / 20%\n\n")
            
            f.write("--- Sample & Sequence Counts ---\n")
            f.write(f"Total Sequence Samples      : {stats['total_samples']}\n")
            f.write(f"Train Sequences             : {stats['train_samples']}\n")
            f.write(f"Validation Sequences        : {stats['validation_samples']}\n\n")
            
            f.write("--- DataLoader Configurations ---\n")
            f.write(f"Batch Size                  : {stats['batch_size']}\n")
            f.write(f"Maximum Sequence Length     : {stats['max_length']}\n")
            f.write(f"Vocabulary Size             : {stats['vocab_size']}\n")
            f.write(f"Pin Memory Enabled          : {torch.cuda.is_available()}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Saved dataloader verification report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write dataloader report: {e}")

def main():
    logger.info("Initializing PyTorch DataLoader Creation Pipeline...")
    try:
        # Prepare loaders (80/20 train/val split, read batch size from global configs)
        train_loader, val_loader, stats = prepare_dataloaders(val_split=0.2, num_workers=0)
        
        # Run assertions
        run_sanity_checks(train_loader, stats)
        
        # Write validation report
        write_report(stats)
        
        logger.info("DataLoader creation pipeline finished successfully.")
        
    except FileNotFoundError as fnf:
        logger.error(f"Input verification failed: {fnf}")
        sys.exit(1)
    except AssertionError as ae:
        logger.error(f"[VALIDATION FAILED] {ae}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
