import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import FLICKR_DIR, OUTPUT_DIR

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataset_exploration.log"
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("explore_dataset")

logger = setup_logging()

def load_dataset(flickr_dir: Path) -> pd.DataFrame:
    """
    Search and load Flickr8k captions file.
    Supports captions.txt, captions.csv, and Flickr8k.token.txt formats.
    """
    possible_files = ["captions.txt", "captions.csv", "Flickr8k.token.txt"]
    captions_file = None
    
    for filename in possible_files:
        path = flickr_dir / filename
        if path.exists():
            captions_file = path
            break
            
    if not captions_file:
        raise FileNotFoundError(
            f"Could not find any captions file in {flickr_dir}. Checked: {possible_files}"
        )
        
    logger.info(f"Found captions file: {captions_file.name}")
    
    try:
        # Load based on extension/format
        if captions_file.suffix == ".txt" and "token" in captions_file.name:
            # Classic Flickr8k.token.txt: image_name.jpg#num \t caption
            df = pd.read_csv(captions_file, sep="\t", header=None, names=["image_idx", "caption"])
            df["image"] = df["image_idx"].apply(lambda x: str(x).split("#")[0])
            df = df[["image", "caption"]]
        else:
            # Standard csv format (comma-separated with header)
            df = pd.read_csv(captions_file)
            # Ensure columns are named properly
            if "image" not in df.columns or "caption" not in df.columns:
                # Try to map columns if they differ
                logger.warning("Columns 'image' and 'caption' not found in headers. Mapping first two columns.")
                df.columns = ["image", "caption"] + list(df.columns[2:])
                df = df[["image", "caption"]]
        
        logger.info(f"Loaded dataset successfully. Initial row count: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error parsing captions file {captions_file}: {e}")
        raise

def validate_integrity(df: pd.DataFrame) -> dict:
    """
    Check dataset for missing values, duplicate captions, and duplicate image names.
    Returns validation findings and a clean version of the DataFrame.
    """
    logger.info("Starting dataset integrity checks...")
    results = {}
    
    # 1. Missing values
    missing_images = df["image"].isna().sum()
    missing_captions = df["caption"].isna().sum()
    
    # Handle empty strings as missing
    empty_images = (df["image"].astype(str).str.strip() == "").sum()
    empty_captions = (df["caption"].astype(str).str.strip() == "").sum()
    
    total_missing_images = missing_images + empty_images
    total_missing_captions = missing_captions + empty_captions
    
    results["missing_images"] = int(total_missing_images)
    results["missing_captions"] = int(total_missing_captions)
    
    if total_missing_images > 0:
        logger.warning(f"Found {total_missing_images} rows with missing or empty image names.")
    if total_missing_captions > 0:
        logger.warning(f"Found {total_missing_captions} rows with missing or empty captions.")
        
    # Drop rows with missing keys to ensure stats don't crash
    df_clean = df.dropna(subset=["image", "caption"]).copy()
    df_clean = df_clean[(df_clean["image"].astype(str).str.strip() != "") & 
                        (df_clean["caption"].astype(str).str.strip() != "")].copy()
    
    # 2. Duplicate rows (identical image AND caption)
    duplicate_rows = df_clean.duplicated(subset=["image", "caption"]).sum()
    results["duplicate_rows"] = int(duplicate_rows)
    if duplicate_rows > 0:
        logger.warning(f"Found {duplicate_rows} exact duplicate image-caption rows.")
        
    # 3. Duplicate captions (same caption text across the dataset)
    duplicate_captions = df_clean.duplicated(subset=["caption"]).sum()
    results["duplicate_captions"] = int(duplicate_captions)
    if duplicate_captions > 0:
        logger.warning(f"Found {duplicate_captions} duplicate caption texts.")
        
    # 4. Distribution of captions per image
    # Let's count how many captions each unique image has
    caption_counts = df_clean.groupby("image")["caption"].count()
    results["caption_counts"] = caption_counts
    
    # Check if there are duplicate image names with unequal caption counts (normally 5 per image)
    non_standard_counts = (caption_counts != 5).sum()
    results["non_standard_captions_per_image"] = int(non_standard_counts)
    if non_standard_counts > 0:
        logger.info(f"Found {non_standard_counts} images that do not have exactly 5 captions.")

    return results, df_clean

def compute_statistics(df: pd.DataFrame, caption_counts: pd.Series) -> dict:
    """
    Compute statistical metrics for unique images, captions, and text lengths.
    """
    logger.info("Computing dataset statistics...")
    stats = {}
    
    # Unique images
    stats["total_images"] = int(df["image"].nunique())
    stats["total_captions"] = int(len(df))
    
    # Captions per image
    stats["min_captions_per_image"] = int(caption_counts.min())
    stats["max_captions_per_image"] = int(caption_counts.max())
    stats["mean_captions_per_image"] = float(caption_counts.mean())
    stats["median_captions_per_image"] = float(caption_counts.median())
    
    # Word count metrics (split by whitespace)
    # Ensure caption is string
    word_counts = df["caption"].apply(lambda x: len(str(x).split()))
    
    stats["avg_caption_length"] = float(word_counts.mean())
    stats["min_caption_length"] = int(word_counts.min())
    stats["max_caption_length"] = int(word_counts.max())
    
    # Shortest and longest captions
    min_idx = word_counts.idxmin()
    max_idx = word_counts.idxmax()
    
    stats["shortest_caption"] = df.loc[min_idx, "caption"]
    stats["shortest_caption_image"] = df.loc[min_idx, "image"]
    stats["longest_caption"] = df.loc[max_idx, "caption"]
    stats["longest_caption_image"] = df.loc[max_idx, "image"]
    
    return stats

def display_samples(df: pd.DataFrame, num_samples: int = 5):
    """Randomly select and print samples from the dataset."""
    logger.info(f"\n--- Randomly Displaying {num_samples} Samples ---")
    if len(df) == 0:
        logger.warning("No data available to display samples.")
        return
        
    sampled_indices = np.random.choice(df.index, size=min(num_samples, len(df)), replace=False)
    for i, idx in enumerate(sampled_indices, 1):
        row = df.loc[idx]
        print(f"Sample {i}:")
        print(f"  [Image]   : {row['image']}")
        print(f"  [Caption] : {row['caption']}")
        print("-" * 50)

def generate_outputs(stats: dict, validation: dict):
    """Write outputs/dataset_summary.csv and outputs/exploration_report.txt."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_csv_path = OUTPUT_DIR / "dataset_summary.csv"
    report_txt_path = OUTPUT_DIR / "exploration_report.txt"
    
    # 1. Generate dataset_summary.csv
    summary_data = {
        "Metric": [
            "Total Images",
            "Total Captions",
            "Mean Captions Per Image",
            "Min Captions Per Image",
            "Max Captions Per Image",
            "Average Caption Length (words)",
            "Min Caption Length (words)",
            "Max Caption Length (words)",
            "Missing Images",
            "Missing Captions",
            "Duplicate Rows",
            "Duplicate Captions",
            "Images with Non-Standard Captions Count"
        ],
        "Value": [
            stats["total_images"],
            stats["total_captions"],
            f"{stats['mean_captions_per_image']:.2f}",
            stats["min_captions_per_image"],
            stats["max_captions_per_image"],
            f"{stats['avg_caption_length']:.2f}",
            stats["min_caption_length"],
            stats["max_caption_length"],
            validation["missing_images"],
            validation["missing_captions"],
            validation["duplicate_rows"],
            validation["duplicate_captions"],
            validation["non_standard_captions_per_image"]
        ]
    }
    
    try:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_csv_path, index=False)
        logger.info(f"Successfully saved summary to: {summary_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")
        
    # 2. Generate exploration_report.txt
    try:
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Flickr8K Dataset Exploration Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- General Statistics ---\n")
            f.write(f"Total Unique Images            : {stats['total_images']}\n")
            f.write(f"Total Caption Annotations      : {stats['total_captions']}\n\n")
            
            f.write("--- Captions Per Image Distribution ---\n")
            f.write(f"Average Captions Per Image     : {stats['mean_captions_per_image']:.2f}\n")
            f.write(f"Median Captions Per Image      : {stats['median_captions_per_image']:.2f}\n")
            f.write(f"Min Captions Per Image         : {stats['min_captions_per_image']}\n")
            f.write(f"Max Captions Per Image         : {stats['max_captions_per_image']}\n")
            f.write(f"Images with != 5 Captions      : {validation['non_standard_captions_per_image']}\n\n")
            
            f.write("--- Text Length Statistics (Words) ---\n")
            f.write(f"Average Caption Length         : {stats['avg_caption_length']:.2f} words\n")
            f.write(f"Shortest Caption Length        : {stats['min_caption_length']} words\n")
            f.write(f"  - Image                      : {stats['shortest_caption_image']}\n")
            f.write(f"  - Caption                    : \"{stats['shortest_caption']}\"\n")
            f.write(f"Longest Caption Length         : {stats['max_caption_length']} words\n")
            f.write(f"  - Image                      : {stats['longest_caption_image']}\n")
            f.write(f"  - Caption                    : \"{stats['longest_caption']}\"\n\n")
            
            f.write("--- Data Integrity & Validation ---\n")
            f.write(f"Missing Image Names Count      : {validation['missing_images']}\n")
            f.write(f"Missing Caption Texts Count    : {validation['missing_captions']}\n")
            f.write(f"Exact Duplicate Rows           : {validation['duplicate_rows']}\n")
            f.write(f"Duplicate Caption Texts        : {validation['duplicate_captions']}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Successfully saved exploration report to: {report_txt_path}")
    except Exception as e:
        logger.error(f"Failed to save exploration report text: {e}")

def main():
    logger.info("Initializing Flickr8k Dataset Exploration...")
    try:
        # Load dataset
        df = load_dataset(FLICKR_DIR)
        
        # Validate integrity
        validation, df_clean = validate_integrity(df)
        
        # Compute statistics
        stats = compute_statistics(df_clean, validation["caption_counts"])
        
        # Output summary files
        generate_outputs(stats, validation)
        
        # Display samples
        display_samples(df_clean, num_samples=5)
        
        logger.info("Flickr8k dataset exploration finished successfully.")
        
    except FileNotFoundError as fnf:
        logger.error(f"Data directory issue: {fnf}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Exploration process crashed with an unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
