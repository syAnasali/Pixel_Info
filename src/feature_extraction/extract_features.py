import logging
import os
import sys
import time
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import FLICKR_DIR, OUTPUT_DIR, IMAGE_SIZE

def setup_logging():
    """Configure logging to console and file."""
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "feature_extraction.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    return logging.getLogger("extract_features")

logger = setup_logging()

class Flickr8kImageDataset(Dataset):
    """
    Custom Dataset to load and preprocess Flickr8k images.
    Gracefully handles corrupted files by returning a success flag.
    """
    def __init__(self, image_dir: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Supported image formats
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            
        # Ensure unique, sorted paths for reproducibility
        self.image_paths = sorted(list(set(self.image_paths)))
        logger.info(f"Dataset scanner found {len(self.image_paths)} candidate images in {image_dir}")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Attempt to open and load image
            img = Image.open(img_path).convert("RGB")
            # Force loading pixel data to trigger decompression errors for corrupted files
            img.verify() 
            
            # Since verify() closes the file, we must reopen it to extract pixels
            img = Image.open(img_path).convert("RGB")
            
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)
                
            return img_tensor, img_path.name, True
        except Exception as e:
            logger.warning(f"Failed to read image {img_path.name}: {e}")
            # Return dummy tensor and set success flag to False
            return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), img_path.name, False

def load_resnet50_extractor(device: torch.device) -> nn.Module:
    """
    Load pretrained ResNet50 and replace classification head with nn.Identity
    to output 2048-dimensional feature maps.
    """
    logger.info("Loading pretrained ResNet50 model...")
    # Load weights using modern weights parameter
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Replace the fully connected classification head with Identity
    model.fc = nn.Identity()
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    return model

def extract_features(image_dir: Path, batch_size: int = 16, sleep_seconds: float = 0.5) -> tuple:
    """
    Scan image directory and extract ResNet50 embeddings.
    Returns:
        features_dict: {image_name: feature_vector}
        stats: dictionary with execution metrics.
    """
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logger.info(f"Extracting features using device: {device} ({gpu_name})")
    
    # 2. Setup transforms (Standard ImageNet specs)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 3. Load dataset & model
    dataset = Flickr8kImageDataset(image_dir, transform=transform)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = load_resnet50_extractor(device)
    
    features_dict = {}
    failed_images = []
    processed_count = 0
    
    start_time = time.time()
    
    # 4. Batch Inference Loop
    logger.info(f"Starting batch feature extraction (batch_size={batch_size}, throttle={sleep_seconds}s)...")
    with torch.no_grad():
        for batch_imgs, batch_names, success_flags in tqdm(dataloader, desc="Extracting features"):
            # Check for failed image loads in this batch
            valid_indices = [i for i, success in enumerate(success_flags) if success]
            failed_indices = [i for i, success in enumerate(success_flags) if not success]
            
            # Log failures
            for i in failed_indices:
                failed_images.append(batch_names[i])
                
            if not valid_indices:
                continue
                
            # Filter batch to process only valid images
            valid_imgs = batch_imgs[valid_indices].to(device)
            valid_names = [batch_names[i] for i in valid_indices]
            
            # Forward pass
            outputs = model(valid_imgs)
            outputs = outputs.cpu().numpy()
            
            # Save features to dictionary
            for name, feat in zip(valid_names, outputs):
                features_dict[name] = feat
                processed_count += 1
                
            # Throttle device temperature
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                
    elapsed_time = time.time() - start_time
    logger.info(f"Finished extraction. Processed: {processed_count}, Failed: {len(failed_images)}")
    
    # Get feature vector dimension
    feat_dim = 0
    if features_dict:
        feat_dim = next(iter(features_dict.values())).shape[0]
        
    stats = {
        "total_processed": processed_count,
        "failed_images": failed_images,
        "feature_dimension": feat_dim,
        "elapsed_time": elapsed_time,
        "device_used": gpu_name
    }
    
    return features_dict, stats

def save_feature_artifacts(features_dict: dict, stats: dict):
    """Save features dictionary to pkl and write validation report to text file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pkl_path = OUTPUT_DIR / "image_features.pkl"
    report_path = OUTPUT_DIR / "feature_extraction_report.txt"
    
    # Save Pickled Features
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(features_dict, f)
        logger.info(f"Saved feature dictionary mapping to: {pkl_path}")
    except Exception as e:
        logger.error(f"Failed to save image features pickle: {e}")
        
    # Save Report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("             Flickr8K Image Feature Extraction Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- Hardware & Environment ---\n")
            f.write(f"Computation Device Used : {stats['device_used']}\n\n")
            
            f.write("--- Extraction Metrics ---\n")
            f.write(f"Total Images Processed  : {stats['total_processed']}\n")
            f.write(f"Total Failed Images     : {len(stats['failed_images'])}\n")
            f.write(f"Feature Dimension       : {stats['feature_dimension']} (ResNet50 Embeddings)\n")
            f.write(f"Total Extraction Time   : {stats['elapsed_time']:.2f} seconds\n")
            if stats['total_processed'] > 0:
                f.write(f"Average Time per Image  : {stats['elapsed_time'] / stats['total_processed'] * 1000:.2f} ms\n")
            f.write("\n")
            
            if stats['failed_images']:
                f.write("--- Failed Images List ---\n")
                for fname in stats['failed_images']:
                    f.write(f"  - {fname}\n")
                f.write("\n")
                
            f.write("=" * 60 + "\n")
            f.write("Report generated successfully.\n")
            
        logger.info(f"Saved feature extraction report to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write feature extraction report: {e}")

def main():
    logger.info("Initializing Image Feature Extraction Pipeline...")
    
    image_dir = FLICKR_DIR / "Images"
    if not image_dir.exists():
        logger.error(f"Images folder not found at: {image_dir}")
        sys.exit(1)
        
    try:
        # Dynamically adjust parameters based on GPU availability
        if torch.cuda.is_available():
            # GPU is fast and handles batching efficiently, 0.1s sleep keeps temperatures stable
            batch_size = 32
            sleep_seconds = 0.1
            logger.info("RTX GPU detected. Using fast GPU extraction parameters.")
        else:
            # CPU is slow and prone to overheating under full load, use strict throttling
            batch_size = 8
            sleep_seconds = 1.0
            logger.info("Running on CPU. Using strict thermal throttling parameters.")
            
        features_dict, stats = extract_features(image_dir, batch_size=batch_size, sleep_seconds=sleep_seconds)
        
        # Save output documents
        save_feature_artifacts(features_dict, stats)
        
        logger.info("Image feature extraction finished successfully.")
        
    except Exception as e:
        logger.critical(f"Feature extraction pipeline crashed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
