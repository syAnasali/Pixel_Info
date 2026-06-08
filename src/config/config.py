import os
from pathlib import Path
import torch

# Directory setup relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Input data and output folders
DATA_DIR = PROJECT_ROOT / "data"
FLICKR_DIR = DATA_DIR / "Flickr8k"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# Ensure output directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration (Auto-detect RTX GPU CUDA support)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration placeholders (future ResNet50 + LSTM)
IMAGE_SIZE = 224  # ResNet standard input size
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
VOCAB_THRESHOLD = 5  # minimum word count to include in vocabulary

# Beam Search configuration for inference
BEAM_SIZE = 3
MAX_CAPTION_LEN = 20

# Hindi translation configuration
HINDI_SUPPORT = True

print(f"[CONFIG] Active device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[CONFIG] GPU Name: {torch.cuda.get_device_name(0)}")
