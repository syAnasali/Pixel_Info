import logging
import pickle
import tempfile
from pathlib import Path
import torch

from src.config.config import DEVICE, OUTPUT_DIR
from src.model.image_caption_model import ImageCaptionModel
from src.inference.caption_utils import beam_search_decode, FeatureExtractor

logger = logging.getLogger("api_predictor")

class CaptionPredictor:
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = OUTPUT_DIR.parent / "checkpoints" / "best_model.pth"
            
        logger.info(f"Initializing CaptionPredictor with weights from: {model_path}")
        
        # Load vocab maps
        w2i_path = OUTPUT_DIR / "word_to_index.pkl"
        i2w_path = OUTPUT_DIR / "index_to_word.pkl"
        
        if not w2i_path.exists() or not i2w_path.exists():
            raise FileNotFoundError(f"Vocabulary mappings not found in {OUTPUT_DIR}. Please run build_vocabulary.py first.")
            
        with open(w2i_path, "rb") as f:
            self.word_to_index = pickle.load(f)
        with open(i2w_path, "rb") as f:
            self.index_to_word = pickle.load(f)
            
        self.vocab_size = len(self.word_to_index)
        logger.info(f"Loaded vocabulary maps. Vocab size: {self.vocab_size}")
        
        # Load model weights
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            
        self.model = ImageCaptionModel(vocab_size=self.vocab_size)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        logger.info(f"Model loaded and set to eval mode on device: {DEVICE}")
        
        # Instantiate FeatureExtractor
        self.feature_extractor = FeatureExtractor(device=DEVICE)
        
    def predict(self, image_bytes: bytes, beam_width: int = 3, alpha: float = 0.75) -> str:
        """
        Extract features from image bytes and generate caption using beam search.
        """
        # Save image bytes to temp file to be consumed by FeatureExtractor
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = Path(tmp_file.name)
            
        try:
            # Extract ResNet50 features
            logger.info("Extracting image features...")
            features = self.feature_extractor.extract(tmp_path)
            
            # Predict caption using beam search
            logger.info(f"Generating caption with beam_width={beam_width}, alpha={alpha}...")
            caption, success = beam_search_decode(
                self.model,
                features,
                self.word_to_index,
                self.index_to_word,
                beam_width=beam_width,
                max_len=38,
                alpha=alpha,
                device=DEVICE
            )
            logger.info(f"Generated caption successfully: '{caption}' (success={success})")
            return caption
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
