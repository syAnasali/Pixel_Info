import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger("caption_utils")

def greedy_decode(
    model: nn.Module,
    image_features: torch.Tensor,
    word_to_index: dict,
    index_to_word: dict,
    max_len: int = 38,
    device: torch.device = torch.device("cpu")
) -> tuple[str, bool]:
    """
    Perform greedy decoding to predict a caption for given image features.
    
    Args:
        model: Trained ImageCaptionModel.
        image_features: Tensor of shape (1, 2048) or (2048,).
        word_to_index: Word to index dictionary.
        index_to_word: Index to word dictionary.
        max_len: Maximum length of caption to generate.
        device: Torch device.
        
    Returns:
        tuple containing:
            - predicted_caption: String of words generated, with startseq/endseq stripped.
            - success: Boolean indicating if endseq was successfully predicted before max_len.
    """
    model.eval()
    
    # Assert features shape
    if len(image_features.shape) == 1:
        image_features = image_features.unsqueeze(0)
    image_features = image_features.to(device)
    
    # Retrieve indices for boundary tags
    start_idx = word_to_index.get("startseq")
    end_idx = word_to_index.get("endseq")
    
    if start_idx is None or end_idx is None:
        raise ValueError("Vocabulary does not contain startseq or endseq tokens.")
        
    # Start sequence with startseq token
    input_seq = [start_idx]
    success = False
    
    # Greedy decoding loop
    with torch.no_grad():
        for _ in range(max_len):
            input_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            logits = model(image_features, input_tensor)  # output shape: (1, vocab_size)
            
            # Predict the next index (greedy argmax)
            pred_idx = torch.argmax(logits, dim=-1).item()
            
            if pred_idx == end_idx:
                success = True
                break
                
            input_seq.append(pred_idx)
            
    # Convert indexes to words (excluding the initial startseq)
    words = []
    for idx in input_seq[1:]:
        word = index_to_word.get(idx)
        if word is not None:
            words.append(word)
            
    return " ".join(words), success

def beam_search_decode(
    model: nn.Module,
    image_features: torch.Tensor,
    word_to_index: dict,
    index_to_word: dict,
    beam_width: int = 3,
    max_len: int = 38,
    alpha: float = 0.75,
    device: torch.device = torch.device("cpu")
) -> tuple[str, bool]:
    """
    Perform beam search decoding to generate caption predictions.
    
    Args:
        model: Trained ImageCaptionModel.
        image_features: Tensor of shape (1, 2048) or (2048,).
        word_to_index: Word to index dictionary.
        index_to_word: Index to word dictionary.
        beam_width: Number of active hypotheses to maintain at each step.
        max_len: Maximum length of caption to generate.
        alpha: Length normalization coefficient (0.0 to 1.0).
        device: Torch device.
        
    Returns:
        tuple containing:
            - predicted_caption: String of words generated, with startseq/endseq stripped.
            - success: Boolean indicating if endseq was successfully predicted.
    """
    # Fallback to greedy decoding if beam_width is 1
    if beam_width <= 1:
        return greedy_decode(model, image_features, word_to_index, index_to_word, max_len=max_len, device=device)
        
    model.eval()
    
    # Assert features shape
    if len(image_features.shape) == 1:
        image_features = image_features.unsqueeze(0)
    image_features = image_features.to(device)
    
    start_idx = word_to_index.get("startseq")
    end_idx = word_to_index.get("endseq")
    
    if start_idx is None or end_idx is None:
        raise ValueError("Vocabulary does not contain startseq or endseq tokens.")
        
    # Each beam is represented as a tuple: (token_sequence_list, cumulative_log_probability)
    beams = [([start_idx], 0.0)]
    completed_beams = []
    
    with torch.no_grad():
        for step in range(max_len):
            candidates = []
            num_active = len(beams)
            
            if num_active == 0:
                break
                
            # Batch inference on all active beams
            img_features_batched = image_features.repeat(num_active, 1)
            input_seqs_batched = torch.tensor([b[0] for b in beams], dtype=torch.long, device=device)
            
            logits = model(img_features_batched, input_seqs_batched)  # shape: (num_active, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)  # shape: (num_active, vocab_size)
            
            # Expand each beam
            for i in range(num_active):
                seq, score = beams[i]
                
                # Get the top-k highest probability word indices for this specific beam
                topk_log_probs, topk_indices = torch.topk(log_probs[i], k=beam_width, dim=-1)
                
                for j in range(beam_width):
                    next_word = topk_indices[j].item()
                    prob = topk_log_probs[j].item()
                    
                    new_seq = seq + [next_word]
                    new_score = score + prob
                    
                    # If endseq is reached, add to completed list
                    if next_word == end_idx:
                        completed_beams.append((new_seq, new_score))
                    else:
                        candidates.append((new_seq, new_score))
                        
            # Sort all candidate expansions by score and keep top-k
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
    # Add any remaining active beams to completed list if loop finishes
    completed_beams.extend(beams)
    
    if not completed_beams:
        # Fallback if no beams generated
        return "", False
        
    # Apply length normalization: score / (length ** alpha)
    normalized_beams = []
    for seq, score in completed_beams:
        # Generated words count (excluding startseq)
        gen_len = max(1, len(seq) - 1)
        norm_score = score / (gen_len ** alpha)
        normalized_beams.append((seq, norm_score))
        
    # Sort by normalized score in descending order
    normalized_beams = sorted(normalized_beams, key=lambda x: x[1], reverse=True)
    best_seq, best_score = normalized_beams[0]
    
    # Check if endseq was successfully predicted
    success = (best_seq[-1] == end_idx)
    
    # Convert indices to words, removing startseq/endseq boundary tokens
    words = []
    start_skip = 1 if best_seq[0] == start_idx else 0
    end_skip = -1 if best_seq[-1] == end_idx else None
    
    target_indices = best_seq[start_skip:end_skip] if end_skip is not None else best_seq[start_skip:]
    for idx in target_indices:
        word = index_to_word.get(idx)
        if word is not None:
            words.append(word)
            
    return " ".join(words), success

class FeatureExtractor:
    """
    Wrapper around pretrained ResNet50 model to extract image features.
    Caches the model internally for performance.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        
        # Standard ImageNet transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def extract(self, image_path: Path) -> torch.Tensor:
        """
        Load an image from disk, preprocess it, and extract its 2048-dimensional features.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        if self.model is None:
            logger.info("Initializing pretrained ResNet50 feature extractor...")
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            # Remove output fully-connected layer
            model.fc = nn.Identity()
            model.eval()
            model.to(self.device)
            self.model = model
            logger.info("ResNet50 feature extractor loaded successfully.")
            
        try:
            # Load and convert image to RGB
            img = Image.open(image_path).convert("RGB")
            # Apply standard transforms and batch shape
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)  # shape (1, 2048)
                
            return features.cpu().squeeze(0)  # Return as 1D feature vector of size 2048
        except Exception as e:
            logger.error(f"Failed to extract features from image {image_path.name}: {e}")
            raise e
