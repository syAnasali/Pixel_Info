import os
import sys
import pickle
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import DEVICE, OUTPUT_DIR
from src.model.caption_decoder import CaptionDecoder

class ImageCaptionModel(nn.Module):
    """
    Complete Image Caption Generation Model.
    
    Coordinates the Image Branch (mapping ResNet50 features from 2048 to 512 dimensions)
    and the Language Branch + Fusion + Classification layers (CaptionDecoder).
    """
    def __init__(self, vocab_size: int, embed_size: int = 512, hidden_size: int = 512, dropout_rate: float = 0.5):
        """
        Initialize the ImageCaptionModel.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_size: Dimensionality of word embeddings and projected image features. Default is 512.
            hidden_size: Hidden dimension of LSTM. Default is 512.
            dropout_rate: Dropout probability. Default is 0.5.
        """
        super(ImageCaptionModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # Image Branch: Projected to embed_size (512) to match language embedding dimension
        self.image_branch = nn.Sequential(
            nn.Linear(2048, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder Branch (Language + Fusion + Prediction)
        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Apply Xavier initialization
        self._init_weights()
        
    def _init_weights(self):
        """Apply Xavier initialization to linear and embedding layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        
    def forward(self, image_features: torch.Tensor, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Image Caption Model.
        
        Args:
            image_features: Tensor of shape (batch_size, 2048) representing ResNet50 features.
            input_sequences: Tensor of shape (batch_size, sequence_length) representing caption token indices.
            
        Returns:
            Logits of shape (batch_size, vocab_size).
        """
        batch_size = image_features.shape[0]
        
        # Shape assertions
        assert image_features.shape == (batch_size, 2048), \
            f"Expected image_features shape (batch_size, 2048), got {image_features.shape}"
        assert input_sequences.shape[0] == batch_size, \
            f"Batch size mismatch between images ({batch_size}) and sequences ({input_sequences.shape[0]})"
            
        # 1. Project Image Features: (batch_size, 2048) -> (batch_size, 512)
        img_emb = self.image_branch(image_features)
        
        # 2. Decode and Predict: (batch_size, 512) and (batch_size, seq_len) -> (batch_size, vocab_size)
        logits = self.decoder(img_emb, input_sequences)
        
        return logits

def get_model_summary(model: nn.Module) -> str:
    """Generate a formatted summary string of the model's architecture and parameters."""
    lines = []
    lines.append("=" * 85)
    lines.append(f"{'Layer (type)':<45} | {'Parameters Count':<20} | {'Requires Grad'}")
    lines.append("=" * 85)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        # Only summarize leaf layers
        if len(list(module.children())) == 0 and not isinstance(module, ImageCaptionModel):
            params = sum(p.numel() for p in module.parameters())
            req_grad = any(p.requires_grad for p in module.parameters())
            layer_name = f"{name} ({module.__class__.__name__})"
            lines.append(f"{layer_name:<45} | {params:<20,} | {str(req_grad)}")
            
    # Include non-leaf parameter summaries (e.g. multi-layer configurations)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lines.append("=" * 85)
    lines.append(f"Total Parameters          : {total_params:,}")
    lines.append(f"Trainable Parameters      : {trainable_params:,}")
    lines.append(f"Non-Trainable Parameters  : {total_params - trainable_params:,}")
    lines.append("=" * 85)
    
    return "\n".join(lines)

def run_verification():
    print("Initializing Image Caption Model Verification...")
    
    # Resolve vocabulary size from preprocessing outputs
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    if w2i_path.exists():
        with open(w2i_path, "rb") as f:
            w2i = pickle.load(f)
        vocab_size = len(w2i)
        print(f"Loaded vocabulary index map. Active Vocabulary Size: {vocab_size}")
    else:
        vocab_size = 2970
        print(f"Vocabulary file not found. Defaulting to vocabulary size: {vocab_size}")
        
    # Instantiate Model
    model = ImageCaptionModel(vocab_size=vocab_size)
    model.to(DEVICE)
    print(f"Model successfully loaded and mapped to device: {DEVICE}")
    
    # Generate summary
    summary_str = get_model_summary(model)
    print("\nModel Summary:")
    print(summary_str)
    
    # Run Dummy Forward Pass Verification
    batch_size = 32
    seq_len = 38
    print(f"\nRunning dummy forward pass validation (batch_size={batch_size}, seq_len={seq_len})...")
    
    dummy_images = torch.randn(batch_size, 2048).to(DEVICE)
    dummy_seqs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        logits = model(dummy_images, dummy_seqs)
        
    output_shape = logits.shape
    expected_shape = (batch_size, vocab_size)
    print(f"Forward Pass Output Shape: {output_shape}")
    
    shape_verified = (output_shape == expected_shape)
    print(f"Shape Verification Status: {'PASSED' if shape_verified else 'FAILED'}")
    
    # Write Report
    report_path = OUTPUT_DIR / "model_architecture_report.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("             Image Captioning Model Architecture Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Device Used                  : {DEVICE}\n")
        f.write(f"Active Vocabulary Size       : {vocab_size}\n")
        f.write(f"Configured Embed Dimension   : 512\n")
        f.write(f"Configured LSTM Hidden Dim   : 512\n\n")
        
        f.write("--- Model Forward Pass Validation ---\n")
        f.write(f"Input Image Features Shape   : ({batch_size}, 2048)\n")
        f.write(f"Input Sequence Shape         : ({batch_size}, {seq_len})\n")
        f.write(f"Predicted Output Shape       : {list(output_shape)}\n")
        f.write(f"Expected Output Shape        : {list(expected_shape)}\n")
        f.write(f"Shape Verification Verdict   : {'PASSED' if shape_verified else 'FAILED'}\n\n")
        
        f.write("--- Detailed Layer Parameters Summary ---\n")
        f.write(summary_str)
        f.write("\n")
        
    print(f"Model architecture report successfully saved to: {report_path}")

if __name__ == "__main__":
    run_verification()
