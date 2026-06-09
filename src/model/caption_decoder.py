import torch
import torch.nn as nn

class CaptionDecoder(nn.Module):
    """
    Caption Decoder Module for Image Caption Generation.
    
    Processes caption token sequences through an embedding layer and a single-layer LSTM,
    fuses the last-step sequence hidden representation with a 512-dimensional projected
    image embedding via element-wise addition, and projects the fused state to vocabulary logits.
    """
    def __init__(self, vocab_size: int, embed_size: int = 512, hidden_size: int = 512, dropout_rate: float = 0.5):
        """
        Initialize the CaptionDecoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_size: Dimensionality of word embeddings. Default is 512.
            hidden_size: Hidden dimension of LSTM and projection layers. Default is 512.
            dropout_rate: Dropout probability. Default is 0.5.
        """
        super(CaptionDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        # Language Branch
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fusion Layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, image_features: torch.Tensor, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for decoder processing, fusion, and prediction.
        
        Args:
            image_features: Projected image embeddings of shape (batch_size, hidden_size).
            input_sequences: Token sequences of shape (batch_size, sequence_length).
            
        Returns:
            Predicted raw vocabulary logits of shape (batch_size, vocab_size).
        """
        # Shape assertions
        batch_size, seq_len = input_sequences.shape
        assert image_features.shape == (batch_size, self.hidden_size), \
            f"Expected image_features shape (batch_size, {self.hidden_size}), got {image_features.shape}"
            
        # 1. Language Branch Forward
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        embedded = self.embedding(input_sequences)
        embedded = self.dropout(embedded)
        
        # LSTM representation processing
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(embedded)
        
        # Extract output of the final sequence time-step: (batch_size, hidden_size)
        lstm_out_last = lstm_out[:, -1, :]
        lstm_out_last = self.dropout(lstm_out_last)
        
        # 2. Fusion (Element-wise Addition)
        fused = image_features + lstm_out_last
        
        # 3. Output Classification Block
        x = self.fc1(fused)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Final shape assertion
        assert logits.shape == (batch_size, self.vocab_size), \
            f"Expected logits shape (batch_size, {self.vocab_size}), got {logits.shape}"
            
        return logits
