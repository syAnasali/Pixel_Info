import torch
from torch.utils.data import Dataset
import numpy as np

class FlickrCaptionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Flickr8k image features,
    pre-padded input token sequences, and target words.
    """
    def __init__(self, image_names, input_sequences, target_words, image_features_dict):
        """
        Args:
            image_names: List or array of image file names.
            input_sequences: NumPy array or list of padded sequence indices.
            target_words: NumPy array or list of target word indices.
            image_features_dict: Dictionary mapping image filename to 2048-d feature array.
        """
        self.image_names = image_names
        self.input_sequences = input_sequences
        self.target_words = target_words
        self.image_features_dict = image_features_dict

    def __len__(self):
        return len(self.target_words)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        # Look up image feature embedding vector
        if image_name in self.image_features_dict:
            feature = self.image_features_dict[image_name]
        else:
            # Fallback in case of skipped/corrupted files during extraction
            feature = np.zeros(2048, dtype=np.float32)
            
        input_seq = self.input_sequences[idx]
        target_val = self.target_words[idx]

        # Convert numpy arrays/values to PyTorch Tensors
        image_features_tensor = torch.tensor(feature, dtype=torch.float32)
        input_sequence_tensor = torch.tensor(input_seq, dtype=torch.long)
        # Wrap target_word in shape (1,) as required: Tensor(shape=(1,))
        target_word_tensor = torch.tensor([target_val], dtype=torch.long)

        return {
            "image_features": image_features_tensor,
            "input_sequence": input_sequence_tensor,
            "target_word": target_word_tensor
        }
