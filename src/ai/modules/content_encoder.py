import torch
import torch.nn as nn

from .wave2vec2 import PretrainedWav2Vec2


class ContentEncoder(nn.Module):
    """
    PyTorch model to load the pretrained Wav2Vec2 model from Hugging Face.
    This model extracts features from raw audio input and applies a linear layer followed by ReLU activation.
    """
    
    def __init__(self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        """
        Args:
            model_name (str): Name of the pretrained Wav2Vec2 model from Hugging Face.
        """
        super(ContentEncoder, self).__init__()
        self.pretrained_wav2vec2 = PretrainedWav2Vec2(model_name)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from raw audio input.
        
        Args:
            inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Extracted feature vectors.
        """
        out = self.pretrained_wav2vec2(inputs)
        return out