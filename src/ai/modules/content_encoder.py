from typing import Optional

import torch
import torch.nn as nn

from ..modules.wave2vec2 import PretrainedWav2Vec2
from ..modules.vector_quantization import VQEmbeddingEMA

class ContentEncoder(nn.Module):
    
    def __init__(self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        super(ContentEncoder, self).__init__()
        self.pretrained_wav2vec2 = PretrainedWav2Vec2(model_name)
        self.adapt = nn.Linear(110, 512)
        self.vector_quantization = VQEmbeddingEMA(64, 512)
        
    def forward(self, 
                inputs: torch.Tensor,
                speaker_embeddings: torch.Tensor,
                waveform_mask: Optional[torch.Tensor]=None,
                mel_mask: Optional[torch.Tensor]=None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to extract features from raw audio input.
        
        Args:
            inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Extracted feature vectors.
        """
        out = self.pretrained_wav2vec2(inputs, waveform_mask)
        x = self.adapt(out.logits)
        x = torch.softmax(x, dim=-1)
        quantization_output, commitment_loss, perplexity = self.vector_quantization(x)
        return {
            "encoder_features": quantization_output,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity
        }