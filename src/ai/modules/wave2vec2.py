from typing import Optional
import torch
import torch.nn as nn
from transformers.models.wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Config


class PretrainedWav2Vec2(nn.Module):
    """
    PyTorch model to load the pretrained Wav2Vec2 model from Hugging Face.
    """
    
    def __init__(self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        """
        Args:
            model_name (str): Name of the pretrained Wav2Vec2 model from Hugging Face.
        """
        super(PretrainedWav2Vec2, self).__init__()
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC(config=self.config)

    def forward(self, inputs: torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        outputs = self.model(inputs, mask)
        return outputs

