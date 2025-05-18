import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


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
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.freeze_feature_extractor()
        
    def forward(self, inputs: torch.Tensor, sampling_rate: int=16000) -> torch.Tensor:
        """
        Forward pass to extract features from raw audio input.
        Args:
            inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
            sampling_rate (int): Sampling rate of the input audio.
        Returns:
            torch.Tensor: Extracted feature vectors.
        """ 
        # Ensure input is 2D: (batch_size, sequence_length)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        
        if isinstance(inputs, torch.Tensor):
            inputs = [audio.cpu().numpy() for audio in inputs]
        
        input_values = self.processor(
            inputs,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        ).input_values
        
        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        return outputs.logits
