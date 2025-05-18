import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor

class PretrainedHuBERT(nn.Module):
    """
    PyTorch model to load the pretrained HuBERT model from Hugging Face.
    """

    def __init__(self, model_path: str = "facebook/hubert-base-ls960"):
        """
        Args:
            model_path (str): Path to the pretrained HuBERT model.
        """
        super(PretrainedHuBERT, self).__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)

    def forward(self, inputs: torch.Tensor, sampling_rate: int=16000) -> torch.Tensor:
        """
        Forward pass to extract features from raw audio input.

        Args:
            raw_audio (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
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
        ).input_values  # shape: (batch_size, sequence_length)

        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        return outputs.last_hidden_state  # shape: (batch_size, sequence_length, feature_dim)