import torch
import torch.nn as nn
from core.settings import AIModelSettings

from ai.modules.hubert import PretrainedHuBERT


class HuBERTVoiceEncoder(nn.Module):
    """
    PyTorch model to load the pretrained HuBERT model from Hugging Face.
    This model extracts features from raw audio input and applies a linear layer followed by ReLU activation.
    """

    def __init__(self, model_path: str = "facebook/hubert-base-ls960"):
        """
        Args:
            model_path (str): Path to the pretrained HuBERT model.
        """
        super(HuBERTVoiceEncoder, self).__init__()
        self.pretrained_hubert = PretrainedHuBERT(model_path)
        self.linear = nn.Linear(in_features=768, out_features=256)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from raw audio input.

        Args:
            inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Extracted feature vectors.
        """
        out = self.pretrained_hubert(inputs)  # shape: (batch_size, sequence_length, feature_dim)q
        embeds_raw = self.relu(self.linear(out.mean(dim=1)))
        
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds 

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "HuBERTVoiceEncoder":
        """
        Load the model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=AIModelSettings.DEVICE)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(AIModelSettings.DEVICE)
        model.eval()
        return model