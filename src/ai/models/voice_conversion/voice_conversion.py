from sympy import N
import torch
import torch.nn as nn

from core.settings import AIModelSettings
from ...modules.content_encoder import ContentEncoder
from ...modules.projection import Projection
from ...modules.flow_matching import CFMFlow

class VoiceConversion(nn.Module):
    def __init__(self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        """
        Args:
            model_name (str): Name of the pretrained Wav2Vec2 model from Hugging Face.
        """
        super(VoiceConversion, self).__init__()
        self.content_encoder = ContentEncoder(model_name)
        self.projection = Projection()
        self.flow = CFMFlow()

# For training
    # def forward(self, inputs: torch.Tensor, speaker_embedding: torch.Tensor, waveform_mask=None, mel_mask=None, target=None):
    #     """
    #     Forward pass to extract features from raw audio input.
    #     Args:
    #         inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
    #         speaker_embedding (torch.Tensor): Speaker embedding for conditioning.
    #     Returns:
    #         torch.Tensor: Extracted feature vectors.
    #     """
    #     encoded_features = self.content_encoder(inputs, waveform_mask)
    #     if target.shape[1] > encoded_features['encoder_features'].shape[1]:
    #         target = target[:, :encoded_features['encoder_features'].shape[1], :]
    #         mel_mask = mel_mask[:, :, :encoded_features['encoder_features'].shape[1]]
    
    #     proj_output = self.projection(encoded_features["encoder_features"], speaker_embedding, mel_mask)
    #     prior_loss = cal_prior_loss(proj_output['mu'].permute(0, 2, 1), target.permute(0, 2, 1), proj_output['mask'])
    #     # Decode the content features
    #     decoded_features = self.flow(proj_output['mu'], speaker_embedding, target, proj_output['mask'])
    #     recon_loss = F.mse_loss(decoded_features["output"], target)
 
    #     # Compute losses if target is provided
    #     result = {
    #         "encoder_features": encoded_features["encoder_features"],
    #         "commitment_loss": encoded_features["commitment_loss"],
    #         "perplexity": encoded_features["perplexity"],
    #         "decoded_features": decoded_features["output"],
    #         "cfm_loss": decoded_features["cfm_loss"],
    #         "prior_loss": prior_loss,
    #         "recon_loss": recon_loss,
    #         "mask": decoded_features["mask"],
    #     }
        
    #     return result

# For inference
    def synthesis(self, x, speaker_embedding, waveform_mask=None, mel_mask=None,
                n_timesteps=10, temperature=0.667, solver = "euler"):
        encoded_features = self.content_encoder(x, waveform_mask)
        if mel_mask is not None and mel_mask.shape[-1] > encoded_features['encoder_features'].shape[1]:
            mel_mask = mel_mask[:, :, :encoded_features['encoder_features'].shape[1]]
        proj_output = self.projection(encoded_features["encoder_features"], speaker_embedding, mel_mask)
        decoded_features = self.flow(proj_output['mu'], speaker_embedding, n_timesteps, proj_output['mask'], temperature, solver=solver)
        return decoded_features

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> 'VoiceConversion':
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