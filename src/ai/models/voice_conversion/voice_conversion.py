import torch
import torch.nn as nn

from ai.modules.content_encoder import ContentEncoder
from ai.modules.cross_attention import CrossAttention
from ai.modules.mel_decoder import MelDecoder
from ai.modules.ot_cfm import ExactOptimalTransportConditionalFlowMatcher
from core.settings import AIModelSettings


class VoiceConversion(nn.Module):
    def __init__(self, model_name: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"):
        """
        Args:
            model_name (str): Name of the pretrained Wav2Vec2 model from Hugging Face.
        """
        super(VoiceConversion, self).__init__()
        self.content_encoder = ContentEncoder(model_name)
        self.encoder_output_dim = 110
        self.embedding_dim = 64
        self.n_embeddings = 256
        self.pre_conv = nn.Conv1d(
            self.encoder_output_dim, self.n_embeddings, kernel_size=1, stride=1)
        self.attn = CrossAttention(self.n_embeddings)
        self.follow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=1e-4)
        self.decoder = MelDecoder(n_up_down=1, n_mid=2, in_channels=256, out_channels=80)
        
    def forward(self, inputs: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features from raw audio input.
        Args:
            inputs (torch.Tensor): Raw audio tensor of shape (batch_size, sequence_length).
            speaker_embedding (torch.Tensor): Speaker embedding for conditioning.
        Returns:
            torch.Tensor: Extracted feature vectors.
        """
        # Extract content features
        content_features = self.content_encoder(inputs)
        
        # Apply pre-quantization convolution
        pre_decode = content_features.permute(0, 2, 1)
        pre_decode = self.pre_conv(pre_decode)
        pre_decode = pre_decode.permute(0, 2, 1)
        
        # Cross-attention
        attention_output = self.attn(pre_decode, speaker_embedding)
        # attention_output = attention_output.permute(0, 2, 1)

        x1 = attention_output
        x0 = torch.randn_like(x1)
        
        t, xt, ut = self.follow_matcher.sample_location_and_conditional_flow(x0, x1)

        # Decode the content features
        decoded_output, cfm_loss = self.decoder(xt, speaker_embedding, t, ut)
        
        return decoded_output, content_features, cfm_loss

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> "VoiceConversion":
        """
        Load the model weights from a checkpoint file.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=AIModelSettings.DEVICE)
        model = cls()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(AIModelSettings.DEVICE)
        model.eval()
        return model