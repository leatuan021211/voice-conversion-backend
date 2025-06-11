import torch
import torch.nn as nn

from ..modules.cross_attention import CrossAttention


class Projection(nn.Module):
    def __init__(self, encoder_out_dim=512, speaker_embed_dim=256, ouput_dim=80, num_heads=64, cross_attention_dim=512):
        super().__init__()
        self.cross_attn = CrossAttention(
            query_dim=speaker_embed_dim,
            kv_dim=encoder_out_dim,
            inner_dim=cross_attention_dim * 4,
            inner_kv_dim=cross_attention_dim * 4,
            out_dim=ouput_dim,
            num_heads=num_heads
        )

    def forward(self, 
                encoded_content: torch.Tensor, 
                speaker_embeddings: torch.Tensor, 
                mask=None):
        speaker_embeddings = speaker_embeddings.unsqueeze(1)
        spks = speaker_embeddings.expand(-1, encoded_content.shape[1], -1)
        attention_mask = mask.unsqueeze(1).expand(-1, 1, spks.shape[1], -1) if mask is not None else None
        mu = self.cross_attn(spks, encoded_content, attention_mask=attention_mask)

        return {
            "mu": mu,
            "mask": mask,
        }