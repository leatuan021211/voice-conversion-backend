import torch
import torch.nn as nn


from ..modules.cross_attention import CrossAttention
from ..modules.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, dim, cross_attention_dim=256):
        super(TransformerBlock, self).__init__()
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.cross_attention = CrossAttention(query_dim=cross_attention_dim, kv_dim=dim, inner_dim=dim*4, inner_kv_dim=dim*4, out_dim=dim)

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=True)
        self.feed_forward = FeedForward(dim, dim)

    def forward(self, x:torch.Tensor, spks:torch.Tensor, attention_mask:torch.Tensor):

        residual = x
        x = self.norm2(x)
        x = self.cross_attention(spks, x, attention_mask=attention_mask)
        x = residual + x

        x = self.norm3(x)
        x = self.feed_forward(x)

        return residual + x
        