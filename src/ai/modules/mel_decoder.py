from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from diffusers.models.lora import LoRACompatibleLinear

from .resnet_block import ResnetBlock1D
from .sinusoidal_positional_embbeding import SinusoidalPosEmb
from .timestep_embedding import TimestepEmbedding
from .cross_attention import CrossAttention


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fn = SnakeBeta(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(LoRACompatibleLinear(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class DecoderBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size, stride, padding):
        super(DecoderBlock, self).__init__()
        self.resnet_block = ResnetBlock1D(in_channels, out_channels, time_emb_dim, kernel_size, stride, padding)
        self.layernorm1 = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.cross_attention = CrossAttention(embed_size=out_channels)
        self.layernorm2 = nn.LayerNorm(out_channels, elementwise_affine=True)
        self.feed_forward = FeedForward(out_channels, out_channels)

    def forward(self, x, context, t):
        # resnet_block connection
        x = x.permute(0, 2, 1)
        res_out = self.resnet_block(x, t)
        res_out = res_out.permute(0, 2, 1)
        
        # Layer normalization
        norm1 = self.layernorm1(res_out)
        
        # Cross attention
        attn = self.cross_attention(norm1, context)
        
        # Layer normalization
        norm2 = self.layernorm2(attn)
        
        # Feed forward network
        x = self.feed_forward(norm2)
        return x + norm1


class MelDecoder(nn.Module):
    """
        Transformer based Unet decoder.
    """
    
    def __init__(self, n_up_down: int, n_mid: int, in_channels: int, out_channels: int):
        """
        Args:
            n_up_down (int): Number of layers in the decoder.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for convolutional layers.
        """
        super(MelDecoder, self).__init__()
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=in_channels
        )

        self.n_up_down = n_up_down
        self.n_mid = n_mid
        
        self.down_layers = nn.ModuleList()
        self.down_layers.append(DecoderBlock(in_channels, in_channels, in_channels, kernel_size=11, stride=1, padding=5))
        for _ in range(1, n_up_down):
            self.down_layers.append(DecoderBlock(in_channels, in_channels, in_channels, kernel_size=11, stride=1, padding=5))
        
        self.mid_layers = nn.ModuleList()
        for _ in range(n_mid):
            self.mid_layers.append(DecoderBlock(in_channels, in_channels, in_channels, kernel_size=11, stride=1, padding=5))
            
        self.up_layers = nn.ModuleList()
        for _ in range(n_up_down):
            self.up_layers.append(ResnetBlock1D(in_channels * 2, in_channels, in_channels, kernel_size=11, stride=1, padding=5))
            self.up_layers.append(DecoderBlock(in_channels, in_channels, in_channels, kernel_size=11, stride=1, padding=5))
            
        self.out_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=16, stride=1, padding=7)          
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    def forward(self, encoder_output, speaker_embedding, time_step, ut):
        """
        Forward pass through the decoder.
        
        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            speaker_embedding (torch.Tensor): Speaker embedding for conditioning.
        
        Returns:
            torch.Tensor: Output of the decoder.
        """
        
        time_emb = self.time_embeddings(time_step)
        time_emb = self.time_mlp(time_emb)
        
        x = encoder_output

        down_layers_output = []
        for layer in self.down_layers:
            x = layer(x, speaker_embedding, time_emb)
            down_layers_output.append(x)
            
        for layer in self.mid_layers:
            x = layer(x, speaker_embedding, time_emb)
            
        for i, layer in enumerate(self.up_layers):
            if i % 2 == 0:
                x = torch.concat([x, down_layers_output[-(i // 2 + 1)]], dim=2)
                x = x.permute(0, 2, 1)
                x = layer(x, time_emb)
                x = x.permute(0, 2, 1)
            else:
                x = layer(x, speaker_embedding, time_emb)

        cfm_loss = F.mse_loss(x, ut, reduction="sum") / ut.shape[1]
        
        x = x.permute(0, 2, 1)
        x = self.out_conv(x)
        x = x.permute(0, 2, 1)
        return x, cfm_loss