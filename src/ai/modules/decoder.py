from typing import Optional

import torch
import torch.nn as nn

from .resnet_block import Resnet1DBlock, Conv1DBlock
from .sinusoidal_positional_embbeding import SinusoidalPosEmb
from .timestep_embedding import TimestepEmbedding
from .transformer_block import TransformerBlock
from .sampling import DownSampleBlock, UpSampleBlock


class Decoder(nn.Module):
    def __init__(self, in_channels=80, out_channels=80, down_up_channels=(512, 512), n_blocks=2, n_mid=2, cross_attention_dim=256):
        super(Decoder, self).__init__()
        down_up_channels = tuple(down_up_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = down_up_channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim
        )
        
        output_channel = in_channels * 2
        for i in range(len(down_up_channels)):
            input_channel = output_channel
            output_channel = down_up_channels[i]
            is_last = i == len(down_up_channels) - 1
            resnet = Resnet1DBlock(in_channels=input_channel, out_channels=output_channel, time_emb_dim=time_embed_dim)
            transformer_block = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=output_channel,
                        cross_attention_dim=cross_attention_dim,
                    )
                    for _ in range(self.n_blocks)
                ]
            )
            downsample = (
                DownSampleBlock(output_channel, output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, kernel_size=3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_block, downsample]))

        for i in range(n_mid):
            input_channel = down_up_channels[-1]
            out_channels = down_up_channels[-1]
            resnet = Resnet1DBlock(in_channels=input_channel, out_channels=output_channel, time_emb_dim=time_embed_dim)
            transformer_block = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=output_channel,
                        cross_attention_dim=cross_attention_dim,
                    )
                    for _ in range(self.n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_block]))

        down_up_channels = down_up_channels[::-1] + (down_up_channels[0],)
        for i in range(len(down_up_channels) - 1):
            input_channel = down_up_channels[i]
            output_channel = down_up_channels[i + 1]
            is_last = i == len(down_up_channels) - 2

            resnet = Resnet1DBlock(
                in_channels=2 * input_channel,
                out_channels=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_block = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=output_channel,
                        cross_attention_dim=cross_attention_dim,
                    )
                    for _ in range(self.n_blocks)
                ]
            )
            upsample = (
                UpSampleBlock(output_channel, output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_block, upsample]))
        
        self.final_block = Conv1DBlock(down_up_channels[-1], down_up_channels[-1] // 2)
        self.final_proj = nn.Conv1d(down_up_channels[-1] // 2, self.out_channels, 1)
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mu: torch.Tensor, speaker_embeddings: torch.Tensor, time_embed:torch.Tensor, xt: torch.Tensor, mask=None):
        '''
        mask: Tensor, shape (bs, 1, T)
        '''
        speaker_embeddings = speaker_embeddings.unsqueeze(1)
        t = self.time_embeddings(time_embed)
        t = self.time_mlp(t)
        x = torch.cat([mu, xt], dim=-1)
        x = x.permute(0, 2, 1)
        content_hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, t, mask_down)
            x = x.permute(0, 2, 1)
            spks = speaker_embeddings.expand(-1, x.shape[1], -1)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    x,
                    spks,
                    mask_down.unsqueeze(1).expand(-1, 1, spks.shape[1], -1) if mask_down is not None else None
                )
            x = x.permute(0, 2, 1)
            content_hiddens.append(x)
            x = x * mask_down if mask_down is not None else x
            x = downsample(x)
            masks.append(mask_down[:, :, ::2] if mask_down is not None else None)

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, t, mask_mid)
            x = x.permute(0, 2, 1)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    x,
                    spks,
                    mask_down.unsqueeze(1).expand(-1, 1, spks.shape[1], -1) if mask_down is not None else None
                )
            x = x.permute(0, 2, 1)

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            hidden = content_hiddens.pop()
            # T_enc = hidden.shape[-1]
            # T_dec = x.shape[-1]
            # if T_enc > T_dec:
            #     diff = T_enc - T_dec
            #     hidden = hidden[:, :, :-diff]
            #     mask_up = mask_up[:, :, :-diff] if mask_up is not None else None

            x = torch.cat([x, hidden], dim=1)
            x = resnet(x, t, mask_up)
            x = x.permute(0, 2, 1)
            spks = speaker_embeddings.expand(-1, x.shape[1], -1)
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    x,
                    spks,
                    mask_up.unsqueeze(1).expand(-1, 1, spks.shape[1], -1)  if mask_up is not None else None
                )
            x = x.permute(0, 2, 1)
            x = x * mask_up if mask_up is not None else x
            x = upsample(x)

        x = self.final_block(x, mask_up)
        x = x * mask_up if mask_up is not None else x
        x = self.final_proj(x)
        x = x * mask_up if mask_up is not None else x
        return {
            "output": x.permute(0, 2, 1),
            "mask": mask_up,
        }