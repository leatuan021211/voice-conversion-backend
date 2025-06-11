import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.group_norm = nn.GroupNorm(groups, out_channels)
        self.mish = nn.Mish()

    def forward(self, x, x_mask=None):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.mish(x)
        return x


class Resnet1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_emb_dim=None, kernel_size=11, stride=1, padding=5, groups=8):
        super(Resnet1DBlock, self).__init__()
        self.mlp = (
            nn.Sequential(
                nn.Mish(),
                torch.nn.Linear(time_emb_dim, out_channels)
            ) if time_emb_dim is not None else None
        )
        self.conv1 = Conv1DBlock(in_channels, out_channels, kernel_size, stride, padding, groups)
        self.conv2 = Conv1DBlock(out_channels, out_channels, kernel_size, stride, padding, groups)
        self.skip_connection = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding=None, x_mask=None):
        residual = self.skip_connection(x)
        x = self.conv1(x, x_mask)

        if time_embedding is not None and self.mlp is not None:
            time_embedding = self.mlp(time_embedding).unsqueeze(-1)
            x += time_embedding

        x = self.conv2(x, x_mask)
        x += residual
        return x