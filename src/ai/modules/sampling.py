import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, x_mask=None):
        if x_mask is not None:
            x = x * x_mask
        x = self.conv(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, x_mask=None):
        if x_mask is not None:
            x = x * x_mask

        x = self.conv(x)
        return x