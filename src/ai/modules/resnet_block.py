import torch
import torch.nn as nn


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, kernel_size, stride, padding, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x):
        output = self.block(x)
        return output


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, kernel_size, stride, padding, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, kernel_size, stride, padding, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, kernel_size, stride, padding, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, time_emb):
        h = self.block1(x)
        t = self.mlp(time_emb).unsqueeze(-1)
        h += t
        h = self.block2(h)
        output = h + self.res_conv(x)
        return output