from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn


num_its = 5
padding_mode = "zeros"


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUp, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels, out_channels, 3, padding=1, padding_mode=padding_mode
            ),
            n_power_iterations=num_its,
        )
        self.norm = nn.GroupNorm(4, out_channels, affine=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip_input):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return torch.cat((x, skip_input), dim=1)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDown, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels, 3, 2, 1, padding_mode=padding_mode
                ),
                n_power_iterations=num_its,
            ),
            nn.GroupNorm(4, out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=1, groups=4
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    padding_mode=padding_mode,
                ),
                n_power_iterations=num_its,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
