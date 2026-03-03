import sys
sys.path.append("./SRW/DiffJPEG")
from DiffJPEG import DiffJPEG
import torch.nn as nn
import torch


class JpegCompression(nn.Module):
    def __init__(self, device, height=128, width=128, quality=50):
        super(JpegCompression, self).__init__()
        self.device = device
        self.height = height
        self.width = width
        self.quality = quality

    def set_params(self, quality=None, height=None, width=None):
        if quality is not None:
            self.quality = quality
        if height is not None:
            self.height = height
        if width is not None:
            self.width = width

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        jpeg = DiffJPEG(
            height=self.height,
            width=self.width,
            differentiable=True,
            quality=self.quality,
        ).to(self.device)
        return torch.clamp(jpeg(x), 0, 1)
