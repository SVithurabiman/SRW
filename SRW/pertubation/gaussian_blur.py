import kornia.filters as kf
import torch.nn as nn


class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=11, sigma=3.0):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = int(4 * sigma + 1) | 1

    def set_params(self, sigma=None, kernel_size=None):
        if sigma is not None:
            self.sigma = sigma
            self.kernel_size = int(4 * sigma + 1) | 1
        if kernel_size is not None:
            self.kernel_size = kernel_size

    def forward(self, x):
        return kf.GaussianBlur2d(
            (self.kernel_size, self.kernel_size), (self.sigma, self.sigma)
        )(x)
