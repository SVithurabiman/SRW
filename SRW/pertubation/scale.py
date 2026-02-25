import torch
import torch.nn as nn
import torch.nn.functional as F

class Scale(nn.Module):
    def __init__(self, device):
        super(Scale, self).__init__()
        self.device = device
        self.scale = 1.0

    def set_params(self, scale=0.9):
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.size()
        scale_matrix = torch.tensor([[self.scale, 0, 0],
                                     [0, self.scale, 0]], dtype=torch.float, device=self.device).unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(scale_matrix, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

