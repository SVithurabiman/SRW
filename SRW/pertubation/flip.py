import torch
import torch.nn as nn
import torch.nn.functional as F


class Flip(nn.Module):
    def __init__(self, device):
        super(Flip, self).__init__()
        self.device = device
        self.flip_type = "horizontal"

    def set_params(self, flip_type="horizontal"):
        assert flip_type in ["horizontal", "vertical"]
        self.flip_type = flip_type

    def forward(self, x):
        B = x.size(0)
        if self.flip_type == "horizontal":
            matrix = torch.tensor(
                [[-1, 0, 0], [0, 1, 0]], dtype=torch.float, device=self.device
            )
        else:
            matrix = torch.tensor(
                [[1, 0, 0], [0, -1, 0]], dtype=torch.float, device=self.device
            )
        affine_matrix = matrix.unsqueeze(0).repeat(B, 1, 1)
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
