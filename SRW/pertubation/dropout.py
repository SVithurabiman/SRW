import torch
import torch.nn as nn


class DropoutAttack(nn.Module):
    def __init__(self, device, dropout_prob=0.3):
        super(DropoutAttack, self).__init__()
        self.dropout_prob = dropout_prob
        self.device = device

    def set_params(self, dropout_prob=None):
        if dropout_prob is not None:
            self.dropout_prob = dropout_prob

    def forward(self, img):
        # img: (B, C, H, W)
        mask = (
            torch.rand(img.shape[0], img.shape[2], img.shape[3], device=img.device)
            > self.dropout_prob
        )
        mask = mask.float().unsqueeze(1)
        return img * mask
