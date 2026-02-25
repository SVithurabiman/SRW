import torch
import torch.nn as nn


class EdgeMaskingLayer(nn.Module):
    def __init__(self, device):
        super(EdgeMaskingLayer, self).__init__()

        self.ratio = 0.05
        self.device = device

    def set_params(self, ratio=None):

        if ratio is not None and ratio >= 0 and ratio <= 0.5:
            self.ratio = ratio

    def forward(self, tensor_image):

        B, C, H, W = tensor_image.shape

        # Calculate border sizes
        border_h = int(H * self.ratio)
        border_w = int(W * self.ratio)

        # Create a mask with 1s in the center and 0s at the edges
        mask = torch.ones((H, W), dtype=tensor_image.dtype, device=tensor_image.device)
        mask[:border_h, :] = 0
        mask[-border_h:, :] = 0
        mask[:, :border_w] = 0
        mask[:, -border_w:] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Apply mask to all images in the batch
        return tensor_image * mask  # Mask is broadcast to (B, C, H, W)
