import torch
import torchvision.transforms.functional as F
import torch.nn as nn


class Rotate(nn.Module):
    def __init__(self, device, angle=0.0):
        super(Rotate, self).__init__()
        self.angle = angle  # Rotation angle in degrees
        self.device = device

    def set_params(self, angle=None):
        if angle is not None:
            self.angle = angle

    def forward(self, img):
        """Rotate the input image tensor by the specified angle."""
        B, C, H, W = img.shape  # Input shape is [B, C, H, W]

        # Rotate each image in the batch
        rotated_imgs = []
        for b in range(B):
            rotated_img = F.rotate(img[b], self.angle)  # Rotate each image individually
            rotated_imgs.append(rotated_img)

        # Stack the rotated images back into a single batch
        rotated_img_batch = torch.stack(rotated_imgs, dim=0)

        return rotated_img_batch
