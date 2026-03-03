import torch
import torch.nn as nn
import math


class DifferentiableCropout(nn.Module):
    """
    A PyTorch module that applies a differentiable cropout (soft blackout)
    to an image tensor.
    """

    def __init__(self, crop_ratio=0.3):
        super(DifferentiableCropout, self).__init__()
        # crop_ratio defines the fraction of the image area to be covered.
        self.crop_ratio = crop_ratio

    def set_params(self, crop_ratio=0.35):
        """Allows dynamic adjustment of parameters."""
        if crop_ratio is not None:
            self.crop_ratio = crop_ratio

    def forward(self, img):
        B, C, H, W = img.shape
        area = H * W * self.crop_ratio
        side = math.sqrt(area)
        radius = side / 2

        # --- 2. Random Center for the Patch (Ensuring it stays inside) ---
        # Center coordinates are randomized within safe boundaries: [radius, H/W - radius]
        cy = torch.rand(B, 1, 1) * (H - 2 * radius) + radius
        cx = torch.rand(B, 1, 1) * (W - 2 * radius) + radius

        # --- 3. Calculate Distances from the Center ---
        yy = torch.arange(H).view(1, H, 1)
        xx = torch.arange(W).view(1, 1, W)
        dy = torch.abs(yy - cy)
        dx = torch.abs(xx - cx)

        # --- 4. Create the Hard Binary Mask ---
        # The mask will be 0 inside the patch (Cutout) and 1 outside.
        # This is the non-differentiable step, treated as a constant coefficient.

        # Check if a pixel is INSIDE the square region (distance < radius)
        is_inside_x = dx < radius
        is_inside_y = dy < radius

        # Intersection: True (1.0) only for the pixels inside the square patch
        is_inside_patch = torch.logical_and(is_inside_x, is_inside_y).float()

        # Final mask (1.0 - 1.0 = 0.0 inside; 1.0 - 0.0 = 1.0 outside)
        mask = (1.0 - is_inside_patch).unsqueeze(1)  # (B, 1, H, W)

        # --- 5. Apply the Mask ---
        return img * mask.to(img.device)
