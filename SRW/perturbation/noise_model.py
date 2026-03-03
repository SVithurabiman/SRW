from SRW.pertubation import (
    jpeg,
    salt_and_pepper,
    gaussian_blur,
    colormanipulation,
    cropout,
    dropout,
    edge_crop,
    identity,
    rotate,
    scale,
    flip,
)
import torch
import torch.nn as nn


class NoiseModel(nn.Module):
    def __init__(self, device):
        super(NoiseModel, self).__init__()
        self.device = device
        self.jpeg = jpeg.JpegCompression(self.device, height=128, width=128)
        self.salt = salt_and_pepper.SaltandPepper(self.device)
        self.blur = gaussian_blur.GaussianBlur(self.device)
        self.color = colormanipulation.ColorManipulation(self.device)
        self.crop = cropout.DifferentiableCropout(self.device)
        self.dropout = dropout.DropoutAttack(self.device)
        self.edgecrop = edge_crop.EdgeMaskingLayer(self.device)
        self.identity = identity.Identity()
        self.rotate = rotate.Rotate(self.device)
        self.scale = scale.Scale(self.device)
        self.flip = flip.Flip(self.device)

        self.noise_layers = nn.ModuleList(
            [
                self.jpeg,
                self.blur,
                self.dropout,
                self.edgecrop,
                self.crop,
                self.rotate,
                self.scale,
                self.flip,
            ]
        )

    def random_params(self, layer):
        if isinstance(layer, jpeg.JpegCompression):
            quality = 40  # torch.randint(50, 60, (1,)).item()
            layer.set_params(quality=quality)

        elif isinstance(layer, salt_and_pepper.SaltandPepper):
            amount = (
                torch.rand(1).item() * 0.09 + 0.01
            )  # Random salt and pepper amount in range [0.01, 0.1]
            layer.set_params(amount)

        elif isinstance(layer, gaussian_blur.GaussianBlur):
            sigma = (
                torch.rand(1).item() * 2 + 1.0
            )  # Random blur intensity in range [0.1, 2.0]
            layer.set_params(sigma=sigma)

        elif isinstance(layer, colormanipulation.ColorManipulation):
            brightness = (
                torch.rand(1).item() * 1.3 + 0.5
            )  # Brightness adjustment in range [0.5, 1.5]
            contrast = (
                torch.rand(1).item() * 1.3 + 0.5
            )  # Contrast adjustment in range [0.5, 1.5]
            saturation = (
                torch.rand(1).item() * 1.3 + 0.5
            )  # Saturation adjustment in range [0.5, 1.5]
            hue = (
                torch.rand(1).item() * 0.2 - 0.1
            )  # Hue adjustment in range [-0.1, 0.1]
            layer.set_params(brightness, contrast, saturation, hue)

        elif isinstance(layer, cropout.DifferentiableCropout):
            crop_ratio = torch.rand(1).item() * 0.0 + 0.35
            layer.set_params(crop_ratio=crop_ratio)

        elif isinstance(layer, dropout.DropoutAttack):
            dropout_prob = torch.rand(1).item() * 0.15 + 0.3
            layer.set_params(dropout_prob=dropout_prob)

        elif isinstance(layer, edge_crop.EdgeMaskingLayer):
            ratio = torch.rand(1).item() * 0.05 + 0.03
            layer.set_params(ratio=ratio)

        elif isinstance(layer, rotate.Rotate):
            angle = torch.randint(-30, 30, (1,)).item()
            layer.set_params(angle=angle)

        elif isinstance(layer, scale.Scale):
            scale_ = torch.empty(1).uniform_(0.7, 1.3).item()
            layer.set_params(scale=scale_)

        elif isinstance(layer, flip.Flip):
            flip_type = "horizontal" if torch.rand(1).item() > 0.5 else "vertical"
            layer.set_params(flip_type=flip_type)

        elif isinstance(layer, identity.Identity):
            pass

        return layer

    def forward(self, x):
        
        idx = torch.randint(0, len(self.noise_layers), (1,)).item()
        selected_layer = self.noise_layers[idx]
        selected_layer = self.random_params(selected_layer)
        pertubed_img = selected_layer((x + 1) / 2)
        return 2 * pertubed_img - 1
