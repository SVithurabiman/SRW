import torch.nn as nn
import torchvision.transforms.functional as TF


class ColorManipulation(nn.Module):
    def __init__(self, device, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super(ColorManipulation, self).__init__()
        self.device = device
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def set_params(self, brightness=None, contrast=None, saturation=None, hue=None):
        if brightness is not None:
            self.brightness = brightness
        if contrast is not None:
            self.contrast = contrast
        if saturation is not None:
            self.saturation = saturation
        if hue is not None:
            self.hue = hue

    def forward(self, img):
        if self.brightness > 0:
            img = TF.adjust_brightness(img, self.brightness)

        if self.contrast > 0:
            img = TF.adjust_contrast(img, self.contrast)

        if self.saturation > 0:
            img = TF.adjust_saturation(img, self.saturation)

        if self.hue > 0:
            img = TF.adjust_hue(img, self.hue)

        return img
