import torch
import torch.nn as nn


class SaltandPepper(nn.Module):
    def __init__(self, device, salt_prob=0.01, pepper_prob=0.01):
        super(SaltandPepper, self).__init__()
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.device = device

    def set_params(self, salt_prob=None, pepper_prob=None):
        if salt_prob is not None:
            self.salt_prob = salt_prob
        if pepper_prob is not None:
            self.pepper_prob = pepper_prob

    def forward(self, img):
        salt_mask = (torch.rand_like(img) < self.salt_prob).float().to(self.device)
        pepper_mask = (torch.rand_like(img) < self.pepper_prob).float().to(self.device)
        img_salt = img + salt_mask
        img_pepper = img_salt - pepper_mask

        return img_pepper
