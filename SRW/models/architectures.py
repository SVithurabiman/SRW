from torchvision.models import resnet50
import torch.nn as nn
from torch.nn.utils import spectral_norm
from SRW.models.basicBlocks import UNetDown, UNetUp, ConvBlock
import torch
from kornia.filters import gaussian_blur2d
import torch.nn.functional as F

num_its = 5
padding_mode = "zeros"


def suppress_low_values(tensor, threshold):
    """
    Sets values with absolute magnitude less than threshold to zero.

    Args:
        tensor (torch.Tensor): Tensor of shape (B, C, H, W), values in [-1, 1]
        threshold (float): Threshold for absolute value

    Returns:
        torch.Tensor: Tensor with small-magnitude values zeroed out
    """

    mask = tensor.abs() > threshold
    return tensor * mask


class LinearMessageExpander(nn.Module):
    def __init__(self, message_dim, target_shape):
        super(LinearMessageExpander, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(message_dim, 64), n_power_iterations=num_its)
        self.fc2 = spectral_norm(
            nn.Linear(64, target_shape[0] * target_shape[1]), n_power_iterations=num_its
        )
        self.conv1 = spectral_norm(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode=padding_mode),
            n_power_iterations=num_its,
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, padding_mode=padding_mode),
            n_power_iterations=num_its,
        )
        self.target_shape = target_shape

    def forward(self, message):
        x = F.relu(self.fc1(message))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 1, self.target_shape[0], self.target_shape[1])
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DeeperSpectralUNetEncoderwithFA_ResNet50(nn.Module):
    def __init__(
        self,
        img_channels=3,
        message_channels=1,
        H=128,
        W=128,
        message_dim=30,
        enhance=False,
        s_f=1.0,
        kernel_size=3,
        sigma=1.4,
        threshold=0.02,
    ):
        super().__init__()
        in_channels = img_channels + message_channels
        self.H = H
        self.W = W
        self.enhance = enhance
        self.s_f = s_f
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.threshold = threshold
        self.message_expander = LinearMessageExpander(message_dim, (H, W))

        base_resnet = resnet50(pretrained=True)

        self.res_layers = nn.ModuleList(
            [
                nn.Sequential(
                    base_resnet.conv1, base_resnet.bn1, base_resnet.relu
                ),  # [B,64,H/2,W/2]
                base_resnet.layer1,  # [B,256,H/4,W/4]
                base_resnet.layer2,  # [B,512,H/8,W/8]
                base_resnet.layer3,  # [B,1024,H/16,W/16]
            ]
        )

        for p in self.res_layers.parameters():
            p.requires_grad = False

        self.down1 = UNetDown(in_channels, 64)
        self.down2 = UNetDown(64 + 256, 128)
        self.down3 = UNetDown(128 + 512, 256)
        self.down4 = UNetDown(256 + 1024, 512)

        self.middle = nn.Sequential(
            spectral_norm(
                nn.Conv2d(512, 1024, 3, 2, 1, padding_mode=padding_mode),
                n_power_iterations=num_its,
            ),
            nn.GroupNorm(4, 1024, affine=False),
            nn.ReLU(inplace=True),
        )

        self.up0 = UNetUp(1024, 512)
        self.up1 = UNetUp(1024, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)

        self.final = spectral_norm(
            nn.Conv2d(128, img_channels, kernel_size=1), n_power_iterations=num_its
        )
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, image, message):
        message = message * 2 - 1
        message_expanded = self.message_expander(message)
        x = torch.cat((image, message_expanded), dim=1)

        r1 = self.res_layers[0](image)  # [B,64,H/2,W/2]
        r2 = self.res_layers[1](r1)  # [B,256,H/4,W/4]
        r3 = self.res_layers[2](r2)  # [B,512,H/8,W/8]
        r4 = self.res_layers[3](r3)  # [B,1024,H/16,W/16]

        d1 = self.down1(x)  # 64 channels
        d2 = self.down2(
            torch.cat([d1, r2], 1)
        )  # 64+256=320 channels input, outputs 128
        d3 = self.down3(torch.cat([d2, r3], 1))  # 128+512=640, outputs 256
        d4 = self.down4(torch.cat([d3, r4], 1))  # 256+1024=1280, outputs 512

        m = self.middle(d4)

        u0 = self.up0(m, d4)
        u1 = self.up1(u0, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        encoded = self.final(u3)
        # out = torch.clamp(self.upsample(encoded),-1,1)
        out = self.upsample(encoded)

        if self.enhance and not self.training:
            out = (out + 1) / 2  # Scale to [0, 1]
            out = gaussian_blur2d(
                out,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma=(self.sigma, self.sigma),
            )
            out = 2 * out - 1  # Scale to [-1, 1]
            out = suppress_low_values(
                out, threshold=self.threshold
            )  # Suppress low values

        out = torch.clamp(self.s_f * out + image, -1.0, 1.0)
        return out


class MessageDecoder(nn.Module):
    def __init__(self, input_channels, H, W, message_dim=30):
        super(MessageDecoder, self).__init__()
        self.initial_block = ConvBlock(input_channels, 32, kernel_size=3, stride=1)
        self.conv1 = ConvBlock(32, 32, kernel_size=3, stride=1)
        self.H = H
        self.W = W
        self.fc = spectral_norm(
            nn.Linear(32 * H * W, message_dim), n_power_iterations=num_its
        )

    def forward(self, x):

        x = self.initial_block(x)
        x1 = self.conv1(x)
        x = x + x1  # torch.cat((x, x1), dim=1)
        x = x.contiguous().view(x.size(0), -1)
        message = self.fc(x)
        return message
