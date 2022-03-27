"""Discrimnator anime"""
from torch import nn
from torch.nn.utils import spectral_norm

from src.models.utils.initialization import initialize_weights


class Discriminator(nn.Module):
    """Discriminator anime"""

    def __init__(self, use_sn=True, d_layers=3):
        super().__init__()
        self.name = "AnimatorGANDiscriminator"
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True),
        ]

        for i in range(d_layers):
            layers += [
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=self.bias,
                ),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    channels * 2,
                    channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.bias,
                ),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias
            ),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if use_sn:
            for i, _ in enumerate(layers):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, img):
        """Forward pass of the discriminator"""
        return self.discriminate(img)
