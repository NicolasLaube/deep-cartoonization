import torch.nn as nn


from src.models.reusable_blocks.convolution_3x3 import conv3x3


NEG_SLOPE = 0.2


class FixedDiscriminator(nn.Module):
    def __init__(self):
        super(FixedDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(3, 32, stride=1),
            nn.LeakyReLU(NEG_SLOPE, inplace=True),
            conv3x3(32, 64, stride=2),
            nn.LeakyReLU(NEG_SLOPE, inplace=True),
            conv3x3(64, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(NEG_SLOPE, inplace=True),
            conv3x3(128, 128, stride=2),
            nn.LeakyReLU(NEG_SLOPE, inplace=True),
            conv3x3(128, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(NEG_SLOPE, inplace=True),
            conv3x3(256, 1, stride=1),  # ??
            nn.Sigmoid(),  # ??
        )

    def forward(self, x):
        x = self.conv(x)
        return x
