"""Utils of AnimeGAN"""
# pylint: disable=R0913
import torch.nn.functional as F
from torch import nn

from src.models.utils.initialization import initialize_weights


class DownConv(nn.Module):
    """DownConv is a class that implements a down-sampling block."""

    def __init__(self, channels, bias=False):
        super().__init__()
        self.conv1 = SeparableConv2D(channels, channels, stride=2, bias=bias)
        self.conv2 = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        """Forward pass of the block."""
        out1 = self.conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        out2 = self.conv2(out2)

        return out1 + out2


class UpConv(nn.Module):
    """UpConv is used in the generator."""

    def __init__(self, channels, bias=False):
        super().__init__()
        self.conv = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        """Forward pass of the upconvolution."""
        out = F.interpolate(x, scale_factor=2.0, mode="bilinear")
        out = self.conv(out)

        return out


class SeparableConv2D(nn.Module):
    """Separable Convolution"""

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=bias
        )
        # self.pad =
        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        """Forward pass of the separable convolution"""
        out = self.depthwise(x)
        out = self.ins_norm1(out)
        out = self.activation1(out)

        out = self.pointwise(out)
        out = self.ins_norm2(out)

        return self.activation2(out)


class ConvBlock(nn.Module):
    """ConvBlock with instance normalization and relu activation."""

    def __init__(
        self, channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        """Forward pass of the block."""
        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class InvertedResBlock(nn.Module):
    """Inverted Residual Block with expansion"""

    def __init__(self, channels=256, out_channels=256, expand_ratio=2, bias=False):
        super().__init__()
        bottleneck_dim = round(expand_ratio * channels)
        self.conv_block = ConvBlock(
            channels, bottleneck_dim, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.depthwise_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            groups=bottleneck_dim,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.conv = nn.Conv2d(
            bottleneck_dim, out_channels, kernel_size=1, stride=1, bias=bias
        )

        self.ins_norm1 = nn.InstanceNorm2d(out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        """Forward pass of the inverted residual block."""
        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x
