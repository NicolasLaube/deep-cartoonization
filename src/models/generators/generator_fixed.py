import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.reusable_blocks.up_block import UpBlock
from src.models.reusable_blocks.resnet_block import add_resblocks
from src.models.reusable_blocks.convolution_3x3 import conv3x3


class FixedGenerator(nn.Module):
    def __init__(self):
        super(FixedGenerator, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=7 // 2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128, stride=2),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv3x3(128, 256, stride=2),
            conv3x3(256, 256, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.res = nn.Sequential(add_resblocks(256, 8))

        self.up = nn.Sequential(
            UpBlock(256, 128, stride=2, add_blur=True),
            conv3x3(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            UpBlock(128, 64, stride=2, add_blur=True),
            conv3x3(64, 64, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=7 // 2),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return x
