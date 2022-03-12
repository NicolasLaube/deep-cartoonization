from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False
        ),
    )
