"""VGG 19"""
import torch
from torch import nn


class VGG19(nn.Module):
    """VGG 19"""

    def __init__(
        self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000
    ):
        """Initialize the VGG19 model"""
        super().__init__()
        self.cfg = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ]
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights is None:
            self.load_state_dict(torch.load(init_weights))

    @staticmethod
    def make_layers(cfg, batch_norm=False):
        """Creates the layers of the VGG19 model"""
        layers = []
        in_channels = 3
        for channel in cfg:
            if channel == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, channel, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(channel), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = channel
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the VGG19 model"""
        if self.feature_mode:
            module_list = list(self.features.modules())
            for module in module_list[1:27]:  # conv4_4
                x = module(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
