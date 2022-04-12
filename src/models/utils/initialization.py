"""Weights initialization for networks."""
from torch import nn


def initialize_weights(net):
    """Wieght initialization for networks."""
    for module in net.modules():
        try:
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
        except AttributeError as error:
            print(f"{error} detected in {module} during initialization")
