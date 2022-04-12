import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=2, add_blur=False):
        super(UpBlock, self).__init__()

        self.shuffle = nn.ConvTranspose2d(
            in_f, out_f, kernel_size=3, stride=stride, padding=0)
        self.has_blur = add_blur
        if self.has_blur:
            self.blur = nn.AvgPool2d(2, 1)

    def forward(self, x):
        x = self.shuffle(x)
        if self.has_blur:
            x = self.blur(x)
        return x
