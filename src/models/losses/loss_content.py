from torchvision.models import vgg16
import torch.nn as nn
import gc  # garbage collector


class ContentLoss(nn.Module):
    def __init__(self, omega=10):
        super(ContentLoss, self).__init__()

        self.base_loss = nn.L1Loss()
        self.omega = omega

        perception = list(vgg16(pretrained=True).features)[:25]
        self.perception = nn.Sequential(*perception).eval()

        for param in self.perception.parameters():
            param.requires_grad = False

        gc.collect()

    def forward(self, x1, x2):
        x1 = self.perception(x1)
        x2 = self.perception(x2)

        return self.omega * self.base_loss(x1, x2)
