import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, real_labels, fake_labels):
        super(AdversarialLoss, self).__init__()
        self.cartoon_labels = real_labels
        self.fake_labels = fake_labels
        self.base_loss = nn.BCELoss()

    def forward(self, real_cartoons, gen_cartoons, edged=None):
        # print(cartoon.shape, self.cartoon_labels.shape)
        real_cartoon_loss = self.base_loss(real_cartoons, self.cartoon_labels)
        gen_cartoons_loss = self.base_loss(gen_cartoons, self.fake_labels)
        if edged is not None:
            edge_loss = self.base_loss(edged, self.fake_labels)
            return real_cartoon_loss + gen_cartoons_loss + edge_loss
        return real_cartoon_loss + gen_cartoons_loss
