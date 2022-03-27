"""Loss anime gan"""
# pylint: disable=R0902, R0913, E1102
import torch
import torch.nn.functional as F
from torch import nn

from src.models.losses.loss_color import ColorLoss
from src.models.vgg.vgg19_anime import Vgg19
from src.preprocessing.anime.gram import gram


class AnimeGanLoss:
    """Animation Loss"""

    def __init__(
        self,
        wadvg: float = 10.0,
        wadvd: float = 10.0,
        wcon: float = 1.5,
        wgra: float = 3.0,
        wcol: float = 30.0,
        gan_loss: str = "lsgan",
        device: str = "cpu",
    ):
        self.device = device

        self.vgg19 = Vgg19().to(self.device).eval()
        self.content_loss = nn.L1Loss().to(self.device)
        self.color_loss = ColorLoss(self.device).to(self.device)
        self.gram_loss = nn.L1Loss().to(self.device)

        self.wadvg = wadvg
        self.wadvd = wadvd
        self.wcon = wcon
        self.wgra = wgra
        self.wcol = wcol
        self.adv_type = gan_loss
        self.bce_loss = nn.BCELoss().to(self.device)

    def compute_loss_generator(self, fake_img, img, fake_logit, anime_gray):
        """
        Compute loss for Generator
        @Arugments:
            - fake_img: generated image
            - img: image
            - fake_logit: output of Discriminator given fake image
            - anime_gray: grayscale of anime image
        @Returns:
            loss
        """
        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img).detach()

        return [
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra
            * self.gram_loss(
                gram(anime_feat).to(self.device), gram(fake_feat).to(self.device)
            ),
            self.wcol * self.color_loss(img, fake_img).to(self.device),
        ]

    def compute_loss_discriminator(
        self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d
    ):
        """Compute loss for Discriminator"""
        return self.wadvd * (
            self.adv_loss_d_real(real_anime_d)
            + self.adv_loss_d_fake(fake_img_d)
            + self.adv_loss_d_fake(real_anime_gray_d)
            + 0.2 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )

    def content_loss_vgg(self, image, recontruction):
        """Content loss for VGG"""
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)

    def adv_loss_d_real(self, pred):
        """Adversarial loss for Discriminator"""
        if self.adv_type == "hinge":
            return torch.mean(F.relu(1.0 - pred))

        if self.adv_type == "lsgan":
            return torch.mean(torch.square(pred - 1.0))

        if self.adv_type == "normal":
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f"Do not support loss type {self.adv_type}")

    def adv_loss_d_fake(self, pred):
        """Adversarial loss for Discriminator"""
        if self.adv_type == "hinge":
            return torch.mean(F.relu(1.0 + pred))

        if self.adv_type == "lsgan":
            return torch.mean(torch.square(pred))

        if self.adv_type == "normal":
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f"Do not support loss type {self.adv_type}")

    def adv_loss_g(self, pred):
        """Adversarial loss for Generator"""
        if self.adv_type == "hinge":
            return -torch.mean(pred)

        if self.adv_type == "lsgan":
            return torch.mean(torch.square(pred - 1.0))

        if self.adv_type == "normal":
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f"Do not support loss type {self.adv_type}")
