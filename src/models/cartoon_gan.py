import os
from typing import Any, List, Optional
import pickle
import numpy as np
from nptyping import NDArray
import torch
import torch.optim as optim
from time import time
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src import config

from src.models.generators.generator_modular import ModularGenerator
from src.models.discriminators.discriminator_modular import ModularDiscriminator
from src.models.utils.parameters import CartoonGanParameters, Architecture
from src.models.utils.vgg19 import VGG19
from src.models.generators.generator_fixed import FixedGenerator
from src.models.discriminators.discriminator_fixed import FixedDiscriminator
from src.models.generators.generator_unet import UNet

torch.backends.cudnn.benchmark = True


class AdversialLoss(nn.Module):
    def __init__(self, cartoon_labels, fake_labels):
        super(AdversialLoss, self).__init__()
        self.cartoon_labels = cartoon_labels
        self.fake_labels = fake_labels
        self.base_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_cartoon, real_cartoon):
        # print(cartoon.shape, self.cartoon_labels.shape)
        D_cartoon_loss = self.base_loss(fake_cartoon, self.cartoon_labels)
        D_generated_fake_loss = self.base_loss(real_cartoon, self.fake_labels)

        return D_cartoon_loss + D_generated_fake_loss


class CartoonGan:
    def __init__(self, architecture: Architecture = Architecture.FIXED) -> None:

        if architecture == Architecture.MODULAR:
            self.generator = ModularGenerator(
                config.NB_CHANNELS_PICTURE,
                config.NB_CHANNELS_CARTOON,
                config.NB_CHANNELS_1st_HIDDEN_LAYER_GEN,
                config.NB_RESNET_BLOCKS,
            )
            self.discriminator = ModularDiscriminator(
                config.NB_CHANNELS_CARTOON, 1, config.NB_CHANNELS_1st_HIDDEN_LAYER_DISC
            )

        elif architecture == Architecture.FIXED:
            self.generator = FixedGenerator()
            self.discriminator = FixedDiscriminator()
        elif architecture == Architecture.UNET:
            self.generator = UNet()
            self.discriminator = FixedDiscriminator()

        self.vgg19 = VGG19(config.VGG_WEIGHTS, feature_mode=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info("Device: %s", self.device)

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.vgg19.to(self.device)

        # initialize other variables
        self.gen_optimizer: Optional[optim.Optimizer] = None
        self.disc_optimizer: Optional[optim.Optimizer] = None
        self.gen_scheduler: Optional[optim.Optimizer] = None
        self.disc_scheduler: Optional[optim.Optimizer] = None

    def load_model(self, generator_path: str, discriminator_path: str) -> None:
        """Loads the model from path"""
        if torch.cuda.is_available():
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            self.generator.load_state_dict(torch.load(generator_path))
        else:
            self.discriminator.load_state_dict(
                torch.load(
                    discriminator_path, map_location=lambda storage, loc: storage
                )
            )
            self.generator.load_state_dict(
                torch.load(generator_path, map_location=lambda storage, loc: storage)
            )

    def __load_optimizers(self, parameters: CartoonGanParameters):
        """Load optimizers"""
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=parameters.gen_lr,
            betas=(parameters.gen_beta1, parameters.gen_beta2),
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=parameters.disc_lr,
            betas=(parameters.disc_beta1, parameters.disc_beta2),
        )
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.gen_optimizer,
            milestones=[parameters.epochs // 2, parameters.epochs // 4 * 3],
            gamma=0.1,
        )
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.disc_optimizer,
            milestones=[parameters.epochs // 2, parameters.epochs // 4 * 3],
            gamma=0.1,
        )

    def __set_train_mode(self):
        """Set model to train mode"""
        self.generator.train()
        self.discriminator.train()
        # vgg19 isn't trained
        self.vgg19.eval()

    def pretrain(
        self,
        pictures_loader: DataLoader,
        parameters: CartoonGanParameters,
    ) -> None:
        """Pretrains model"""
        self.__load_optimizers(parameters)
        self.__set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)

        last_save_time = datetime.now()

        for epoch in range(parameters.epochs):
            logging.info("Epoch %s/%s", epoch, parameters.epochs)
            epoch_start_time = time()
            reconstruction_losses = []
            for picture_batch in tqdm(pictures_loader):
                picture_batch = picture_batch.to(self.device)
                self.gen_optimizer.zero_grad()

                cartoons_fake = self.generator(picture_batch)

                features_pictures = self.vgg19((picture_batch + 1) / 2)
                features_fake = self.vgg19((cartoons_fake + 1) / 2)

                # image reconstruction with only L1 loss function
                reconstruction_loss = 10 * l1_loss(
                    features_fake, features_pictures.detach()
                )
                reconstruction_losses.append(reconstruction_loss.item())

                reconstruction_loss.backward()
                self.gen_optimizer.step()

                if ((datetime.now() - last_save_time).seconds / 60) > 30:
                    # reset time
                    last_save_time = datetime.now()
                    # save all 30 minutes
                    self.save_model(
                        os.path.join(
                            config.WEIGHTS_FOLDER, f"pretrained_gen_{epoch}.pkl"
                        ),
                        os.path.join(
                            config.WEIGHTS_FOLDER, f"pretrained_disc_{epoch}.pkl"
                        ),
                    )
            self.save_model(
                os.path.join(config.WEIGHTS_FOLDER, f"pretrained_gen_{epoch}.pkl"),
                os.path.join(config.WEIGHTS_FOLDER, f"pretrained_disc_{epoch}.pkl"),
            )
            per_epoch_time = time() - epoch_start_time

            logging.info(
                "[%d/%d] - time: %.2f, Recon loss: %.3f"
                % (
                    (epoch + 1),
                    parameters.epochs,
                    per_epoch_time,
                    torch.mean(torch.FloatTensor(reconstruction_losses)),
                )
            )

    def train(
        self,
        *,
        cartoons_loader: DataLoader,
        pictures_loader: DataLoader,
        parameters: CartoonGanParameters,
    ) -> None:
        """Train function"""

        assert len(cartoons_loader) == len(
            pictures_loader
        ), "Lengths should be identical"

        real = torch.ones(
            cartoons_loader.batch_size,
            1,
            parameters.input_size // 4,
            parameters.input_size // 4,
        ).to(self.device)
        fake = torch.zeros(
            cartoons_loader.batch_size,
            1,
            parameters.input_size // 4,
            parameters.input_size // 4,
        ).to(self.device)

        self.__load_optimizers(parameters)
        self.__set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)
        bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        adv_loss = AdversialLoss(real, fake).to(self.device)

        self.vgg19.eval()
        last_save_time = datetime.now()

        for epoch in range(parameters.epochs):
            logging.info("Epoch %s/%s", epoch, parameters.epochs)

            epoch_start_time = time()
            # self.generator.train()

            # self.gen_scheduler.step()
            # self.disc_scheduler.step()
            disc_losses = []
            gen_losses = []
            conditional_losses = []

            for picture_batch, cartoon_batch in tqdm(
                zip(pictures_loader, cartoons_loader), total=len(pictures_loader)
            ):

                # e = cartoon_batch[:, :, :, parameters.input_size:]
                cartoon_batch = cartoon_batch  # [:, :, :, : parameters.input_size]
                picture_batch = picture_batch.to(self.device)
                cartoon_batch = cartoon_batch.to(self.device)
                # e = e.to(self.device)

                # Discriminator training
                self.disc_optimizer.zero_grad()
                for param in self.discriminator.parameters():  # added
                    param.requires_grad = True  # added

                with torch.cuda.amp.autocast():
                    gen_cartoons = self.generator(picture_batch)
                    disc_fake_cartoons = self.discriminator(gen_cartoons.detach())

                    disc_real_cartoons = self.discriminator(cartoon_batch)
                    disc_loss = adv_loss(disc_fake_cartoons, disc_real_cartoons)

                # D_edge = self.discriminator(e)
                # disc_edged_cartoon_loss = bce_loss(D_edge, fake)

                # + disc_edged_cartoon_loss
                disc_losses.append(disc_loss.item())

                disc_loss.backward()
                self.disc_optimizer.step()

                # Generator training
                self.gen_optimizer.zero_grad()
                for param in self.discriminator.parameters():
                    param.requires_grad = False

                with torch.cuda.amp.autocast():

                    generated_image = self.generator(cartoon_batch)
                    disc_fake = self.discriminator(generated_image)
                    disc_fake_loss = bce_loss(disc_fake, real)

                    picture_features = self.vgg19((picture_batch + 1) / 2)
                    cartoon_features = self.vgg19((generated_image + 1) / 2)
                    conditionnal_loss = parameters.conditional_lambda * l1_loss(
                        cartoon_features, picture_features.detach()
                    )

                gen_loss = disc_fake_loss + conditionnal_loss
                gen_losses.append(disc_fake_loss.item())

                conditional_losses.append(conditionnal_loss.item())

                gen_loss.backward()
                self.gen_optimizer.step()

                if ((datetime.now() - last_save_time).seconds / 60) > 30:
                    # reset time
                    last_save_time = datetime.now()
                    # save all 30 minutes
                    self.save_model(
                        os.path.join(config.WEIGHTS_FOLDER, f"trained_gen_{epoch}.pkl"),
                        os.path.join(
                            config.WEIGHTS_FOLDER, f"trained_disc_{epoch}.pkl"
                        ),
                    )
            self.save_model(
                os.path.join(config.WEIGHTS_FOLDER, f"trained_gen_{epoch}.pkl"),
                os.path.join(config.WEIGHTS_FOLDER, f"trained_disc_{epoch}.pkl"),
            )

            per_epoch_time = time() - epoch_start_time

            logging.info(
                "[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f"
                % (
                    (epoch + 1),
                    parameters.epochs,
                    per_epoch_time,
                    torch.mean(torch.FloatTensor(disc_losses)),
                    torch.mean(torch.FloatTensor(gen_losses)),
                    torch.mean(torch.FloatTensor(conditional_losses)),
                )
            )

    def save_model(self, generator_path: str, discriminator_path: str) -> None:
        """Save a model"""
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def cartoonize_dataset(
        self, pictures_loader: DataLoader
    ) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Cartoonize pictures"""
        cartoons = []
        with torch.no_grad():
            self.generator.eval()
            for pictures_batch in tqdm(pictures_loader):
                pictures_batch = pictures_batch.to(self.device)
                cartoons.extend(self.generator(pictures_batch))

        return cartoons
