"""Fixed GAN Trainer"""
import logging
import os
from datetime import datetime
from time import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.base.base_trainer import Trainer
from src.models.discriminators.discriminator_modular import ModularDiscriminator
from src.models.generators.generator_modular import ModularGenerator
from src.models.utils.parameters import ArchitectureParams, TrainerParams
from src.models.utils.vgg19 import VGG19


class ModularGANTrainer(Trainer):
    """Fixed Cartoon GAN"""

    def __init__(self, device: str, architecture_params: ArchitectureParams) -> None:
        self.architecture_params = architecture_params
        self.vgg19 = VGG19()

        Trainer.__init__(self, device=device)

    def load_discriminator(self) -> nn.Module:
        return ModularDiscriminator(
            in_nc=self.architecture_params.nb_channels_cartoon,
            out_nc=self.architecture_params.nb_channels_1st_hidden_layer_disc,
        )

    def load_generator(self) -> nn.Module:
        return ModularGenerator(
            in_nc=self.architecture_params.nb_channels_picture,
            out_nc=self.architecture_params.nb_channels_1st_hidden_layer_gen,
        )

    def pretrain(
        self,
        *,
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        batch_callback: Callable[[], Any],
        validation_callback: Callable[[], Any],
        pretrain_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> Any:
        """Pretrains model"""
        self._init_optimizers(pretrain_params, epochs)
        self._set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)

        for epoch in range(epoch_start, epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, epochs)
            reconstruction_losses = []
            for picture_batch in tqdm(pictures_loader_train):
                picture_batch = picture_batch.to(self.device)
                self.gen_optimizer.zero_grad()

                cartoons_fake = self.generator(picture_batch)

                features_pictures = self.vgg19((picture_batch + 1) / 2)
                features_fake = self.vgg19((cartoons_fake + 1) / 2)

                # image reconstruction with only L1 loss function
                reconstruction_loss = 10 * l1_loss(
                    features_fake, features_pictures.detach()
                )

                reconstruction_loss.backward()
                self.gen_optimizer.step()

                self._save_weights(
                    os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
                )

                callback_args = {
                    "epoch": epoch,
                    "step": step,
                    "losses": {
                        "reconstruction_loss": reconstruction_loss,
                    },
                }
                self._callback(batch_callback, callback_args)

                step += 1

            reconstruction_losses = []
            with torch.no_grad():
                for pictures in tqdm(pictures_loader_validation):

                    pictures = pictures.to(self.device)

                    self.generator.zero_grad()

                    with torch.autocast(self.device):

                        gen_cartoons = self.generator(pictures)
                        reconstruction_loss = 10 * l1_loss(gen_cartoons, pictures)

            mean_reconstruction_loss = np.mean(reconstruction_losses)

            callback_args = {
                "epoch": epoch,
                "losses": {
                    "reconstruction_loss": mean_reconstruction_loss,
                },
            }

            self._callback(validation_callback, callback_args)

            self._save_model(
                os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
            )

    def train(
        self,
        *,
        pictures_loader: DataLoader,
        cartoons_loader: DataLoader,
        batch_callback: Callable[[], Any],
        train_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> None:
        """Train function"""
        self._init_optimizers(train_params, epochs)
        weights_folder = self._init_weight_folder(weights_folder)
        self._set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)
        bce_loss = nn.BCELoss().to(self.device)

        self.vgg19.eval()
        last_save_time = datetime.now()

        real = torch.ones(
            cartoons_loader.batch_size,
            1,
            train_params.input_size // 4,
            train_params.input_size // 4,
        ).to(self.device)
        fake = torch.zeros(
            cartoons_loader.batch_size,
            1,
            train_params.input_size // 4,
            train_params.input_size // 4,
        ).to(self.device)

        for epoch in range(epoch_start, epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, epochs)

            epoch_start_time = time()
            self.generator.train()

            self.gen_scheduler.step()
            self.disc_scheduler.step()
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

                disc_real_cartoons = self.discriminator(cartoon_batch)
                disc_real_cartoon_loss = bce_loss(disc_real_cartoons, real)

                gen_cartoons = self.generator(picture_batch)
                disc_fake_cartoons = self.discriminator(gen_cartoons)
                disc_fake_cartoon_loss = bce_loss(disc_fake_cartoons, fake)

                # D_edge = self.discriminator(e)
                # disc_edged_cartoon_loss = bce_loss(D_edge, fake)

                disc_loss = (
                    disc_fake_cartoon_loss + disc_real_cartoon_loss
                )  # + disc_edged_cartoon_loss
                disc_losses.append(disc_loss.item())

                disc_loss.backward()
                self.disc_optimizer.step()

                # Generator training
                self.gen_optimizer.zero_grad()

                generated_image = self.generator(cartoon_batch)
                disc_fake = self.discriminator(generated_image)
                disc_fake_loss = bce_loss(disc_fake, real)

                picture_features = self.vgg19((picture_batch + 1) / 2)
                cartoon_features = self.vgg19((generated_image + 1) / 2)
                conditionnal_loss = train_params.conditional_lambda * l1_loss(
                    cartoon_features, picture_features.detach()
                )

                gen_loss = disc_fake_loss + conditionnal_loss
                gen_losses.append(disc_fake_loss.item())

                conditional_losses.append(conditionnal_loss.item())

                gen_loss.backward()
                self.gen_optimizer.step()
                self._callback(batch_callback)

                if (
                    (datetime.now() - last_save_time).seconds / 60
                ) > config.SAVE_EVERY_MIN:
                    # reset time
                    last_save_time = datetime.now()
                    # save all n minutes
                    self.save_model(
                        os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                        os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
                    )
            self.save_model(
                os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
            )

            per_epoch_time = time() - epoch_start_time

            logging.info(
                "[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f"
                % (
                    (epoch + 1),
                    weights_folder.epochs,
                    per_epoch_time,
                    torch.mean(torch.FloatTensor(disc_losses)),
                    torch.mean(torch.FloatTensor(gen_losses)),
                    torch.mean(torch.FloatTensor(conditional_losses)),
                )
            )
        return {
            "discriminator_loss": torch.mean(torch.FloatTensor(disc_losses)),
            "generator_loss": torch.mean(torch.FloatTensor(gen_losses)),
            "conditional_loss": torch.mean(torch.FloatTensor(conditional_losses)),
        }
