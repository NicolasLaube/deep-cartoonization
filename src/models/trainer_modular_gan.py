"""Modular GAN Trainer"""
# pylint: disable=R0915, E1102, R0914
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.base.base_trainer import Trainer
from src.models.discriminators.discriminator_modular import ModularDiscriminator
from src.models.generators.generator_modular import ModularGenerator
from src.models.losses.loss_content import ContentLoss
from src.models.utils.parameters import (
    ArchitectureParamsModular,
    PretrainingParams,
    TrainingParams,
)


class ModularGANTrainer(Trainer):
    """Modular Cartoon GAN"""

    def __init__(
        self, device: str, architecture_params: ArchitectureParamsModular
    ) -> None:
        self.architecture_params = architecture_params

        Trainer.__init__(self, device=device)

    def load_discriminator(self) -> nn.Module:
        return ModularDiscriminator(
            in_nc=self.architecture_params.nb_channels_disc_input,
            out_nc=self.architecture_params.nb_channels_disc_output,
            nf=self.architecture_params.nb_channels_1st_hidden_layer_disc,
        )

    def load_generator(self) -> nn.Module:
        return ModularGenerator(
            in_nc=self.architecture_params.nb_channels_gen_input,
            out_nc=self.architecture_params.nb_channels_gen_output,
            nf=self.architecture_params.nb_channels_1st_hidden_layer_gen,
            nb=self.architecture_params.nb_resnet_blocks,
        )

    def pretrain(
        self,
        *,
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        pretrain_params: PretrainingParams,
        batch_callback: Optional[Callable] = None,
        validation_callback: Optional[Callable] = None,
        epoch_start: int = 0,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> Any:
        """Pretrains model"""
        self._init_optimizers(pretrain_params, epochs)
        self._set_train_mode()

        content_loss = ContentLoss().to(self.device)

        step = (epoch_start - 1) * len(pictures_loader_train)

        for epoch in range(epoch_start, epochs + epoch_start):
            for picture_batch in tqdm(pictures_loader_train):
                picture_batch = picture_batch.to(self.device)
                self.gen_optimizer.zero_grad()

                cartoons_fake = self.generator(picture_batch)

                # image reconstruction with only L1 loss function
                reconstruction_loss = content_loss(cartoons_fake, picture_batch)

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

            reconstruction_losses: List[float] = []
            with torch.no_grad():
                for pictures in tqdm(pictures_loader_validation):

                    pictures = pictures.to(self.device)

                    self.generator.zero_grad()

                    with self.autocast(self.device):

                        gen_cartoons = self.generator(pictures)
                        reconstruction_loss = content_loss(
                            gen_cartoons, pictures.detach()
                        )

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
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        cartoons_loader_train: DataLoader,
        cartoons_loader_validation: DataLoader,
        train_params: TrainingParams,
        batch_callback: Optional[Callable] = None,
        validation_callback: Optional[Callable[[], Any]] = None,
        epoch_start: int = 0,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> None:
        """Train function"""
        self._init_optimizers(train_params, epochs)
        self._set_train_mode()

        bce_loss = nn.BCELoss().to(self.device)
        content_loss = ContentLoss().to(self.device)

        real = torch.ones(
            cartoons_loader_train.batch_size
            if cartoons_loader_train.batch_size is not None
            else config.DEFAULT_BATCH_SIZE,
            1,
            train_params.input_size // 4,
            train_params.input_size // 4,
        ).to(self.device)
        fake = torch.zeros(
            cartoons_loader_train.batch_size
            if cartoons_loader_train.batch_size is not None
            else config.DEFAULT_BATCH_SIZE,
            1,
            train_params.input_size // 4,
            train_params.input_size // 4,
        ).to(self.device)

        step = (epoch_start - 1) * len(pictures_loader_train)

        for epoch in range(epoch_start, epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, epochs)

            self.generator.train()

            self.gen_scheduler.step()
            self.disc_scheduler.step()

            #############################
            # Train the discriminator #
            #############################

            for pictures, cartoons in tqdm(
                zip(pictures_loader_train, cartoons_loader_train),
                total=len(pictures_loader_train),
            ):

                pictures = pictures.to(self.device)
                cartoons = pictures.to(self.device)
                # e = e.to(self.device)

                ##########################
                # Discriminator training #
                ##########################

                self.disc_optimizer.zero_grad()

                disc_real_cartoons = self.discriminator(cartoons)
                disc_real_cartoon_loss = bce_loss(disc_real_cartoons, real)

                gen_cartoons = self.generator(pictures)
                disc_fake_cartoons = self.discriminator(gen_cartoons)
                disc_fake_cartoon_loss = bce_loss(disc_fake_cartoons, fake)

                # D_edge = self.discriminator(e)
                # disc_edged_cartoon_loss = bce_loss(D_edge, fake)

                disc_loss = (
                    disc_fake_cartoon_loss + disc_real_cartoon_loss
                )  # + disc_edged_cartoon_loss

                disc_loss.backward()
                self.disc_optimizer.step()

                ######################
                # Generator training #
                ######################

                self.gen_optimizer.zero_grad()

                gen_cartoons = self.generator(pictures)
                disc_fake = self.discriminator(gen_cartoons)
                disc_fake_loss = bce_loss(disc_fake, real)

                reconstruction_loss = content_loss(gen_cartoons, pictures)

                gen_loss = (
                    train_params.weight_generator_bce_loss * disc_fake_loss
                    + train_params.weight_generator_content_loss * reconstruction_loss
                )

                gen_loss.backward()
                self.gen_optimizer.step()

                callback_args = {
                    "epoch": epoch,
                    "step": step,
                    "losses": {
                        "disc_loss": disc_loss,
                        "gen_loss": gen_loss,
                    },
                }

                self._callback(batch_callback, callback_args)

                self._save_weights(
                    os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
                )

                step += 1

            ##############################
            # Validation on validation set #
            ##############################

            losses_lists: Dict[str, List[float]] = {
                "disc_loss": [],
                "bce_loss": [],
                "content_loss": [],
                "gen_loss": [],
            }
            with torch.no_grad():
                for pictures, cartoons in tqdm(
                    zip(pictures_loader_validation, cartoons_loader_validation),
                    total=len(pictures_loader_validation),
                ):

                    pictures = pictures.to(self.device)
                    cartoons = cartoons.to(self.device)

                    ############################
                    # Discriminator validation #
                    ############################

                    with self.autocast(self.device):
                        disc_real_cartoons = self.discriminator(cartoons)
                        disc_real_cartoon_loss = bce_loss(disc_real_cartoons, real)

                        gen_cartoons = self.generator(pictures)
                        disc_fake_cartoons = self.discriminator(gen_cartoons)
                        disc_fake_cartoon_loss = bce_loss(disc_fake_cartoons, fake)

                        disc_loss = disc_fake_cartoon_loss + disc_real_cartoon_loss

                    losses_lists["disc_loss"].append(disc_loss.cpu().detach().numpy())

                    ########################
                    # Generator validation #
                    ########################

                    with self.autocast(self.device):
                        generated_image = self.generator(cartoons)
                        disc_fake = self.discriminator(generated_image)
                        disc_fake_loss = bce_loss(disc_fake, real)
                        reconstruction_loss = content_loss(
                            generated_image, pictures.detach()
                        )

                        gen_loss = disc_fake_loss + reconstruction_loss

                    losses_lists["gen_loss"].append(gen_loss.cpu().detach().numpy())

            mean_losses = {
                loss_name: np.mean(values)
                for (loss_name, values) in losses_lists.items()
            }

            callback_args = {
                "epoch": epoch,
                "losses": mean_losses,
            }
            self._callback(validation_callback, callback_args)

            self.save_model(
                os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
            )
