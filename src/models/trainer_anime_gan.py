"""Trainer Anim GAN"""
# pylint: disable=E0401, R0914
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

# import Variable
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.base.base_trainer import Trainer
from src.models.discriminators.discriminator_anime import Discriminator
from src.models.generators.generator_anim import Generator
from src.models.losses.loss_anime_gan import AnimeGanLoss
from src.models.utils.parameters import (
    ArchitectureParams,
    PretrainingParams,
    TrainingParams,
)


class TrainerAnimeGAN(Trainer):
    """Fixed GAN"""

    def __init__(
        self,
        architecture_params: ArchitectureParams,
        device: str = "cpu",
    ) -> None:
        super().__init__(device)
        self.architecture_params = architecture_params

    def load_discriminator(self) -> nn.Module:
        """Loads discriminator"""
        return Discriminator()

    def load_generator(self) -> nn.Module:
        """Loads generator"""
        return Generator()

    def pretrain(
        self,
        *,
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        pretrain_params: PretrainingParams,
        batch_callback: Optional[Callable] = None,
        validation_callback: Optional[Callable] = None,
        epoch_start: int = 0,
        weights_folder: str = "",
        epochs: int = 10,
    ) -> None:
        raise NotImplementedError("Pretraining is not implemented for Anime GAN")

    # pylint: disable-msg=too-many-statements
    def train(
        self,
        *,
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        cartoons_loader_train: DataLoader,
        cartoons_loader_validation: DataLoader,
        train_params: TrainingParams,
        batch_callback: Optional[Callable[[], Any]] = None,
        validation_callback: Optional[Callable[[], Any]] = None,
        epoch_start: int = 1,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> None:
        """Train function"""
        self._init_optimizers(train_params, epochs)

        self._set_train_mode()
        self._reset_timer()

        # Intialize losses
        loss_fn = AnimeGanLoss(device=self.device)
        # anime gan loss params could be changed in the future

        step = (epoch_start - 1) * len(pictures_loader_train)

        for epoch in range(epoch_start, epochs + epoch_start):

            self.generator.train()

            for pictures, (cartoons, cartoons_gray, cartoons_gray_smothed) in tqdm(
                zip(pictures_loader_train, cartoons_loader_train),
                total=len(pictures_loader_train),
            ):
                # verify that tensor values are in [-1, 1]
                assert (pictures.min() >= -1) and (pictures.max() <= 1)
                assert (cartoons.min() >= -1) and (cartoons.max() <= 1)
                assert (cartoons_gray.min() >= -1) and (cartoons_gray.max() <= 1)
                assert (cartoons_gray_smothed.min() >= -1) and (
                    cartoons_gray_smothed.max() <= 1
                )

                pictures = pictures.to(self.device)
                cartoons = cartoons.to(self.device)
                cartoons_gray = cartoons_gray.to(self.device)
                cartoons_gray_smothed = cartoons_gray_smothed.to(self.device)

                ##########################
                # Discriminator training #
                ##########################

                self.discriminator.zero_grad()
                # for param in self.discriminator.parameters():
                #     param.requires_grad = True

                with self.autocast():
                    gen_cartoons = self.generator(pictures)

                    disc_fake = self.discriminator(gen_cartoons.detach())
                    disc_cartoons = self.discriminator(cartoons)
                    disc_cartoons_gray = self.discriminator(cartoons_gray)
                    disc_cartoons_gray_smoothed = self.discriminator(
                        cartoons_gray_smothed
                    )

                    disc_loss = Variable(
                        loss_fn.compute_loss_discriminator(
                            disc_fake,
                            disc_cartoons,
                            disc_cartoons_gray,
                            disc_cartoons_gray_smoothed,
                        ),
                        requires_grad=True,
                    )

                    disc_loss.backward()
                self.disc_optimizer.step()

                ######################
                # Generator training #
                ######################

                self.generator.zero_grad()
                for param in self.discriminator.parameters():
                    param.requires_grad = False

                with self.autocast():
                    gen_cartoons = self.generator(pictures)

                    disc_fake = self.discriminator(gen_cartoons)

                    (
                        adv_loss,
                        con_loss,
                        gra_loss,
                        col_loss,
                    ) = loss_fn.compute_loss_generator(
                        gen_cartoons, pictures, disc_fake, cartoons_gray
                    )

                    gen_loss = adv_loss + con_loss + gra_loss + col_loss

                    gen_loss.backward()
                self.gen_optimizer.step()

                self._save_weights(
                    os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
                )

                callback_args = {
                    "epoch": epoch,
                    "step": step,
                    "losses": {
                        "disc_loss": disc_loss,
                        "adv_loss": adv_loss,
                        "con_loss": con_loss,
                        "gra_loss": gra_loss,
                        "col_loss": col_loss,
                        "gen_loss": gen_loss,
                    },
                }
                self._callback(batch_callback, callback_args)

                step += 1

            losses_lists: Dict[str, List[float]] = {
                "disc_loss": [],
                "bce_loss": [],
                "content_loss": [],
                "gen_loss": [],
                "adv_loss": [],
                "con_loss": [],
                "gra_loss": [],
                "col_loss": [],
            }
            with torch.no_grad():
                for pictures, (cartoons, cartoons_gray, cartoons_gray_smoothed) in tqdm(
                    zip(pictures_loader_validation, cartoons_loader_validation),
                    total=len(pictures_loader_validation),
                ):

                    pictures = pictures.to(self.device)
                    cartoons = cartoons.to(self.device)
                    cartoons_gray = cartoons_gray.to(self.device)
                    cartoons_gray_smoothed = cartoons_gray_smoothed.to(self.device)

                    ############################
                    # Discriminator validation #
                    ############################

                    with self.autocast():
                        gen_cartoons = self.generator(pictures)

                        disc_fake = self.discriminator(gen_cartoons.detach())
                        disc_cartoons = self.discriminator(cartoons)
                        disc_cartoons_gray = self.discriminator(cartoons_gray)
                        disc_cartoons_gray_smoothed = self.discriminator(
                            cartoons_gray_smoothed
                        )

                        disc_loss = loss_fn.compute_loss_discriminator(
                            disc_fake,
                            disc_cartoons,
                            disc_cartoons_gray,
                            disc_cartoons_gray_smoothed,
                        )
                    losses_lists["disc_loss"].append(disc_loss.cpu().detach().numpy())

                    ########################
                    # Generator validation #
                    ########################

                    with self.autocast():
                        gen_cartoons = self.generator(pictures)

                        disc_fake = self.discriminator(gen_cartoons)

                        (
                            adv_loss,
                            con_loss,
                            gra_loss,
                            col_loss,
                        ) = loss_fn.compute_loss_generator(
                            gen_cartoons, pictures, disc_fake, cartoons_gray
                        )

                        gen_loss = adv_loss + con_loss + gra_loss + col_loss

                    losses_lists["adv_loss"].append(adv_loss.cpu().detach().numpy())
                    losses_lists["con_loss"].append(con_loss.cpu().detach().numpy())
                    losses_lists["gra_loss"].append(gra_loss.cpu().detach().numpy())
                    losses_lists["col_loss"].append(col_loss.cpu().detach().numpy())
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

            self._save_model(
                os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
            )
