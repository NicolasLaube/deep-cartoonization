"""Fixed GAN"""
# pylint: disable=E1102,R0914
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.base.base_trainer import Trainer
from src.models.discriminators.discriminator_fixed import FixedDiscriminator
from src.models.generators.generator_fixed import FixedGenerator
from src.models.losses import AdversarialLoss, ContentLoss
from src.models.utils.parameters import PretrainingParams, TrainingParams


class FixedCartoonGANTrainer(Trainer):
    """Fixed GAN"""

    def __init__(
        self,
        *args,
        **kargs,
    ) -> None:
        Trainer.__init__(self, *args, **kargs)

    def load_discriminator(self) -> nn.Module:
        """Loads discriminator"""
        return FixedDiscriminator()

    def load_generator(self) -> nn.Module:
        """Loads generator"""
        return FixedGenerator()

    def pretrain(
        self,
        *,
        pictures_loader_train: DataLoader,
        pictures_loader_validation: DataLoader,
        pretrain_params: PretrainingParams,
        batch_callback: Optional[Callable] = None,
        validation_callback: Optional[Callable] = None,
        epoch_start: int = 1,
        weights_folder: str = config.WEIGHTS_FOLDER,
        epochs: int = 10,
    ) -> None:
        """Pretrains model"""
        self._init_optimizers(pretrain_params, epochs)
        self._set_train_mode()
        self._reset_timer()

        content_loss = ContentLoss().to(self.device)
        scaler = torch.cuda.amp.GradScaler()

        step = (epoch_start - 1) * len(pictures_loader_train)

        for epoch in range(epoch_start, epochs + epoch_start):

            for pictures in tqdm(pictures_loader_train):

                pictures = pictures.to(self.device)

                self.generator.zero_grad()

                with torch.autocast(self.device):

                    gen_cartoons = self.generator(pictures)
                    reconstruction_loss = content_loss(gen_cartoons, pictures)

                scaler.scale(reconstruction_loss).backward()
                scaler.step(self.gen_optimizer)
                scaler.update()

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
                        reconstruction_loss = content_loss(gen_cartoons, pictures)

                    reconstruction_losses.append(
                        reconstruction_loss.cpu().detach().numpy()
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

        # Intialize losses
        content_loss = ContentLoss().to(self.device)
        bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        adversarial_loss = AdversarialLoss(real, fake).to(self.device)

        scaler = torch.cuda.amp.GradScaler()

        step = (epoch_start - 1) * len(pictures_loader_train)

        for epoch in range(epoch_start, epochs + epoch_start):

            self.gen_scheduler.step()
            self.disc_scheduler.step()

            for pictures, cartoons in tqdm(
                zip(pictures_loader_train, cartoons_loader_train),
                total=len(pictures_loader_train),
            ):

                pictures = pictures.to(self.device)
                cartoons = cartoons.to(self.device)

                ##########################
                # Discriminator training #
                ##########################

                self.discriminator.zero_grad()
                for param in self.discriminator.parameters():
                    param.requires_grad = True

                with torch.autocast(self.device):
                    gen_cartoons = self.generator(pictures)

                    disc_fake = self.discriminator(gen_cartoons.detach())
                    disc_true = self.discriminator(cartoons)

                    disc_loss = adversarial_loss(disc_true, disc_fake)

                scaler.scale(disc_loss).backward()
                scaler.step(self.disc_optimizer)

                ######################
                # Generator training #
                ######################

                self.generator.zero_grad()
                for param in self.discriminator.parameters():
                    param.requires_grad = False

                with torch.autocast(self.device):
                    disc_fake = self.discriminator(gen_cartoons)

                    gen_bce_loss = bce_loss(disc_fake, real)
                    gen_content_loss = content_loss(gen_cartoons, pictures)
                    gen_loss = (
                        train_params.weight_generator_bce_loss * gen_bce_loss
                        + train_params.weight_generator_content_loss * gen_content_loss
                    )

                scaler.scale(gen_loss).backward()
                scaler.step(self.gen_optimizer)

                scaler.update()

                self._save_weights(
                    os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
                )

                callback_args = {
                    "epoch": epoch,
                    "step": step,
                    "losses": {
                        "disc_loss": disc_loss,
                        "content_loss": gen_content_loss,
                        "bce_loss": gen_bce_loss,
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

                    with torch.autocast(self.device):
                        gen_cartoons = self.generator(pictures)

                        disc_fake = self.discriminator(gen_cartoons.detach())
                        disc_true = self.discriminator(cartoons)

                        disc_loss = adversarial_loss(disc_true, disc_fake)
                    losses_lists["disc_loss"].append(disc_loss.cpu().detach().numpy())

                    ########################
                    # Generator validation #
                    ########################

                    with torch.autocast(self.device):
                        disc_fake = self.discriminator(pictures)

                        gen_bce_loss = bce_loss(disc_fake, fake)
                        gen_content_loss = content_loss(gen_cartoons, pictures)
                        gen_loss = gen_bce_loss + gen_content_loss
                    losses_lists["bce_loss"].append(gen_bce_loss.cpu().detach().numpy())
                    losses_lists["content_loss"].append(
                        gen_content_loss.cpu().detach().numpy()
                    )
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
