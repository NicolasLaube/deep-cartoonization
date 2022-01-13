import os
from typing import Optional
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import config
from src.models.utils.parameters import TrainerParams
from src.models.generators.generator_fixed import FixedGenerator
from src.models.discriminators.discriminator_fixed import FixedDiscriminator
from src.models.trainer import Trainer
from src.models.losses import AdversarialLoss, ContentLoss


class FixedCartoonGANTrainer(Trainer):
    def __init__(
        self,
        *args,
        **kargs,
    ) -> None:
        Trainer.__init__(self, *args, **kargs)

    def load_discriminator(self) -> nn.Module:
        return FixedDiscriminator()

    def load_generator(self) -> nn.Module:
        return FixedGenerator()

    def pretrain(
        self,
        *,
        pictures_loader: DataLoader,
        pretrain_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: Optional[str] = config.WEIGHTS_FOLDER,
        batch_callback: Optional[callable] = None,
    ) -> float:
        """Pretrains model"""
        self._init_optimizers(pretrain_params)
        self._set_train_mode()
        self._reset_timer()

        content_loss = ContentLoss().to(self.device)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epoch_start, pretrain_params.epochs + epoch_start):

            for pictures in tqdm(pictures_loader):

                pictures = pictures.to(self.device)

                self.generator.zero_grad()

                with torch.autocast(self.device):

                    gen_cartoons = self.generator(pictures)
                    reconstruction_loss = content_loss(gen_cartoons, pictures)

                scaler.scale(reconstruction_loss).backward()
                scaler.step(self.gen_optimizer)

                self._save_weights(
                    os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
                )

                callback_args = {
                    "epoch": epoch,
                    "losses": {
                        "reconstruction_loss": reconstruction_loss,
                    },
                }
                self._callback(batch_callback, callback_args)

            self._save_model(
                os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
            )

    def train(
        self,
        *,
        pictures_loader: DataLoader,
        cartoons_loader: DataLoader,
        batch_callback: callable,
        train_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: Optional[str] = None,
    ) -> None:
        """Train function"""
        self._init_optimizers(train_params)
        weights_folder = self._init_weight_folder(weights_folder)
        self._set_train_mode()
        self._reset_timer()

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

        # Intialize losses
        content_loss = ContentLoss().to(self.device)
        bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        adversarial_loss = AdversarialLoss(real, fake).to(self.device)

        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epoch_start, train_params.epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, train_params.epochs)

            self.gen_scheduler.step()
            self.disc_scheduler.step()

            for pictures, cartoons in tqdm(
                zip(pictures_loader, cartoons_loader), total=len(pictures_loader)
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
                    disc_fake = self.discriminator(pictures)

                    gen_bce_loss = bce_loss(disc_fake, fake)
                    gen_content_loss = content_loss(gen_cartoons, pictures)
                    gen_loss = gen_bce_loss + gen_content_loss

                scaler.scale(gen_loss).backward()
                scaler.step(self.gen_optimizer)

                scaler.update()

                self._save_weights(
                    os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                    os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
                )

                callback_args = {
                    "epoch": epoch,
                    "losses": {
                        "disc_loss": disc_loss,
                        "content_loss": gen_content_loss,
                        "bce_loss": gen_bce_loss,
                        "gen_loss": gen_loss,
                    },
                }
                self._callback(batch_callback, callback_args)

            self._save_model(
                os.path.join(weights_folder, f"trained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"trained_disc_{epoch}.pkl"),
            )
