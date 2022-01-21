import os
from typing import Optional

import torch
from time import time
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src import config
from src.models.generators.generator_fixed import FixedGenerator
from src.models.discriminators.discriminator_fixed import FixedDiscriminator
from src.models.trainer import Trainer
from src.models.utils.parameters import TrainerParams


class FixedCartoonGan(Trainer):
    def __init__(
        self,
        *args,
        **kargs,
    ) -> None:
        Trainer.__init__(*args, **kargs)

    def _Trainer__load_discriminator(self) -> nn.Module:
        return FixedDiscriminator()

    def _Trainer__load_generator(self) -> nn.Module:
        return FixedGenerator()

    def pretrain(
        self,
        *,
        pictures_loader: DataLoader,
        batch_callback: callable,
        pretrain_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: Optional[str] = config.WEIGHTS_FOLDER,
    ) -> float:
        """Pretrains model"""
        self.__init_optimizers(pretrain_params)
        self.__set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)

        last_save_time = datetime.now()

        for epoch in range(epoch_start, pretrain_params.epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, pretrain_params.epochs)
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

                if (
                    (datetime.now() - last_save_time).seconds / 60
                ) > config.SAVE_EVERY_MIN:
                    # reset time
                    last_save_time = datetime.now()
                    # save all n minutes
                    self.save_model(
                        os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                        os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
                    )
            self.save_model(
                os.path.join(weights_folder, f"pretrained_gen_{epoch}.pkl"),
                os.path.join(weights_folder, f"pretrained_disc_{epoch}.pkl"),
            )
            per_epoch_time = time() - epoch_start_time

            logging.info(
                "[%d/%d] - time: %.2f, Recon loss: %.3f"
                % (
                    (epoch + 1),
                    pretrain_params.epochs,
                    per_epoch_time,
                    torch.mean(torch.FloatTensor(reconstruction_losses)),
                )
            )
            return torch.mean(torch.FloatTensor(reconstruction_losses))

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
        self.__set_train_mode()

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

        for epoch in range(epoch_start, train_params.epochs + epoch_start):
            logging.info("Epoch %s/%s", epoch, train_params.epochs)

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
            self.save(
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


"""
def __load_architecture(self, architecture):
        Load GAN architecure

        if architecture == CartoonGanArchitecture.MODULAR:
            return ModularDiscriminator(
                (self.model_parameters.nb_channels_cartoon,),
                1,
                self.model_parameters.nb_channels_1st_hidden_layer_disc,
            ), ModularGenerator(
                (self.model_parameters.nb_channels_picture,),
                (self.model_parameters.nb_channels_cartoon,),
                (self.model_parameters.nb_channels_1st_hidden_layer_gen,),
                (self.model_parameters.nb_resnet_blocks,),
            )

        if architecture == CartoonGanArchitecture.FIXED:
            return FixedDiscriminator(), FixedGenerator()

        return FixedDiscriminator(), UNet()
"""
