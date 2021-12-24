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

from src import config
import src.models.networks as networks
from src.models.networks.generator import Generator
from src.models.networks.discriminator import Discriminator
from src.models.parameters import CartoonGanParameters
from src.models.networks.vgg19 import VGG19


class CartoonGan():
    def __init__(self) -> None:
        self.generator = Generator(3, 3)
        self.discriminator = Discriminator(3, 1)
        self.vgg19 = VGG19()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.vgg19.to(self.device)

        # initialize other variables
        self.gen_optimizer: Optional[optim.Optimizer] = None
        self.disc_optimizer: Optional[optim.Optimizer] = None
        self.gen_scheduler: Optional[optim.Optimizer] = None
        self.disc_scheduler: Optional[optim.Optimizer] = None

    def __load_optimizers(self, parameters: CartoonGanParameters):
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), 
            lr=parameters.gen_lr, 
            betas=(parameters.gen_beta1, parameters.gen_beta2)
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), 
            lr=parameters.disc_lr,
            betas=(parameters.disc_beta1, parameters.disc_beta2)
        )
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.gen_optimizer, 
            milestones=[parameters.epochs // 2, parameters.epochs // 4 * 3], 
            gamma=0.1
        )
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.disc_optimizer, 
            milestones=[parameters.epochs // 2, parameters.epochs // 4 * 3],
            gamma=0.1
        )

    def __set_train_mode(self):
        """Set model to train mode"""
        self.generator.train()
        self.discriminator.train()
        # vgg19 isn't trained
        self.vgg19.eval()

    def pretrain(self,
        *,
        pictures_loader: DataLoader, 
        parameters: CartoonGanParameters
        ) -> None:
        """Pretrains model"""
        self.__load_optimizers(parameters)
        self.__set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)

        for epoch in range(parameters.epochs):
            epoch_start_time = time()
            reconstruction_losses = []
            for picture_batch in pictures_loader:
                picture_batch = picture_batch.to(self.device)
                self.gen_optimizer.zero_grad()

                fake_cartoon = self.generator(picture_batch)

                picture_features = self.vgg19((picture_batch + 1) / 2)
                fake_features = self.vgg19((fake_cartoon + 1) / 2)

                # image reconstruction with only L1 loss function
                reconstruction_loss = 10 * l1_loss(fake_features, picture_features.detach())
                reconstruction_losses.append(reconstruction_loss.item())

                reconstruction_loss.backward()
                self.gen_optimizer.step()

            per_epoch_time = time() - epoch_start_time

            logging.info(
                '[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), 
                parameters.epochs, 
                per_epoch_time,
                torch.mean(torch.FloatTensor(reconstruction_losses)))
            )

            self.save_model(os.path.join(config.WEIGHTS_FOLDER, f"pretrained_gen_{epoch}.pkl"), os.path.join(config.WEIGHTS_FOLDER, f"pretrained_disc_{epoch}.pkl"))


    def train(self,
        *,
        cartoon_loader: DataLoader, 
        picture_loader: DataLoader, 
        parameters: CartoonGanParameters
        ) -> None:
        """Train function"""
        assert len(cartoon_loader) == len(picture_loader), "Lengths should be identical"
        
        self.__load_optimizers(parameters)
        self.__set_train_mode()

        l1_loss = nn.L1Loss().to(self.device)
        bce_loss = nn.BCELoss().to(self.device)

        self.vgg19.eval()

        real = torch.ones(cartoon_loader.batch_size, 1, parameters.input_size // 4, parameters.input_size // 4).to(self.device)
        fake = torch.zeros(cartoon_loader.batch_size, 1, parameters.input_size // 4, parameters.input_size // 4).to(self.device)

        for epoch in range(parameters.epochs):
            epoch_start_time = time()
            self.generator.train()

            parameters.generator_scheduler.step()
            parameters.discriminator_scheduler.step()
            disc_losses = []
            gen_losses = []
            conditional_losses = []
            for picture_batch, cartoon_batch in zip(picture_loader, cartoon_loader):

                # e = cartoon_batch[:, :, :, parameters.input_size:]
                cartoon_batch = cartoon_batch[:, :, :, : parameters.input_size]
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

                disc_loss = disc_fake_cartoon_loss + disc_real_cartoon_loss # + disc_edged_cartoon_loss
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
                conditionnal_loss = parameters.conditional_lambda * l1_loss(cartoon_features, picture_features.detach())

                gen_loss = disc_fake_loss + conditionnal_loss
                gen_losses.append(disc_fake_loss.item())

                conditional_losses.append(conditionnal_loss.item())

                gen_loss.backward()
                self.gen_optimizer.step()

            per_epoch_time = time() - epoch_start_time

            logging.info('[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % 
                ((epoch + 1), 
                parameters.epochs, 
                per_epoch_time, 
                torch.mean(torch.FloatTensor(disc_losses)),
                torch.mean(torch.FloatTensor(gen_losses)), 
                torch.mean(torch.FloatTensor(conditional_losses))
            ))

    def save_model(self, generator_path: str, discriminator_path: str) -> None:
        """Save a model"""
        # torch.save(G.state_dict(), os.path.join(args.name + '_results', 'generator_latest.pkl'))
        # torch.save(D.state_dict(), os.path.join(args.name + '_results', 'discriminator_latest.pkl'))
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def cartoonize(self, pictures: List[NDArray[(3, Any, Any)]]) -> List[NDArray[(3, Any, Any), np.int32]]:
        """Predict """

    def __repr__(self) -> str:
        """Prints the model's architecture"""
        return f"""-------- Networks ------- \n
        {networks.print_network(self.generator)}\n
        {networks.print_network(self.discriminator)}\n
        {networks.print_network(self.vgg19)}     
        """
