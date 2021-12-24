from typing import Any, List, Optional
import numpy as np
from nptyping import NDArray
import torch
import torch.optim as optim
from time import time
import logging
import torch.nn as nn
from torch.optim.optimizer import Optimizer

import src.models.networks as networks
from src.models.networks.generator import Generator
from src.models.networks.discriminator import Discriminator
from src.dataset.cartoon_loader import CartoonDatasetLoader
from src.dataset.pictures_loader import PicturesDatasetLoader
from src.models.parameters import CartoonGanParameters
from src.models.networks.vgg19 import VGG19


class CartoonGan():
    def __init__(self) -> None:
        self.generator = Generator()
        self.discriminator = Discriminator()
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

    def load_weights(self, pretrain_path: str = None, train_path: str = None) -> None:
        pass

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
    
    def __set_eval_mode(self):
        pass

    def pretrain(self,
        *,
        picture_data: PicturesDatasetLoader, 
        parameters: CartoonGanParameters
        ) -> None:
        """Pretrain model"""
        self.__load_optimizers(parameters)
        self.__set_train_mode()

        L1_loss = nn.L1_loss().to(self.device)

        for epoch in range(parameters.epochs):
            epoch_start_time = time.time()
            Recon_losses = []
            for x in picture_data:
                x = x.to(self.device)

                # train generator G
                parameters.generator_optimizer.zero_grad()

                x_feature = self.vgg19((x + 1) / 2)
                generator_ = self.generator(x)
                generator_feature = self.vgg19((generator_ + 1) / 2)

                Recon_loss = 10 * L1_loss(generator_feature, x_feature.detach())
                Recon_losses.append(Recon_loss.item())

                Recon_loss.backward()
                parameters.generator_optimizer.step()

            per_epoch_time = time.time() - epoch_start_time

            logging.info(
                '[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), 
                parameters.epochs, 
                per_epoch_time, 
                torch.mean(torch.FloatTensor(Recon_losses)))
            )


    def train(self, 
        cartoon_data: CartoonDatasetLoader, 
        picture_data: PicturesDatasetLoader, 
        parameters: CartoonGanParameters
        ) -> None:
        """Train function"""
        self.__load_optimizers(parameters)

        L1_loss = nn.L1_loss().to(self.device)
        BCE_loss = nn.BCELoss().to(self.device)

        self.vgg19.eval()

        real = torch.ones(parameters.batch_size, 1, parameters.input_size // 4, parameters.input_size // 4).to(self.device)
        fake = torch.zeros(parameters.batch_size, 1, parameters.input_size // 4, parameters.input_size // 4).to(self.device)

        for epoch in range(parameters.epochs):
            epoch_start_time = time.time()
            self.generator.train()

            parameters.generator_scheduler.step()
            parameters.discriminator_scheduler.step()
            disc_losses = []
            gen_losses = []
            conditional_losses = []
            for picture, cartoon in zip(picture_data, cartoon_data):
                e = cartoon[:, :, :, parameters.input_size:]
                cartoon = cartoon[:, :, :, : parameters.input_size]
                picture, cartoon, e = picture.to(self.device), cartoon.to(self.device), e.to(self.device)

                # Discriminator training
                parameters.discriminator_optimizer.zero_grad()

                disc_real_cartoon = self.discriminator(cartoon)
                disc_real_cartoon_loss = BCE_loss(disc_real_cartoon, real)

                gen_cartoon = self.generator(picture)
                disc_fake_cartoon = self.discriminator(gen_cartoon)
                disc_fake_cartoon_loss = BCE_loss(disc_fake_cartoon, fake)

                D_edge = self.discriminator(e)
                disc_edged_cartoon_loss = BCE_loss(D_edge, fake)

                disc_loss = disc_fake_cartoon_loss + disc_real_cartoon_loss + disc_edged_cartoon_loss
                disc_losses.append(disc_loss.item())

                disc_loss.backward()
                parameters.discriminator_optimizer.step()

                # Generator training
                parameters.generator_optimizer.zero_grad()

                generated_image = self.generator(cartoon)
                disc_fake = self.discriminator(generated_image)
                disc_fake_loss = BCE_loss(disc_fake, real)

                picture_feature = self.vgg19((picture + 1) / 2)
                cartoon_feature = self.vgg19((generated_image + 1) / 2)
                conditionnal_loss = parameters.conditional_lambda * L1_loss(cartoon_feature, picture_feature.detach())

                gen_loss = disc_fake_loss + conditionnal_loss
                gen_losses.append(disc_fake_loss.item())

                conditional_losses.append(conditionnal_loss.item())

                gen_loss.backward()
                parameters.generator_optimizer.step()


            per_epoch_time = time.time() - epoch_start_time
            # train_hist['per_epoch_time'].append(per_epoch_time)
            logging.info('[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % 
                ((epoch + 1), 
                parameters.epochs, 
                per_epoch_time, 
                torch.mean(torch.FloatTensor(disc_losses)),
                torch.mean(torch.FloatTensor(gen_losses)), 
                torch.mean(torch.FloatTensor(conditional_losses))
            ))        

    """def validate(self, train_pictures: PicturesDatasetLoader, test_pictures: PicturesDatasetLoader) -> None:
        with torch.no_grad():
            self.generator.eval()
            for n, (x, _) in enumerate(train_pictures):
                x = x.to(self.device)
                G_recon = self.generator(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer', str(epochs+1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            for picture in enumerate(test_pictures):
                picture = picture.to(self.device)
                G_recon = self.generator(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)           
    """

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
        {networks.print_network(generator)}\n
        {networks.print_network(discriminator)}\n
        {networks.print_network(vgg19)}     
        """
