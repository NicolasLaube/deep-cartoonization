from abc import ABC, abstractmethod
from datetime import datetime
import torch
from torch._C import device
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from src.models.utils.params_trainer import TrainerParams
from src import config
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def assertsize(func):
    def wrapper(*args, **kwargs):
        assert len(args[0]) == len(args[1]), "Lengths should be identical"
        return func(*args, **kwargs)


class Trainer(ABC):
    """Generic trainer class"""

    def __init__(self, architecture_params, device: str = "cpu") -> None:
        self.device = device
        self.architecture_params = architecture_params
        self.generator = self.__load_generator()
        self.discriminator = self.__load_discriminator()
        self.writer = SummaryWriter()

        self.generator.to(device)
        self.discriminator.to(device)

        # initialize other variables
        self.gen_optimizer: Optional[optim.Optimizer] = None
        self.disc_optimizer: Optional[optim.Optimizer] = None
        self.gen_scheduler: Optional[optim.Optimizer] = None
        self.disc_scheduler: Optional[optim.Optimizer] = None

    @abstractmethod
    @assertsize
    def train(
        self,
        *,
        pictures_loader: DataLoader,
        cartoons_loader: DataLoader,
        batch_callback: callable,
        train_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: Optional[str] = config.WEIGHTS_FOLDER,
    ) -> None:
        """Train the model"""
        pass

    @abstractmethod
    def pretrain(
        self,
        *,
        pictures_loader: DataLoader,
        batch_callback: callable,
        pretrain_params: TrainerParams,
        epoch_start: int = 0,
        weights_folder: Optional[str] = config.WEIGHTS_FOLDER,
    ) -> None:
        """Pretrain the model"""
        pass

    def save(self, disc_path: str, gen_path: str) -> None:
        """Save the model"""
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)

    def load(self, disc_path: str, gen_path: str) -> None:
        """Load the model from weights"""
        if torch.cuda.is_available():
            self.discriminator.load_state_dict(torch.load(disc_path))
            self.generator.load_state_dict(torch.load(gen_path))
        else:
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=lambda storage, loc: storage)
            )
            self.generator.load_state_dict(
                torch.load(gen_path, map_location=lambda storage, loc: storage)
            )

    @abstractmethod
    def __load_generator(self) -> nn.Module:
        pass

    @abstractmethod
    def __load_discriminator(self) -> nn.Module:
        pass

    def __save_loss(self, step: int, **kargs):
        for key, loss in kargs:
            self.writer.add_scalar(key, loss, global_step=step)

    def __init_timer(self):
        self.last_save = datetime.now()

    def __save_weights(self, gen_path, disc_path):
        if ((datetime.now() - self.last_save).seconds / 60) > config.SAVE_EVERY_MIN:
            self.last_save = datetime.now()
            self.save_model(gen_path, disc_path)

    def __set_train_mode(self):
        """Set model to train mode"""
        self.generator.train()
        self.discriminator.train()

    def __init_optimizers(self, params: TrainerParams):
        """Load optimizers"""
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=params.gen_lr,
            betas=(params.gen_beta1, params.gen_beta2),
        )
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=params.disc_lr,
            betas=(params.disc_beta1, params.disc_beta2),
        )
        self.gen_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.gen_optimizer,
            milestones=[params.epochs // 2, params.epochs // 4 * 3],
            gamma=0.1,
        )
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.disc_optimizer,
            milestones=[params.epochs // 2, params.epochs // 4 * 3],
            gamma=0.1,
        )
