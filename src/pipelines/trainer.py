from abc import ABC, abstractmethod
import torch
from torch._C import device
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


def assertsize(func):
    def wrapper(*args, **kwargs):
        assert len(args[0]) == len(args[1]), "Lengths should be identical"
        return func(*args, **kwargs)


class Trainer(ABC):
    """Generic trainer class"""

    def __init__(self, base_params, device: str = "cpu") -> None:
        self.device = device
        self.generator = self.__load_generator()
        self.discriminator = self.__load_discriminator()

        self.generator.to(device)
        self.discriminator.to(device)

    @abstractmethod
    @assertsize
    def train(
        self,
        pictures_loader: DataLoader,
        cartoon_loader: DataLoader,
        batch_callback: callable,
        epoch_start: int = 0,
    ):
        """Train the model"""
        pass

    @abstractmethod
    def pretrain(
        self,
        pictures_loader: DataLoader,
        batch_callback: callable,
        epoch_start: int = 0,
    ):
        """Pretrain the model"""
        pass

    @abstractmethod
    def save(self, disc_path: str, gen_path: str) -> None:
        """Save the model"""
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)

    @abstractmethod
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

    def __load_generator(self) -> nn.Module:
        pass

    def __load_discriminator(self) -> nn.Module:
        pass
