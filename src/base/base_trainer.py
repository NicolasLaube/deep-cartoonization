"""Base Trainer class"""
# pylint: disable=R0902
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from src import config
from src.models.utils.parameters import (
    BaseTrainingParams,
    PretrainingParams,
    TrainingParams,
)


class Trainer(ABC):
    """Generic trainer class"""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.generator = self.load_generator()
        self.discriminator = self.load_discriminator()

        self.last_save_time = datetime.now()

        self.generator.to(device)
        self.discriminator.to(device)

        # initialize other variables
        self.gen_optimizer: optim.Optimizer = None  # type: ignore
        self.disc_optimizer: optim.Optimizer = None  # type: ignore
        self.gen_scheduler: optim.lr_scheduler.MultiStepLR = None  # type: ignore
        self.disc_scheduler: optim.lr_scheduler.MultiStepLR = None  # type: ignore

    @abstractmethod
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
        epochs: int = 10
    ) -> None:
        """Train the model"""

    @abstractmethod
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
        epochs: int = 10
    ) -> None:
        """Pretrain the model"""

    def save_model(
        self,
        gen_path: str,
        disc_path: str,
    ) -> None:
        """Save the model"""
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)

    def load_model(self, gen_path: str, disc_path: str) -> None:
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
    def load_generator(self) -> nn.Module:
        """Load generator"""

    @abstractmethod
    def load_discriminator(self) -> nn.Module:
        """Load discriminator"""

    def _reset_timer(self) -> None:
        """Reset timer"""
        self.last_save_time = datetime.now()

    @staticmethod
    def _callback(callback: Optional[Callable], kwargs: Dict[str, Any]):
        """Call callback function if defined"""
        if callback is not None:
            callback(**kwargs)

    def _save_weights(self, gen_path: str, disc_path: str) -> None:
        """Save weights"""
        if (
            (datetime.now() - self.last_save_time).seconds / 60
        ) > config.SAVE_EVERY_MIN:
            self._reset_timer()
            self._save_model(gen_path, disc_path)

    def _save_model(self, generator_path: str, discriminator_path: str) -> None:
        """Save a model"""
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

    def _set_train_mode(self) -> None:
        """Set model to train mode"""
        self.generator.train()
        self.discriminator.train()

    def _init_optimizers(self, params: BaseTrainingParams, epochs: int) -> None:
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
            milestones=[epochs // 2, epochs // 4 * 3],
            gamma=0.1,
        )
        self.disc_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.disc_optimizer,
            milestones=[epochs // 2, epochs // 4 * 3],
            gamma=0.1,
        )
