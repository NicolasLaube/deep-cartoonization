"""Parameters for the model."""
# pylint: disable=C0103, R0913, R0902
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class Architecture(enum.Enum):
    """Architectures"""

    GANStyle = "Style Gan"
    GANUNet = "UNet GAN"
    GANFixed = "Fixed GAN"
    GANModular = "Modular GAN"
    GANAnime = "Anime GAN"


#############################
#### Architecture params ####
#############################


@dataclass
class ArchitectureParams:
    """Architecture params"""


@dataclass
class ArchitectureParamsNULL(ArchitectureParams):
    """Null architecture params"""


@dataclass
class ArchitectureParamsModular(ArchitectureParams):
    """Architecture params for Modular GAN"""

    nb_resnet_blocks: Optional[int] = 8
    nb_channels_picture: Optional[int] = 3
    nb_channels_cartoon: Optional[int] = 3
    nb_channels_1st_hidden_layer_gen: Optional[int] = 64
    nb_channels_1st_hidden_layer_disc: Optional[int] = 32


@dataclass
class TrainerParams:
    """Trainer parameters"""

    gen_lr: float = 1e-4
    disc_lr: float = 1e-4
    batch_size: int = 16
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    disc_beta1: float = 0.5
    disc_beta2: float = 0.999
    input_size: int = 256


@dataclass
class CartoonGanLossParameters:
    """Cartoon GAN loss parameters"""

    discriminator_loss: float
    generator_loss: float
    conditional_loss: float
