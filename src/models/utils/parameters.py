"""Parameters for the model."""
# pylint: disable=R0902, C0103
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class Architecture(enum.Enum):
    """Enum for the architecture"""

    GANStyle = "Style Gan"
    GANUNet = "UNet GAN"
    GANFixed = "Fixed GAN"
    GANModular = "Modular GAN"


#############################
#### Architecture params ####
#############################


@dataclass
class ArchitectureParams:
    """Parameters for the architecture"""


@dataclass
class ArchitectureParamsNULL(ArchitectureParams):
    """Null architecture params"""


@dataclass
class ArchitectureParamsModular(ArchitectureParams):
    """Architecture params for the modular GAN"""

    nb_resnet_blocks: Optional[int] = 8
    nb_channels_gen_input: Optional[int] = 3
    nb_channels_gen_output: Optional[int] = 3
    nb_channels_1st_hidden_layer_gen: Optional[int] = 64
    nb_channels_1st_hidden_layer_disc: Optional[int] = 32
    nb_channels_disc_input: Optional[int] = 3
    nb_channels_disc_output: Optional[int] = 1


@dataclass
class TrainerParams:
    """Parameters for training"""

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
    """Parameters for the CartoonGAN loss"""

    discriminator_loss: float
    generator_loss: float
    conditional_loss: float
