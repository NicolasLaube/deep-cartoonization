from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import enum


@dataclass
class CartoonGanModelParameters:
    nb_resnet_blocks: Optional[int] = 8
    nb_channels_picture: Optional[int] = 3
    nb_channels_cartoon: Optional[int] = 3
    nb_channels_1st_hidden_layer_gen: Optional[int] = 64
    nb_channels_1st_hidden_layer_disc: Optional[int] = 32

    def null_object() -> CartoonGanModelParameters:
        return CartoonGanModelParameters(
            nb_resnet_blocks=-1,
            nb_channels_picture=-1,
            nb_channels_cartoon=-1,
            nb_channels_1st_hidden_layer_gen=-1,
            nb_channels_1st_hidden_layer_disc=-1,
        )


@dataclass
class CartoonGanParameters:
    gen_lr: float
    disc_lr: float
    epochs: int
    batch_size: int
    conditional_lambda: float
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    disc_beta1: float = 0.5
    disc_beta2: float = 0.999
    input_size: int = 256


@dataclass
class CartoonGanBaseParameters:
    gen_lr: float
    disc_lr: float
    batch_size: int
    conditional_lambda: float
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    disc_beta1: float = 0.5
    disc_beta2: float = 0.999
    input_size: int = 256


@dataclass
class CartoonGanLossParameters:
    discriminator_loss: float
    generator_loss: float
    conditional_loss: float


class CartoonGanArchitecture(enum.Enum):
    UNET = "UNET architecture"
    MODULAR = "Modular architecture"
    FIXED = "Fixed architecture"
