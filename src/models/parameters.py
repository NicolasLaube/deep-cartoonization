from typing import Optional
from torch.optim import Optimizer
from dataclasses import dataclass


@dataclass
class CartoonGanModelParameters:
    nb_resnet_blocks: int = 8
    nb_channels_picture: int = 3
    nb_channels_cartoon: int = 3
    nb_channels_1st_hidden_layer_gen: int = 64
    nb_channels_1st_hidden_layer_disc: int = 32


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
