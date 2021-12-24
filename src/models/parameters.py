from typing import Optional
from torch.optim import Optimizer
from dataclasses import dataclass


class CartoonGanParameters(dataclass):
    gen_lr: float
    disc_lr: float
    
    generator_optimizer: Optimizer
    discriminator_optimizer: Optimizer
    generator_scheduler: Optimizer
    discriminator_scheduler: Optimizer
    epochs: int
    nb_resnet_blocks: int
    batch_size: int
    conditional_lambda: float
