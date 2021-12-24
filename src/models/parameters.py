from typing import Optional
from torch.optim import Optimizer
from dataclasses import dataclass


class CartoonGanParameters(dataclass):
    gen_lr: float
    disc_lr: float

    epochs: int
    nb_resnet_blocks: int
    batch_size: int
    conditional_lambda: float
