from typing import Optional
from torch.optim import Optimizer
from dataclasses import dataclass

@dataclass
class CartoonGanParameters():
    gen_lr: float
    disc_lr: float
    epochs: int
    batch_size: int
    conditional_lambda: float
    gen_beta1: float = 0.5
    gen_beta2: float = 0.999
    disc_beta1: float = 0.5
    disc_beta2: float = 0.999
