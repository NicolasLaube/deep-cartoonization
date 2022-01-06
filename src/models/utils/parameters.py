from dataclasses import dataclass
import enum


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


class Architecture(enum.Enum):
    UNET = "UNET architecture"
    MODULAR = "Modular architecture"
    FIXED = "Fixed architecture"
