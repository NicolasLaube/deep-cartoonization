from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from src.pipelines.pipeline import ArchitectureParams


@dataclass
class TrainerParams:
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
