from dataclasses import dataclass
import enum
from typing import Optional


class Architecture(enum.Enum):
    GANStyle = "Style Gan"
    GANUNet = "UNet GAN"
    GANFixed = "Fixed GAN"
    GANModular = "Modular GAN"


#############################
#### Architecture params ####
#############################


@dataclass
class ArchitectureParams:
    pass


@dataclass
class NULLArhcitectureParams(ArchitectureParams):
    pass


@dataclass
class ArchitectureParamsNULL(ArchitectureParams):
    pass


@dataclass
class ArchitectureParamsModular(ArchitectureParams):
    nb_resnet_blocks: Optional[int] = 8
    nb_channels_picture: Optional[int] = 3
    nb_channels_cartoon: Optional[int] = 3
    nb_channels_1st_hidden_layer_gen: Optional[int] = 64
    nb_channels_1st_hidden_layer_disc: Optional[int] = 32
