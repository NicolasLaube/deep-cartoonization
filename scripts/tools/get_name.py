from typing import Literal


def get_gen_name(epoch: int) -> str:
    """To get a generator name"""
    return f"trained_gen_{epoch}.pkl"


def get_disc_name(epoch: int) -> str:
    """To get a discriminator name"""
    return f"trained_disc_{epoch}.pkl"


def get_model_img_name(img_n: int, kind: Literal["picture", "cartoon"]) -> str:
    """To get an image name"""
    return f"image_{img_n}_{kind}.png"
