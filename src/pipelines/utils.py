"""Utils"""
import logging

import torch


def init_device() -> str:
    """To find the GPU if it exists"""
    cuda = torch.cuda.is_available()
    if cuda:
        logging.info("Nvidia card available, running on GPU")
        logging.info(torch.cuda.get_device_name(0))
    else:
        logging.info("Nvidia card unavailable, running on CPU")
    return "cuda" if cuda else "cpu"
