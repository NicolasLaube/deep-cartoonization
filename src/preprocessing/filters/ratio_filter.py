"""Ration filter"""
from enum import Enum

import pandas as pd


class RatioFilterMode(Enum):
    """Ratio filter enum"""

    NO_FILTER = None
    FILTER_PORTRAIT = "portrait"
    FILTER_LANDSCAPE = "landscape"


def filter_ratio(
    df_images: pd.DataFrame, ratio_filter_mode: RatioFilterMode
) -> pd.DataFrame:
    """
    To filter images with a specific ratio (portrait, landscape...)
    """
    if ratio_filter_mode == RatioFilterMode.FILTER_PORTRAIT.value:
        return df_images[df_images["width"] / df_images["height"] > 1]
    if ratio_filter_mode == RatioFilterMode.FILTER_LANDSCAPE.value:
        return df_images[df_images["width"] / df_images["height"] < 1]

    return df_images
