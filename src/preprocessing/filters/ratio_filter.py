from typing import NewType
from enum import Enum
import pandas as pd


class RatioFilterMode(Enum):
    NO_FILTER = None
    FILTER_PORTRAIT = "portrait"
    FILTER_LANDSCAPE = "landscape"


RatioFilterModes = NewType("RatioFilterModes", RatioFilterMode)


def filter_ratio(
    df_images: pd.DataFrame, ratio_filter_mode: RatioFilterModes
) -> pd.DataFrame:
    """
    To filter images with a specific ratio (portrait, landscape...)
    """
    if ratio_filter_mode == RatioFilterMode.NO_FILTER.value:
        return df_images
    elif ratio_filter_mode == RatioFilterMode.FILTER_PORTRAIT.value:
        return df_images[df_images["width"] / df_images["height"] > 1]
    elif ratio_filter_mode == RatioFilterMode.FILTER_LANDSCAPE.value:
        return df_images[df_images["width"] / df_images["height"] < 1]
