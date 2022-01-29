"""Low quality filter"""
from typing import Optional, Tuple

import pandas as pd


def filter_low_quality(
    df_images: pd.DataFrame, new_size: Optional[Tuple[int, int]]
) -> pd.DataFrame:
    """
    To filter images with a lower quality than the one required
    """
    if new_size is None:
        return df_images

    height, width = new_size[0], new_size[1]
    df_images = df_images[
        (df_images["height"] >= height) & (df_images["width"] >= width)
    ]
    return df_images
