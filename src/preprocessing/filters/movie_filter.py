from typing import List
import pandas as pd
from src import config


def filter_movies(df_images: pd.DataFrame, movies: List[config.Movie]) -> pd.DataFrame:
    """
    To take images only in specific movies (portrait, landscape...)
    """
    return df_images[df_images["movie"].isin([movie.name for movie in movies])]
