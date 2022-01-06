import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# from tqdm.auto import tqdm  # for notebooks

from src import config


def extract_frames(movie: str) -> None:
    """
    Extract information about all frames of a specific movie.
    """
    folder_path = os.path.join(config.CARTOONS_FOLDER, movie)
    frames_movie = [
        {"movie": movie, "name": frame, "path": os.path.join(folder_path, frame)}
        for frame in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, frame))
    ]
    return frames_movie


def extract_size(row):
    """
    Extract image size
    """
    image = Image.open(row["path"])
    return image.size


def create_all_cartoons_csv():
    """
    Create a csv file with all the cartoons, and information on them
    """
    # Create new `pandas` methods which use `tqdm` progress
    tqdm.pandas()
    frames_list = []
    for movie in config.MOVIES:
        frames_list.extend(extract_frames(movie.name))
    df = pd.DataFrame(frames_list)
    # use progress_apply instead of apply to get progress bar
    df[["width", "height"]] = df.progress_apply(
        extract_size, axis=1, result_type="expand"
    )
    df.to_csv(config.CARTOONS_ALL_CSV)


def create_all_pictures_csv():
    """
    Create a csv file with all the flickr pictures, and information on them
    """
    # Create new `pandas` methods which use `tqdm` progress
    tqdm.pandas()
    df = pd.read_csv(config.PICTURES_CSV)
    df = df.rename(columns={"image": "name"})
    df["path"] = df["name"].apply(
        lambda name: os.path.join(config.PICTURES_FOLDER, name)
    )
    # use progress_apply instead of apply to get progress bar
    df[["width", "height"]] = df.progress_apply(
        extract_size, axis=1, result_type="expand"
    )
    df.to_csv(config.PICTURES_ALL_CSV)
